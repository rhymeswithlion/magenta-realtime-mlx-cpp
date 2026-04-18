#include "magenta_realtime_mlx/depthformer/modules.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "magenta_realtime_mlx/nn_ops.h"
#include "mlx/fast.h"

namespace magenta_realtime_mlx::depthformer {

namespace mx = mlx::core;

namespace {

mx::Shape S(std::initializer_list<int> dims) {
  mx::Shape s;
  s.reserve(dims.size());
  for (int v : dims) s.push_back(static_cast<int32_t>(v));
  return s;
}

std::string join(std::string_view a, std::string_view b) {
  std::string out;
  out.reserve(a.size() + 1 + b.size());
  out.append(a);
  out.push_back('.');
  out.append(b);
  return out;
}

mx::array load_param(const WeightBundle& bundle, std::string_view key,
                     mx::Dtype dtype) {
  const auto& t = bundle.tensor(key);
  // Bundles are written as float32; cast lazily to the compute dtype. Keep
  // integer tensors (none expected here, but safe) untouched.
  if (t.dtype() == dtype) return t;
  return mx::astype(t, dtype);
}

}  // namespace

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

RMSNorm::RMSNorm(const WeightBundle& bundle, std::string_view prefix, int dim,
                 float eps, mx::Dtype dtype)
    : weight_(load_param(bundle, join(prefix, "weight"), dtype)),
      eps_arr_(eps),
      eps_(eps) {
  if (static_cast<int>(weight_.shape(0)) != dim) {
    throw std::runtime_error("RMSNorm weight shape mismatch");
  }
}

mx::array RMSNorm::operator()(const mx::array& x) const {
  // ``mx::fast::rms_norm`` is a single fused Metal kernel. Tested both the
  // hand-rolled manual chain (which ``mx::compile`` *can* fuse with
  // surrounding ops) and the fast kernel: the fast kernel won by ~110
  // ms/chunk on M2 Pro / bf16 (2155 vs 2265 ms). The manual chain expands
  // to ~9 ops; even after compile-fusion, the fast kernel's hand-tuned
  // layout dominates. RMSNorm runs ~13k times per chunk so this matters.
  // Note: this is a manual RMSNorm matching the Magenta RealTime
  // checkpoint convention (T5-style, no eps inside the rsqrt). MLX's
  // stock RMSNorm module uses a different formulation and is not
  // interchangeable.
  return mx::fast::rms_norm(x, weight_, eps_);
}

// ---------------------------------------------------------------------------
// Linear (bias-free)
// ---------------------------------------------------------------------------

Linear::Linear(const WeightBundle& bundle, std::string_view prefix,
               mx::Dtype dtype)
    : weight_t_(mx::transpose(load_param(bundle, join(prefix, "weight"), dtype))),
      // Materialise the (out, in) view once; it's a lazy stride flip so
      // sharing it across calls costs zero memory but saves an
      // ``mx::array`` construction in the hot weights-as-args dispatcher.
      weight_view_(mx::transpose(weight_t_, {1, 0})) {}

mx::array Linear::operator()(const mx::array& x) const {
  return mx::matmul(x, weight_t_);
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

Embedding::Embedding(const WeightBundle& bundle, std::string_view prefix,
                     mx::Dtype dtype)
    : weight_(load_param(bundle, join(prefix, "weight"), dtype)) {}

mx::array Embedding::operator()(const mx::array& ids) const {
  return mx::take(weight_, ids, /*axis=*/0);
}

// ---------------------------------------------------------------------------
// RelativePositionBias
// ---------------------------------------------------------------------------

RelativePositionBias::RelativePositionBias(const WeightBundle& bundle,
                                           std::string_view prefix,
                                           int num_heads, int num_buckets,
                                           int max_distance, bool bidirectional,
                                           mx::Dtype dtype)
    : num_heads_(num_heads),
      num_buckets_(num_buckets),
      max_distance_(max_distance),
      bidirectional_(bidirectional),
      bias_table_(load_param(
          bundle, join(prefix, "relative_attention_bias.weight"), dtype)),
      dtype_(dtype) {
  if (static_cast<int>(bias_table_.shape(0)) != num_buckets_ ||
      static_cast<int>(bias_table_.shape(1)) != num_heads_) {
    throw std::runtime_error("RelativePositionBias weight shape mismatch");
  }
}

namespace {

// T5-style bucketing -- mirrors the static method in modules.py exactly.
mx::array relative_position_bucket(const mx::array& relative_position,
                                   bool bidirectional, int num_buckets,
                                   int max_distance) {
  int eff_buckets = num_buckets;
  mx::array ret = mx::zeros(relative_position.shape(), mx::int32);
  mx::array n = mx::negative(relative_position);

  if (bidirectional) {
    eff_buckets /= 2;
    mx::array bumped =
        mx::astype(mx::where(mx::less(n, mx::array(0)), mx::array(eff_buckets),
                             mx::array(0)),
                   mx::int32);
    ret = mx::add(ret, bumped);
    n = mx::abs(n);
  } else {
    n = mx::maximum(n, mx::array(0));
  }

  const int max_exact = eff_buckets / 2;
  mx::array is_small = mx::less(n, mx::array(max_exact));

  const float log_ratio = std::log(static_cast<float>(max_distance) /
                                   static_cast<float>(max_exact));
  mx::array val_if_large =
      mx::astype(mx::add(mx::array(static_cast<float>(max_exact)),
                         mx::multiply(mx::divide(mx::log(mx::divide(
                                                     mx::astype(n, mx::float32),
                                                     mx::array(static_cast<float>(max_exact)))),
                                                 mx::array(log_ratio)),
                                      mx::array(static_cast<float>(eff_buckets - max_exact)))),
                 mx::int32);
  val_if_large = mx::minimum(val_if_large, mx::array(eff_buckets - 1));

  return mx::add(ret, mx::where(is_small, mx::astype(n, mx::int32), val_if_large));
}

}  // namespace

mx::array RelativePositionBias::operator()(int query_length, int key_length,
                                           int offset) const {
  mx::array ctx = mx::arange(offset, offset + query_length, mx::int32);
  mx::array mem = mx::arange(0, key_length, mx::int32);
  mx::array rel =
      mx::subtract(mx::expand_dims(mem, 0), mx::expand_dims(ctx, 1));  // (Q, K)
  mx::array buckets = relative_position_bucket(rel, bidirectional_,
                                               num_buckets_, max_distance_);
  // bias_table: (num_buckets, num_heads) -> gather rows by bucket idx.
  mx::array values = mx::take(bias_table_, buckets, /*axis=*/0);  // (Q, K, H)
  // Transpose to (H, Q, K) and add a leading batch dim.
  mx::array transposed = mx::transpose(values, {2, 0, 1});
  return mx::expand_dims(transposed, 0);  // (1, H, Q, K)
}

// ---------------------------------------------------------------------------
// causal_mask
// ---------------------------------------------------------------------------

mx::array causal_mask(int seq_len, mx::Dtype dtype) {
  // Triu of -inf above the diagonal. We build the mask in float32 first to
  // avoid generating ``-inf`` values in lower-precision dtypes (which may
  // surface as NaN in fast SDPA); cast at the end if requested.
  const float neg_inf = -std::numeric_limits<float>::infinity();
  mx::array neg = mx::full(S({seq_len, seq_len}), mx::array(neg_inf), mx::float32);
  mx::array masked = mx::triu(neg, /*k=*/1);
  mx::array out = mx::expand_dims(mx::expand_dims(masked, 0), 0);
  if (dtype != mx::float32) {
    out = mx::astype(out, dtype);
  }
  return out;
}

// ---------------------------------------------------------------------------
// GatedFeedForward (T5 1.1 GeGLU)
// ---------------------------------------------------------------------------

GatedFeedForward::GatedFeedForward(const WeightBundle& bundle,
                                   std::string_view prefix, int /*d_model*/,
                                   int /*d_ff*/, mx::Dtype dtype)
    : wi_0_(bundle, join(prefix, "wi_0"), dtype),
      wi_1_(bundle, join(prefix, "wi_1"), dtype),
      wo_(bundle, join(prefix, "wo"), dtype) {}

mx::array GatedFeedForward::operator()(const mx::array& x) const {
  mx::array a = wi_0_(x);
  mx::array b = wi_1_(x);
  return wo_(mx::multiply(gelu_approx(a), b));
}

// ---------------------------------------------------------------------------
// MultiHeadAttention
// ---------------------------------------------------------------------------

MultiHeadAttention::MultiHeadAttention(const WeightBundle& bundle,
                                       std::string_view prefix, int d_model,
                                       int num_heads, int d_kv, mx::Dtype dtype)
    : d_model_(d_model),
      num_heads_(num_heads),
      d_kv_(d_kv),
      q_proj_(bundle, join(prefix, "q_proj"), dtype),
      k_proj_(bundle, join(prefix, "k_proj"), dtype),
      v_proj_(bundle, join(prefix, "v_proj"), dtype),
      o_proj_(bundle, join(prefix, "o_proj"), dtype) {}

mx::array MultiHeadAttention::forward(
    const mx::array& x, const std::optional<mx::array>& context,
    const std::optional<mx::array>& mask,
    const std::optional<mx::array>& position_bias,
    const std::optional<KVCache>& past_kv, KVCache& out_kv) const {
  const int B = static_cast<int>(x.shape(0));
  const int T = static_cast<int>(x.shape(1));

  // Q always comes from the input. Reshape (B, T, H*D) -> (B, T, H, D) -> (B, H, T, D).
  mx::array q = mx::transpose(
      mx::reshape(q_proj_(x), S({B, T, num_heads_, d_kv_})), {0, 2, 1, 3});

  mx::array k = mx::array({0});
  mx::array v = mx::array({0});

  if (past_kv.has_value() && context.has_value()) {
    // Cross-attention with cached K/V — reuse the cache verbatim.
    k = past_kv->k;
    v = past_kv->v;
  } else {
    const mx::array& kv_input = context.has_value() ? *context : x;
    const int Sl = static_cast<int>(kv_input.shape(1));
    k = mx::transpose(
        mx::reshape(k_proj_(kv_input), S({B, Sl, num_heads_, d_kv_})),
        {0, 2, 1, 3});
    v = mx::transpose(
        mx::reshape(v_proj_(kv_input), S({B, Sl, num_heads_, d_kv_})),
        {0, 2, 1, 3});
    if (past_kv.has_value()) {
      // Self-attention with cache — concat along the sequence axis.
      k = mx::concatenate({past_kv->k, k}, /*axis=*/2);
      v = mx::concatenate({past_kv->v, v}, /*axis=*/2);
    }
  }

  out_kv = KVCache{k, v};

  std::optional<mx::array> attn_mask;
  if (position_bias.has_value() && mask.has_value()) {
    attn_mask = mx::add(*position_bias, *mask);
  } else if (position_bias.has_value()) {
    attn_mask = *position_bias;
  } else if (mask.has_value()) {
    attn_mask = *mask;
  }

  mx::array attn_out = mx::fast::scaled_dot_product_attention(
      q, k, v, /*scale=*/1.0f, /*mask_mode=*/"", attn_mask);
  // (B, H, T, D) -> (B, T, H, D) -> (B, T, H*D)
  attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}),
                         S({B, T, num_heads_ * d_kv_}));
  return o_proj_(attn_out);
}

mx::array MultiHeadAttention::forward_with_explicit_kv(
    const mx::array& x, const mx::array& k, const mx::array& v,
    const std::optional<mx::array>& mask,
    const std::optional<mx::array>& position_bias) const {
  const int B = static_cast<int>(x.shape(0));
  const int T = static_cast<int>(x.shape(1));

  mx::array q = mx::transpose(
      mx::reshape(q_proj_(x), S({B, T, num_heads_, d_kv_})), {0, 2, 1, 3});

  std::optional<mx::array> attn_mask;
  if (position_bias.has_value() && mask.has_value()) {
    attn_mask = mx::add(*position_bias, *mask);
  } else if (position_bias.has_value()) {
    attn_mask = *position_bias;
  } else if (mask.has_value()) {
    attn_mask = *mask;
  }

  mx::array attn_out = mx::fast::scaled_dot_product_attention(
      q, k, v, /*scale=*/1.0f, /*mask_mode=*/"", attn_mask);
  attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}),
                         S({B, T, num_heads_ * d_kv_}));
  return o_proj_(attn_out);
}

KVCache MultiHeadAttention::project_kv(const mx::array& context) const {
  const int B = static_cast<int>(context.shape(0));
  const int Sl = static_cast<int>(context.shape(1));
  mx::array k = mx::transpose(
      mx::reshape(k_proj_(context), S({B, Sl, num_heads_, d_kv_})),
      {0, 2, 1, 3});
  mx::array v = mx::transpose(
      mx::reshape(v_proj_(context), S({B, Sl, num_heads_, d_kv_})),
      {0, 2, 1, 3});
  return KVCache{k, v};
}

// ---------------------------------------------------------------------------
// EncoderLayer
// ---------------------------------------------------------------------------

EncoderLayer::EncoderLayer(const WeightBundle& bundle,
                           std::string_view prefix, int d_model, int num_heads,
                           int d_kv, int d_ff, mx::Dtype dtype)
    : pre_attn_norm_(bundle, join(prefix, "pre_attn_norm"), d_model, 1e-6f,
                     dtype),
      self_attn_(bundle, join(prefix, "self_attn"), d_model, num_heads, d_kv,
                 dtype),
      pre_ffn_norm_(bundle, join(prefix, "pre_ffn_norm"), d_model, 1e-6f, dtype),
      ffn_(bundle, join(prefix, "ffn"), d_model, d_ff, dtype) {}

mx::array EncoderLayer::operator()(
    const mx::array& x, const std::optional<mx::array>& mask) const {
  KVCache discard{mx::array(0.0f), mx::array(0.0f)};
  mx::array h = self_attn_.forward(pre_attn_norm_(x), /*context=*/std::nullopt,
                                   mask, /*position_bias=*/std::nullopt,
                                   /*past_kv=*/std::nullopt, discard);
  mx::array y = mx::add(x, h);
  mx::array f = ffn_(pre_ffn_norm_(y));
  return mx::add(y, f);
}

// ---------------------------------------------------------------------------
// DecoderLayer
// ---------------------------------------------------------------------------

DecoderLayer::DecoderLayer(const WeightBundle& bundle,
                           std::string_view prefix, int d_model, int num_heads,
                           int d_kv, int d_ff, bool has_cross_attention,
                           mx::Dtype dtype)
    : has_cross_attention_(has_cross_attention),
      pre_self_attn_norm_(bundle, join(prefix, "pre_self_attn_norm"), d_model,
                          1e-6f, dtype),
      self_attn_(bundle, join(prefix, "self_attn"), d_model, num_heads, d_kv,
                 dtype),
      pre_ffn_norm_(bundle, join(prefix, "pre_ffn_norm"), d_model, 1e-6f,
                    dtype),
      ffn_(bundle, join(prefix, "ffn"), d_model, d_ff, dtype) {
  if (has_cross_attention_) {
    pre_cross_attn_norm_.emplace(bundle, join(prefix, "pre_cross_attn_norm"),
                                 d_model, 1e-6f, dtype);
    cross_attn_.emplace(bundle, join(prefix, "cross_attn"), d_model, num_heads,
                        d_kv, dtype);
  }
}

mx::array DecoderLayer::forward(
    const mx::array& x, const std::optional<mx::array>& encoder_output,
    const std::optional<mx::array>& mask,
    const std::optional<mx::array>& position_bias,
    const std::optional<KVCache>& self_attn_cache,
    const std::optional<KVCache>& cross_attn_cache, KVCache& out_self_kv,
    std::optional<KVCache>& out_cross_kv) const {
  mx::array h = self_attn_.forward(pre_self_attn_norm_(x),
                                   /*context=*/std::nullopt, mask,
                                   position_bias, self_attn_cache, out_self_kv);
  mx::array y = mx::add(x, h);

  out_cross_kv.reset();
  if (has_cross_attention_ && encoder_output.has_value()) {
    KVCache new_cross{mx::array(0.0f), mx::array(0.0f)};
    mx::array h2 = cross_attn_->forward(
        (*pre_cross_attn_norm_)(y), encoder_output, /*mask=*/std::nullopt,
        /*position_bias=*/std::nullopt, cross_attn_cache, new_cross);
    out_cross_kv = new_cross;
    y = mx::add(y, h2);
  }

  mx::array f = ffn_(pre_ffn_norm_(y));
  return mx::add(y, f);
}

std::tuple<mx::array, mx::array, mx::array> DecoderLayer::forward_compiled(
    const mx::array& x, const mx::array& position_bias,
    const mx::array& self_K_in, const mx::array& self_V_in,
    const std::optional<KVCache>& cross_kv) const {
  // Self-attention with explicit cache: project Q/K/V from norm(x), concat
  // new K/V with the prior cache along axis=2, run SDPA. Bypasses the
  // ``MultiHeadAttention::forward`` host-side branch logic so the trace MLX
  // sees is a single fused op chain.
  const mx::array x_norm = pre_self_attn_norm_(x);
  const int B = static_cast<int>(x_norm.shape(0));
  const int T = static_cast<int>(x_norm.shape(1));
  const int H = self_attn_.num_heads();
  const int D = self_attn_.d_kv();

  // K/V projection from input (only the new T positions).
  // Reuse MultiHeadAttention::project_kv via a tiny inline equivalent so we
  // can also concat with the prior cache before handing to SDPA.
  KVCache new_self_kv = self_attn_.project_kv(x_norm);
  mx::array new_self_K = mx::concatenate({self_K_in, new_self_kv.k}, /*axis=*/2);
  mx::array new_self_V = mx::concatenate({self_V_in, new_self_kv.v}, /*axis=*/2);

  mx::array h = self_attn_.forward_with_explicit_kv(
      x_norm, new_self_K, new_self_V, /*mask=*/std::nullopt, position_bias);
  (void)B; (void)T; (void)H; (void)D;
  mx::array y = mx::add(x, h);

  if (has_cross_attention_ && cross_kv.has_value()) {
    mx::array h2 = cross_attn_->forward_with_explicit_kv(
        (*pre_cross_attn_norm_)(y), cross_kv->k, cross_kv->v,
        /*mask=*/std::nullopt, /*position_bias=*/std::nullopt);
    y = mx::add(y, h2);
  }

  mx::array f = ffn_(pre_ffn_norm_(y));
  mx::array out = mx::add(y, f);
  return {out, new_self_K, new_self_V};
}

KVCache DecoderLayer::precompute_cross_kv(
    const mx::array& encoder_output) const {
  if (!has_cross_attention_ || !cross_attn_.has_value()) {
    throw std::runtime_error(
        "precompute_cross_kv called on a layer without cross-attention");
  }
  return cross_attn_->project_kv(encoder_output);
}

void DecoderLayer::append_mlxfn_weights(std::vector<mx::array>& out) const {
  // Order MUST match the layout the ``.mlxfn`` source-graph bundles
  // were exported with -- the exporter and this dispatcher feed the
  // same flat list to the compiled function. See ``cpp/README.md`` for
  // how the published bundles are built. See modules.h doc comment for
  // the full list and rationale for excluding cross_attn k/v
  // projections.
  out.push_back(pre_self_attn_norm_.weight());
  out.push_back(self_attn_.q_proj().weight());
  out.push_back(self_attn_.k_proj().weight());
  out.push_back(self_attn_.v_proj().weight());
  out.push_back(self_attn_.o_proj().weight());
  if (has_cross_attention_) {
    if (!pre_cross_attn_norm_.has_value() || !cross_attn_.has_value()) {
      throw std::runtime_error(
          "append_mlxfn_weights: layer reports has_cross_attention but "
          "submodules are unset");
    }
    out.push_back(pre_cross_attn_norm_->weight());
    out.push_back(cross_attn_->q_proj().weight());
    out.push_back(cross_attn_->o_proj().weight());
  }
  out.push_back(pre_ffn_norm_.weight());
  out.push_back(ffn_.wi_0().weight());
  out.push_back(ffn_.wi_1().weight());
  out.push_back(ffn_.wo().weight());
}

}  // namespace magenta_realtime_mlx::depthformer

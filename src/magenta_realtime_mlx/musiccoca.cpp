#include "magenta_realtime_mlx/musiccoca.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "magenta_realtime_mlx/nn_ops.h"
#include "mlx/fast.h"

namespace magenta_realtime_mlx::musiccoca {

namespace mx = mlx::core;

namespace {

template <class... Ts>
mx::Shape S(Ts... vs) {
  return mx::Shape{static_cast<int32_t>(vs)...};
}

std::string join(std::string_view prefix, std::string_view suffix) {
  std::string out;
  out.reserve(prefix.size() + 1 + suffix.size());
  out.append(prefix.data(), prefix.size());
  out.push_back('.');
  out.append(suffix.data(), suffix.size());
  return out;
}

mx::array load_param(const WeightBundle& bundle, std::string_view key,
                     mx::Dtype dtype) {
  return mx::astype(bundle.tensor(std::string(key)), dtype);
}

// Softplus: log(1 + exp(x)), in its numerically stable form:
//   softplus(x) = max(x, 0) + log1p(exp(-|x|))
mx::array softplus(const mx::array& x) {
  mx::array zero = mx::array(0.0f);
  mx::array max_part = mx::maximum(x, zero);
  mx::array abs_x = mx::abs(x);
  mx::array neg_abs = mx::negative(abs_x);
  mx::array log1pexp = mx::log1p(mx::exp(neg_abs));
  return mx::add(max_part, log1pexp);
}

}  // namespace

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

LayerNorm::LayerNorm(const WeightBundle& bundle, std::string_view prefix,
                     int /*dim*/, float eps, mx::Dtype dtype)
    : weight_(load_param(bundle, join(prefix, "weight"), dtype)),
      bias_(load_param(bundle, join(prefix, "bias"), dtype)),
      eps_(eps) {}

mx::array LayerNorm::operator()(const mx::array& x) const {
  mx::array x_f32 = mx::astype(x, mx::float32);
  const int last_axis = static_cast<int>(x_f32.ndim()) - 1;
  mx::array mean =
      mx::mean(x_f32, /*axis=*/last_axis, /*keepdims=*/true);
  mx::array centered = mx::subtract(x_f32, mean);
  mx::array var =
      mx::mean(mx::multiply(centered, centered), last_axis, /*keepdims=*/true);
  mx::array norm = mx::multiply(centered, mx::rsqrt(mx::add(var, mx::array(eps_))));
  mx::array y = mx::astype(norm, x.dtype());
  return mx::add(mx::multiply(y, weight_), bias_);
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

Linear::Linear(const WeightBundle& bundle, std::string_view prefix,
               bool has_bias, mx::Dtype dtype)
    : weight_t_(mx::transpose(load_param(bundle, join(prefix, "weight"), dtype))) {
  if (has_bias) {
    bias_ = load_param(bundle, join(prefix, "bias"), dtype);
  }
}

mx::array Linear::operator()(const mx::array& x) const {
  mx::array y = mx::matmul(x, weight_t_);
  if (bias_.has_value()) y = mx::add(y, *bias_);
  return y;
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

Embedding::Embedding(const WeightBundle& bundle, std::string_view prefix,
                     mx::Dtype dtype)
    : weight_(load_param(bundle, join(prefix, "weight"), dtype)) {}

mx::array Embedding::operator()(const mx::array& ids) const {
  // take: gather rows from weight_ using integer indices.
  return mx::take(weight_, ids, /*axis=*/0);
}

// ---------------------------------------------------------------------------
// CoCaMHA
// ---------------------------------------------------------------------------

CoCaMHA::CoCaMHA(const WeightBundle& bundle, std::string_view prefix,
                 int /*d_model*/, int num_heads, int d_head, bool per_dim_scale,
                 float atten_logit_cap, mx::Dtype dtype)
    : num_heads_(num_heads),
      d_head_(d_head),
      atten_logit_cap_(atten_logit_cap),
      has_per_dim_scale_(per_dim_scale),
      q_proj_(bundle, join(prefix, "q_proj"), /*has_bias=*/true, dtype),
      k_proj_(bundle, join(prefix, "k_proj"), /*has_bias=*/true, dtype),
      v_proj_(bundle, join(prefix, "v_proj"), /*has_bias=*/true, dtype),
      o_proj_(bundle, join(prefix, "o_proj"), /*has_bias=*/true, dtype) {
  if (per_dim_scale) {
    per_dim_scale_ = load_param(bundle, join(prefix, "_per_dim_scale"), dtype);
  }
}

mx::array CoCaMHA::operator()(
    const mx::array& query, const mx::array& key, const mx::array& value,
    const std::optional<mx::array>& additive_mask) const {
  const int B = static_cast<int>(query.shape(0));
  const int Sq = static_cast<int>(query.shape(1));
  const int Sk = static_cast<int>(key.shape(1));
  const int H = num_heads_;
  const int D = d_head_;

  mx::array q = q_proj_(query);  // (B, Sq, H*D)
  mx::array k = k_proj_(key);
  mx::array v = v_proj_(value);

  // (B, S, H*D) -> (B, H, S, D) via reshape + transpose.
  q = mx::transpose(mx::reshape(q, S(B, Sq, H, D)), {0, 2, 1, 3});
  k = mx::transpose(mx::reshape(k, S(B, Sk, H, D)), {0, 2, 1, 3});
  v = mx::transpose(mx::reshape(v, S(B, Sk, H, D)), {0, 2, 1, 3});

  if (has_per_dim_scale_) {
    // scale = softplus(per_dim_scale) / log(2)
    mx::array sp = softplus(*per_dim_scale_);
    mx::array scale = mx::divide(sp, mx::array(std::log(2.0f)));
    q = mx::multiply(q, scale);
  }

  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(D));
  mx::array attn = mx::multiply(
      mx::matmul(q, mx::transpose(k, {0, 1, 3, 2})), mx::array(attn_scale));

  if (atten_logit_cap_ > 0.0f) {
    mx::array cap = mx::array(atten_logit_cap_);
    attn = mx::multiply(cap, mx::tanh(mx::divide(attn, cap)));
  }

  if (additive_mask.has_value()) {
    attn = mx::add(attn, *additive_mask);
  }

  attn = mx::softmax(attn, /*axis=*/-1);
  mx::array out = mx::matmul(attn, v);  // (B, H, Sq, D)
  out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), S(B, Sq, H * D));
  return o_proj_(out);
}

// ---------------------------------------------------------------------------
// TransformerLayer
// ---------------------------------------------------------------------------

TransformerLayer::TransformerLayer(const WeightBundle& bundle,
                                   std::string_view prefix, int d_model,
                                   int num_heads, int d_head, int d_ff,
                                   mx::Dtype dtype)
    : attn_ln_(bundle, join(prefix, "attn_ln"), d_model, 1e-5f, dtype),
      attn_(bundle, join(prefix, "attn"), d_model, num_heads, d_head,
            /*per_dim_scale=*/false, kAttenLogitCap, dtype),
      ffn_ln_(bundle, join(prefix, "ffn_ln"), d_model, 1e-5f, dtype),
      ffn_w1_(bundle, join(prefix, "ffn_w1"), /*has_bias=*/true, dtype),
      ffn_w2_(bundle, join(prefix, "ffn_w2"), /*has_bias=*/true, dtype) {}

mx::array TransformerLayer::operator()(
    const mx::array& x, const std::optional<mx::array>& mask) const {
  mx::array h = attn_ln_(x);
  h = attn_(h, h, h, mask);
  mx::array y = mx::add(x, h);
  h = ffn_ln_(y);
  h = ffn_w1_(h);
  h = gelu(h);
  h = ffn_w2_(h);
  return mx::add(y, h);
}

// ---------------------------------------------------------------------------
// AttentionPooler
// ---------------------------------------------------------------------------

AttentionPooler::AttentionPooler(const WeightBundle& bundle,
                                 std::string_view prefix, int d_model,
                                 int num_heads, int d_head, mx::Dtype dtype)
    : query_(load_param(bundle, join(prefix, "query"), dtype)),
      ln_(bundle, join(prefix, "ln"), d_model, 1e-5f, dtype),
      attn_(bundle, join(prefix, "attn"), d_model, num_heads, d_head,
            /*per_dim_scale=*/true, /*atten_logit_cap=*/0.0f, dtype) {}

mx::array AttentionPooler::operator()(const mx::array& encoder_output) const {
  const int B = static_cast<int>(encoder_output.shape(0));
  const int D = static_cast<int>(encoder_output.shape(2));
  mx::array q = mx::broadcast_to(query_, S(B, 1, D));
  mx::array normed = ln_(encoder_output);
  mx::array pooled = attn_(q, normed, normed);  // (B, 1, D)
  return mx::squeeze(pooled, /*axis=*/1);
}

// ---------------------------------------------------------------------------
// MusicCoCaEncoder
// ---------------------------------------------------------------------------

namespace {

std::vector<TransformerLayer> build_layers(const WeightBundle& bundle,
                                           std::string_view stack_prefix,
                                           int num_layers, int d_model,
                                           int num_heads, int d_head, int d_ff,
                                           mx::Dtype dtype) {
  std::vector<TransformerLayer> out;
  out.reserve(num_layers);
  for (int i = 0; i < num_layers; ++i) {
    std::string prefix =
        std::string(stack_prefix) + "." + std::to_string(i);
    out.emplace_back(bundle, prefix, d_model, num_heads, d_head, d_ff, dtype);
  }
  return out;
}

}  // namespace

MusicCoCaEncoder::MusicCoCaEncoder(const WeightBundle& bundle,
                                   const MusicCoCaConfig& config,
                                   mx::Dtype dtype)
    : config_(config),
      dtype_(dtype),
      patch_proj_(bundle, "patch_proj", /*has_bias=*/true, dtype),
      pos_embedding_(load_param(bundle, "pos_embedding", dtype)),
      token_embedding_(bundle, "token_embedding", dtype),
      audio_layers_(build_layers(bundle, "audio_layers", config.num_layers,
                                 config.d_model, config.num_heads,
                                 config.d_head, config.d_ff, dtype)),
      audio_exit_ln_(bundle, "audio_exit_ln", config.d_model, 1e-5f, dtype),
      text_layers_(build_layers(bundle, "text_layers", config.num_layers,
                                config.d_model, config.num_heads,
                                config.d_head, config.d_ff, dtype)),
      text_exit_ln_(bundle, "text_exit_ln", config.d_model, 1e-5f, dtype),
      music_pooler_(bundle, "music_pooler", config.d_model, config.num_heads,
                    config.pooler_d_head, dtype),
      text_pooler_(bundle, "text_pooler", config.d_model, config.num_heads,
                   config.pooler_d_head, dtype) {}

mx::array MusicCoCaEncoder::embed_text(const mx::array& token_ids,
                                       const mx::array& padding_mask) const {
  if (token_ids.ndim() != 2) {
    throw std::invalid_argument("embed_text: token_ids must be (B, L)");
  }
  mx::array x = token_embedding_(token_ids);
  x = mx::astype(x, dtype_);
  x = mx::multiply(
      x, mx::array(std::sqrt(static_cast<float>(config_.d_model))));

  // padding_mask is 1.0 where padded, 0.0 where real. Expand to (B, 1, 1, L)
  // and scale by -1e9 to zero-out the attention logits for those keys.
  mx::array pad = mx::astype(padding_mask, dtype_);
  pad = mx::expand_dims(pad, /*axis=*/1);
  pad = mx::expand_dims(pad, /*axis=*/2);
  mx::array attn_mask = mx::multiply(pad, mx::array(-1e9f));

  for (const auto& layer : text_layers_) {
    x = layer(x, attn_mask);
  }
  x = text_exit_ln_(x);
  return text_pooler_(x);
}

mx::array MusicCoCaEncoder::embed_audio(const mx::array& log_mel) const {
  if (log_mel.ndim() != 3) {
    throw std::invalid_argument("embed_audio: log_mel must be (B, 992, 128)");
  }
  const int B = static_cast<int>(log_mel.shape(0));
  mx::array x = mx::astype(log_mel, dtype_);
  x = mx::reshape(x, S(B, config_.num_patches, config_.patch_dim));
  x = patch_proj_(x);
  x = mx::add(x, pos_embedding_);
  for (const auto& layer : audio_layers_) {
    x = layer(x, std::nullopt);
  }
  x = audio_exit_ln_(x);
  return music_pooler_(x);
}

}  // namespace magenta_realtime_mlx::musiccoca

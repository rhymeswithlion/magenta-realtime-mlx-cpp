#pragma once

// Depthformer transformer building blocks (RMSNorm, Linear, attention,
// gated FFN, encoder/decoder layers, sinusoidal position embeddings).
// Everything except the high-level chunk-decode loop; the modules
// here support both the encoder forward pass and the depth/temporal layer
// forwards used by ``depth_forward_full`` / ``temporal_step``.
//
// Conventions:
//   * Linear layers store the weight transposed to ``(in, out)`` so that
//     ``y = mx::matmul(x, weight)`` matches the standard transposed-weight
//     ``x @ w.T`` convention without a per-call transpose.
//   * Embedding tables remain in their bundled ``(vocab, dim)`` layout
//     and are gathered with ``mx::take(weight, ids, axis=0)``.
//   * KV cache is a ``std::optional<KVCache>`` per attention call; values
//     have shape ``(B, H, T, d_kv)``.

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "magenta_realtime_mlx/weights.h"
#include "mlx/mlx.h"

namespace magenta_realtime_mlx::depthformer {

struct KVCache {
  mlx::core::array k;  // (B, H, T, d_kv)
  mlx::core::array v;  // (B, H, T, d_kv)
};

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

class RMSNorm {
 public:
  RMSNorm(const WeightBundle& bundle, std::string_view prefix, int dim,
          float eps, mlx::core::Dtype dtype);
  mlx::core::array operator()(const mlx::core::array& x) const;

  // Raw weight tensor (shape ``(dim,)``). Exposed for the weights-as-args
  // ``.mlxfn`` dispatcher (see ``Depthformer::append_*_step_weights``) so
  // the imported function can be called with externally supplied weights.
  const mlx::core::array& weight() const noexcept { return weight_; }

 private:
  mlx::core::array weight_;  // (dim,)
  // ``eps`` cached as both a 0-d fp32 array (for any future fallback that
  // wants an array operand) and a plain float (passed to
  // ``mx::fast::rms_norm``). Today only ``eps_`` is used.
  mlx::core::array eps_arr_;
  float eps_;
};

class Linear {
 public:
  // Loads ``prefix.weight`` (shape (out, in)) and stores it pre-transposed
  // to (in, out). No bias support — every Depthformer linear is bias-free.
  Linear(const WeightBundle& bundle, std::string_view prefix,
         mlx::core::Dtype dtype);
  mlx::core::array operator()(const mlx::core::array& x) const;

  // Returns the weight in the standard ``(out, in)`` convention. Internally
  // we store the transpose (``(in, out)``) to avoid a per-call transpose
  // during ``mx::matmul``, but the ``.mlxfn`` bundles embed the
  // ``x @ weight.T`` trace and therefore expect callers to pass the
  // ``(out, in)`` layout. The transpose is materialised once at
  // construction time as a lazy strided view (``mx::transpose`` is just a
  // stride flip in MLX -- no kernel launch, no extra buffer); subsequent
  // calls return that cached view by reference. Caching matters in the
  // weights-as-args ``.mlxfn`` dispatcher hot path: a chunk pushes
  // ~42k weight references through ``Linear::weight()``, and avoiding a
  // fresh ``mx::array`` (smart-pointer + transpose-spec construction)
  // per call shaves measurable CPU dispatch time.
  const mlx::core::array& weight() const noexcept { return weight_view_; }

 private:
  mlx::core::array weight_t_;  // (in, out)
  mlx::core::array weight_view_;  // lazy transpose of weight_t_, (out, in)
};

class Embedding {
 public:
  Embedding(const WeightBundle& bundle, std::string_view prefix,
            mlx::core::Dtype dtype);
  // ``ids`` shape (B, T) int32 -> (B, T, dim).
  mlx::core::array operator()(const mlx::core::array& ids) const;

 private:
  mlx::core::array weight_;  // (vocab, dim) -- kept in original dtype
};

class RelativePositionBias {
 public:
  RelativePositionBias(const WeightBundle& bundle, std::string_view prefix,
                       int num_heads, int num_buckets, int max_distance,
                       bool bidirectional, mlx::core::Dtype dtype);

  // Returns (1, num_heads, query_length, key_length) additive bias.
  // ``offset`` allows asking for a single-frame query (T=1) at an
  // arbitrary position, matching ``temporal_rel_pos(1, frame_idx+1, offset=frame_idx)``.
  mlx::core::array operator()(int query_length, int key_length,
                              int offset = 0) const;

  int num_heads() const noexcept { return num_heads_; }

 private:
  int num_heads_;
  int num_buckets_;
  int max_distance_;
  bool bidirectional_;
  mlx::core::array bias_table_;  // (num_buckets, num_heads)
  mlx::core::Dtype dtype_;
};

// Causal additive mask: 0 below the diagonal, -inf above. Returns
// ``(1, 1, T, T)`` in the requested dtype.
mlx::core::array causal_mask(int seq_len, mlx::core::Dtype dtype);

class GatedFeedForward {
 public:
  GatedFeedForward(const WeightBundle& bundle, std::string_view prefix,
                   int d_model, int d_ff, mlx::core::Dtype dtype);
  mlx::core::array operator()(const mlx::core::array& x) const;

  // Submodule accessors for the weights-as-args ``.mlxfn`` dispatcher.
  const Linear& wi_0() const noexcept { return wi_0_; }
  const Linear& wi_1() const noexcept { return wi_1_; }
  const Linear& wo() const noexcept { return wo_; }

 private:
  Linear wi_0_;
  Linear wi_1_;
  Linear wo_;
};

class MultiHeadAttention {
 public:
  MultiHeadAttention(const WeightBundle& bundle, std::string_view prefix,
                     int d_model, int num_heads, int d_kv,
                     mlx::core::Dtype dtype);

  // ``x``        : (B, T, d_model). Used as queries always; doubles as KV
  //                source when ``context`` is empty (self-attention).
  // ``context``  : (B, S, d_model) for cross-attention. ``std::nullopt`` for
  //                self-attention.
  // ``mask``     : optional additive mask broadcastable to (B, H, T, K).
  // ``position_bias``: optional additive bias (typically (1, H, T, K)).
  // ``past_kv``  : optional KV cache. For self-attention we concatenate the
  //                fresh K/V along the sequence axis. For cross-attention we
  //                reuse the cached K/V verbatim (matching Python).
  mlx::core::array forward(
      const mlx::core::array& x,
      const std::optional<mlx::core::array>& context,
      const std::optional<mlx::core::array>& mask,
      const std::optional<mlx::core::array>& position_bias,
      const std::optional<KVCache>& past_kv,
      KVCache& out_kv) const;

  // Compiled-friendly variant: runs SDPA with caller-provided K/V tensors and
  // skips the K/V projection. Used when K/V is already cached (cross-attn
  // after the first temporal step) or when self-attn K/V has been concatenated
  // outside this call. Pure ``arrays -> array`` shape -- safe inside
  // ``mx::compile``.
  mlx::core::array forward_with_explicit_kv(
      const mlx::core::array& x,
      const mlx::core::array& k,
      const mlx::core::array& v,
      const std::optional<mlx::core::array>& mask,
      const std::optional<mlx::core::array>& position_bias) const;

  // Project ``context`` through ``k_proj`` / ``v_proj`` and reshape into the
  // attention layout ``(B, H, S, d_kv)``. Used to pre-compute cross K/V once
  // per chunk so the autoregressive temporal loop never re-runs the
  // encoder-output projection.
  KVCache project_kv(const mlx::core::array& context) const;

  int num_heads() const noexcept { return num_heads_; }
  int d_kv() const noexcept { return d_kv_; }

  // Submodule accessors for the weights-as-args ``.mlxfn`` dispatcher.
  const Linear& q_proj() const noexcept { return q_proj_; }
  const Linear& k_proj() const noexcept { return k_proj_; }
  const Linear& v_proj() const noexcept { return v_proj_; }
  const Linear& o_proj() const noexcept { return o_proj_; }

 private:
  int d_model_;
  int num_heads_;
  int d_kv_;
  Linear q_proj_;
  Linear k_proj_;
  Linear v_proj_;
  Linear o_proj_;
};

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

class EncoderLayer {
 public:
  EncoderLayer(const WeightBundle& bundle, std::string_view prefix,
               int d_model, int num_heads, int d_kv, int d_ff,
               mlx::core::Dtype dtype);
  mlx::core::array operator()(
      const mlx::core::array& x,
      const std::optional<mlx::core::array>& mask = std::nullopt) const;

 private:
  RMSNorm pre_attn_norm_;
  MultiHeadAttention self_attn_;
  RMSNorm pre_ffn_norm_;
  GatedFeedForward ffn_;
};

struct DecoderLayerCache {
  std::optional<KVCache> self_attn;
  std::optional<KVCache> cross_attn;
};

class DecoderLayer {
 public:
  DecoderLayer(const WeightBundle& bundle, std::string_view prefix,
               int d_model, int num_heads, int d_kv, int d_ff,
               bool has_cross_attention, mlx::core::Dtype dtype);

  mlx::core::array forward(
      const mlx::core::array& x,
      const std::optional<mlx::core::array>& encoder_output,
      const std::optional<mlx::core::array>& mask,
      const std::optional<mlx::core::array>& position_bias,
      const std::optional<KVCache>& self_attn_cache,
      const std::optional<KVCache>& cross_attn_cache,
      KVCache& out_self_kv,
      std::optional<KVCache>& out_cross_kv) const;

  // Compiled-friendly forward (pure ``arrays -> arrays``). Intended for use
  // inside ``mx::compile`` autoregressive bodies. ``self_K_in`` / ``self_V_in``
  // are the prior self-attention K/V (shape ``(B, H, T, d_kv)``); the new
  // K/V is concatenated along axis=2 internally and returned. ``cross_kv``
  // must be supplied (already projected from ``encoder_output``) for layers
  // with cross-attention; depth layers pass ``std::nullopt``. Returns
  // ``(layer_output, new_self_K, new_self_V)``.
  std::tuple<mlx::core::array, mlx::core::array, mlx::core::array>
  forward_compiled(const mlx::core::array& x,
                   const mlx::core::array& position_bias,
                   const mlx::core::array& self_K_in,
                   const mlx::core::array& self_V_in,
                   const std::optional<KVCache>& cross_kv) const;

  // Pre-project the encoder output through this layer's cross-attention
  // K/V projections. Throws if this layer doesn't have cross-attention.
  KVCache precompute_cross_kv(
      const mlx::core::array& encoder_output) const;

  bool has_cross_attention() const noexcept { return has_cross_attention_; }

  // Append this layer's weights, in canonical mlxfn order, to ``out``.
  // The ordering MUST stay in lockstep with the way the published
  // ``.mlxfn`` bundles were exported -- the exporter and the C++
  // dispatcher feed the same list to the
  // imported function and any drift produces a shape error or, worse,
  // silently transposed weights.
  //
  // For depth layers (no cross-attn) this emits 9 weights:
  //   pre_self_attn_norm,
  //   self_attn.{q,k,v,o}_proj,
  //   pre_ffn_norm,
  //   ffn.{wi_0, wi_1, wo}
  //
  // For temporal layers (has cross-attn) this emits 12 weights:
  //   pre_self_attn_norm,
  //   self_attn.{q,k,v,o}_proj,
  //   pre_cross_attn_norm,
  //   cross_attn.{q,o}_proj,           # k/v intentionally excluded
  //   pre_ffn_norm,
  //   ffn.{wi_0, wi_1, wo}
  //
  // Cross-attn k_proj/v_proj are omitted because the temporal_step
  // ``.mlxfn`` is traced with precomputed cross-attn K/V passed in as
  // separate inputs; the layer body short-circuits before touching
  // those projections, so the trace never references them.
  void append_mlxfn_weights(std::vector<mlx::core::array>& out) const;

 private:
  bool has_cross_attention_;
  RMSNorm pre_self_attn_norm_;
  MultiHeadAttention self_attn_;
  std::optional<RMSNorm> pre_cross_attn_norm_;
  std::optional<MultiHeadAttention> cross_attn_;
  RMSNorm pre_ffn_norm_;
  GatedFeedForward ffn_;
};

}  // namespace magenta_realtime_mlx::depthformer

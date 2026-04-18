#pragma once

// High-level Depthformer model: encoder + decoder, ``encode`` /
// ``temporal_step`` / ``depth_step`` entry points used by the streaming
// chunk loop. Loads weights from a ``WeightBundle`` (see
// ``weights.h``) and exposes optional fast paths backed by the
// published ``.mlxfn`` source-graph bundles.
//
// Stage 4 implements:
//   * Configuration (base / large)
//   * Sinusoidal position embeddings (computed once at construction)
//   * ``encode``                   — full encoder forward
//   * ``depth_forward_full``       — depth decoder, single pass with causal
//                                    mask. Used directly by speculative
//                                    decoding (Stage 5) and as the parity
//                                    target for this stage.
//
// Stage 5 will add ``temporal_step`` / ``depth_step`` (KV cache + sampling).

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "magenta_realtime_mlx/depthformer/modules.h"
#include "magenta_realtime_mlx/weights.h"
#include "mlx/mlx.h"

namespace magenta_realtime_mlx::depthformer {

struct DepthformerConfig {
  int vocab_size = 29824;
  int d_model = 768;
  int d_kv = 64;
  int d_ff = 2048;
  int num_heads = 12;

  int num_encoder_layers = 12;
  int num_temporal_layers = 20;
  int num_depth_layers = 4;

  int max_encoder_length = 1006;
  int rvq_depth = 16;

  int temporal_num_buckets = 128;
  int temporal_max_distance = 128;
  int depth_num_buckets = 16;
  int depth_max_distance = 16;

  static DepthformerConfig base();
  static DepthformerConfig large();
};

using LayerCache = std::pair<KVCache, std::optional<KVCache>>;

class Depthformer {
 public:
  Depthformer(const WeightBundle& bundle, const DepthformerConfig& config,
              mlx::core::Dtype dtype);

  const DepthformerConfig& config() const noexcept { return config_; }
  mlx::core::Dtype dtype() const noexcept { return dtype_; }

  // ---- Encoder forward ---------------------------------------------------
  mlx::core::array encode(const mlx::core::array& encoder_input_tokens) const;

  // ---- Optional ``mx::compile`` wrapping for the autoregressive hot paths -
  // Mirrors the upstream ``Depthformer.compile_for_inference`` /
  // ``optimize()`` step. When called, future ``encode`` / ``temporal_step`` /
  // ``depth_step`` calls dispatch through ``mx::compile``-wrapped versions
  // that fuse op chains into single Metal kernels (~2x autoregressive
  // throughput on Apple Silicon). Without it the eager paths still work --
  // tests and fixture comparisons rely on bit-for-bit eager output.
  void compile_for_inference(bool compile_decode_steps = true);

  // ---- Single-pass depth decoder forward (no KV cache) --------------------
  // ``token_embeddings`` shape (B, K, d_model). Returns logits (B, K, vocab).
  mlx::core::array depth_forward_full(
      const mlx::core::array& token_embeddings,
      const std::optional<mlx::core::array>& position_bias = std::nullopt,
      bool causal = true) const;

  // ---- Autoregressive step API (KV cache) ---------------------------------
  // Empty caches sized to the right number of layers. Each entry's
  // ``self_attn`` is initialised as nullopt; callers (or the layer) treat
  // nullopt as "no prior K/V".
  std::vector<LayerCache> empty_temporal_cache() const;
  std::vector<LayerCache> empty_depth_cache() const;

  // Pre-computed relative-position biases for a chunk so each step doesn't
  // re-bucket. Indexing matches frame_idx / depth_idx (one entry per step).
  std::vector<mlx::core::array> precompute_temporal_biases(int max_frames) const;
  std::vector<mlx::core::array> precompute_depth_biases(int max_depth) const;

  // Single temporal decoder step. ``frame_embedding`` shape (B, 1, d_model).
  // Updates ``cache`` in place; returns the layer-stack output (B, 1, d_model).
  mlx::core::array temporal_step(
      const mlx::core::array& frame_embedding,
      const mlx::core::array& encoder_output,
      std::vector<LayerCache>& cache, int frame_idx,
      const std::optional<mlx::core::array>& position_bias = std::nullopt) const;

  // Single depth decoder step. ``token_embedding`` shape (B, 1, d_model).
  // Updates ``cache`` in place; returns logits (B, vocab) (final dim squeezed).
  mlx::core::array depth_step(
      const mlx::core::array& token_embedding,
      std::vector<LayerCache>& cache, int depth_idx,
      const std::optional<mlx::core::array>& position_bias = std::nullopt) const;

  // Helpers exposed mainly for tests / Stage 5 callers.
  const Embedding& token_embedding() const noexcept { return token_embedding_; }
  mlx::core::array position_embedding(int seq_len) const;

  RelativePositionBias& temporal_rel_pos() noexcept { return *temporal_rel_pos_; }
  RelativePositionBias& depth_rel_pos() noexcept { return *depth_rel_pos_; }
  const RelativePositionBias& temporal_rel_pos() const noexcept { return *temporal_rel_pos_; }
  const RelativePositionBias& depth_rel_pos() const noexcept { return *depth_rel_pos_; }

  const std::vector<EncoderLayer>& encoder_layers() const noexcept { return encoder_layers_; }
  const std::vector<DecoderLayer>& temporal_layers() const noexcept { return temporal_layers_; }
  const std::vector<DecoderLayer>& depth_layers() const noexcept { return depth_layers_; }

  // ---- Weights-as-args ``.mlxfn`` weight collectors -----------------------
  // Append the canonical-order weight arrays a pre-traced
  // ``depth_step`` / ``temporal_step`` ``.mlxfn`` (format_version >= 2)
  // expects as trailing inputs. The full input vector for those bundles
  // is ``[token_emb, position_bias, ...kvs..., ...weights...]``; the
  // dispatch sites in ``model.cpp`` first push the kv inputs and then
  // call one of these to push the weights.
  //
  // The arrays returned are the same lazy ``mx::array`` handles that the
  // model uses for its eager forward path -- no copy, no transpose
  // kernel -- so calling these is effectively free (a few dozen smart
  // pointer copies).
  //
  // Counts (base config):
  //   depth_step:    L_depth * 9 + 2 = 38   (4 layers + decoder_norm + lm_head)
  //   temporal_step: L_temporal * 12 = 240  (20 layers, no tail)
  //
  // The lists are precomputed once at construction (see
  // ``rebuild_mlxfn_weight_caches_``) and the ``append_*`` methods just do
  // a bulk insert of cached references. A chunk pushes ~42k weight refs
  // through these methods (50 frames * 240 + 800 depth-steps * 38), so
  // avoiding per-call layer-tree traversal + per-weight ``mx::array``
  // construction is a measurable C++-side win.
  void append_depth_step_weights(std::vector<mlx::core::array>& out) const;
  void append_temporal_step_weights(std::vector<mlx::core::array>& out) const;

  // Convenience accessors for ``decoder_norm`` and ``lm_head`` (mainly
  // used by the depth_step collector but exposed for tests).
  const RMSNorm& decoder_norm() const noexcept { return decoder_norm_; }
  const Linear& lm_head() const noexcept { return lm_head_; }

 private:
  DepthformerConfig config_;
  mlx::core::Dtype dtype_;
  Embedding token_embedding_;
  mlx::core::array position_embedding_;  // (max_encoder_length, d_model) fp32
  std::vector<EncoderLayer> encoder_layers_;
  RMSNorm encoder_norm_;
  std::unique_ptr<RelativePositionBias> temporal_rel_pos_;
  std::vector<DecoderLayer> temporal_layers_;
  std::unique_ptr<RelativePositionBias> depth_rel_pos_;
  std::vector<DecoderLayer> depth_layers_;
  RMSNorm decoder_norm_;
  Linear lm_head_;

  // ``mx::compile``-wrapped versions of the hot paths. Populated by
  // ``compile_for_inference``; nullopt while running eager. Each one takes a
  // flat ``vector<array>`` of inputs and returns a flat ``vector<array>`` of
  // outputs (MLX's C++ compile API doesn't support nested structures).
  using CompiledFn = std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&)>;
  std::optional<CompiledFn> compiled_encode_;
  std::optional<CompiledFn> compiled_temporal_step_;
  std::optional<CompiledFn> compiled_depth_step_;

  // Optional per-cache-length compiled depth_step / temporal_step functions
  // loaded from pre-traced ``.mlxfn`` files. Index = cache_length (=
  // ``depth_idx`` / ``frame_idx`` when the corresponding step is invoked).
  // Index 0 is left unset; the first step at each level stays on the eager
  // path because no prior cache exists yet (and for ``temporal_step`` the
  // eager first step seeds the cross-attn K/V cache that the imported
  // function consumes from then on).
  //
  // Populated by ``compile_for_inference`` when
  // ``MRT_DEPTHFORMER_DEPTH_MLXFN_DIR`` / ``MRT_DEPTHFORMER_TEMPORAL_MLXFN_DIR``
  // points to a directory containing
  // ``depth_step_<tag>_<dtype>_cl<NN>.mlxfn`` /
  // ``temporal_step_<tag>_<dtype>_cl<NN>.mlxfn`` bundles published
  // alongside the weights on Hugging Face (downloaded into
  // ``<weights-dir>/mlxfn/``). See ``cpp/README.md`` for how the
  // bundles are produced and consumed.
  std::vector<std::optional<CompiledFn>> depth_step_by_cl_;
  std::vector<std::optional<CompiledFn>> temporal_step_by_cl_;

  // Precomputed weight reference lists for the weights-as-args dispatcher.
  // Built once at construction (after all layers + lm_head exist) and
  // reused for every imported-fast-path call.
  std::vector<mlx::core::array> depth_step_weight_args_;
  std::vector<mlx::core::array> temporal_step_weight_args_;

  // Padded single-graph dispatchers (mlxfn manifest format_version 3).
  // When loaded these REPLACE the per-cl tables: K/V buffers are kept
  // at the maximum cache length for the whole chunk, attention masks
  // are constructed at runtime to ignore positions past the real
  // cache_length, and the same imported function services every
  // cache_length. Bundle drops from 65 .mlxfn files to 3.
  //
  //   MRT_DEPTHFORMER_TEMPORAL_PADDED_MLXFN=/abs/path/temporal_step_padded_*.mlxfn
  //   MRT_DEPTHFORMER_DEPTH_PADDED_MLXFN=/abs/path/depth_step_padded_*.mlxfn
  //
  // Bit-exact vs the per-cl variant but ~25% slower in mlx-stream
  // end-to-end on M3 Ultra (per-cl wins because Apple's fast SDPA is
  // per-shape-fused). Loaded only as a fallback when the per-cl bundle
  // is absent.
  std::optional<CompiledFn> padded_depth_step_;
  std::optional<CompiledFn> padded_temporal_step_;
  // Maximum K/V cache length the padded buffers are sized to. Equal to
  // ``rvq_depth`` for depth and ``chunk_length_frames`` for temporal.
  int padded_depth_max_ = 0;
  int padded_temporal_max_ = 0;
};

}  // namespace magenta_realtime_mlx::depthformer

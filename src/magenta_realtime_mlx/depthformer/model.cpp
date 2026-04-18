#include "magenta_realtime_mlx/depthformer/model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <filesystem>
#include <regex>

#include "mlx/export.h"

namespace magenta_realtime_mlx::depthformer {

namespace mx = mlx::core;

DepthformerConfig DepthformerConfig::base() {
  return DepthformerConfig{};
}

DepthformerConfig DepthformerConfig::large() {
  DepthformerConfig c;
  c.d_model = 1024;
  c.d_kv = 64;
  c.d_ff = 2816;
  c.num_heads = 16;
  c.num_encoder_layers = 24;
  c.num_temporal_layers = 40;
  c.num_depth_layers = 8;
  return c;
}

namespace {

// Build the sinusoidal table on the host (CPU floats) so we match the
// upstream NumPy reference bit-for-bit. This is run once at construction.
mx::array build_sinusoidal_position_embedding(int max_length, int features,
                                              float min_scale = 1.0f,
                                              float max_scale = 10000.0f) {
  std::vector<float> data(static_cast<size_t>(max_length) * features, 0.0f);
  const int half = features / 2;
  if (half <= 1) {
    throw std::runtime_error("features // 2 must be >= 2 for sinusoidal PE");
  }
  const float scale_factor =
      -std::log(max_scale / min_scale) / static_cast<float>(half - 1);
  std::vector<float> div_term(half);
  for (int i = 0; i < half; ++i) {
    div_term[i] = min_scale * std::exp(static_cast<float>(i) * scale_factor);
  }
  for (int pos = 0; pos < max_length; ++pos) {
    float* row = &data[static_cast<size_t>(pos) * features];
    for (int i = 0; i < half; ++i) {
      const float angle = static_cast<float>(pos) * div_term[i];
      row[i] = std::sin(angle);
      row[half + i] = std::cos(angle);
    }
  }
  return mx::array(data.data(),
                   mx::Shape{static_cast<int32_t>(max_length),
                             static_cast<int32_t>(features)},
                   mx::float32);
}

std::vector<EncoderLayer> build_encoder_layers(const WeightBundle& bundle,
                                               const DepthformerConfig& cfg,
                                               mx::Dtype dtype) {
  std::vector<EncoderLayer> layers;
  layers.reserve(cfg.num_encoder_layers);
  for (int i = 0; i < cfg.num_encoder_layers; ++i) {
    std::string prefix = "encoder_layers." + std::to_string(i);
    layers.emplace_back(bundle, prefix, cfg.d_model, cfg.num_heads, cfg.d_kv,
                        cfg.d_ff, dtype);
  }
  return layers;
}

std::vector<DecoderLayer> build_decoder_layers(const WeightBundle& bundle,
                                               const std::string& family,
                                               int count,
                                               const DepthformerConfig& cfg,
                                               bool has_cross_attention,
                                               mx::Dtype dtype) {
  std::vector<DecoderLayer> layers;
  layers.reserve(count);
  for (int i = 0; i < count; ++i) {
    std::string prefix = family + "." + std::to_string(i);
    layers.emplace_back(bundle, prefix, cfg.d_model, cfg.num_heads, cfg.d_kv,
                        cfg.d_ff, has_cross_attention, dtype);
  }
  return layers;
}

// ---------------------------------------------------------------------------
// Padded ``.mlxfn`` helpers (mlxfn manifest format_version 3, see g29)
// ---------------------------------------------------------------------------

// Pad a self-attention K or V tensor from its current seq-length axis (axis=2)
// out to ``max_len`` with zeros. No-op if already at ``max_len``. The padded
// positions hold junk from the model's perspective -- the runtime mask zeroes
// them out before they ever reach softmax.
mx::array pad_kv_to_max(const mx::array& kv, int max_len, mx::Dtype dtype) {
  const int cur_len = static_cast<int>(kv.shape(2));
  if (cur_len == max_len) return kv;
  if (cur_len > max_len) {
    throw std::runtime_error("pad_kv_to_max: cur_len > max_len");
  }
  const auto& shape = kv.shape();
  mx::Shape pad_shape{shape[0], shape[1],
                      static_cast<int32_t>(max_len - cur_len), shape[3]};
  mx::array zeros = mx::zeros(pad_shape, dtype);
  return mx::concatenate({kv, zeros}, /*axis=*/2);
}

// Build the additive self-attention mask the padded ``temporal_step`` /
// ``depth_step`` mlxfn expects. Combines the relative-position bias for
// the FULL ``max_len`` (query at ``cache_length``, keys at ``0..max_len-1``)
// with a validity mask that injects ``-inf`` at every position past
// ``cache_length``. Returns shape ``(1, H, 1, max_len)`` in ``dtype``.
mx::array build_padded_attn_mask(const RelativePositionBias& rel_pos,
                                 int cache_length, int max_len,
                                 mx::Dtype dtype) {
  // position_bias: (1, H, 1, max_len). Built in the rel_pos's storage dtype.
  mx::array bias = rel_pos(/*query_length=*/1, /*key_length=*/max_len,
                           /*offset=*/cache_length);
  if (bias.dtype() != dtype) bias = mx::astype(bias, dtype);
  // valid_mask: (1, 1, 1, max_len). 0 at positions <= cache_length, -inf
  // elsewhere. Built in fp32 first to avoid generating ``-inf`` in lower
  // precision (which can manifest as NaN through fast SDPA), then cast.
  const float neg_inf = -std::numeric_limits<float>::infinity();
  mx::array positions = mx::arange(0, max_len, mx::int32);
  mx::array valid = mx::less_equal(positions, mx::array(cache_length));
  mx::array valid_mask = mx::where(valid, mx::array(0.0f),
                                   mx::array(neg_inf));
  valid_mask = mx::reshape(valid_mask, {1, 1, 1, max_len});
  if (valid_mask.dtype() != dtype) {
    valid_mask = mx::astype(valid_mask, dtype);
  }
  return mx::add(bias, valid_mask);
}

// ``insert_indicator`` of shape ``(1, 1, max_len, 1)``: 1.0 at position
// ``cache_length`` (where the new token's K/V should be scattered), 0.0
// elsewhere. The padded mlxfn uses this as ``K_full = K_pad * (1 - ind) +
// k_new * ind`` to inject the new K without dynamic indexing.
mx::array build_insert_indicator(int cache_length, int max_len,
                                 mx::Dtype dtype) {
  mx::array positions = mx::arange(0, max_len, mx::int32);
  mx::array indicator = mx::equal(positions, mx::array(cache_length));
  indicator = mx::reshape(indicator, {1, 1, max_len, 1});
  return mx::astype(indicator, dtype);
}

}  // namespace

Depthformer::Depthformer(const WeightBundle& bundle,
                         const DepthformerConfig& config, mx::Dtype dtype)
    : config_(config),
      dtype_(dtype),
      token_embedding_(bundle, "token_embedding", dtype),
      position_embedding_(build_sinusoidal_position_embedding(
          config.max_encoder_length, config.d_model)),
      encoder_layers_(build_encoder_layers(bundle, config, dtype)),
      encoder_norm_(bundle, "encoder_norm", config.d_model, 1e-6f, dtype),
      temporal_rel_pos_(std::make_unique<RelativePositionBias>(
          bundle, "temporal_rel_pos", config.num_heads,
          config.temporal_num_buckets, config.temporal_max_distance,
          /*bidirectional=*/false, dtype)),
      temporal_layers_(build_decoder_layers(bundle, "temporal_layers",
                                            config.num_temporal_layers, config,
                                            /*has_cross_attention=*/true,
                                            dtype)),
      depth_rel_pos_(std::make_unique<RelativePositionBias>(
          bundle, "depth_rel_pos", config.num_heads,
          config.depth_num_buckets, config.depth_max_distance,
          /*bidirectional=*/false, dtype)),
      depth_layers_(build_decoder_layers(bundle, "depth_layers",
                                         config.num_depth_layers, config,
                                         /*has_cross_attention=*/false, dtype)),
      decoder_norm_(bundle, "decoder_norm", config.d_model, 1e-6f, dtype),
      lm_head_(bundle, "lm_head", dtype) {
  // Precompute the weights-as-args reference lists once. The dispatch
  // hot path then just bulk-inserts these into the input vector instead
  // of walking the layer tree on every call.
  depth_step_weight_args_.reserve(depth_layers_.size() * 9 + 2);
  for (const auto& layer : depth_layers_) {
    layer.append_mlxfn_weights(depth_step_weight_args_);
  }
  depth_step_weight_args_.push_back(decoder_norm_.weight());
  depth_step_weight_args_.push_back(lm_head_.weight());

  temporal_step_weight_args_.reserve(temporal_layers_.size() * 12);
  for (const auto& layer : temporal_layers_) {
    layer.append_mlxfn_weights(temporal_step_weight_args_);
  }
}

mx::array Depthformer::position_embedding(int seq_len) const {
  if (seq_len > config_.max_encoder_length) {
    throw std::out_of_range("seq_len exceeds max_encoder_length");
  }
  // Slice the first ``seq_len`` rows.
  mx::Shape start{0, 0};
  mx::Shape stop{static_cast<int32_t>(seq_len),
                 static_cast<int32_t>(config_.d_model)};
  return mx::slice(position_embedding_, start, stop);
}

mx::array Depthformer::encode(const mx::array& encoder_input_tokens) const {
  if (compiled_encode_.has_value()) {
    return (*compiled_encode_)({encoder_input_tokens})[0];
  }
  const int seq_len = static_cast<int>(encoder_input_tokens.shape(1));
  mx::array x = token_embedding_(encoder_input_tokens);
  mx::array pe = position_embedding(seq_len);
  if (pe.dtype() != x.dtype()) {
    pe = mx::astype(pe, x.dtype());
  }
  x = mx::add(x, pe);
  for (const auto& layer : encoder_layers_) {
    x = layer(x);
  }
  return encoder_norm_(x);
}

mx::array Depthformer::depth_forward_full(
    const mx::array& token_embeddings,
    const std::optional<mx::array>& position_bias, bool causal) const {
  const int k_len = static_cast<int>(token_embeddings.shape(1));

  mx::array bias = position_bias.has_value()
                       ? *position_bias
                       : (*depth_rel_pos_)(k_len, k_len, /*offset=*/0);

  std::optional<mx::array> mask;
  if (causal) {
    mask = causal_mask(k_len, token_embeddings.dtype());
  }

  mx::array x = token_embeddings;
  for (const auto& layer : depth_layers_) {
    KVCache discard_self{mx::array(0.0f), mx::array(0.0f)};
    std::optional<KVCache> discard_cross;
    x = layer.forward(x, /*encoder_output=*/std::nullopt, mask, bias,
                      /*self_attn_cache=*/std::nullopt,
                      /*cross_attn_cache=*/std::nullopt, discard_self,
                      discard_cross);
  }
  x = decoder_norm_(x);
  return lm_head_(x);
}

// ---------------------------------------------------------------------------
// Autoregressive step API
// ---------------------------------------------------------------------------

namespace {
LayerCache make_empty_layer_cache() {
  return LayerCache{KVCache{mx::array(0.0f), mx::array(0.0f)}, std::nullopt};
}
}  // namespace

std::vector<LayerCache> Depthformer::empty_temporal_cache() const {
  // We use ``std::optional<KVCache>`` semantics on the call sites of layer
  // forward; this helper merely sizes the vector to the layer count. Each
  // entry's ``second`` (cross-attn) starts as nullopt; we only track that
  // they have not been populated yet via a separate flag (cross_done_).
  std::vector<LayerCache> cache;
  cache.reserve(temporal_layers_.size());
  for (size_t i = 0; i < temporal_layers_.size(); ++i) {
    cache.push_back(make_empty_layer_cache());
  }
  return cache;
}

std::vector<LayerCache> Depthformer::empty_depth_cache() const {
  std::vector<LayerCache> cache;
  cache.reserve(depth_layers_.size());
  for (size_t i = 0; i < depth_layers_.size(); ++i) {
    cache.push_back(make_empty_layer_cache());
  }
  return cache;
}

std::vector<mx::array> Depthformer::precompute_temporal_biases(int max_frames) const {
  std::vector<mx::array> out;
  out.reserve(max_frames);
  for (int f = 0; f < max_frames; ++f) {
    out.push_back((*temporal_rel_pos_)(/*query_length=*/1, /*key_length=*/f + 1,
                                       /*offset=*/f));
  }
  return out;
}

std::vector<mx::array> Depthformer::precompute_depth_biases(int max_depth) const {
  std::vector<mx::array> out;
  out.reserve(max_depth);
  for (int d = 0; d < max_depth; ++d) {
    out.push_back((*depth_rel_pos_)(/*query_length=*/1, /*key_length=*/d + 1,
                                    /*offset=*/d));
  }
  return out;
}

mx::array Depthformer::temporal_step(
    const mx::array& frame_embedding, const mx::array& encoder_output,
    std::vector<LayerCache>& cache, int frame_idx,
    const std::optional<mx::array>& position_bias) const {
  if (cache.size() != temporal_layers_.size()) {
    throw std::runtime_error("temporal_step: cache size mismatch");
  }

  mx::array bias = position_bias.has_value()
                       ? *position_bias
                       : (*temporal_rel_pos_)(1, frame_idx + 1, frame_idx);

  const bool first_step = (frame_idx == 0);

  // Padded single-graph fast path (g29). When loaded, services every
  // frame_idx > 0 from a single .mlxfn. K/V buffers stay at
  // ``padded_temporal_max_`` for the whole chunk; the runtime mask
  // tells SDPA to ignore positions past the real cache length. The
  // input vector layout is:
  //   inputs[0]                = frame_embedding (B, 1, d_model)
  //   inputs[1]                = combined_self_mask (1, H, 1, M)
  //   inputs[2]                = insert_indicator   (1, 1, M, 1)
  //   inputs[3 + 4i + 0/1]     = self_K_pad / self_V_pad (B, H, M, D)
  //   inputs[3 + 4i + 2/3]     = cross_K   / cross_V    (B, H, ENC_LEN, D)
  //   inputs[3 + 4*L .. end]   = per-layer weights (canonical order; see
  //                               ``append_temporal_step_weights``).
  if (!first_step && padded_temporal_step_.has_value()) {
    const size_t L = temporal_layers_.size();
    const int M = padded_temporal_max_;
    std::vector<mx::array> inputs;
    inputs.reserve(3 + 4 * L + L * 12);
    mx::array fe = frame_embedding;
    if (fe.dtype() != dtype_) fe = mx::astype(fe, dtype_);
    inputs.push_back(fe);
    inputs.push_back(
        build_padded_attn_mask(*temporal_rel_pos_, frame_idx, M, dtype_));
    inputs.push_back(build_insert_indicator(frame_idx, M, dtype_));
    for (size_t i = 0; i < L; ++i) {
      if (!cache[i].second.has_value()) {
        throw std::runtime_error(
            "temporal_step: padded path requires cross-K/V cache to be "
            "populated by an earlier eager step");
      }
      mx::array sk = cache[i].first.k;
      mx::array sv = cache[i].first.v;
      if (sk.dtype() != dtype_) sk = mx::astype(sk, dtype_);
      if (sv.dtype() != dtype_) sv = mx::astype(sv, dtype_);
      // Pad to MAX_T on the first padded call (frame_idx == 1, where the
      // eager frame=0 produced K/V at length 1). Subsequent calls already
      // hold MAX_T-sized buffers because every padded dispatch returns
      // K/V at MAX_T.
      sk = pad_kv_to_max(sk, M, dtype_);
      sv = pad_kv_to_max(sv, M, dtype_);
      // Cross-attn K/V: same one-time cast as the per-cl path. Stored
      // back into the cache so subsequent frames in the same chunk see
      // bf16 directly (avoids ~12 GB / chunk of redundant cast work).
      if (cache[i].second->k.dtype() != dtype_) {
        cache[i].second->k = mx::astype(cache[i].second->k, dtype_);
      }
      if (cache[i].second->v.dtype() != dtype_) {
        cache[i].second->v = mx::astype(cache[i].second->v, dtype_);
      }
      inputs.push_back(sk);
      inputs.push_back(sv);
      inputs.push_back(cache[i].second->k);
      inputs.push_back(cache[i].second->v);
    }
    append_temporal_step_weights(inputs);
    std::vector<mx::array> outputs = (*padded_temporal_step_)(inputs);
    for (size_t i = 0; i < L; ++i) {
      cache[i].first = KVCache{outputs[1 + 2 * i + 0], outputs[1 + 2 * i + 1]};
    }
    return outputs[0];
  }

  // Per-cache-length imported fast path (g21). Same dispatch pattern as
  // depth_step (see g20). Cast K/V to model dtype on input because the
  // ``.mlxfn`` was traced with bf16 K/V; the eager C++ path stores fp32.
  //
  // Format: weights-as-args (mlxfn manifest format_version 2). The
  // imported function expects:
  //   inputs[0]                  = frame_embedding (B, 1, d_model)
  //   inputs[1]                  = position_bias   (1, H, 1, key_len)
  //   inputs[2 + 4i + 0/1]       = self_K_in / self_V_in
  //   inputs[2 + 4i + 2/3]       = cross_K   / cross_V   (precomputed)
  //   inputs[2 + 4*L .. end]     = per-layer weight tensors in canonical
  //                                order (see Depthformer::append_temporal_step_weights).
  // The C++ runtime mirrors the order baked into the ``.mlxfn`` exactly.
  // Any drift produces a shape error from MLX (or worse, a silent garbage
  // result), so changes here MUST track the exporter that produced the bundle.
  const size_t fidx_us = static_cast<size_t>(frame_idx);
  if (!first_step && fidx_us < temporal_step_by_cl_.size()
      && temporal_step_by_cl_[fidx_us].has_value()) {
    const size_t L = temporal_layers_.size();
    std::vector<mx::array> inputs;
    inputs.reserve(2 + 4 * L + L * 12);
    // frame_embedding may arrive as fp32 from the encode pipeline (the
    // encoder upcasts in RMSNorm and the C++ encode wrapper preserves
    // that); the ``.mlxfn`` was traced with bf16 input. Cast at the
    // boundary.
    mx::array fe = frame_embedding;
    if (fe.dtype() != dtype_) fe = mx::astype(fe, dtype_);
    inputs.push_back(fe);
    inputs.push_back(bias);
    for (size_t i = 0; i < L; ++i) {
      if (!cache[i].second.has_value()) {
        throw std::runtime_error(
            "temporal_step: imported path requires cross-K/V cache to be "
            "populated by an earlier eager step");
      }
      mx::array sk = cache[i].first.k;
      mx::array sv = cache[i].first.v;
      // Self-attn K/V are stored at the model dtype after the very first
      // dispatch (we drop the cast-back-to-fp32 the original temporal_step
      // dispatcher used). The only remaining cast is on frame_idx=1, where
      // the eager frame=0 path may still produce non-model-dtype K/V.
      if (sk.dtype() != dtype_) sk = mx::astype(sk, dtype_);
      if (sv.dtype() != dtype_) sv = mx::astype(sv, dtype_);
      // Cross-attn K/V are constant across all frames in a chunk -- they
      // come from encoder output and never get re-projected. Cast once
      // (and store back into the cache) so every subsequent frame in the
      // same chunk reuses the bf16 version. Without storing back, 20
      // layers x 2 (K,V) x 49 frames x ~6 MB / cast = ~12 GB of wasted
      // cast bandwidth per chunk.
      if (cache[i].second->k.dtype() != dtype_) {
        cache[i].second->k = mx::astype(cache[i].second->k, dtype_);
      }
      if (cache[i].second->v.dtype() != dtype_) {
        cache[i].second->v = mx::astype(cache[i].second->v, dtype_);
      }
      inputs.push_back(sk);
      inputs.push_back(sv);
      inputs.push_back(cache[i].second->k);
      inputs.push_back(cache[i].second->v);
    }
    append_temporal_step_weights(inputs);
    std::vector<mx::array> outputs = (*temporal_step_by_cl_[fidx_us])(inputs);
    for (size_t i = 0; i < L; ++i) {
      cache[i].first = KVCache{outputs[1 + 2 * i + 0], outputs[1 + 2 * i + 1]};
    }
    return outputs[0];
  }

  // Compiled fast path. We require frame_idx > 0 so the eager first step has
  // already populated cross-attn K/V (and self-attn K/V) for every layer.
  // The compiled body never re-projects cross K/V -- it threads it through as
  // an input -- which is the dominant per-frame win.
  if (compiled_temporal_step_.has_value() && !first_step) {
    const size_t L = temporal_layers_.size();
    std::vector<mx::array> inputs;
    inputs.reserve(2 + 4 * L);
    inputs.push_back(frame_embedding);
    inputs.push_back(bias);
    for (size_t i = 0; i < L; ++i) {
      if (!cache[i].second.has_value()) {
        throw std::runtime_error(
            "temporal_step: compiled path requires cross-K/V cache to be "
            "populated by an earlier eager step");
      }
      inputs.push_back(cache[i].first.k);
      inputs.push_back(cache[i].first.v);
      inputs.push_back(cache[i].second->k);
      inputs.push_back(cache[i].second->v);
    }
    std::vector<mx::array> outputs = (*compiled_temporal_step_)(inputs);
    for (size_t i = 0; i < L; ++i) {
      cache[i].first = KVCache{outputs[1 + 2 * i + 0], outputs[1 + 2 * i + 1]};
    }
    return outputs[0];
  }

  // Eager path (also used to seed the cross-K/V cache on frame 0).
  mx::array x = frame_embedding;
  for (size_t i = 0; i < temporal_layers_.size(); ++i) {
    std::optional<KVCache> self_in;
    std::optional<KVCache> cross_in;
    if (!first_step) {
      self_in = cache[i].first;
      cross_in = cache[i].second;
    }

    KVCache new_self{mx::array(0.0f), mx::array(0.0f)};
    std::optional<KVCache> new_cross;
    x = temporal_layers_[i].forward(
        x, encoder_output, /*mask=*/std::nullopt, bias, self_in, cross_in,
        new_self, new_cross);
    cache[i].first = new_self;
    if (new_cross.has_value()) {
      cache[i].second = new_cross;
    }
  }
  return x;
}

mx::array Depthformer::depth_step(
    const mx::array& token_embedding, std::vector<LayerCache>& cache,
    int depth_idx, const std::optional<mx::array>& position_bias) const {
  if (cache.size() != depth_layers_.size()) {
    throw std::runtime_error("depth_step: cache size mismatch");
  }

  mx::array bias = position_bias.has_value()
                       ? *position_bias
                       : (*depth_rel_pos_)(1, depth_idx + 1, depth_idx);

  const bool first_step = (depth_idx == 0);

  // Padded single-graph fast path (g29). Single .mlxfn services every
  // depth_idx > 0; K/V buffers stay at ``padded_depth_max_`` for the
  // whole depth pass. Inputs:
  //   inputs[0]                = token_embedding (B, 1, d_model)
  //   inputs[1]                = combined_self_mask (1, H, 1, M)
  //   inputs[2]                = insert_indicator   (1, 1, M, 1)
  //   inputs[3 + 2i + 0/1]     = self_K_pad / self_V_pad (B, H, M, D)
  //   inputs[3 + 2*L .. end]   = per-layer weights + decoder_norm + lm_head
  if (!first_step && padded_depth_step_.has_value()) {
    const size_t L = depth_layers_.size();
    const int M = padded_depth_max_;
    std::vector<mx::array> inputs;
    inputs.reserve(3 + 2 * L + L * 9 + 2);
    mx::array te = token_embedding;
    if (te.dtype() != dtype_) te = mx::astype(te, dtype_);
    inputs.push_back(te);
    inputs.push_back(
        build_padded_attn_mask(*depth_rel_pos_, depth_idx, M, dtype_));
    inputs.push_back(build_insert_indicator(depth_idx, M, dtype_));
    for (size_t i = 0; i < L; ++i) {
      mx::array sk = cache[i].first.k;
      mx::array sv = cache[i].first.v;
      if (sk.dtype() != dtype_) sk = mx::astype(sk, dtype_);
      if (sv.dtype() != dtype_) sv = mx::astype(sv, dtype_);
      sk = pad_kv_to_max(sk, M, dtype_);
      sv = pad_kv_to_max(sv, M, dtype_);
      inputs.push_back(sk);
      inputs.push_back(sv);
    }
    append_depth_step_weights(inputs);
    std::vector<mx::array> outputs = (*padded_depth_step_)(inputs);
    for (size_t i = 0; i < L; ++i) {
      cache[i].first = KVCache{outputs[1 + 2 * i + 0], outputs[1 + 2 * i + 1]};
    }
    return outputs[0];
  }

  // Per-cache-length imported fast path. When the user has pre-exported
  // ``depth_step_*_cl<NN>.mlxfn`` bundles (g20, see compile_for_inference),
  // we dispatch by ``depth_idx`` (= cache_length): each .mlxfn was traced at
  // a specific K/V seq-len, and ``mx::fast::scaled_dot_product_attention``
  // bakes that length into its Metal kernel (g19a). The microbench (g19c)
  // shows ~0.78 ms/call vs ~2.06 ms/call for the C++ lambda path -- a 37%
  // per-call win that projects to several hundred ms / chunk.
  const size_t didx_us = static_cast<size_t>(depth_idx);
  if (!first_step && didx_us < depth_step_by_cl_.size()
      && depth_step_by_cl_[didx_us].has_value()) {
    const size_t L = depth_layers_.size();
    // The ``.mlxfn`` was traced with K/V in the model's loaded dtype
    // (bf16 by default). The C++ eager path, however, stores K/V as
    // float32 -- the depth_layer's K/V projections get upcast somewhere
    // along the way (a latent eager-vs-traced parity divergence flagged
    // in g19b). Cast back at the boundary before dispatching. The cast
    // itself is cheap (a single small Metal kernel per layer) compared
    // to the ~0.78 ms / call we save from using the better fused kernel.
    // Format: weights-as-args (mlxfn manifest format_version 2). Inputs:
    //   inputs[0]              = token_embedding (B, 1, d_model)
    //   inputs[1]              = position_bias   (1, H, 1, key_len)
    //   inputs[2 + 2i + 0/1]   = K_in / V_in for layer i
    //   inputs[2 + 2*L .. end] = per-layer weight tensors (canonical order)
    //                            + decoder_norm + lm_head; see
    //                            ``Depthformer::append_depth_step_weights``.
    std::vector<mx::array> inputs;
    inputs.reserve(2 + 2 * L + L * 9 + 2);
    inputs.push_back(token_embedding);
    inputs.push_back(bias);
    for (size_t i = 0; i < L; ++i) {
      mx::array k = cache[i].first.k;
      mx::array v = cache[i].first.v;
      // Cache K/V are stored at model dtype after the first dispatch
      // (we no longer cast back to fp32 on the way out -- see g23). The
      // cast here is a runtime no-op except on the very first depth_idx=1
      // call after the eager depth_idx=0 step.
      if (k.dtype() != dtype_) k = mx::astype(k, dtype_);
      if (v.dtype() != dtype_) v = mx::astype(v, dtype_);
      inputs.push_back(k);
      inputs.push_back(v);
    }
    append_depth_step_weights(inputs);
    std::vector<mx::array> outputs = (*depth_step_by_cl_[didx_us])(inputs);
    for (size_t i = 0; i < L; ++i) {
      // Store K/V as the model dtype (bf16) -- not float32 -- so subsequent
      // dispatches don't have to cast inputs back. The original code cast
      // back to float32 to match the eager path; the upcasting has since
      // been removed.
      cache[i].first = KVCache{outputs[1 + 2 * i + 0], outputs[1 + 2 * i + 1]};
    }
    return outputs[0];
  }

  // Compiled fast path. Same constraint as temporal_step: depth_idx > 0 so
  // the cache has been seeded by the eager first step.
  if (compiled_depth_step_.has_value() && !first_step) {
    const size_t L = depth_layers_.size();
    std::vector<mx::array> inputs;
    inputs.reserve(2 + 2 * L);
    inputs.push_back(token_embedding);
    inputs.push_back(bias);
    for (size_t i = 0; i < L; ++i) {
      inputs.push_back(cache[i].first.k);
      inputs.push_back(cache[i].first.v);
    }
    std::vector<mx::array> outputs = (*compiled_depth_step_)(inputs);
    for (size_t i = 0; i < L; ++i) {
      cache[i].first = KVCache{outputs[1 + 2 * i + 0], outputs[1 + 2 * i + 1]};
    }
    return outputs[0];
  }

  mx::array x = token_embedding;
  for (size_t i = 0; i < depth_layers_.size(); ++i) {
    std::optional<KVCache> self_in;
    if (!first_step) {
      self_in = cache[i].first;
    }
    KVCache new_self{mx::array(0.0f), mx::array(0.0f)};
    std::optional<KVCache> discard_cross;
    x = depth_layers_[i].forward(
        x, /*encoder_output=*/std::nullopt, /*mask=*/std::nullopt, bias,
        self_in, /*cross_attn_cache=*/std::nullopt, new_self, discard_cross);
    cache[i].first = new_self;
  }

  x = decoder_norm_(x);
  mx::array logits = lm_head_(x);  // (B, 1, vocab)
  // Squeeze the singleton T axis -> (B, vocab) (logically a ``.squeeze(1)``).
  const int B = static_cast<int>(logits.shape(0));
  const int V = static_cast<int>(logits.shape(2));
  return mx::reshape(logits, mx::Shape{static_cast<int32_t>(B),
                                       static_cast<int32_t>(V)});
}

// ---------------------------------------------------------------------------
// Weights-as-args ``.mlxfn`` weight collectors
// ---------------------------------------------------------------------------
//
// These build the flat weight-arg lists fed to the ``.mlxfn``
// compiled functions; the ordering must stay in lockstep with the way
// the published bundles were exported. Any change to the per-layer
// ordering must track the upstream exporter that produced the bundles
// (otherwise the ``.mlxfn`` will receive shape-mismatched weights and
// either error out or silently produce garbage).
//
// These methods are pure metadata operations: ``mx::array`` is a smart
// pointer to the underlying buffer, and ``Linear::weight()`` returns a
// lazy strided view (``mx::transpose`` does not launch a kernel). The
// per-call cost is therefore dominated by the ``vector<>::push_back``
// allocations, which we amortise by reserving up front.

void Depthformer::append_depth_step_weights(
    std::vector<mx::array>& out) const {
  // Bulk insert the precomputed reference list (built once in the
  // constructor). Each ``mx::array`` copy is a cheap atomic refcount
  // bump on the underlying buffer + view metadata. With base config
  // this is 38 refs per call, called 800x per chunk.
  out.insert(out.end(), depth_step_weight_args_.begin(),
             depth_step_weight_args_.end());
}

void Depthformer::append_temporal_step_weights(
    std::vector<mx::array>& out) const {
  // Same precomputed-list pattern as ``append_depth_step_weights``; 240
  // refs per call, called 49x per chunk.
  out.insert(out.end(), temporal_step_weight_args_.begin(),
             temporal_step_weight_args_.end());
}

// ---------------------------------------------------------------------------
// compile_for_inference
// ---------------------------------------------------------------------------
//
// Wraps the depthformer's hot paths in ``mx::compile`` so each call
// fires a single fused Metal kernel instead of dispatching dozens of
// small ones. On first call to a compiled hot path, MLX traces the op
// graph, generates a fused Metal kernel,
// and caches it. Subsequent calls with matching input shapes hit the cached
// kernel. The KV-cache K/V grow by one token per autoregressive step, so MLX
// retraces per shape; the per-trace cost is paid once per chunk and amortised
// across subsequent chunks (the same shape sequence repeats).

void Depthformer::compile_for_inference(bool compile_decode_steps) {
  namespace mxc = mlx::core;
  // Diagnostic: allow toggling compile mode at runtime via env var to
  // measure the contribution of the simplify vs fuse passes vs eager.
  // Each pass is small here -- the dominant cost lives in GPU compute,
  // not the host-side fusion passes -- but the knob is useful for A/B.
  //   MRT_COMPILE_MODE=enabled|no_simplify|no_fuse|disabled
  if (const char* m = std::getenv("MRT_COMPILE_MODE")) {
    std::string mode_str(m);
    if (mode_str == "no_fuse") mxc::set_compile_mode(mxc::CompileMode::no_fuse);
    else if (mode_str == "no_simplify") mxc::set_compile_mode(mxc::CompileMode::no_simplify);
    else if (mode_str == "disabled") mxc::set_compile_mode(mxc::CompileMode::disabled);
    else if (mode_str == "enabled") mxc::set_compile_mode(mxc::CompileMode::enabled);
    fprintf(stderr, "[mrt] compile mode: %s\n", mode_str.c_str());
  }
  if (!compiled_encode_.has_value()) {
    // Optional: load a pre-traced encode graph from a .mlxfn file
    // instead of rebuilding it via the C++ lambda below. The hypothesis
    // (g15-g17 / g18) is that the pre-traced graph produces a slightly
    // better source graph than the C++ capturing lambda, and wrapping
    // the imported function in ``mx::compile`` lets MLX produce
    // identical fused kernels.
    //
    //   MRT_DEPTHFORMER_ENCODE_MLXFN=/abs/path/to/encode_<tag>_<dtype>.mlxfn
    //
    // Loaded from the published ``.mlxfn`` bundles under
    // ``<weights-dir>/mlxfn/`` (downloaded from Hugging Face by
    // ``scripts/download_weights_from_hf.py``). The dtype must match
    // this Depthformer's loaded weight dtype; otherwise weight
    // tensors baked into the .mlxfn won't match the rest of the model
    // and parity tests will fail.
    if (const char* p = std::getenv("MRT_DEPTHFORMER_ENCODE_MLXFN");
        p != nullptr && p[0] != '\0') {
      auto imported = std::make_shared<mxc::ImportedFunction>(
          mxc::import_function(p));
      fprintf(stderr, "[mrt] loaded encode graph from %s\n", p);
      // The pre-traced encode graph returns the encoder output in
      // the model's loaded dtype (bf16 by default). Previously we cast
      // back to float32 here because the eager temporal_step path was
      // built assuming fp32 cross-attention input. Now that
      // ``temporal_step_by_cl_`` handles every frame_idx > 0 in bf16 and
      // the eager frame=0 path tolerates bf16 encoder_output (cross-attn
      // kv_input projects through bf16 weights -> bf16 K/V), keeping the
      // encoder output at the model dtype eliminates a ~10 MB downcast on
      // the boundary plus ~40 cross-K/V casts inside the first imported
      // temporal_step.
      compiled_encode_ = mxc::compile(
          [imported](const std::vector<mxc::array>& inputs)
              -> std::vector<mxc::array> {
            return (*imported)(inputs);
          });
    } else {
      // Capture ``this`` by raw pointer; the Depthformer outlives the
      // compiled function (System owns both with matched lifetimes).
      auto self = this;
      compiled_encode_ = mxc::compile(
          [self](const std::vector<mxc::array>& inputs)
              -> std::vector<mxc::array> {
            // Inline the eager body: ``encode`` itself short-circuits to the
            // compiled path when set, so calling it here would recurse. The
            // op graph below is what MLX traces and fuses.
            const mxc::array& encoder_input_tokens = inputs[0];
            const int seq_len =
                static_cast<int>(encoder_input_tokens.shape(1));
            mxc::array x = self->token_embedding_(encoder_input_tokens);
            mxc::array pe = self->position_embedding(seq_len);
            if (pe.dtype() != x.dtype()) {
              pe = mxc::astype(pe, x.dtype());
            }
            x = mxc::add(x, pe);
            for (const auto& layer : self->encoder_layers_) {
              x = layer(x);
            }
            return {self->encoder_norm_(x)};
          });
    }
  }
  if (compile_decode_steps) {
    // Padded single-graph dispatchers (g29). When set, these REPLACE
    // both the per-cl ``temporal_step_by_cl_`` table and the lambda
    // ``compiled_temporal_step_`` for every frame_idx > 0.
    //
    //   MRT_DEPTHFORMER_TEMPORAL_PADDED_MLXFN=/abs/path/temporal_step_padded_*.mlxfn
    //   MRT_DEPTHFORMER_DEPTH_PADDED_MLXFN  =/abs/path/depth_step_padded_*.mlxfn
    //
    // Bundle is 3 files (encode + 2 padded steps) instead of 65.
    if (const char* p = std::getenv("MRT_DEPTHFORMER_TEMPORAL_PADDED_MLXFN");
        p != nullptr && p[0] != '\0') {
      auto imported = std::make_shared<mxc::ImportedFunction>(
          mxc::import_function(p));
      // Padded buffer length = chunk_length_frames. The current config
      // hard-codes 50 (System::chunk_length_frames). We use the same
      // value here -- if the .mlxfn was traced at a different M, the
      // first dispatch will fail loudly with a shape error.
      padded_temporal_max_ = 50;
      padded_temporal_step_ =
          [imported](const std::vector<mxc::array>& inputs)
              -> std::vector<mxc::array> { return (*imported)(inputs); };
      fprintf(stderr,
              "[mrt] loaded padded temporal_step graph from %s (M=%d)\n",
              p, padded_temporal_max_);
    }
    if (const char* p = std::getenv("MRT_DEPTHFORMER_DEPTH_PADDED_MLXFN");
        p != nullptr && p[0] != '\0') {
      auto imported = std::make_shared<mxc::ImportedFunction>(
          mxc::import_function(p));
      // Padded depth buffer length = decoder_codec_rvq_depth (16 in
      // base config). We don't have direct access to System config
      // here, but it equals ``config_.rvq_depth``.
      padded_depth_max_ = config_.rvq_depth;
      padded_depth_step_ =
          [imported](const std::vector<mxc::array>& inputs)
              -> std::vector<mxc::array> { return (*imported)(inputs); };
      fprintf(stderr,
              "[mrt] loaded padded depth_step graph from %s (M=%d)\n",
              p, padded_depth_max_);
    }
    // Optional: per-cache-length temporal_step bundles (g21). Mirrors the
    // depth_step loader below. Unset slots fall back to
    // ``compiled_temporal_step_``.
    //
    //   MRT_DEPTHFORMER_TEMPORAL_MLXFN_DIR=/abs/path/to/dir
    if (const char* dir_env = std::getenv("MRT_DEPTHFORMER_TEMPORAL_MLXFN_DIR");
        dir_env != nullptr && dir_env[0] != '\0' &&
        temporal_step_by_cl_.empty()) {
      namespace fs = std::filesystem;
      fs::path dir(dir_env);
      if (!fs::is_directory(dir)) {
        fprintf(stderr, "[mrt] WARNING: not a directory: %s\n", dir_env);
      } else {
        const std::regex re(R"(temporal_step_.*_cl(\d{2})\.mlxfn)");
        size_t loaded = 0;
        // Upper bound on T_max; current configs use up to 50 frames per
        // chunk.
        temporal_step_by_cl_.assign(128, std::nullopt);
        for (const auto& entry : fs::directory_iterator(dir)) {
          if (!entry.is_regular_file()) continue;
          std::smatch m;
          const std::string name = entry.path().filename().string();
          if (!std::regex_match(name, m, re)) continue;
          const int cl = std::stoi(m[1].str());
          if (cl <= 0) continue;
          if (cl >= static_cast<int>(temporal_step_by_cl_.size())) continue;
          auto imported = std::make_shared<mxc::ImportedFunction>(
              mxc::import_function(entry.path().string()));
          temporal_step_by_cl_[cl] = mxc::compile(
              [imported](const std::vector<mxc::array>& inputs)
                  -> std::vector<mxc::array> {
                return (*imported)(inputs);
              });
          ++loaded;
        }
        fprintf(stderr,
                "[mrt] loaded %zu temporal_step .mlxfn bundles from %s\n",
                loaded, dir_env);
      }
    }
    if (!compiled_temporal_step_.has_value()) {
      auto self = this;
      // Body layout (in/out described in model.h next to the dispatcher):
      //   inputs[0]            = frame_embedding (B, 1, d_model)
      //   inputs[1]            = position_bias   (1, H, 1, key_len)
      //   inputs[2 + 4i + 0/1] = self_K_in / self_V_in
      //   inputs[2 + 4i + 2/3] = cross_K   / cross_V (precomputed)
      //   outputs[0]           = layer-stack output (B, 1, d_model)
      //   outputs[1 + 2i + 0/1] = new self_K / self_V (T+1 along axis=2)
      compiled_temporal_step_ = mxc::compile(
          [self](const std::vector<mxc::array>& inputs)
              -> std::vector<mxc::array> {
            const mxc::array& frame_emb = inputs[0];
            const mxc::array& pos_bias = inputs[1];
            const size_t L = self->temporal_layers_.size();
            std::vector<mxc::array> outputs;
            outputs.reserve(1 + 2 * L);
            outputs.emplace_back(frame_emb);  // overwritten below
            mxc::array x = frame_emb;
            for (size_t i = 0; i < L; ++i) {
              const mxc::array& self_K_in = inputs[2 + 4 * i + 0];
              const mxc::array& self_V_in = inputs[2 + 4 * i + 1];
              const mxc::array& cross_K = inputs[2 + 4 * i + 2];
              const mxc::array& cross_V = inputs[2 + 4 * i + 3];
              auto [out, new_K, new_V] =
                  self->temporal_layers_[i].forward_compiled(
                      x, pos_bias, self_K_in, self_V_in,
                      KVCache{cross_K, cross_V});
              x = out;
              outputs.emplace_back(new_K);
              outputs.emplace_back(new_V);
            }
            outputs[0] = x;
            return outputs;
          });
    }
    // Optional: per-cache-length depth_step bundles, one ``.mlxfn``
    // per cache length 0..15. Files are named
    // ``depth_step_<tag>_<dtype>_cl<NN>.mlxfn``. We scan the
    // directory once and slot each file into ``depth_step_by_cl_[NN]``;
    // ``depth_step`` then dispatches by depth_idx. Missing slots fall back
    // to the lambda-based ``compiled_depth_step_`` (built immediately
    // after this block) -- so partial bundles work fine.
    //
    //   MRT_DEPTHFORMER_DEPTH_MLXFN_DIR=/abs/path/to/dir/with/depth_step_*.mlxfn
    if (const char* dir_env = std::getenv("MRT_DEPTHFORMER_DEPTH_MLXFN_DIR");
        dir_env != nullptr && dir_env[0] != '\0' &&
        depth_step_by_cl_.empty()) {
      namespace fs = std::filesystem;
      fs::path dir(dir_env);
      if (!fs::is_directory(dir)) {
        fprintf(stderr, "[mrt] WARNING: not a directory: %s\n", dir_env);
      } else {
        // Match e.g. ``depth_step_base_bf16_cl08.mlxfn`` and capture the
        // 2-digit cache length. We don't enforce <tag>/<dtype> here -- if
        // the user points us at a directory with mismatched bundles, the
        // bf16/f32 tensors baked into the .mlxfn won't match the loaded
        // weights and the parity test will catch it loudly.
        const std::regex re(R"(depth_step_.*_cl(\d{2})\.mlxfn)");
        size_t loaded = 0;
        // Resize once to the max possible cache length the temporal stack
        // ever consumes (depth has up to ``decoder_codec_rvq_depth + 1``;
        // we don't have direct access to that constant here, but 64 is a
        // safe upper bound for any current config).
        depth_step_by_cl_.assign(64, std::nullopt);
        for (const auto& entry : fs::directory_iterator(dir)) {
          if (!entry.is_regular_file()) continue;
          std::smatch m;
          const std::string name = entry.path().filename().string();
          if (!std::regex_match(name, m, re)) continue;
          const int cl = std::stoi(m[1].str());
          if (cl <= 0) continue;  // first step stays on the eager path
          if (cl >= static_cast<int>(depth_step_by_cl_.size())) continue;
          auto imported = std::make_shared<mxc::ImportedFunction>(
              mxc::import_function(entry.path().string()));
          depth_step_by_cl_[cl] = mxc::compile(
              [imported](const std::vector<mxc::array>& inputs)
                  -> std::vector<mxc::array> {
                return (*imported)(inputs);
              });
          ++loaded;
        }
        fprintf(stderr,
                "[mrt] loaded %zu depth_step .mlxfn bundles from %s\n",
                loaded, dir_env);
      }
    }
    if (!compiled_depth_step_.has_value()) {
      auto self = this;
      // Body layout:
      //   inputs[0]            = token_embedding (B, 1, d_model)
      //   inputs[1]            = position_bias   (1, H, 1, key_len)
      //   inputs[2 + 2i + 0/1] = self_K_in / self_V_in (no cross)
      //   outputs[0]           = squeezed logits (B, vocab)
      //   outputs[1 + 2i + 0/1] = new self_K / self_V
      compiled_depth_step_ = mxc::compile(
          [self](const std::vector<mxc::array>& inputs)
              -> std::vector<mxc::array> {
            const mxc::array& token_emb = inputs[0];
            const mxc::array& pos_bias = inputs[1];
            const size_t L = self->depth_layers_.size();
            mxc::array x = token_emb;
            std::vector<mxc::array> outputs;
            outputs.reserve(1 + 2 * L);
            outputs.emplace_back(x);  // overwritten
            for (size_t i = 0; i < L; ++i) {
              const mxc::array& self_K_in = inputs[2 + 2 * i + 0];
              const mxc::array& self_V_in = inputs[2 + 2 * i + 1];
              auto [out, new_K, new_V] =
                  self->depth_layers_[i].forward_compiled(
                      x, pos_bias, self_K_in, self_V_in,
                      /*cross_kv=*/std::nullopt);
              x = out;
              outputs.emplace_back(new_K);
              outputs.emplace_back(new_V);
            }
            x = self->decoder_norm_(x);
            mxc::array logits = self->lm_head_(x);  // (B, 1, vocab)
            const int B = static_cast<int>(logits.shape(0));
            const int V = static_cast<int>(logits.shape(2));
            outputs[0] = mxc::reshape(
                logits, mxc::Shape{static_cast<int32_t>(B),
                                   static_cast<int32_t>(V)});
            return outputs;
          });
    }
  }
}

}  // namespace magenta_realtime_mlx::depthformer

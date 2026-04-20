// Copyright 2026 Brian Cruz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// MusicCoCa joint text/audio encoder used to turn a style prompt into
// a 768-dim style
// embedding that the Depthformer conditions on. Both the text and audio
// paths are ported; the text path is the hot one (warmup + every generate).
//
// Layout matches the upstream Magenta RealTime implementation exactly:
// pre-norm transformer layers with GELU feed-forward and attention-logit capping
// (tanh soft-cap at 50.0). The attention pooler uses a learned per-dim
// scale on the query, computed via ``softplus(v) / log(2)``.

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "magenta_realtime_mlx/weights.h"
#include "mlx/mlx.h"

namespace magenta_realtime_mlx::musiccoca {

// ---------------------------------------------------------------------------
// Leaf modules
// ---------------------------------------------------------------------------

class LayerNorm {
 public:
  LayerNorm(const WeightBundle& bundle, std::string_view prefix, int dim,
            float eps, mlx::core::Dtype dtype);
  mlx::core::array operator()(const mlx::core::array& x) const;

 private:
  mlx::core::array weight_;
  mlx::core::array bias_;
  float eps_;
};

class Linear {
 public:
  // If ``bias`` is true, ``prefix + ".bias"`` must exist.
  Linear(const WeightBundle& bundle, std::string_view prefix, bool has_bias,
         mlx::core::Dtype dtype);
  mlx::core::array operator()(const mlx::core::array& x) const;

 private:
  mlx::core::array weight_t_;  // (in, out), pre-transposed.
  std::optional<mlx::core::array> bias_;
};

class Embedding {
 public:
  Embedding(const WeightBundle& bundle, std::string_view prefix,
            mlx::core::Dtype dtype);
  mlx::core::array operator()(const mlx::core::array& ids) const;
  mlx::core::Shape weight_shape() const { return weight_.shape(); }

 private:
  mlx::core::array weight_;
};

// ---------------------------------------------------------------------------
// CoCa-style multi-head attention (bias on every proj, optional logit cap,
// optional per-dim query scaling).
// ---------------------------------------------------------------------------

class CoCaMHA {
 public:
  CoCaMHA(const WeightBundle& bundle, std::string_view prefix, int d_model,
          int num_heads, int d_head, bool per_dim_scale, float atten_logit_cap,
          mlx::core::Dtype dtype);

  mlx::core::array operator()(
      const mlx::core::array& query, const mlx::core::array& key,
      const mlx::core::array& value,
      const std::optional<mlx::core::array>& additive_mask = std::nullopt) const;

 private:
  int num_heads_;
  int d_head_;
  float atten_logit_cap_;
  bool has_per_dim_scale_;
  Linear q_proj_;
  Linear k_proj_;
  Linear v_proj_;
  Linear o_proj_;
  std::optional<mlx::core::array> per_dim_scale_;  // (d_head,)
};

// ---------------------------------------------------------------------------
// Pre-norm transformer encoder layer (self-attention + GELU FFN).
// ---------------------------------------------------------------------------

class TransformerLayer {
 public:
  static constexpr float kAttenLogitCap = 50.0f;

  TransformerLayer(const WeightBundle& bundle, std::string_view prefix,
                   int d_model, int num_heads, int d_head, int d_ff,
                   mlx::core::Dtype dtype);

  mlx::core::array operator()(
      const mlx::core::array& x,
      const std::optional<mlx::core::array>& mask = std::nullopt) const;

 private:
  LayerNorm attn_ln_;
  CoCaMHA attn_;
  LayerNorm ffn_ln_;
  Linear ffn_w1_;
  Linear ffn_w2_;
};

// ---------------------------------------------------------------------------
// Attention pooler: cross-attention with a learned query.
// ---------------------------------------------------------------------------

class AttentionPooler {
 public:
  AttentionPooler(const WeightBundle& bundle, std::string_view prefix,
                  int d_model, int num_heads, int d_head, mlx::core::Dtype dtype);

  // ``encoder_output`` shape (B, S, D); returns (B, D).
  mlx::core::array operator()(const mlx::core::array& encoder_output) const;

 private:
  mlx::core::array query_;  // (1, 1, d_model)
  LayerNorm ln_;
  CoCaMHA attn_;
};

// ---------------------------------------------------------------------------
// Full encoder: text + audio stacks sharing infrastructure.
// ---------------------------------------------------------------------------

struct MusicCoCaConfig {
  int d_model = 768;
  int num_heads = 12;
  int d_head = 64;
  int d_ff = 3072;
  int num_layers = 12;
  int pooler_d_head = 256;
  int num_patches = 496;
  int patch_dim = 256;
  int vocab_size = 64000;
  int max_text_len = 128;
};

class MusicCoCaEncoder {
 public:
  MusicCoCaEncoder(const WeightBundle& bundle,
                   const MusicCoCaConfig& config = {},
                   mlx::core::Dtype dtype = mlx::core::float32);

  // ``token_ids`` shape (B, L), dtype int32 (or any integer); ``padding_mask``
  // shape (B, L), dtype float where 1 means "padding token" (mirrors the
  // upstream protocol). Returns (B, d_model).
  mlx::core::array embed_text(const mlx::core::array& token_ids,
                              const mlx::core::array& padding_mask) const;

  // ``log_mel`` shape (B, 992, 128). Returns (B, d_model).
  mlx::core::array embed_audio(const mlx::core::array& log_mel) const;

  const MusicCoCaConfig& config() const noexcept { return config_; }

 private:
  MusicCoCaConfig config_;
  mlx::core::Dtype dtype_;
  Linear patch_proj_;
  mlx::core::array pos_embedding_;  // (num_patches, d_model)
  Embedding token_embedding_;
  std::vector<TransformerLayer> audio_layers_;
  LayerNorm audio_exit_ln_;
  std::vector<TransformerLayer> text_layers_;
  LayerNorm text_exit_ln_;
  AttentionPooler music_pooler_;
  AttentionPooler text_pooler_;
};

}  // namespace magenta_realtime_mlx::musiccoca

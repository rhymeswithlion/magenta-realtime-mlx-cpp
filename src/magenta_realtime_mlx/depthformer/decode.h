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

// Depthformer decode loop: temperature/top-k sampling, classifier-free
// guidance, optional vocab masks, and the block-autoregressive
// ``generate_tokens`` driver used by the streaming system.
//
// Covers the production path used by the ``mlx-stream`` binary -- the
// instrumentation, intervention, and probing helpers from the Python module
// are intentionally out of scope (those exist for offline experiments).
//
// What we ship:
//   * ``top_k_logits``, ``sample_with_temperature`` -- direct ports.
//   * ``build_depth_draft``                          -- speculative draft helper.
//   * ``speculative_depth_decode``                   -- one-pass verification.
//   * ``generate_tokens``                            -- block-autoregressive
//                                                      decode (50 frames x 16
//                                                      RVQ levels by default),
//                                                      with optional speculative
//                                                      depth decoding when a
//                                                      previous-chunk context is
//                                                      provided.

#include <cstdint>
#include <optional>
#include <vector>

#include "magenta_realtime_mlx/depthformer/model.h"
#include "mlx/mlx.h"

namespace magenta_realtime_mlx::depthformer {

// Defaults match the Magenta RealTime fine-tuned base model
// (``mrt_merged_base.gin`` in the upstream reference repo).
struct GenerateOptions {
  int num_frames = 50;
  int rvq_depth = 16;
  float temperature = 1.1f;
  int top_k = 40;
  float guidance_weight = 4.0f;
  // ``seed.value()`` is forwarded to ``mx::random::seed`` exactly once before
  // the loop so that two runs with the same seed produce identical tokens.
  std::optional<uint64_t> seed;
};

// ``logits`` -> ``logits`` with everything outside the top-``k`` set to -inf.
// k <= 0 returns the input unchanged (no-op; semantics chosen to make
// the call site safe regardless of whether the user opted into top-k).
mlx::core::array top_k_logits(const mlx::core::array& logits, int k);

// Sample one token per row from (B, vocab) logits. Returns shape (B,) int32.
//   * temperature == 0  -> argmax
//   * temperature  > 0  -> categorical(logits / T) after optional top-k mask
mlx::core::array sample_with_temperature(const mlx::core::array& logits,
                                         float temperature, int top_k);

// Build a length-``rvq_depth`` int32 draft from a previous chunk's tokens, or
// nullopt when the slot is invalid (no prior context, off the front, or any
// negative entry). ``context_tokens_llm`` is shape ``(context_frames, rvq_depth)``.
std::optional<std::vector<int32_t>> build_depth_draft(
    const std::vector<std::vector<int32_t>>& context_tokens_llm, int frame_idx,
    int rvq_depth, int chunk_length_frames);

// Result of ``speculative_depth_decode``.
struct SpeculativeResult {
  std::vector<int32_t> accepted;  // length rvq_depth, only ``num_accepted`` plus the rejection are valid
  int num_accepted = 0;            // number of *matched* draft tokens (rejection slot at index num_accepted)
};

// Verify ``draft_tokens`` against the model with one ``depth_forward_full`` and
// return the longest matching prefix plus the resampled rejection token at
// position ``num_accepted``. ``vocab_masks`` may contain nullopt entries.
SpeculativeResult speculative_depth_decode(
    const Depthformer& model, const mlx::core::array& temporal_out,
    const std::vector<int32_t>& draft_tokens,
    const std::vector<std::optional<mlx::core::array>>& vocab_masks,
    float cfg_scale_cond, float cfg_scale_uncond, float temperature,
    int top_k);

// Block-autoregressive token decoder.
//
// ``encoder_input_tokens`` is shape ``(2, enc_len)`` int32. Row 0 is the
// conditioned encoding, row 1 is the unconditioned (CFG) encoding.
//
// Returns shape ``(num_frames, rvq_depth)`` int32 (LLM token IDs).
//
// ``vocab_masks`` (optional) has length ``rvq_depth``; each entry may be a
// (vocab,) bool / int mask tensor. ``nullopt`` means no mask for that depth.
//
// ``context_tokens_llm`` enables speculative depth decoding when non-empty.
//
// ``precomputed_encoder_output`` is an optional shortcut for the cross-chunk
// CPU encoder pipeline (see ``System::generate_chunk`` /
// ``GenerateChunkOptions::pipeline_encoder``). When set, the function skips
// ``model.encode(encoder_input_tokens)`` and uses the supplied tensor
// directly. Shape must match what ``model.encode`` would produce
// ``(B=2, seq_len, d_model)``. ``encoder_input_tokens`` is still required
// for shape inference / API symmetry (and is not touched in this path).
mlx::core::array generate_tokens(
    const Depthformer& model, const mlx::core::array& encoder_input_tokens,
    const GenerateOptions& options = {},
    const std::vector<std::optional<mlx::core::array>>& vocab_masks = {},
    const std::vector<std::vector<int32_t>>& context_tokens_llm = {},
    int chunk_length_frames = 50,
    const std::optional<mlx::core::array>& precomputed_encoder_output =
        std::nullopt);

}  // namespace magenta_realtime_mlx::depthformer

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

#include "magenta_realtime_mlx/depthformer/decode.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mlx/random.h"
#include "mlx/transforms.h"

namespace magenta_realtime_mlx::depthformer {

namespace mx = mlx::core;

namespace {

// Granular profiling toggle. Set ``MRT_PROFILE_DECODE=1`` (or any non-empty
// value) to print a per-chunk breakdown of where wall-time goes. Off by
// default and zero-overhead when off (single env lookup at chunk start).
struct ProfileToggle {
  bool enabled;
  ProfileToggle()
      : enabled(std::getenv("MRT_PROFILE_DECODE") != nullptr &&
                std::getenv("MRT_PROFILE_DECODE")[0] != '\0') {}
};

bool profile_enabled() {
  static const ProfileToggle kToggle{};
  return kToggle.enabled;
}

using Clock = std::chrono::steady_clock;
using NS = std::chrono::nanoseconds;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Hot-path constant: ``mx::where(..., neg_inf, ...)`` is used ~800x/chunk.
// Holding a single 0-d fp32 array avoids constructing a fresh
// ``mx::full(shape, -inf, dtype)`` on every call, which would build a fresh
// op node + scalar literal each time.
// ``mx::where`` broadcasts the 0-d operand so the result shape is right.
const mx::array& neg_inf_scalar() {
  static const mx::array kNegInf{-std::numeric_limits<float>::infinity()};
  return kNegInf;
}

// CFG: row 0 = conditioned, row 1 = unconditioned. Returns (1, vocab) fp32.
// Scale factors are passed as already-constructed 0-d arrays so the caller
// can hoist them out of the per-frame inner loop (called ~800x per chunk).
mx::array cfg_combine(const mx::array& logits_2_v,
                      const mx::array& cfg_scale_cond,
                      const mx::array& cfg_scale_uncond) {
  // logits shape (2, vocab); slice rows 0:1 and 1:2 to keep shape (1, vocab).
  mx::array cond =
      mx::astype(mx::slice(logits_2_v, mx::Shape{0, 0},
                           mx::Shape{1, static_cast<int32_t>(logits_2_v.shape(1))}),
                 mx::float32);
  mx::array unc =
      mx::astype(mx::slice(logits_2_v, mx::Shape{1, 0},
                           mx::Shape{2, static_cast<int32_t>(logits_2_v.shape(1))}),
                 mx::float32);
  return mx::subtract(mx::multiply(cfg_scale_cond, cond),
                      mx::multiply(cfg_scale_uncond, unc));
}

mx::array apply_vocab_mask(const mx::array& logits,
                           const std::optional<mx::array>& mask) {
  if (!mask.has_value()) return logits;
  return mx::where(*mask, logits, neg_inf_scalar());
}

// Materialise (B,) int32 token to (2, 1) int32 ready for token_embedding lookup.
// Mirrors ``_broadcast_cfg_token``: same id used for both CFG rows.
mx::array broadcast_cfg_token(const mx::array& token_b1) {
  // token_b1: shape (1,) int32. Broadcast to (2, 1).
  mx::array t = mx::reshape(mx::astype(token_b1, mx::int32),
                            mx::Shape{1, 1});
  return mx::broadcast_to(t, mx::Shape{2, 1});
}

}  // namespace

mx::array top_k_logits(const mx::array& logits, int k) {
  if (k <= 0) return logits;
  // Standard top-k masking: threshold = min of the top-k values, mask
  // anything strictly below it. ``mx::min`` keeps this correct
  // regardless of ``mx::topk``'s sort convention (it currently returns
  // ascending, so ``values[..., 0]`` would also work, but ``min`` makes
  // intent obvious and is order-independent).
  //
  // Note: prior to fixing this, both Python and C++ used
  // ``values[..., -1:]`` as the threshold, which picks the *max* of the
  // top-k -- effectively collapsing every call to top-1 / argmax-with-
  // randomness. With temperature=1.1, top_k=40 the bug produced much
  // less diverse audio than intended. The bug was fixed in the
  // ``top_k_logits was effective top-1`` commit; this comment is left
  // as a tombstone for git archaeology.
  mx::array values = mx::topk(logits, k, /*axis=*/-1);
  mx::array threshold = mx::min(values, /*axis=*/-1, /*keepdims=*/true);
  return mx::where(mx::less(logits, threshold), neg_inf_scalar(), logits);
}

mx::array sample_with_temperature(const mx::array& logits, float temperature,
                                  int top_k) {
  if (temperature == 0.0f) {
    return mx::argmax(logits, /*axis=*/-1);
  }
  mx::array scaled = mx::divide(mx::astype(logits, mx::float32),
                                mx::array(temperature));
  if (top_k > 0) {
    scaled = top_k_logits(scaled, top_k);
  }
  return mx::random::categorical(scaled, /*axis=*/-1);
}

// Hot-path-friendly variant: caller passes a pre-built temperature scalar so
// the inner depth-decode loop doesn't construct a fresh ``mx::array(t)`` for
// every one of its ~800 calls per chunk.
mx::array sample_with_temperature(const mx::array& logits,
                                  const mx::array& temperature,
                                  int top_k) {
  mx::array scaled = mx::divide(mx::astype(logits, mx::float32), temperature);
  if (top_k > 0) {
    scaled = top_k_logits(scaled, top_k);
  }
  return mx::random::categorical(scaled, /*axis=*/-1);
}

std::optional<std::vector<int32_t>> build_depth_draft(
    const std::vector<std::vector<int32_t>>& context_tokens_llm, int frame_idx,
    int rvq_depth, int chunk_length_frames) {
  if (context_tokens_llm.empty()) return std::nullopt;
  const int total = static_cast<int>(context_tokens_llm.size());
  const int context_frame = -chunk_length_frames + frame_idx;
  if (context_frame < -total || context_frame >= 0) return std::nullopt;
  // Translate negative index to positive.
  const int pos = total + context_frame;
  const auto& row = context_tokens_llm.at(pos);
  if (static_cast<int>(row.size()) < rvq_depth) return std::nullopt;
  std::vector<int32_t> draft(row.begin(), row.begin() + rvq_depth);
  for (int v : draft) {
    if (v < 0) return std::nullopt;
  }
  return draft;
}

SpeculativeResult speculative_depth_decode(
    const Depthformer& model, const mx::array& temporal_out,
    const std::vector<int32_t>& draft_tokens,
    const std::vector<std::optional<mx::array>>& vocab_masks,
    float cfg_scale_cond, float cfg_scale_uncond, float temperature,
    int top_k) {
  const int k_len = static_cast<int>(draft_tokens.size());
  const int B = static_cast<int>(temporal_out.shape(0));
  if (k_len <= 0) {
    return SpeculativeResult{};
  }
  // Build per-call scalar arrays once; cheap because k_len <= rvq_depth.
  const mx::array cfg_scale_cond_arr(cfg_scale_cond);
  const mx::array cfg_scale_uncond_arr(cfg_scale_uncond);
  const mx::array temperature_arr(temperature);

  // Build draft embedding for tokens 0 .. k_len-2 (the last token is verified
  // against the rejection sampling result, never fed back as an input).
  mx::array draft_emb = temporal_out;  // placeholder; will be overwritten
  if (k_len > 1) {
    std::vector<int32_t> tail(draft_tokens.begin(),
                              draft_tokens.begin() + (k_len - 1));
    mx::array tail_arr(tail.data(), mx::Shape{static_cast<int32_t>(k_len - 1)},
                       mx::int32);
    mx::array idx2 = mx::broadcast_to(mx::expand_dims(tail_arr, 0),
                                      mx::Shape{static_cast<int32_t>(B),
                                                static_cast<int32_t>(k_len - 1)});
    draft_emb = model.token_embedding()(idx2);
  } else {
    // Empty trailing embedding: shape (B, 0, d_model).
    draft_emb = mx::zeros(mx::Shape{static_cast<int32_t>(B), 0,
                                    static_cast<int32_t>(temporal_out.shape(2))},
                          temporal_out.dtype());
  }
  mx::array inputs = mx::concatenate({temporal_out, draft_emb}, /*axis=*/1);
  mx::array all_logits = model.depth_forward_full(inputs);  // (B, k_len, V)
  mx::eval(all_logits);

  SpeculativeResult result;
  result.accepted.assign(draft_tokens.begin(), draft_tokens.end());
  result.num_accepted = k_len;

  for (int i = 0; i < k_len; ++i) {
    // Slice (B, 1, V) at depth i.
    mx::array slice_logits = mx::slice(
        all_logits,
        mx::Shape{0, static_cast<int32_t>(i), 0},
        mx::Shape{static_cast<int32_t>(B), static_cast<int32_t>(i + 1),
                  static_cast<int32_t>(all_logits.shape(2))});
    // Reshape -> (B, V).
    mx::array logits_bv = mx::reshape(
        slice_logits, mx::Shape{static_cast<int32_t>(B),
                                static_cast<int32_t>(all_logits.shape(2))});
    mx::array guided = cfg_combine(logits_bv, cfg_scale_cond_arr,
                                   cfg_scale_uncond_arr);
    if (i < static_cast<int>(vocab_masks.size())) {
      guided = apply_vocab_mask(guided, vocab_masks[i]);
    }
    mx::array token = sample_with_temperature(guided, temperature_arr, top_k);
    mx::eval(token);
    const int32_t model_id = token.item<int32_t>();
    if (draft_tokens[i] != model_id) {
      result.accepted[i] = model_id;
      result.num_accepted = i;
      break;
    }
    result.accepted[i] = model_id;
  }
  return result;
}

mx::array generate_tokens(
    const Depthformer& model, const mx::array& encoder_input_tokens,
    const GenerateOptions& options,
    const std::vector<std::optional<mx::array>>& vocab_masks,
    const std::vector<std::vector<int32_t>>& context_tokens_llm,
    int chunk_length_frames,
    const std::optional<mx::array>& precomputed_encoder_output) {
  if (options.seed.has_value()) {
    mx::random::seed(*options.seed);
  }
  if (!vocab_masks.empty() &&
      static_cast<int>(vocab_masks.size()) != options.rvq_depth) {
    throw std::runtime_error(
        "generate_tokens: vocab_masks size must equal rvq_depth (or be empty)");
  }

  const float cfg_scale_cond = 1.0f + options.guidance_weight;
  const float cfg_scale_uncond = options.guidance_weight;

  // Hot-path scalars: build once per chunk so the inner depth loop reuses
  // the same 0-d arrays each step instead of constructing fresh literals
  // (~3000 redundant scalar arrays per chunk pre-hoist).
  const mx::array cfg_scale_cond_arr(cfg_scale_cond);
  const mx::array cfg_scale_uncond_arr(cfg_scale_uncond);
  // ``rvq_depth_arr`` divides ``frame_embed_sum`` (model dtype, e.g. bf16)
  // to produce ``prev_frame_mean``. Constructing it as ``float`` (fp32)
  // promotes the result to fp32, which then cascades through every
  // downstream temporal_step input -> K/V projection -> hidden state and
  // forces the per-call fp32 -> bf16 casts in the .mlxfn dispatch path
  // (see g22 trace investigation). The upstream ``frame_embed_sum /
  // rvq_depth`` uses an integer divisor and so preserves the array's
  // dtype; we build this scalar directly in the model dtype to match.
  const mx::array rvq_depth_arr =
      mx::astype(mx::array(static_cast<float>(options.rvq_depth)),
                 model.dtype());
  // ``temperature == 0`` short-circuits to ``argmax`` (no divide), so we only
  // build the temperature scalar when it would actually be used. Constructing
  // ``mx::array(0.0f)`` here would silently propagate to the array overload
  // of ``sample_with_temperature`` and produce a divide-by-zero (NaN logits,
  // garbage tokens). Keep ``options.temperature`` for the float-overload
  // dispatch in the temp==0 path.
  const bool temp_is_zero = (options.temperature == 0.0f);
  const mx::array temperature_arr(temp_is_zero ? 1.0f : options.temperature);

  const bool prof = profile_enabled();
  Clock::time_point chunk_t0 = prof ? Clock::now() : Clock::time_point{};
  double encoder_ms = 0.0;
  double bias_ms = 0.0;
  double temporal_step_ms = 0.0;
  double depth_step_ms = 0.0;
  double sample_ms = 0.0;
  double frame_eval_ms = 0.0;
  double frame_post_ms = 0.0;
  int profiled_frames = 0;

  Clock::time_point t_phase = prof ? Clock::now() : Clock::time_point{};
  // Use the precomputed encoder output when the caller supplied one
  // (cross-chunk CPU encoder pipeline -- see ``System::generate_chunk``
  // / ``GenerateChunkOptions::pipeline_encoder``). The cached array was
  // ``async_eval``'d at the end of the previous chunk on the CPU stream;
  // here we just adopt it. The first temporal_step's eval implicitly
  // waits for it via the cross-attn dependency, so we don't need an
  // explicit ``mx::eval`` to gate downstream GPU work -- the very first
  // ``frame_eval`` will sync against the (probably-already-finished) CPU
  // result. When no cache is supplied (first chunk, or pipelining
  // disabled) we fall back to the original synchronous GPU encode.
  mx::array encoder_output = precomputed_encoder_output.has_value()
                                 ? *precomputed_encoder_output
                                 : model.encode(encoder_input_tokens);
  if (!precomputed_encoder_output.has_value()) {
    mx::eval(encoder_output);
  }
  if (prof) encoder_ms += ms_since(t_phase);

  if (prof) t_phase = Clock::now();
  std::vector<mx::array> temporal_biases =
      model.precompute_temporal_biases(options.num_frames);
  std::vector<mx::array> depth_biases =
      model.precompute_depth_biases(options.rvq_depth);
  // No-op materialisation hint -- biases are tiny but we want them off the
  // critical path for the first temporal_step.
  for (auto& b : temporal_biases) mx::eval(b);
  for (auto& b : depth_biases) mx::eval(b);
  if (prof) bias_ms += ms_since(t_phase);

  const int d_model = model.config().d_model;
  const int B = static_cast<int>(encoder_input_tokens.shape(0));  // expected 2
  const int V = model.config().vocab_size;

  // Output buffer: (num_frames * rvq_depth) flat int32.
  std::vector<int32_t> generated(static_cast<size_t>(options.num_frames) *
                                     static_cast<size_t>(options.rvq_depth),
                                 0);

  std::vector<LayerCache> temporal_cache = model.empty_temporal_cache();
  mx::array prev_frame_mean =
      mx::zeros(mx::Shape{static_cast<int32_t>(B), 1,
                          static_cast<int32_t>(d_model)},
                model.dtype());

  // Initial pad token (zeros) for the first temporal step.
  mx::array pad_tokens =
      mx::zeros(mx::Shape{static_cast<int32_t>(B), 1}, mx::int32);

  // ``MRT_PROFILE_DECODE=2`` enables the deep mode that adds a synthetic
  // ``mx::eval`` after each temporal_step so we can attribute per-frame work
  // between temporal and depth phases. Wall time becomes worse in this mode
  // (more host syncs) but the breakdown is more informative.
  const bool prof_deep =
      prof && std::getenv("MRT_PROFILE_DECODE") &&
      std::string(std::getenv("MRT_PROFILE_DECODE")) == "2";
  double temporal_eval_ms = 0.0;

  // Async pipelining: in the non-speculative path we ``async_eval``
  // ``prev_frame_mean`` (the only frame-N output the next iteration
  // consumes) and accumulate the per-frame token vectors for a single
  // host readback after the loop. This lets the GPU keep frame N+1's
  // work in flight while the CPU is still building frame N+2's graph,
  // hiding the C++ binding's per-call dispatch overhead.
  // ``MRT_DISABLE_ASYNC_EVAL=1`` falls back to the original blocking
  // per-frame eval for A/B comparison and profile-mode honesty.
  const bool disable_async = std::getenv("MRT_DISABLE_ASYNC_EVAL") != nullptr &&
                             std::getenv("MRT_DISABLE_ASYNC_EVAL")[0] != '\0';
  std::vector<mx::array> stacked_frames;
  if (!disable_async) stacked_frames.reserve(options.num_frames);
  for (int frame_idx = 0; frame_idx < options.num_frames; ++frame_idx) {
    Clock::time_point t_temporal = prof ? Clock::now() : Clock::time_point{};
    mx::array temporal_input = (frame_idx == 0)
                                   ? model.token_embedding()(pad_tokens)
                                   : prev_frame_mean;
    mx::array temporal_out = model.temporal_step(
        temporal_input, encoder_output, temporal_cache, frame_idx,
        temporal_biases[frame_idx]);
    if (prof) temporal_step_ms += ms_since(t_temporal);
    if (prof_deep) {
      Clock::time_point t_te = Clock::now();
      mx::eval(temporal_out);
      temporal_eval_ms += ms_since(t_te);
    }

    // Optional speculative path.
    std::optional<std::vector<int32_t>> draft = build_depth_draft(
        context_tokens_llm, frame_idx, options.rvq_depth, chunk_length_frames);

    mx::array frame_embed_sum =
        mx::zeros(mx::Shape{static_cast<int32_t>(B), 1,
                            static_cast<int32_t>(d_model)},
                  temporal_out.dtype());

    if (draft.has_value()) {
      SpeculativeResult sr = speculative_depth_decode(
          model, temporal_out, *draft, vocab_masks, cfg_scale_cond,
          cfg_scale_uncond, options.temperature, options.top_k);

      const int fill_end =
          (sr.num_accepted < options.rvq_depth) ? sr.num_accepted + 1 : options.rvq_depth;
      for (int d = 0; d < fill_end; ++d) {
        generated[static_cast<size_t>(frame_idx) * options.rvq_depth + d] =
            sr.accepted[d];
        const int32_t tid = sr.accepted[d];
        std::vector<int32_t> bid{tid, tid};
        mx::array tok_2(bid.data(), mx::Shape{2, 1}, mx::int32);
        frame_embed_sum =
            mx::add(frame_embed_sum, model.token_embedding()(tok_2));
      }

      if (fill_end < options.rvq_depth) {
        // Resume the depth cache up through the accepted tokens, then
        // continue generating to fill the frame.
        std::vector<LayerCache> depth_cache = model.empty_depth_cache();
        for (int d = 0; d < fill_end; ++d) {
          if (d == 0) {
            (void)model.depth_step(temporal_out, depth_cache, d,
                                   depth_biases[d]);
          } else {
            const int32_t prev = sr.accepted[d - 1];
            std::vector<int32_t> bid{prev, prev};
            mx::array step_in = model.token_embedding()(
                mx::array(bid.data(), mx::Shape{2, 1}, mx::int32));
            (void)model.depth_step(step_in, depth_cache, d, depth_biases[d]);
          }
        }
        const int32_t last = sr.accepted[fill_end - 1];
        std::vector<int32_t> last_bid{last, last};
        mx::array depth_input = model.token_embedding()(
            mx::array(last_bid.data(), mx::Shape{2, 1}, mx::int32));
        for (int depth_idx = fill_end; depth_idx < options.rvq_depth;
             ++depth_idx) {
          mx::array logits =
              model.depth_step(depth_input, depth_cache, depth_idx,
                               depth_biases[depth_idx]);
          mx::array guided =
              cfg_combine(logits, cfg_scale_cond_arr, cfg_scale_uncond_arr);
          if (depth_idx < static_cast<int>(vocab_masks.size())) {
            guided = apply_vocab_mask(guided, vocab_masks[depth_idx]);
          }
          mx::array token =
              temp_is_zero
                  ? mx::argmax(guided, /*axis=*/-1)
                  : sample_with_temperature(guided, temperature_arr,
                                            options.top_k);
          mx::eval(token);
          const int32_t tid = token.item<int32_t>();
          generated[static_cast<size_t>(frame_idx) * options.rvq_depth +
                    depth_idx] = tid;
          std::vector<int32_t> bid{tid, tid};
          depth_input = model.token_embedding()(
              mx::array(bid.data(), mx::Shape{2, 1}, mx::int32));
          frame_embed_sum = mx::add(frame_embed_sum, depth_input);
        }
      }
      prev_frame_mean = mx::divide(frame_embed_sum, rvq_depth_arr);
      mx::eval(prev_frame_mean);
    } else {
      // Plain non-speculative depth decode; one ``mx.eval`` per frame.
      std::vector<LayerCache> depth_cache = model.empty_depth_cache();
      mx::array depth_input = temporal_out;
      std::vector<mx::array> depth_tokens;
      depth_tokens.reserve(options.rvq_depth);
      for (int depth_idx = 0; depth_idx < options.rvq_depth; ++depth_idx) {
        Clock::time_point t_depth = prof ? Clock::now() : Clock::time_point{};
        mx::array logits =
            model.depth_step(depth_input, depth_cache, depth_idx,
                             depth_biases[depth_idx]);
        if (prof) depth_step_ms += ms_since(t_depth);
        Clock::time_point t_samp = prof ? Clock::now() : Clock::time_point{};
        mx::array guided = cfg_combine(logits, cfg_scale_cond_arr,
                                       cfg_scale_uncond_arr);
        if (depth_idx < static_cast<int>(vocab_masks.size())) {
          guided = apply_vocab_mask(guided, vocab_masks[depth_idx]);
        }
        mx::array token =
            temp_is_zero
                ? mx::argmax(guided, /*axis=*/-1)
                : sample_with_temperature(guided, temperature_arr,
                                          options.top_k);
        depth_tokens.push_back(token);
        depth_input = model.token_embedding()(broadcast_cfg_token(token));
        frame_embed_sum = mx::add(frame_embed_sum, depth_input);
        if (prof) sample_ms += ms_since(t_samp);
      }

      mx::array stacked = mx::reshape(
          mx::stack(depth_tokens, /*axis=*/0),
          mx::Shape{static_cast<int32_t>(options.rvq_depth)});
      prev_frame_mean = mx::divide(frame_embed_sum, rvq_depth_arr);
      Clock::time_point t_eval = prof ? Clock::now() : Clock::time_point{};
      if (disable_async) {
        // Original blocking path: eval both per-frame and copy stacked
        // synchronously. Used for A/B benchmarking and as a safety net.
        mx::eval(stacked, prev_frame_mean);
        if (prof) frame_eval_ms += ms_since(t_eval);
        Clock::time_point t_post = prof ? Clock::now() : Clock::time_point{};
        std::vector<int32_t> stacked_host(options.rvq_depth);
        const int32_t* src = stacked.data<int32_t>();
        std::memcpy(stacked_host.data(), src,
                    sizeof(int32_t) * options.rvq_depth);
        std::memcpy(&generated[static_cast<size_t>(frame_idx) *
                               options.rvq_depth],
                    stacked_host.data(), sizeof(int32_t) * options.rvq_depth);
        if (prof) frame_post_ms += ms_since(t_post);
      } else {
        // Async path: kick off ``prev_frame_mean`` so the GPU starts the
        // depth-mean reduction while the CPU keeps building the next
        // frame's temporal_step graph. Defer the ``stacked`` host
        // readback until after the loop -- nothing in subsequent frames
        // consumes per-frame tokens (speculative draft uses the prior
        // chunk's tokens, not in-progress ones).
        mx::async_eval(prev_frame_mean);
        stacked_frames.push_back(stacked);
        if (prof) frame_eval_ms += ms_since(t_eval);
      }
      if (prof) ++profiled_frames;
    }
  }

  if (!disable_async && !stacked_frames.empty()) {
    // Single readback for all frame token vectors. ``stack`` builds a
    // (num_frames, rvq_depth) int32 array; one ``eval`` waits for the
    // entire chunk's GPU work to complete (including all the
    // ``async_eval``'d ``prev_frame_mean`` arrays they share work with).
    Clock::time_point t_final = prof ? Clock::now() : Clock::time_point{};
    mx::array all_stacked = mx::stack(stacked_frames, /*axis=*/0);
    mx::eval(all_stacked);
    const int32_t* src = all_stacked.data<int32_t>();
    std::memcpy(generated.data(), src,
                sizeof(int32_t) * generated.size());
    if (prof) frame_post_ms += ms_since(t_final);
  }

    if (prof) {
      const double total_ms = ms_since(chunk_t0);
      const double accounted = encoder_ms + bias_ms + temporal_step_ms +
                               depth_step_ms + sample_ms + frame_eval_ms +
                               frame_post_ms + temporal_eval_ms;
      const double overhead = total_ms - accounted;
      std::fprintf(
          stderr,
          "[mrt] decode profile: total %.1f ms (encoder %.1f, bias %.1f, "
          "temporal_step %.1f, depth_step %.1f, sample+cfg+embed %.1f, "
          "frame_eval %.1f, post %.1f, other %.1f)\n",
          total_ms, encoder_ms, bias_ms, temporal_step_ms, depth_step_ms,
          sample_ms, frame_eval_ms, frame_post_ms, overhead);
      if (prof_deep) {
        std::fprintf(stderr,
                     "[mrt] deep: temporal_eval %.1f ms (depth+sample = "
                     "frame_eval - temporal_eval = %.1f ms)\n",
                     temporal_eval_ms, frame_eval_ms);
      }
      if (profiled_frames > 0) {
        std::fprintf(stderr,
                     "[mrt] per-frame: temporal %.2f ms, depth_step %.2f ms "
                     "(x%d), sample+cfg %.2f ms, eval %.2f ms\n",
                     temporal_step_ms / profiled_frames,
                     depth_step_ms / (profiled_frames * options.rvq_depth),
                     options.rvq_depth, sample_ms / profiled_frames,
                     frame_eval_ms / profiled_frames);
        if (prof_deep) {
          std::fprintf(
              stderr,
              "[mrt] deep per-frame: temporal_eval %.2f ms, depth_eval %.2f "
              "ms\n",
              temporal_eval_ms / profiled_frames,
              (frame_eval_ms - 0.0) / profiled_frames);
        }
      }
    }

  (void)V;
  // Wrap the host buffer as a (num_frames, rvq_depth) int32 array.
  return mx::array(generated.data(),
                   mx::Shape{static_cast<int32_t>(options.num_frames),
                             static_cast<int32_t>(options.rvq_depth)},
                   mx::int32);
}

}  // namespace magenta_realtime_mlx::depthformer

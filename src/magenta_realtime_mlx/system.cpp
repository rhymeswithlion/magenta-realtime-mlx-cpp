#include "magenta_realtime_mlx/system.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "magenta_realtime_mlx/audio.h"
#include "magenta_realtime_mlx/rvq.h"

namespace magenta_realtime_mlx {

namespace mx = mlx::core;
namespace df = magenta_realtime_mlx::depthformer;

namespace {

template <class... Ts>
mx::Shape S(Ts... vs) {
  return mx::Shape{static_cast<int32_t>(vs)...};
}

// Extract (num_frames, rvq_depth) int32 host buffer from an already-evaluated
// mx::array.
std::vector<int32_t> to_host_i32(const mx::array& a) {
  const int n = static_cast<int>(a.size());
  std::vector<int32_t> out(n);
  std::memcpy(out.data(), a.data<int32_t>(), sizeof(int32_t) * n);
  return out;
}

// Build the flat ``(2 * encoder_seq_len,)`` int32 host buffer that the
// Depthformer encoder consumes -- two rows: row 0 is conditioned
// (codec context + style), row 1 is the CFG-unconditioned variant
// (codec context + masked style). Extracted out of ``generate_chunk``
// so the cross-chunk encoder pipeline can call it twice per chunk
// (once for the current input, once with the just-updated context for
// the next chunk's pre-encode).
//
// ``context_tokens_host`` is shape ``(ctx_frames, rvq_depth)`` -- the
// raw int32 codec tokens stored on ``SystemState``. Negative entries
// (empty slots) are mapped to ``MASK``. ``style_pos_host`` is the
// length-``encoder_style_rvq_depth`` host vector of style-vocab token
// IDs (already ``mx::eval``'d on the caller side).
std::vector<int32_t> build_encoder_inputs_host(
    const std::vector<int32_t>& context_tokens_host, int ctx_frames,
    int rvq_depth, const SystemConfig& config,
    const std::vector<int32_t>& style_pos_host) {
  const int enc_cdepth = config.encoder_codec_rvq_depth;
  const int style_depth = config.encoder_style_rvq_depth;
  const int enc_len = ctx_frames * enc_cdepth + style_depth;
  std::vector<int32_t> out(2 * enc_len);

  // Map context (only the first ``enc_cdepth`` levels are visible to
  // the encoder) to LLM vocab IDs, with MASK for empty (-1) slots.
  // We write directly into ``out`` row 0's leading region rather than
  // building a separate ``encoder_codec`` buffer.
  const int32_t mask = config.vocab_mask_token();
  const int32_t codec_off = config.vocab_codec_offset();
  const int codebook = config.codec_rvq_codebook_size;
  for (int f = 0; f < ctx_frames; ++f) {
    for (int d = 0; d < enc_cdepth; ++d) {
      const int32_t v = context_tokens_host[f * rvq_depth + d];
      out[f * enc_cdepth + d] =
          (v >= 0) ? (codec_off + d * codebook + v) : mask;
    }
  }
  std::memcpy(&out[ctx_frames * enc_cdepth], style_pos_host.data(),
              sizeof(int32_t) * style_depth);

  // Row 1: same codec context, masked style. The codec block is
  // identical to row 0 -- copy it rather than re-deriving from
  // ``context_tokens_host``.
  std::memcpy(&out[enc_len], &out[0],
              sizeof(int32_t) * ctx_frames * enc_cdepth);
  std::fill(&out[enc_len + ctx_frames * enc_cdepth], &out[2 * enc_len], mask);
  return out;
}

}  // namespace

System::System(const std::filesystem::path& cache_root, std::string_view tag,
               mx::Dtype llm_dtype)
    : config_(), llm_dtype_(llm_dtype) {
  paths_ = resolve_inference_bundle(cache_root, tag);

  ss_decoder_bundle_ =
      std::make_unique<WeightBundle>(paths_.spectrostream_decoder);
  ss_codebooks_bundle_ =
      std::make_unique<WeightBundle>(paths_.spectrostream_codebooks);
  ss_encoder_bundle_ =
      std::make_unique<WeightBundle>(paths_.spectrostream_encoder);
  mc_encoder_bundle_ =
      std::make_unique<WeightBundle>(paths_.musiccoca_encoder);
  mc_codebooks_bundle_ =
      std::make_unique<WeightBundle>(paths_.musiccoca_codebooks);
  df_bundle_ = std::make_unique<WeightBundle>(paths_.depthformer);

  // Codec and MusicCoCa always run in float32 to mirror Python -- only the
  // depthformer honours the user-selected dtype.
  decoder_ =
      std::make_unique<SpectroStreamDecoder>(*ss_decoder_bundle_, mx::float32);
  llm_ = std::make_unique<df::Depthformer>(
      *df_bundle_, df::DepthformerConfig::base(), llm_dtype_);
  style_model_ = std::make_unique<StyleModel>(
      *mc_encoder_bundle_, *mc_codebooks_bundle_, paths_.musiccoca_vocab,
      mx::float32);

  // Wrap the depthformer's hot paths in ``mx::compile`` so each call
  // fires a single fused Metal kernel instead of
  // tens of small kernel launches per op. Without this, C++ runs ~2x slower
  // than ``make mlx-stream`` (eager dispatch) on the same machine.
  llm_->compile_for_inference(/*compile_decode_steps=*/true);

  build_vocab_masks();
}

void System::build_vocab_masks() {
  const int depth = config_.decoder_codec_rvq_depth;
  const int vocab = config_.llm_vocab_size;
  vocab_masks_.clear();
  vocab_masks_.reserve(depth);
  for (int level = 0; level < depth; ++level) {
    const int start = config_.vocab_codec_offset() +
                      level * config_.codec_rvq_codebook_size;
    const int end = start + config_.codec_rvq_codebook_size;
    std::vector<bool> host(vocab, false);
    for (int i = start; i < end; ++i) host[i] = true;
    // Pack into an int8 buffer because MLX C++ doesn't construct arrays
    // directly from ``bool*``. ``where`` accepts int (truthy) masks.
    std::vector<int8_t> packed(vocab, 0);
    for (int i = 0; i < vocab; ++i) packed[i] = host[i] ? 1 : 0;
    mx::array mask(packed.data(), mx::Shape{static_cast<int32_t>(vocab)},
                   mx::int8);
    // Cast to bool once to make ``where`` broadcast behaviour explicit.
    mx::array as_bool = mx::astype(mask, mx::bool_);
    mx::eval(as_bool);
    vocab_masks_.emplace_back(as_bool);
  }
}

mx::array System::embed_style(std::string_view prompt) const {
  return style_model_->style_tokens_lm(prompt, config_.encoder_style_rvq_depth,
                                        config_.style_rvq_codebook_size,
                                        config_.vocab_style_offset());
}

SystemState System::empty_state() const {
  SystemState s;
  s.context_frames = config_.context_length_frames();
  s.rvq_depth = config_.decoder_codec_rvq_depth;
  s.context_tokens.assign(
      static_cast<size_t>(s.context_frames) * s.rvq_depth, -1);
  s.chunk_index = 0;
  s.num_channels = config_.codec_num_channels;
  s.crossfade_num_samples = 0;  // crossfade only applies after first chunk
  return s;
}

GenerateResult System::generate_chunk(
    SystemState& state, const std::optional<mx::array>& style_tokens_lm,
    const GenerateChunkOptions& options) const {
  if (state.context_frames != config_.context_length_frames() ||
      state.rvq_depth != config_.decoder_codec_rvq_depth) {
    throw std::runtime_error(
        "System::generate_chunk: state is not configured for this System");
  }

  const int ctx_frames = state.context_frames;
  const int rvq_depth = state.rvq_depth;
  const int chunk_frames = config_.chunk_length_frames();
  const int xf_frames = config_.crossfade_length_frames();
  const int xf_samples = config_.crossfade_length_samples();

  // ---- Build encoder input tokens (conditioned + unconditioned) ------------
  // Eval the style tokens once (typically a no-op -- the caller in
  // ``mlx-stream`` evals once at session start and reuses the same
  // array). Stored as a host vector so both the current call and the
  // pipelined next-chunk pre-encode can reuse it.
  std::vector<int32_t> style_pos(config_.encoder_style_rvq_depth,
                                 config_.vocab_mask_token());
  if (style_tokens_lm.has_value()) {
    mx::array evald = *style_tokens_lm;
    mx::eval(evald);
    if (static_cast<int>(evald.size()) != config_.encoder_style_rvq_depth) {
      throw std::runtime_error(
          "System::generate_chunk: style_tokens_lm size != "
          "encoder_style_rvq_depth");
    }
    std::memcpy(style_pos.data(), evald.data<int32_t>(),
                sizeof(int32_t) * config_.encoder_style_rvq_depth);
  }

  std::vector<int32_t> encoder_inputs = build_encoder_inputs_host(
      state.context_tokens, ctx_frames, rvq_depth, config_, style_pos);
  const int enc_len =
      ctx_frames * config_.encoder_codec_rvq_depth +
      config_.encoder_style_rvq_depth;

  mx::array encoder_tokens(encoder_inputs.data(),
                           mx::Shape{2, static_cast<int32_t>(enc_len)},
                           mx::int32);

  // ---- Cross-chunk encoder pipeline pickup --------------------------------
  // If the previous ``generate_chunk`` call ran with
  // ``pipeline_encoder=true`` it stashed the *next* chunk's
  // encoder_output in ``state.precomputed_encoder_output``. Validate
  // by element-wise comparing the input fingerprint -- if the caller
  // changed ``style_tokens_lm`` between calls, or somehow externally
  // mutated ``state.context_tokens``, the cache is stale and we drop
  // it. Always clear the slot so the cache is at most one chunk old
  // (we'll repopulate it at the end of this call if pipelining stays
  // on).
  std::optional<mx::array> precomputed_encoder_output;
  if (state.precomputed_encoder_output.has_value() &&
      state.precomputed_encoder_inputs == encoder_inputs) {
    precomputed_encoder_output = state.precomputed_encoder_output;
  }
  state.precomputed_encoder_output.reset();
  state.precomputed_encoder_inputs.clear();

  // ---- Generate tokens -----------------------------------------------------
  df::GenerateOptions gen_opts;
  gen_opts.num_frames = chunk_frames;
  gen_opts.rvq_depth = rvq_depth;
  gen_opts.temperature = options.temperature;
  gen_opts.top_k = options.top_k;
  gen_opts.guidance_weight = options.guidance_weight;
  gen_opts.seed = options.seed;

  // Speculative decoding: wire in the previous chunk's LLM tokens when we
  // have them. Only kicks in for chunk_index > 0 (build_depth_draft enforces
  // this by checking for negative entries) AND when the caller explicitly
  // opts in via ``options.speculative``. We default to off because the
  // verification path needs ~16 host syncs per frame (mx::eval + .item() per
  // draft token), which on Apple Silicon costs more wall time than it
  // saves at typical acceptance rates.
  std::vector<std::vector<int32_t>> context_tokens_llm_for_spec;
  if (options.speculative && state.chunk_index > 0 && state.last_num_frames > 0 &&
      static_cast<int>(state.last_frame_tokens_llm.size()) ==
          state.last_num_frames * rvq_depth) {
    context_tokens_llm_for_spec.reserve(state.last_num_frames);
    for (int f = 0; f < state.last_num_frames; ++f) {
      std::vector<int32_t> row(
          state.last_frame_tokens_llm.begin() + f * rvq_depth,
          state.last_frame_tokens_llm.begin() + (f + 1) * rvq_depth);
      context_tokens_llm_for_spec.push_back(std::move(row));
    }
  }

  // Per-phase wall-clock split (MRT_PROFILE_SYSTEM=1). Useful for sizing
  // pipeline-parallelism opportunities: ``depthformer`` is the dominant
  // GPU phase, ``codec`` (rvq_dequantize + decoder + iSTFT) is the only
  // other meaningful chunk-local work.
  using SClock = std::chrono::steady_clock;
  const bool sys_prof = std::getenv("MRT_PROFILE_SYSTEM") != nullptr;
  auto sp_t0 = sys_prof ? SClock::now() : SClock::time_point{};
  mx::array llm_tokens = df::generate_tokens(
      *llm_, encoder_tokens, gen_opts, vocab_masks_,
      context_tokens_llm_for_spec, chunk_frames,
      precomputed_encoder_output);
  mx::eval(llm_tokens);
  auto sp_t1 = sys_prof ? SClock::now() : SClock::time_point{};

  // Save LLM-vocab tokens for speculative decoding on the next call before we
  // convert to plain RVQ indices (that is the format build_depth_draft wants).
  std::vector<int32_t> generated_llm_host = to_host_i32(llm_tokens);
  state.last_frame_tokens_llm = generated_llm_host;
  state.last_num_frames = chunk_frames;

  // ---- Translate LLM token IDs to RVQ indices ------------------------------
  mx::array rvq_tokens = llm_to_rvq(llm_tokens, config_.codec_rvq_codebook_size,
                                     config_.vocab_codec_offset(),
                                     /*safe=*/true);
  mx::eval(rvq_tokens);
  std::vector<int32_t> rvq_host = to_host_i32(rvq_tokens);

  // ---- Build decoder input: [xfade_frames, rvq_tokens] ---------------------
  // Pull last xf_frames from context_tokens (zeros on first chunk).
  std::vector<int32_t> xfade_in(xf_frames * rvq_depth, 0);
  if (state.chunk_index > 0) {
    for (int f = 0; f < xf_frames; ++f) {
      for (int d = 0; d < rvq_depth; ++d) {
        const int src_frame = ctx_frames - xf_frames + f;
        const int32_t v =
            state.context_tokens[src_frame * rvq_depth + d];
        xfade_in[f * rvq_depth + d] = std::max<int32_t>(v, 0);
      }
    }
  }
  const int xfade_total_frames = xf_frames + chunk_frames;
  std::vector<int32_t> xfade_host(xfade_total_frames * rvq_depth);
  std::memcpy(&xfade_host[0], xfade_in.data(),
              sizeof(int32_t) * xf_frames * rvq_depth);
  std::memcpy(&xfade_host[xf_frames * rvq_depth], rvq_host.data(),
              sizeof(int32_t) * chunk_frames * rvq_depth);

  // ---- Compute next chunk's context_tokens (will be assigned to state
  //      below; computed early so the cross-chunk encoder pipeline can
  //      build its input from it before we enter the codec phase). The
  //      crossfade buffer building above is the last place we read the
  //      *current* ``state.context_tokens``.
  std::vector<int32_t> new_context(ctx_frames * rvq_depth);
  for (int f = 0; f < ctx_frames; ++f) {
    const int src = (f + chunk_frames) % ctx_frames;
    for (int d = 0; d < rvq_depth; ++d) {
      new_context[f * rvq_depth + d] =
          state.context_tokens[src * rvq_depth + d];
    }
  }
  for (int f = 0; f < chunk_frames; ++f) {
    for (int d = 0; d < rvq_depth; ++d) {
      new_context[(ctx_frames - chunk_frames + f) * rvq_depth + d] =
          rvq_host[f * rvq_depth + d];
    }
  }

  // ---- Cross-chunk encoder pipeline kick-off ------------------------------
  // Fired BEFORE the codec phase: the GPU encoder gets queued behind
  // the codec on the same Metal command queue, so the two run
  // back-to-back without a host roundtrip in between. The next chunk
  // sees an already-evaluated ``encoder_output`` and skips the
  // 89 ms encoder phase entirely. Net gain is small (~13 ms / chunk)
  // because the encoder still runs on GPU and competes for the same
  // command queue as the codec, but it strictly removes one
  // synchronisation point per chunk.
  //
  // We initially routed the encoder through MLX's CPU stream via
  // ``mx::StreamContext`` to overlap with codec on a different
  // device. That didn't work: the compiled / ``.mlxfn``-imported
  // encoder graph silently stays on GPU regardless of the outer
  // stream context (CPU usage never spikes), and the eager CPU
  // fallback is ~10x slower than GPU on Apple Silicon for the
  // encoder's matmul shapes -- a net regression. Keeping the
  // pre-codec scheduling as a GPU-side optimisation.
  if (options.pipeline_encoder) {
    std::vector<int32_t> next_encoder_inputs = build_encoder_inputs_host(
        new_context, ctx_frames, rvq_depth, config_, style_pos);
    mx::array next_encoder_tokens(
        next_encoder_inputs.data(),
        mx::Shape{2, static_cast<int32_t>(enc_len)}, mx::int32);
    auto pe_t0 = sys_prof ? SClock::now() : SClock::time_point{};
    mx::array next_encoder_output = llm_->encode(next_encoder_tokens);
    mx::async_eval(next_encoder_output);
    if (sys_prof) {
      auto pe_t1 = SClock::now();
      std::fprintf(
          stderr, "[mrt-sys]   pipelined encoder kick-off %.2f ms\n",
          std::chrono::duration<double, std::milli>(pe_t1 - pe_t0).count());
    }
    state.precomputed_encoder_output = std::move(next_encoder_output);
    state.precomputed_encoder_inputs = std::move(next_encoder_inputs);
  }

  // ---- Dequantize via SpectroStream RVQ codebooks and decode --------------
  mx::array codebooks =
      ss_codebooks_bundle_->tensor("codebooks");  // (K, C, D) float32
  mx::array tokens_2d(xfade_host.data(),
                      mx::Shape{static_cast<int32_t>(xfade_total_frames),
                                static_cast<int32_t>(rvq_depth)},
                      mx::int32);
  mx::array embeddings = rvq_dequantization(tokens_2d, codebooks);  // (S, 256)
  embeddings = mx::expand_dims(embeddings, /*axis=*/0);  // (1, S, 256)

  mx::array stft_out = (*decoder_)(embeddings);  // (1, H, 480, 4)

  const int target_T =
      xfade_total_frames * (config_.codec_sample_rate /
                            static_cast<int>(config_.codec_frame_rate));
  mx::array waveform = istft_postprocess(stft_out, /*B=*/1, target_T);
  // waveform shape: (1, target_T, 2).
  waveform = mx::squeeze(waveform, /*axis=*/0);  // (T, 2)
  mx::eval(waveform);
  auto sp_t2 = sys_prof ? SClock::now() : SClock::time_point{};
  if (sys_prof) {
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::fprintf(stderr,
                 "[mrt-sys] chunk %lld: depthformer %.1f ms, codec %.1f ms\n",
                 static_cast<long long>(state.chunk_index),
                 ms(sp_t0, sp_t1), ms(sp_t1, sp_t2));
  }

  const int T = static_cast<int>(waveform.shape(0));
  const int C = static_cast<int>(waveform.shape(1));

  // Host copy so we can do the crossfade + slicing in plain C++.
  std::vector<float> wave_host(T * C);
  const float* wsrc = waveform.data<float>();
  std::memcpy(wave_host.data(), wsrc, sizeof(float) * T * C);

  // ---- Slice out next crossfade tail and the decoded samples --------------
  std::vector<float> next_crossfade(xf_samples * C, 0.0f);
  if (xf_samples > 0) {
    std::memcpy(next_crossfade.data(),
                wave_host.data() + (T - xf_samples) * C,
                sizeof(float) * xf_samples * C);
  }
  const int decoded_len = T - xf_samples;
  std::vector<float> decoded(decoded_len * C);
  std::memcpy(decoded.data(), wave_host.data(),
              sizeof(float) * decoded_len * C);

  // Apply crossfade ramp against the previous chunk's tail.
  if (state.chunk_index > 0 && xf_samples > 0 &&
      state.crossfade_num_samples == xf_samples &&
      decoded_len >= xf_samples) {
    for (int i = 0; i < xf_samples; ++i) {
      const float t = static_cast<float>(i) /
                      static_cast<float>(xf_samples - 1 == 0 ? 1 : xf_samples - 1);
      const float angle = t * 1.5707963267948966f;  // pi/2
      const float fade_in = std::sin(angle);
      const float fade_out = std::sin((1.0f - t) * 1.5707963267948966f);
      for (int c = 0; c < C; ++c) {
        decoded[i * C + c] = decoded[i * C + c] * fade_in +
                             state.crossfade_samples[i * C + c] * fade_out;
      }
    }
  }

  state.crossfade_samples = std::move(next_crossfade);
  state.crossfade_num_samples = xf_samples;

  // ---- Commit next chunk's context_tokens (computed pre-codec above) -----
  state.context_tokens = std::move(new_context);
  state.chunk_index += 1;

  // ---- Package result ------------------------------------------------------
  mx::array decoded_mx(decoded.data(),
                       mx::Shape{static_cast<int32_t>(decoded_len),
                                 static_cast<int32_t>(C)},
                       mx::float32);
  mx::array ctx_mx(state.context_tokens.data(),
                   mx::Shape{static_cast<int32_t>(ctx_frames),
                             static_cast<int32_t>(rvq_depth)},
                   mx::int32);

  Waveform wf{decoded_mx, config_.codec_sample_rate};
  GeneratorState ns{ctx_mx, state.chunk_index};
  return GenerateResult{wf, ns, std::nullopt};
}

}  // namespace magenta_realtime_mlx

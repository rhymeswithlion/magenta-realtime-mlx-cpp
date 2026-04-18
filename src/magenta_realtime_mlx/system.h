#pragma once

// End-to-end MagentaRT system orchestrator. Wires SpectroStream,
// MusicCoCa, and the Depthformer LLM into the ``generate_chunk`` loop used
// by the ``mlx-stream`` binary.
//
// Only the production path is ported -- probe capture, interventions, and
// fused-decode variants in the Python file are out of scope for the C++
// runtime (see the Stage-7 section of the implementation plan).

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "magenta_realtime_mlx/depthformer/decode.h"
#include "magenta_realtime_mlx/depthformer/model.h"
#include "magenta_realtime_mlx/musiccoca.h"
#include "magenta_realtime_mlx/schema.h"
#include "magenta_realtime_mlx/spectrostream.h"
#include "magenta_realtime_mlx/style_model.h"
#include "magenta_realtime_mlx/weights.h"
#include "mlx/mlx.h"

namespace magenta_realtime_mlx {

struct SystemConfig {
  float chunk_length = 2.0f;             // seconds
  float context_length = 10.0f;          // seconds
  float crossfade_length = 0.04f;        // seconds
  int codec_sample_rate = 48000;
  float codec_frame_rate = 25.0f;
  int codec_num_channels = 2;
  int codec_rvq_codebook_size = 1024;
  int style_rvq_codebook_size = 1024;
  int encoder_codec_rvq_depth = 4;
  int encoder_style_rvq_depth = 6;
  int decoder_codec_rvq_depth = 16;
  int llm_vocab_size = 29824;

  // Derived quantities matching the upstream ``MagentaRealtimeConfig``.
  int chunk_length_samples() const {
    return static_cast<int>(chunk_length * codec_sample_rate + 0.5f);
  }
  int chunk_length_frames() const {
    return static_cast<int>(chunk_length * codec_frame_rate + 0.5f);
  }
  int context_length_frames() const {
    return static_cast<int>(context_length * codec_frame_rate + 0.5f);
  }
  int crossfade_length_samples() const {
    return static_cast<int>(crossfade_length * codec_sample_rate + 0.5f);
  }
  int crossfade_length_frames() const {
    return static_cast<int>(crossfade_length * codec_frame_rate + 0.5f);
  }

  int vocab_mask_token() const { return 1; }
  int vocab_codec_offset() const { return 2; }  // PAD + MASK
  int vocab_codec_size() const {
    return decoder_codec_rvq_depth * codec_rvq_codebook_size;
  }
  int vocab_style_offset() const {
    return vocab_codec_offset() + vocab_codec_size() + 1024;
  }
};

// Options for a single ``generate_chunk`` call.
struct GenerateChunkOptions {
  float temperature = 1.1f;
  int top_k = 40;
  float guidance_weight = 5.0f;
  std::optional<uint64_t> seed;
  // Use speculative depth decoding on chunks > 0 (drafts depth tokens from the
  // previous chunk; verified in one big depth-forward pass + per-token
  // rejection sampling). The verification path needs a host sync per draft
  // token (16/frame) so on Apple Silicon it is *slower* than the plain decode
  // unless the acceptance rate is very high. Defaults to ``false``;
  // the streaming binary leaves it off because the per-draft sync
  // dominates the savings on Apple Silicon.
  bool speculative = false;

  // Cross-chunk encoder pipelining. When true, ``generate_chunk`` builds
  // the *next* chunk's encoder input (using the just-updated
  // ``state.context_tokens`` plus the current ``style_tokens_lm``) and
  // dispatches the encoder on MLX's CPU stream via ``mx::async_eval``
  // immediately before returning. The next call sees a populated
  // ``state.precomputed_encoder_output`` (if its input fingerprint
  // matches) and skips the GPU encode entirely, freeing ~90 ms / chunk
  // on the base config (Apple Silicon CPU keeps up with the GPU on a
  // 12-layer / d_model=768 encoder, so the work is genuinely hidden
  // under the previous chunk's codec phase).
  //
  // Cache invalidation is automatic: if the next call's encoder inputs
  // differ from the fingerprint stashed in ``SystemState`` (e.g. the
  // caller swapped ``style_tokens_lm``), we fall back to the
  // synchronous GPU encode and pay the original ~90 ms once.
  //
  // Defaults to ``false`` to keep the API behavior bit-for-bit
  // backwards compatible. ``mlx-stream`` opts in.
  bool pipeline_encoder = false;
};

// Mutable state carried across chunks. Matches the upstream
// ``GeneratorState`` semantics plus the ``_crossfade_samples`` buffer the
// upstream implementation hides on the state dataclass.
struct SystemState {
  // Shape (context_length_frames, decoder_codec_rvq_depth) int32.
  // Entries < 0 mean "empty" (mirrors the upstream -1 initialisation).
  std::vector<int32_t> context_tokens;
  int context_frames = 0;
  int rvq_depth = 0;
  std::int64_t chunk_index = 0;

  // Shape (crossfade_length_samples, num_channels) float32. Empty on first
  // chunk; populated by ``generate_chunk`` once we have past audio.
  std::vector<float> crossfade_samples;  // row-major (samples, channels)
  int crossfade_num_samples = 0;
  int num_channels = 0;

  // Shape (chunk_length_frames, decoder_codec_rvq_depth) int32. The tokens
  // produced by the last ``generate_chunk`` call (populated on both success
  // paths). Used for speculative decoding on subsequent calls.
  std::vector<int32_t> last_frame_tokens_llm;
  int last_num_frames = 0;

  // Cross-chunk CPU encoder pipeline. When the previous
  // ``generate_chunk`` call ran with ``GenerateChunkOptions::
  // pipeline_encoder = true``, it stashed the *next* chunk's encoder
  // output here (computed on MLX's CPU stream, ``async_eval``'d so it
  // runs concurrently with the previous chunk's codec). The next call
  // verifies the fingerprint matches its actual encoder inputs; on
  // match it skips the GPU encode and uses the cached array directly.
  // Both fields are cleared on consumption (and on fingerprint miss),
  // so the cache is always at most one chunk old.
  //
  // Shape: ``(2, encoder_seq_len, d_model)``. ``mlx::core::array`` is a
  // refcounted handle; storing it here keeps the underlying buffer
  // (and its in-flight CPU computation) alive until consumed.
  std::optional<mlx::core::array> precomputed_encoder_output;
  // Flat ``(2 * encoder_seq_len,)`` int32 fingerprint of the encoder
  // inputs that produced ``precomputed_encoder_output``. Compared
  // element-wise against the next call's freshly built encoder inputs.
  std::vector<int32_t> precomputed_encoder_inputs;
};

class System {
 public:
  // ``cache_root`` is typically ``<repo>/.weights-cache``.
  //
  // ``llm_dtype`` controls the Depthformer's compute precision
  // (default ``bfloat16`` on Apple Silicon -- ~2x faster than fp32
  // with no audible quality loss). The
  // SpectroStream codec and MusicCoCa style encoder always run in float32
  // because the Python implementation does so regardless of ``llm_dtype``;
  // mismatched dtypes here would silently change the audio character relative
  // to ``mlx-stream``.
  System(const std::filesystem::path& cache_root,
         std::string_view tag = "base",
         mlx::core::Dtype llm_dtype = mlx::core::bfloat16);

  const SystemConfig& config() const noexcept { return config_; }
  int sample_rate() const noexcept { return config_.codec_sample_rate; }
  int num_channels() const noexcept { return config_.codec_num_channels; }
  double chunk_length() const noexcept { return config_.chunk_length; }

  // Compute the LLM-vocabulary style tokens for a prompt once; reuse across
  // chunks (tokenize prompt once, feed every call).
  // Returns shape ``(encoder_style_rvq_depth,)`` int32.
  mlx::core::array embed_style(std::string_view prompt) const;

  // Build a fresh state consistent with ``config_``. Subsequent calls should
  // pass the returned state back in; it is mutated in place.
  SystemState empty_state() const;

  // One chunk. Mutates ``state`` in place and returns the decoded waveform
  // plus the next logical ``chunk_index``.
  //
  // ``style_tokens_lm`` is the ``(encoder_style_rvq_depth,)`` int32 tensor
  // produced by ``embed_style``. ``nullopt`` means "unconditioned" (matches
  // the Python code's ``style is None`` path: fill with MASK).
  GenerateResult generate_chunk(
      SystemState& state,
      const std::optional<mlx::core::array>& style_tokens_lm,
      const GenerateChunkOptions& options = {}) const;

 private:
  // One-time: build the per-level vocab masks used by CFG decoding so the
  // LLM can only produce tokens on the right RVQ level.
  void build_vocab_masks();

  SystemConfig config_;
  mlx::core::Dtype llm_dtype_;
  InferenceBundlePaths paths_;

  // Owned bundles (kept alive so the modules can reference their tensors).
  std::unique_ptr<WeightBundle> ss_decoder_bundle_;
  std::unique_ptr<WeightBundle> ss_codebooks_bundle_;
  std::unique_ptr<WeightBundle> ss_encoder_bundle_;
  std::unique_ptr<WeightBundle> mc_encoder_bundle_;
  std::unique_ptr<WeightBundle> mc_codebooks_bundle_;
  std::unique_ptr<WeightBundle> df_bundle_;

  std::unique_ptr<SpectroStreamDecoder> decoder_;
  std::unique_ptr<depthformer::Depthformer> llm_;
  std::unique_ptr<StyleModel> style_model_;

  std::vector<std::optional<mlx::core::array>> vocab_masks_;
};

}  // namespace magenta_realtime_mlx

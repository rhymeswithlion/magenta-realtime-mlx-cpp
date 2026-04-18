#pragma once

// Glue around ``MusicCoCaEncoder``: tokenises a text prompt with
// SentencePiece, runs the text encoder, RVQ-quantises the embedding against
// the MusicCoCa codebooks, and returns LLM-vocabulary style tokens ready for
// the Depthformer encoder.

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "magenta_realtime_mlx/musiccoca.h"
#include "magenta_realtime_mlx/weights.h"
#include "mlx/mlx.h"

namespace sentencepiece {
class SentencePieceProcessor;
}

namespace magenta_realtime_mlx {

class StyleModel {
 public:
  static constexpr int kMaxTextLength = 128;
  static constexpr int kEmbeddingDim = 768;
  static constexpr int kRVQDepth = 12;
  static constexpr int kRVQCodebookSize = 1024;

  // ``encoder_bundle`` owns the MusicCoCa weights, ``codebooks_bundle`` owns
  // the RVQ codebooks (key ``"codebooks"`` shape ``(K, C, D)``) that the
  // Python side loads from ``musiccoca_codebooks.npy``. ``vocab_path`` is a
  // SentencePiece ``.model`` file.
  StyleModel(const WeightBundle& encoder_bundle,
             const WeightBundle& codebooks_bundle,
             const std::filesystem::path& vocab_path,
             mlx::core::Dtype dtype = mlx::core::float32);
  ~StyleModel();

  StyleModel(const StyleModel&) = delete;
  StyleModel& operator=(const StyleModel&) = delete;

  // Text embedding (B=1): returns shape (kEmbeddingDim,) float32.
  mlx::core::array embed_text(std::string_view prompt) const;

  // RVQ-quantise an embedding of shape (kEmbeddingDim,) into (kRVQDepth,)
  // int32 tokens, using the first ``kRVQDepth`` codebooks.
  mlx::core::array tokenize(const mlx::core::array& embedding) const;

  // Map the first ``encoder_style_rvq_depth`` RVQ tokens to LLM-vocabulary
  // token IDs using the per-level offsets Magenta RT expects.
  // Returns shape ``(encoder_style_rvq_depth,)`` int32.
  mlx::core::array style_tokens_lm(std::string_view prompt,
                                   int encoder_style_rvq_depth,
                                   int style_rvq_codebook_size,
                                   int vocab_style_offset) const;

 private:
  std::pair<std::vector<int32_t>, std::vector<float>> tokenize_text_ids(
      std::string_view prompt) const;

  musiccoca::MusicCoCaEncoder encoder_;
  mlx::core::array codebooks_;  // (K, C, D) float32
  std::unique_ptr<sentencepiece::SentencePieceProcessor> spm_;
};

}  // namespace magenta_realtime_mlx

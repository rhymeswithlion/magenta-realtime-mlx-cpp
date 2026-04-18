#include "magenta_realtime_mlx/style_model.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include <sentencepiece_processor.h>

#include "magenta_realtime_mlx/rvq.h"

namespace magenta_realtime_mlx {

namespace mx = mlx::core;

namespace {

std::string lowercase(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

}  // namespace

StyleModel::StyleModel(const WeightBundle& encoder_bundle,
                       const WeightBundle& codebooks_bundle,
                       const std::filesystem::path& vocab_path,
                       mx::Dtype dtype)
    : encoder_(encoder_bundle, musiccoca::MusicCoCaConfig{}, dtype),
      codebooks_(mx::astype(codebooks_bundle.tensor("codebooks"), mx::float32)),
      spm_(std::make_unique<sentencepiece::SentencePieceProcessor>()) {
  auto status = spm_->Load(vocab_path.string());
  if (!status.ok()) {
    throw std::runtime_error("StyleModel: failed to load SentencePiece model " +
                             vocab_path.string() + ": " + status.ToString());
  }
}

StyleModel::~StyleModel() = default;

std::pair<std::vector<int32_t>, std::vector<float>>
StyleModel::tokenize_text_ids(std::string_view prompt) const {
  const std::string lowered = lowercase(prompt);
  std::vector<int> raw_ids;
  auto status = spm_->Encode(lowered, &raw_ids);
  if (!status.ok()) {
    throw std::runtime_error("StyleModel::tokenize_text: Encode failed: " +
                             status.ToString());
  }
  // Trim to leave room for the BOS token (matches the upstream behaviour).
  const int keep = std::min<int>(static_cast<int>(raw_ids.size()),
                                 kMaxTextLength - 1);

  std::vector<int32_t> ids(kMaxTextLength, 0);
  std::vector<float> padding(kMaxTextLength, 1.0f);
  ids[0] = 1;  // BOS
  padding[0] = 0.0f;
  for (int i = 0; i < keep; ++i) {
    ids[i + 1] = static_cast<int32_t>(raw_ids[static_cast<size_t>(i)]);
    padding[i + 1] = 0.0f;
  }
  return {ids, padding};
}

mx::array StyleModel::embed_text(std::string_view prompt) const {
  auto [ids_host, padding_host] = tokenize_text_ids(prompt);
  mx::array ids(ids_host.data(), mx::Shape{1, kMaxTextLength}, mx::int32);
  mx::array padding(padding_host.data(), mx::Shape{1, kMaxTextLength},
                    mx::float32);
  mx::array emb = encoder_.embed_text(ids, padding);  // (1, 768)
  mx::eval(emb);
  return mx::astype(mx::squeeze(emb, /*axis=*/0), mx::float32);  // (768,)
}

mx::array StyleModel::tokenize(const mx::array& embedding) const {
  mx::array e = embedding;
  if (e.ndim() == 1) e = mx::expand_dims(e, /*axis=*/0);
  if (e.ndim() != 2) {
    throw std::invalid_argument(
        "StyleModel::tokenize: embedding must be rank 1 or 2");
  }
  // rvq_quantization uses the *first* kRVQDepth codebooks (depth is inferred
  // from codebooks.shape[0]). Our codebooks tensor already has depth 12.
  auto [tokens, residual] = rvq_quantization(e, codebooks_);
  (void)residual;
  mx::eval(tokens);
  // Drop the batch axis to match the 1-D semantics of the Python wrapper.
  if (tokens.shape(0) == 1) {
    return mx::reshape(tokens, mx::Shape{kRVQDepth});
  }
  return tokens;
}

mx::array StyleModel::style_tokens_lm(std::string_view prompt,
                                      int encoder_style_rvq_depth,
                                      int style_rvq_codebook_size,
                                      int vocab_style_offset) const {
  mx::array emb = embed_text(prompt);
  mx::array tokens = tokenize(emb);  // (kRVQDepth,) int32
  // Take the first encoder_style_rvq_depth levels.
  mx::array trimmed =
      mx::slice(tokens, mx::Shape{0},
                mx::Shape{static_cast<int32_t>(encoder_style_rvq_depth)});
  return rvq_to_llm(trimmed, style_rvq_codebook_size, vocab_style_offset);
}

}  // namespace magenta_realtime_mlx

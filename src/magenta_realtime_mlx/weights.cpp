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

#include "magenta_realtime_mlx/weights.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "mlx/io.h"

namespace magenta_realtime_mlx {

namespace fs = std::filesystem;
namespace mx = mlx::core;

WeightBundle::WeightBundle(const fs::path& path) : path_(path) {
  if (!fs::is_regular_file(path_)) {
    throw std::runtime_error("WeightBundle: not a file: " + path_.string());
  }
  auto loaded = mx::load_safetensors(path_.string());
  tensors_ = std::move(loaded.first);
}

const mx::array& WeightBundle::tensor(std::string_view key) const {
  auto it = tensors_.find(std::string(key));
  if (it == tensors_.end()) {
    throw std::out_of_range("WeightBundle " + path_.string() +
                            ": missing key \"" + std::string(key) + "\"");
  }
  return it->second;
}

bool WeightBundle::contains(std::string_view key) const {
  return tensors_.find(std::string(key)) != tensors_.end();
}

std::vector<std::string> WeightBundle::keys() const {
  std::vector<std::string> out;
  out.reserve(tensors_.size());
  for (const auto& [k, _] : tensors_) {
    out.push_back(k);
  }
  std::sort(out.begin(), out.end());
  return out;
}

WeightBundle& WeightCache::load(const fs::path& path) {
  fs::path canonical = fs::weakly_canonical(path);
  std::string key = canonical.string();
  auto it = bundles_.find(key);
  if (it == bundles_.end()) {
    auto inserted =
        bundles_.emplace(key, std::make_unique<WeightBundle>(canonical));
    it = inserted.first;
  }
  return *it->second;
}

InferenceBundlePaths resolve_inference_bundle(const fs::path& cache_root,
                                              std::string_view tag) {
  if (!fs::is_directory(cache_root)) {
    throw std::runtime_error("resolve_inference_bundle: not a directory: " +
                             cache_root.string());
  }

  InferenceBundlePaths p;
  p.spectrostream_encoder = cache_root / "spectrostream_encoder.safetensors";
  p.spectrostream_decoder = cache_root / "spectrostream_decoder.safetensors";
  p.spectrostream_codebooks =
      cache_root / "spectrostream_codebooks.safetensors";
  p.musiccoca_encoder = cache_root / "musiccoca_encoder.safetensors";
  p.musiccoca_codebooks = cache_root / "musiccoca_codebooks.safetensors";
  p.musiccoca_vocab = cache_root / "musiccoca_vocab.model";
  p.depthformer = cache_root / "depthformer" /
                  ("depthformer_" + std::string(tag) + ".safetensors");

  std::vector<std::pair<std::string, fs::path>> required = {
      {"SpectroStream encoder", p.spectrostream_encoder},
      {"SpectroStream decoder", p.spectrostream_decoder},
      {"SpectroStream codebooks", p.spectrostream_codebooks},
      {"MusicCoCa encoder", p.musiccoca_encoder},
      {"MusicCoCa codebooks", p.musiccoca_codebooks},
      {"MusicCoCa SentencePiece vocab", p.musiccoca_vocab},
      {"Depthformer checkpoint", p.depthformer},
  };

  std::vector<std::string> missing;
  for (const auto& [label, path] : required) {
    if (!fs::is_regular_file(path)) {
      std::ostringstream os;
      os << "  - " << label << ": " << path.string();
      missing.push_back(os.str());
    }
  }
  if (!missing.empty()) {
    std::ostringstream os;
    os << "Incomplete Magenta RT C++ inference bundle (looked under "
       << cache_root.string() << "):\n";
    for (const auto& line : missing) {
      os << line << "\n";
    }
    os << "Run `make ensure-weights-cache` to download + bundle.";
    throw std::runtime_error(os.str());
  }

  return p;
}

}  // namespace magenta_realtime_mlx

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

// Lightweight wrapper around ``mlx::core::load_safetensors``. The Python
// runtime has a much richer ``weights`` module that walks the filesystem and
// validates layouts; the C++ runtime takes its weights-cache root as a CLI
// argument and only needs:
//   * ``WeightBundle::tensor(key)`` -- read tensor by name (memoized).
//   * ``WeightBundle::keys()``       -- inspect what is available.
//   * Free functions to discover the layout under a cache root.

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "mlx/mlx.h"

namespace magenta_realtime_mlx {

// Holds the contents of a single ``.safetensors`` file. Lazy: the file is
// parsed once on construction and cached by ``WeightCache``.
class WeightBundle {
 public:
  explicit WeightBundle(const std::filesystem::path& path);

  // Throws ``std::out_of_range`` if ``key`` is not present.
  const mlx::core::array& tensor(std::string_view key) const;

  // True if ``key`` is present.
  bool contains(std::string_view key) const;

  // Sorted list of available tensor keys.
  std::vector<std::string> keys() const;

  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
  std::unordered_map<std::string, mlx::core::array> tensors_;
};

// Memoizes ``WeightBundle`` instances by absolute path so callers can cheaply
// share bundles across modules.
class WeightCache {
 public:
  WeightBundle& load(const std::filesystem::path& path);

 private:
  std::unordered_map<std::string, std::unique_ptr<WeightBundle>> bundles_;
};

// Layout discovery for the C++ CLI. A ``cache_root`` is something
// like ``<repo>/.weights-cache``.
struct InferenceBundlePaths {
  std::filesystem::path spectrostream_encoder;
  std::filesystem::path spectrostream_decoder;
  std::filesystem::path spectrostream_codebooks;
  std::filesystem::path musiccoca_encoder;
  std::filesystem::path musiccoca_codebooks;
  std::filesystem::path musiccoca_vocab;
  std::filesystem::path depthformer;
};

// Resolve every ``.safetensors`` (or ``.model`` for the SentencePiece vocab)
// the runtime needs from ``cache_root``. ``tag`` selects the depthformer
// checkpoint (e.g. ``"base"``).
//
// Throws ``std::runtime_error`` listing every missing file if the layout is
// incomplete, mirroring the upstream ``require_inference_weights`` helper.
InferenceBundlePaths resolve_inference_bundle(
    const std::filesystem::path& cache_root, std::string_view tag = "base");

}  // namespace magenta_realtime_mlx

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

#include "magenta_realtime_mlx/rvq.h"

#include <stdexcept>
#include <vector>

namespace magenta_realtime_mlx {

namespace mx = mlx::core;

std::pair<mx::array, mx::array> rvq_quantization(
    const mx::array& vectors, const mx::array& codebooks) {
  if (codebooks.ndim() != 3) {
    throw std::invalid_argument("rvq_quantization: codebooks must be (K, C, D)");
  }
  if (vectors.ndim() != 2) {
    throw std::invalid_argument("rvq_quantization: vectors must be (N, D)");
  }
  const int num_levels = static_cast<int>(codebooks.shape(0));
  const int dim = static_cast<int>(codebooks.shape(2));
  if (vectors.shape(1) != dim) {
    throw std::invalid_argument(
        "rvq_quantization: vector dim mismatches codebook dim");
  }

  mx::array residual = mx::astype(vectors, mx::float32);
  const mx::array cbs = mx::astype(codebooks, mx::float32);

  std::vector<mx::array> tokens_per_level;
  tokens_per_level.reserve(num_levels);

  for (int k = 0; k < num_levels; ++k) {
    // ``take(a, scalar, axis)`` removes the indexed axis (numpy-style),
    // so cb_k already has shape (C, D).
    mx::array cb_k = mx::take(cbs, k, /*axis=*/0);
    mx::array r2 = mx::sum(mx::multiply(residual, residual),
                           /*axis=*/-1, /*keepdims=*/true);          // (N, 1)
    mx::array c2 = mx::sum(mx::multiply(cb_k, cb_k),
                           /*axis=*/-1, /*keepdims=*/false);          // (C,)
    mx::array cross = mx::matmul(residual, mx::transpose(cb_k, {1, 0}));
    mx::array dists =
        mx::add(mx::subtract(r2, mx::multiply(mx::array(2.0f), cross)), c2);
    mx::array tk = mx::astype(
        mx::argmin(dists, /*axis=*/-1, /*keepdims=*/false), mx::int32);  // (N,)
    tokens_per_level.push_back(tk);
    // ``take(cb_k, tk, axis=0)`` is gather: result shape (N, D).
    mx::array picked = mx::take(cb_k, tk, /*axis=*/0);
    residual = mx::subtract(residual, picked);
  }

  mx::array tokens = mx::stack(tokens_per_level, /*axis=*/-1);
  return {tokens, residual};
}

mx::array rvq_dequantization(const mx::array& tokens,
                             const mx::array& codebooks) {
  if (codebooks.ndim() != 3) {
    throw std::invalid_argument("rvq_dequantization: codebooks must be (K, C, D)");
  }
  if (tokens.ndim() != 2) {
    throw std::invalid_argument("rvq_dequantization: tokens must be (N, K)");
  }
  const int num_levels = static_cast<int>(tokens.shape(1));
  const int dim = static_cast<int>(codebooks.shape(2));
  const int n = static_cast<int>(tokens.shape(0));

  mx::array vectors = mx::zeros({n, dim}, mx::float32);
  for (int k = 0; k < num_levels; ++k) {
    mx::array cb_k = mx::take(codebooks, k, /*axis=*/0);     // (C, D)
    mx::array tk = mx::take(tokens, k, /*axis=*/1);          // (N,)
    vectors = mx::add(vectors, mx::take(cb_k, tk, /*axis=*/0));
  }
  return vectors;
}

namespace {

// Build a (..., K) tensor whose last axis is ``[0, 1, ..., K-1]``,
// broadcastable against ``tokens``.
mx::array level_indices_like(const mx::array& tokens) {
  const int last = static_cast<int>(tokens.shape(-1));
  mx::array levels = mx::astype(mx::arange(last), mx::int32);
  mx::array reshaped = levels;
  for (int i = 0; i < tokens.ndim() - 1; ++i) {
    reshaped = mx::expand_dims(reshaped, 0);
  }
  return reshaped;
}

}  // namespace

mx::array rvq_to_llm(const mx::array& tokens, int codebook_size, int offset) {
  mx::array levels = level_indices_like(tokens);
  mx::array tokens_i32 = mx::astype(tokens, mx::int32);
  mx::array scaled = mx::multiply(levels, mx::array(codebook_size));
  return mx::add(mx::add(mx::array(offset), scaled), tokens_i32);
}

mx::array llm_to_rvq(const mx::array& tokens, int codebook_size, int offset,
                     bool safe) {
  mx::array tokens_i32 = mx::astype(tokens, mx::int32);
  mx::array shifted = mx::subtract(tokens_i32, mx::array(offset));
  mx::array rvq_tokens = mx::remainder(shifted, mx::array(codebook_size));
  if (safe) {
    // Equivalent to ``shifted // codebook_size == expected_levels``,
    // computed as ``shifted - rvq_tokens == expected_levels * codebook_size``
    // so we don't depend on a floor_divide op (MLX C++ doesn't expose one).
    mx::array residual = mx::subtract(shifted, rvq_tokens);
    mx::array expected =
        mx::multiply(level_indices_like(tokens), mx::array(codebook_size));
    mx::array expected_b = mx::broadcast_to(expected, residual.shape());
    mx::array diff =
        mx::sum(mx::abs(mx::subtract(residual, expected_b)));
    mx::eval(diff);
    if (diff.item<int>() != 0) {
      throw std::invalid_argument(
          "llm_to_rvq: token levels do not match expected RVQ structure");
    }
  }
  return mx::astype(rvq_tokens, mx::int32);
}

}  // namespace magenta_realtime_mlx

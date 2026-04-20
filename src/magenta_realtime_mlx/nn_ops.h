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

// Tiny activation helpers that ``mlx::core`` doesn't ship as standalone ops.
// We bias toward inlining the body to avoid creating a separate translation
// unit just to call ``mx::where`` once.

#include <functional>
#include <vector>

#include "mlx/compile.h"
#include "mlx/mlx.h"

namespace magenta_realtime_mlx {

// ELU activation (default alpha = 1):
//
//   elu(x) = x                          if x > 0
//   elu(x) = alpha * (exp(x) - 1)       otherwise
inline mlx::core::array elu(const mlx::core::array& x, float alpha = 1.0f) {
  namespace mx = mlx::core;
  mx::array zero = mx::array(0.0f);
  mx::array e = mx::multiply(mx::array(alpha),
                             mx::subtract(mx::exp(x), mx::array(1.0f)));
  return mx::where(mx::greater(x, zero), x, e);
}

// Exact GELU:
//
//   gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
inline mlx::core::array gelu(const mlx::core::array& x) {
  namespace mx = mlx::core;
  mx::array half = mx::array(0.5f);
  mx::array one = mx::array(1.0f);
  mx::array inv_sqrt2 = mx::array(0.7071067811865475f);  // 1/sqrt(2)
  mx::array y = mx::multiply(
      mx::multiply(half, x),
      mx::add(one, mx::erf(mx::multiply(x, inv_sqrt2))));
  return y;
}

// Tanh-approximate GELU:
//
//   gelu_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))
//
// We pre-wrap the body in ``mx::compile(..., shapeless=true)`` so that the
// 9-op expansion fuses into a single Metal kernel. Without this the FFN
// inside the Depthformer pays for ~9 individual ops per layer x every step
// (~19k gelu calls per chunk), which was a substantial slice of measured
// per-frame overhead.
namespace detail {

inline std::vector<mlx::core::array> gelu_approx_body(
    const std::vector<mlx::core::array>& inputs) {
  namespace mx = mlx::core;
  const mx::array& x = inputs[0];
  // Scalar literals MUST be cast to ``x.dtype()`` -- ``mx::array(0.5f)`` is
  // fp32 and would silently promote bf16 inputs to fp32 (matmul/multiply
  // promote to the wider dtype). Without this cast, every FFN in the
  // Depthformer upcasts to fp32 starting at layer 0, propagating to all
  // downstream K/V caches and breaking parity with both the per-cl and
  // padded ``.mlxfn`` graphs (which were traced in bf16).
  const mx::array half = mx::astype(mx::array(0.5f), x.dtype());
  const mx::array one = mx::astype(mx::array(1.0f), x.dtype());
  const mx::array c0 = mx::astype(mx::array(0.7978845608028654f), x.dtype());  // sqrt(2/pi)
  const mx::array c1 = mx::astype(mx::array(0.044715f), x.dtype());
  mx::array x2 = mx::multiply(x, x);
  mx::array inner = mx::multiply(
      c0, mx::multiply(x, mx::add(one, mx::multiply(c1, x2))));
  return {mx::multiply(half, mx::multiply(x, mx::add(one, mx::tanh(inner))))};
}

inline const std::function<std::vector<mlx::core::array>(
    const std::vector<mlx::core::array>&)>&
gelu_approx_compiled() {
  static const auto kFn = mlx::core::compile(&gelu_approx_body,
                                             /*shapeless=*/true);
  return kFn;
}

}  // namespace detail

inline mlx::core::array gelu_approx(const mlx::core::array& x) {
  return detail::gelu_approx_compiled()({x})[0];
}

}  // namespace magenta_realtime_mlx

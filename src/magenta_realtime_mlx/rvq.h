#pragma once

// Residual Vector Quantization (RVQ) helpers shared by SpectroStream and
// MusicCoCa. Same algorithm/shapes/dtypes as the upstream Magenta
// RealTime reference; arrays here are
// ``mlx::core::array`` instead of ``np.ndarray``.

#include <utility>

#include "mlx/mlx.h"

namespace magenta_realtime_mlx {

// Quantize ``vectors`` (N, D) using ``codebooks`` (K, C, D).
// Returns ``{tokens (N, K) int32, residuals (N, D) float32}``.
std::pair<mlx::core::array, mlx::core::array> rvq_quantization(
    const mlx::core::array& vectors,
    const mlx::core::array& codebooks);

// Dequantize ``tokens`` (N, K) using ``codebooks`` (K_total, C, D).
// Only the first ``K = tokens.shape(-1)`` codebooks are used, matching
// the upstream reference.
mlx::core::array rvq_dequantization(const mlx::core::array& tokens,
                                    const mlx::core::array& codebooks);

// Convert RVQ token grid (..., K) to flat LLM vocabulary indices using the
// per-level offset ``offset + level * codebook_size + token``.
//
// 1-D inputs are treated as a single frame of K codes, matching numpy's
// ``np.arange(tokens.shape[0])`` branch.
mlx::core::array rvq_to_llm(const mlx::core::array& tokens,
                            int codebook_size,
                            int offset);

// Inverse of ``rvq_to_llm``. With ``safe = true`` we verify each token is
// addressed by its expected level (per the upstream reference), throwing
// ``std::invalid_argument`` if the layout is wrong.
mlx::core::array llm_to_rvq(const mlx::core::array& tokens,
                            int codebook_size,
                            int offset,
                            bool safe = true);

}  // namespace magenta_realtime_mlx

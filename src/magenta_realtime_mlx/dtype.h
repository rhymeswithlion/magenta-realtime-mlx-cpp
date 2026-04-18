#pragma once

// Runtime dtype selection for the MLX C++ port.
//
// A single CLI flag (`--dtype fp32|fp16|bf16`, default fp32) controls the
// dtype that every module casts its weights to at construction time.
// Activations inherit the dtype naturally because MLX ops preserve the
// dtype of their inputs. Parity tests against the upstream reference all
// run with fp32; lower precisions are smoke-tested only.

#include <string>
#include <string_view>

#include "mlx/dtype.h"

namespace magenta_realtime_mlx {

// Parse a CLI dtype string. Throws std::invalid_argument on unknown input.
// Accepted spellings: "fp32" | "float32" | "f32",
//                     "fp16" | "float16" | "f16" | "half",
//                     "bf16" | "bfloat16".
mlx::core::Dtype parse_dtype(std::string_view name);

// Inverse of parse_dtype, returning the canonical short name ("fp32", etc.).
std::string dtype_name(mlx::core::Dtype dtype);

}  // namespace magenta_realtime_mlx

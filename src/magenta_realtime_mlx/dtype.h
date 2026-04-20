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

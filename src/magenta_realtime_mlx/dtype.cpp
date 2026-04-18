#include "magenta_realtime_mlx/dtype.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace magenta_realtime_mlx {

namespace {

std::string to_lower(std::string_view s) {
  std::string out(s);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return out;
}

}  // namespace

mlx::core::Dtype parse_dtype(std::string_view name) {
  const std::string n = to_lower(name);
  if (n == "fp32" || n == "float32" || n == "f32") {
    return mlx::core::float32;
  }
  if (n == "fp16" || n == "float16" || n == "f16" || n == "half") {
    return mlx::core::float16;
  }
  if (n == "bf16" || n == "bfloat16") {
    return mlx::core::bfloat16;
  }
  throw std::invalid_argument(
      "unknown dtype \"" + std::string(name) +
      "\"; expected fp32|fp16|bf16");
}

std::string dtype_name(mlx::core::Dtype dtype) {
  if (dtype == mlx::core::float32) return "fp32";
  if (dtype == mlx::core::float16) return "fp16";
  if (dtype == mlx::core::bfloat16) return "bf16";
  return "unsupported";
}

}  // namespace magenta_realtime_mlx

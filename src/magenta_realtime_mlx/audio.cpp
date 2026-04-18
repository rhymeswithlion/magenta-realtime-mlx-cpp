#include "magenta_realtime_mlx/audio.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace magenta_realtime_mlx {

namespace mx = mlx::core;

mx::array crossfade_ramp(int length, std::string_view style) {
  if (length <= 0) {
    throw std::invalid_argument("crossfade_ramp: length must be > 0");
  }
  // ``linspace(0, 1, length)`` matches ``np.linspace(0., 1., length)``.
  auto t = mx::linspace(0.0, 1.0, length, mx::float32);
  if (style == "eqpower") {
    static const float k_pi_over_2 = static_cast<float>(M_PI) * 0.5f;
    return mx::sin(mx::multiply(t, mx::array(k_pi_over_2)));
  }
  if (style == "linear") {
    return t;
  }
  throw std::invalid_argument("crossfade_ramp: unknown style \"" +
                              std::string(style) + "\"");
}

mx::array to_mono(const mx::array& samples, std::string_view strategy) {
  if (samples.ndim() == 1) {
    return samples;
  }
  if (samples.ndim() != 2) {
    throw std::invalid_argument("to_mono: input must be rank 1 or 2");
  }
  if (samples.shape(1) == 1) {
    return mx::reshape(samples, {static_cast<int>(samples.shape(0))});
  }
  if (strategy == "average") {
    return mx::astype(mx::mean(samples, /*axis=*/1, /*keepdims=*/false),
                      mx::float32);
  }
  if (strategy == "left") {
    return mx::astype(mx::take(samples, 0, /*axis=*/1), mx::float32);
  }
  throw std::invalid_argument("to_mono: unknown strategy \"" +
                              std::string(strategy) + "\"");
}

}  // namespace magenta_realtime_mlx

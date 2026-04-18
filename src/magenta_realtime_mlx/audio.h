#pragma once

// Small audio helpers used by the orchestrator and the playback loop. The
// Python module also exposes ``resample`` (librosa-based) for style-audio
// conditioning; we will only port that path if/when the C++ CLI grows
// ``--style-wav`` support, which the implementation plan defers past stage 7.

#include <vector>

#include "mlx/mlx.h"

namespace magenta_realtime_mlx {

// Build a length-N crossfade ramp. Two styles, matching the Python helper:
//   * "eqpower": sin((t * pi) / 2)  -- equal-power ramp used by the chunk
//                                      cross-fader
//   * "linear":  t                   -- linear ramp, used by tests
//
// Returned dtype is float32. ``length`` must be > 0.
mlx::core::array crossfade_ramp(int length, std::string_view style = "eqpower");

// Convert (T, C) stereo audio to (T,) mono. Two strategies, matching Python:
//   * "average": mean across channels (default, matches the Python default)
//   * "left":    take channel 0
//
// Single-channel inputs (rank 1, or rank 2 with C == 1) are passed through.
mlx::core::array to_mono(const mlx::core::array& samples,
                         std::string_view strategy = "average");

}  // namespace magenta_realtime_mlx

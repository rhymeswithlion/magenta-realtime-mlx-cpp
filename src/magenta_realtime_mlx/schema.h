#pragma once

// Result types returned by the system orchestrator: ``Waveform``,
// ``GeneratorState``, and the ``GenerateChunkResult`` aggregate.

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

#include "mlx/mlx.h"

namespace magenta_realtime_mlx {

// Audio samples shape: (num_samples, num_channels) float32.
struct Waveform {
  mlx::core::array samples;
  int sample_rate;

  int num_samples() const {
    return samples.ndim() == 0 ? 0 : static_cast<int>(samples.shape(0));
  }
  int num_channels() const {
    return samples.ndim() == 2 ? static_cast<int>(samples.shape(1)) : 1;
  }
  double duration() const {
    return num_samples() / static_cast<double>(sample_rate);
  }
};

// Opaque between-chunk state. ``context_tokens`` shape:
// (context_frames, rvq_depth) int32.
struct GeneratorState {
  mlx::core::array context_tokens;
  std::int64_t chunk_index;
};

// Result from a single ``generate_chunk`` call. ``probe_states`` is used by
// debug builds to surface intermediate tensors; production callers can
// ignore it.
struct GenerateResult {
  Waveform waveform;
  GeneratorState state;
  std::optional<std::unordered_map<std::string, mlx::core::array>> probe_states;
};

}  // namespace magenta_realtime_mlx

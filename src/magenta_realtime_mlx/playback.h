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

// PortAudio live-playback glue. The decoder pushes interleaved float32
// stereo samples at 48 kHz into ``PlaybackQueue``; a PortAudio callback
// thread drains them into the audio device. A lock-free SPSC ring buffer
// is overkill here -- there is exactly one producer (decoder) and one
// consumer (callback) and we already tolerate >100 ms of latency, so the
// implementation uses a minimal mutex-guarded deque. If we later need
// tighter latency we can swap the internals without changing the call
// sites.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <portaudio.h>

namespace magenta_realtime_mlx::playback {

// One chunk of interleaved stereo float32 samples.
struct Chunk {
  std::vector<float> samples;  // (num_frames * num_channels), row-major
  int num_frames;
  int num_channels;
};

class PlaybackQueue {
 public:
  PlaybackQueue() = default;
  PlaybackQueue(const PlaybackQueue&) = delete;
  PlaybackQueue& operator=(const PlaybackQueue&) = delete;

  // Push one chunk onto the queue. Thread-safe wrt the consumer.
  void push(Chunk chunk);

  // Signal end-of-stream; the consumer drains remaining chunks and exits.
  void close();

  // Pop ``num_frames * num_channels`` samples into ``out``, filling with
  // zeros if the queue is empty. Returns the number of samples actually
  // drawn from real chunks (vs. silence).
  std::size_t fill(float* out, std::size_t num_samples);

  // How many chunks are currently buffered (for diagnostics).
  std::size_t size() const;

  // True once ``close()`` was called AND the queue is empty.
  bool drained() const;

 private:
  mutable std::mutex m_;
  std::deque<Chunk> chunks_;
  std::vector<float> partial_;       // leftover samples from a chunk
  std::size_t partial_pos_ = 0;
  std::atomic<bool> closed_{false};
};

struct PlaybackConfig {
  int sample_rate = 48000;
  int num_channels = 2;
  std::string device_substring;  // empty -> default output device
  bool list_devices_and_exit = false;
};

// Scoped PortAudio playback stream. Opening the stream starts pulling from
// the queue on the PortAudio callback thread. ``stop()`` is called by the
// destructor but can be invoked explicitly to flush before shutdown.
class PortAudioStream {
 public:
  PortAudioStream(PlaybackQueue& queue, const PlaybackConfig& config);
  ~PortAudioStream();

  PortAudioStream(const PortAudioStream&) = delete;
  PortAudioStream& operator=(const PortAudioStream&) = delete;

  void start();
  void stop();

 private:
  static int pa_callback(const void* input, void* output,
                         unsigned long frame_count,
                         const PaStreamCallbackTimeInfo* time_info,
                         PaStreamCallbackFlags status_flags, void* user_data);

  PlaybackQueue& queue_;
  PlaybackConfig config_;
  PaStream* stream_ = nullptr;
  bool started_ = false;
};

// Print a table of PortAudio devices to stdout (for --list-devices).
void list_devices();

}  // namespace magenta_realtime_mlx::playback

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

#include "magenta_realtime_mlx/playback.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <portaudio.h>

namespace magenta_realtime_mlx::playback {

namespace {

struct PaGuard {
  PaGuard() {
    PaError e = Pa_Initialize();
    if (e != paNoError) {
      throw std::runtime_error(std::string("Pa_Initialize failed: ") +
                               Pa_GetErrorText(e));
    }
  }
  ~PaGuard() { Pa_Terminate(); }
  PaGuard(const PaGuard&) = delete;
  PaGuard& operator=(const PaGuard&) = delete;
};

// Keep PortAudio initialised for the whole program. PortAudio ref-counts
// Pa_Initialize / Pa_Terminate calls so extra invocations inside the stream
// object are harmless, but we still want a top-level guard so the CLI can
// call ``list_devices()`` without constructing a stream.
PaGuard& pa_guard() {
  static PaGuard g;
  return g;
}

int resolve_output_device(const std::string& substring) {
  if (substring.empty()) {
    return Pa_GetDefaultOutputDevice();
  }
  int count = Pa_GetDeviceCount();
  for (int i = 0; i < count; ++i) {
    const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
    if (!info || info->maxOutputChannels <= 0) continue;
    std::string name = info->name ? info->name : "";
    if (name.find(substring) != std::string::npos) return i;
  }
  throw std::runtime_error("no output device matching \"" + substring + "\"");
}

}  // namespace

// ---------------------------------------------------------------------------
// PlaybackQueue
// ---------------------------------------------------------------------------

void PlaybackQueue::push(Chunk chunk) {
  std::lock_guard<std::mutex> lk(m_);
  chunks_.emplace_back(std::move(chunk));
}

void PlaybackQueue::close() { closed_.store(true); }

std::size_t PlaybackQueue::fill(float* out, std::size_t num_samples) {
  std::size_t written = 0;
  std::size_t real_samples = 0;
  while (written < num_samples) {
    if (partial_pos_ < partial_.size()) {
      const std::size_t take =
          std::min(partial_.size() - partial_pos_, num_samples - written);
      std::memcpy(out + written, partial_.data() + partial_pos_,
                  sizeof(float) * take);
      partial_pos_ += take;
      written += take;
      real_samples += take;
      continue;
    }
    // Need a new chunk.
    Chunk next;
    {
      std::lock_guard<std::mutex> lk(m_);
      if (chunks_.empty()) break;
      next = std::move(chunks_.front());
      chunks_.pop_front();
    }
    partial_ = std::move(next.samples);
    partial_pos_ = 0;
  }
  if (written < num_samples) {
    std::memset(out + written, 0, sizeof(float) * (num_samples - written));
  }
  return real_samples;
}

std::size_t PlaybackQueue::size() const {
  std::lock_guard<std::mutex> lk(m_);
  return chunks_.size();
}

bool PlaybackQueue::drained() const {
  std::lock_guard<std::mutex> lk(m_);
  return closed_.load() && chunks_.empty() && partial_pos_ >= partial_.size();
}

// ---------------------------------------------------------------------------
// PortAudioStream
// ---------------------------------------------------------------------------

PortAudioStream::PortAudioStream(PlaybackQueue& queue,
                                 const PlaybackConfig& config)
    : queue_(queue), config_(config) {
  pa_guard();  // ensure PortAudio is initialised

  int device = resolve_output_device(config_.device_substring);
  if (device == paNoDevice) {
    throw std::runtime_error("no default output device available");
  }
  const PaDeviceInfo* info = Pa_GetDeviceInfo(device);
  if (!info) {
    throw std::runtime_error("Pa_GetDeviceInfo failed for resolved device");
  }

  PaStreamParameters out_params{};
  out_params.device = device;
  out_params.channelCount = config_.num_channels;
  out_params.sampleFormat = paFloat32;
  out_params.suggestedLatency = info->defaultLowOutputLatency;
  out_params.hostApiSpecificStreamInfo = nullptr;

  PaError err = Pa_OpenStream(
      &stream_, /*inputParameters=*/nullptr, &out_params, config_.sample_rate,
      paFramesPerBufferUnspecified, paNoFlag, &PortAudioStream::pa_callback,
      this);
  if (err != paNoError) {
    throw std::runtime_error(std::string("Pa_OpenStream failed: ") +
                             Pa_GetErrorText(err));
  }

  std::cerr << "playback: device=" << info->name
            << " sr=" << config_.sample_rate
            << " ch=" << config_.num_channels << "\n";
}

PortAudioStream::~PortAudioStream() {
  stop();
  if (stream_) {
    Pa_CloseStream(stream_);
    stream_ = nullptr;
  }
}

void PortAudioStream::start() {
  if (started_) return;
  PaError err = Pa_StartStream(stream_);
  if (err != paNoError) {
    throw std::runtime_error(std::string("Pa_StartStream failed: ") +
                             Pa_GetErrorText(err));
  }
  started_ = true;
}

void PortAudioStream::stop() {
  if (!started_) return;
  Pa_StopStream(stream_);
  started_ = false;
}

int PortAudioStream::pa_callback(const void* /*input*/, void* output,
                                 unsigned long frame_count,
                                 const PaStreamCallbackTimeInfo* /*time_info*/,
                                 PaStreamCallbackFlags /*status_flags*/,
                                 void* user_data) {
  auto* self = static_cast<PortAudioStream*>(user_data);
  float* out = static_cast<float*>(output);
  const std::size_t total = static_cast<std::size_t>(frame_count) *
                            static_cast<std::size_t>(self->config_.num_channels);
  self->queue_.fill(out, total);
  return paContinue;
}

// ---------------------------------------------------------------------------
// list_devices
// ---------------------------------------------------------------------------

void list_devices() {
  pa_guard();
  int count = Pa_GetDeviceCount();
  std::cout << "PortAudio devices (" << count << "):\n";
  const int default_out = Pa_GetDefaultOutputDevice();
  for (int i = 0; i < count; ++i) {
    const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
    if (!info) continue;
    std::cout << (i == default_out ? "* " : "  ") << "[" << i << "] "
              << info->name << "  (in=" << info->maxInputChannels
              << ", out=" << info->maxOutputChannels
              << ", sr=" << info->defaultSampleRate << ")\n";
  }
}

}  // namespace magenta_realtime_mlx::playback

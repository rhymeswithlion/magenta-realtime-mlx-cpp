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

// ``mlx-stream`` -- C++ MLX MagentaRT streaming binary.
//
// Generates continuous music from a text prompt and either plays it out via
// PortAudio or runs a dry-run smoke test.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "magenta_realtime_mlx/dtype.h"
#include "magenta_realtime_mlx/playback.h"
#include "magenta_realtime_mlx/system.h"

namespace {

namespace mrt = magenta_realtime_mlx;
namespace mx = mlx::core;
namespace pb = magenta_realtime_mlx::playback;
namespace fs = std::filesystem;

struct Args {
  std::string prompt = "deep house";
  std::uint64_t seed = 0;
  float temperature = 1.1f;
  int top_k = 40;
  float guidance_weight = 5.0f;

  std::optional<fs::path> weights_dir;
  std::string dtype = "bf16";
  std::string tag = "base";

  int warmup_chunks = 1;
  std::optional<int> max_chunks;

  std::string device;
  bool list_devices = false;
  bool dry_run = false;
  std::optional<fs::path> record;
  bool no_record = false;

  // Off by default. The per-draft sync (``mx::eval`` + ``.item()``
  // every depth step) dominates the savings on Apple Silicon at typical
  // RVQ acceptance rates, making speculation a net-negative for live
  // streaming. See ``System::generate_chunk``.
  bool speculative = false;

  // Live-streaming pre-buffer (chunks of audio to accumulate before
  // PortAudio starts pulling). Without a pre-buffer, PortAudio starts
  // immediately on stream open and plays silence until the first chunk
  // is generated (~1.3 s on M3 Ultra), then any per-chunk jitter that
  // exceeds the chunk duration causes underruns. Default is 2 chunks
  // (~4 s of audio @ 2 s / chunk) which makes startup audibly clean and
  // leaves headroom for the rare 2x-RTF chunk. Set to 0 to disable.
  int prebuffer_chunks = 2;

  // Cap on the number of chunks the queue is allowed to hold *while
  // playback is running*. Once the queue is full the generation loop
  // throttles itself (50 ms polls) until the audio callback drains a
  // chunk. Prevents the queue from growing unbounded when generation
  // is faster than realtime (RTF 1.55x → +0.7 s of audio per chunk
  // wall time → unbounded memory & latency over a long session).
  // Default 3 keeps the queue at 2-3 in steady state, which is
  // ``prebuffer_chunks (2) + 1 chunk of headroom``. Set to 0 to
  // disable the cap (recover the unbounded-queue behaviour).
  int max_queue_chunks = 3;

  // Auto-discover ``encode/depth_step/temporal_step`` ``.mlxfn`` bundles
  // under ``<weights-dir>/mlxfn/`` and load them at System construction
  // time. Without these the C++ Depthformer falls back to its
  // capturing-lambda path which is ~50 % slower (RTF ~1.0 vs ~1.5 on
  // M3 Ultra). On by default since ``make cpp-stream`` users almost
  // always want the fast path; pass ``--no-mlxfn`` to disable for A/B.
  bool use_mlxfn = true;
};

void print_usage() {
  std::cout <<
      "Usage: mlx-stream [options]\n"
      "Options:\n"
      "  --prompt <text>           style prompt (default: \"deep house\")\n"
      "  --seed <int>              base RNG seed, incremented per chunk\n"
      "  --temperature <float>     sampling temperature (default 1.1)\n"
      "  --top-k <int>             top-k filter (default 40)\n"
      "  --guidance-weight <float> CFG weight (default 5.0)\n"
      "  --weights-dir <path>      weights cache root (default "
      "<repo>/.weights-cache)\n"
      "  --dtype <fp32|fp16|bf16>  Depthformer compute dtype (default bf16; "
      "the codec and MusicCoCa always run in fp32)\n"
      "  --tag <name>              depthformer checkpoint tag (default base)\n"
      "  --warmup-chunks <int>     chunks generated before playback starts\n"
      "  --max-chunks <int>        stop after N chunks beyond warmup\n"
      "  --device <name>           PortAudio output device substring\n"
      "  --list-devices            print PortAudio devices and exit\n"
      "  --dry-run                 skip playback; requires --max-chunks\n"
      "  --record <path>           write concatenated played audio to WAV\n"
      "  --no-record               do not write a WAV\n"
      "  --speculative             enable speculative depth decoding "
      "(slower on Apple Silicon for typical acceptance rates; off by default)\n"
      "  --prebuffer-chunks <int>  chunks of audio to accumulate before live\n"
      "                            playback starts (default 2 = ~4 s; 0 = off)\n"
      "  --max-queue-chunks <int>  cap on queued audio while playing (default 3;\n"
      "                            0 = unbounded). Generator throttles when full.\n"
      "  --no-mlxfn                disable auto-discovery of .mlxfn bundles\n"
      "                            (forces the slower capturing-lambda path)\n"
      "  -h, --help                show this help\n";
}

bool starts_with(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && std::memcmp(s.data(), p.data(), p.size()) == 0;
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto take = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[++i];
    };
    if (arg == "-h" || arg == "--help") {
      print_usage();
      std::exit(0);
    } else if (arg == "--prompt") {
      a.prompt = take("--prompt");
    } else if (arg == "--seed") {
      a.seed = std::stoull(take("--seed"));
    } else if (arg == "--temperature") {
      a.temperature = std::stof(take("--temperature"));
    } else if (arg == "--top-k") {
      a.top_k = std::stoi(take("--top-k"));
    } else if (arg == "--guidance-weight") {
      a.guidance_weight = std::stof(take("--guidance-weight"));
    } else if (arg == "--weights-dir") {
      a.weights_dir = fs::path(take("--weights-dir"));
    } else if (arg == "--dtype") {
      a.dtype = take("--dtype");
    } else if (arg == "--tag") {
      a.tag = take("--tag");
    } else if (arg == "--warmup-chunks") {
      a.warmup_chunks = std::stoi(take("--warmup-chunks"));
    } else if (arg == "--max-chunks") {
      a.max_chunks = std::stoi(take("--max-chunks"));
    } else if (arg == "--device") {
      a.device = take("--device");
    } else if (arg == "--list-devices") {
      a.list_devices = true;
    } else if (arg == "--dry-run") {
      a.dry_run = true;
    } else if (arg == "--record") {
      a.record = fs::path(take("--record"));
    } else if (arg == "--no-record") {
      a.no_record = true;
    } else if (arg == "--speculative") {
      a.speculative = true;
    } else if (arg == "--prebuffer-chunks") {
      a.prebuffer_chunks = std::stoi(take("--prebuffer-chunks"));
      if (a.prebuffer_chunks < 0) a.prebuffer_chunks = 0;
    } else if (arg == "--max-queue-chunks") {
      a.max_queue_chunks = std::stoi(take("--max-queue-chunks"));
      if (a.max_queue_chunks < 0) a.max_queue_chunks = 0;
    } else if (arg == "--no-mlxfn") {
      a.use_mlxfn = false;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return a;
}

mx::Dtype parse_dtype(const std::string& name) {
  if (name == "fp32" || name == "float32") return mx::float32;
  if (name == "fp16" || name == "float16") return mx::float16;
  if (name == "bf16" || name == "bfloat16") return mx::bfloat16;
  throw std::runtime_error("--dtype must be one of fp32|fp16|bf16");
}

fs::path default_weights_dir() {
  // Walk up from the current working directory looking for ``.weights-cache``
  // or ``<repo>/.weights-cache``. This matches the Python launcher's
  // expectation of being run from inside the repo.
  fs::path cwd = fs::current_path();
  for (int up = 0; up <= 5; ++up) {
    fs::path candidate = cwd;
    for (int i = 0; i < up; ++i) candidate = candidate.parent_path();
    fs::path cache = candidate / ".weights-cache";
    if (fs::is_directory(cache)) return cache;
  }
  return cwd / ".weights-cache";
}

// Tiny WAV writer: interleaved float32 stereo -> PCM16 RIFF file, because
// ``soundfile`` / ``libsndfile`` would be a new dependency we don't otherwise
// need. Called once at shutdown, so performance is not a concern.
void write_wav_pcm16(const fs::path& path, const std::vector<float>& samples,
                     int sample_rate, int num_channels) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path.string());

  const std::uint32_t n_frames =
      static_cast<std::uint32_t>(samples.size() / num_channels);
  const std::uint32_t data_bytes = n_frames * num_channels * 2;
  const std::uint32_t riff_size = 36 + data_bytes;

  auto w32 = [&](std::uint32_t v) {
    f.put(static_cast<char>(v & 0xff));
    f.put(static_cast<char>((v >> 8) & 0xff));
    f.put(static_cast<char>((v >> 16) & 0xff));
    f.put(static_cast<char>((v >> 24) & 0xff));
  };
  auto w16 = [&](std::uint16_t v) {
    f.put(static_cast<char>(v & 0xff));
    f.put(static_cast<char>((v >> 8) & 0xff));
  };

  f.write("RIFF", 4);
  w32(riff_size);
  f.write("WAVE", 4);
  f.write("fmt ", 4);
  w32(16);            // PCM fmt chunk size
  w16(1);             // PCM format
  w16(static_cast<std::uint16_t>(num_channels));
  w32(static_cast<std::uint32_t>(sample_rate));
  w32(static_cast<std::uint32_t>(sample_rate * num_channels * 2));
  w16(static_cast<std::uint16_t>(num_channels * 2));
  w16(16);
  f.write("data", 4);
  w32(data_bytes);

  for (float x : samples) {
    float clipped = x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
    std::int16_t s = static_cast<std::int16_t>(clipped * 32767.0f);
    f.put(static_cast<char>(s & 0xff));
    f.put(static_cast<char>((s >> 8) & 0xff));
  }
}

std::atomic<bool> g_stop_requested{false};
void handle_sigint(int /*sig*/) { g_stop_requested.store(true); }

std::vector<float> copy_float(const mx::array& waveform) {
  std::vector<float> out(waveform.size());
  std::memcpy(out.data(), waveform.data<float>(),
              sizeof(float) * waveform.size());
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  try {
    args = parse_args(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "error parsing args: " << e.what() << "\n";
    return 2;
  }

  if (args.list_devices) {
    try {
      pb::list_devices();
    } catch (const std::exception& e) {
      std::cerr << "list_devices failed: " << e.what() << "\n";
      return 3;
    }
    return 0;
  }

  if (args.dry_run && !args.max_chunks) {
    std::cerr << "--dry-run requires --max-chunks N (N >= 1)\n";
    return 2;
  }

  mx::Dtype dtype = mx::float32;
  try {
    dtype = parse_dtype(args.dtype);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 2;
  }

  fs::path weights_dir = args.weights_dir.value_or(default_weights_dir());
  std::cerr << "weights: " << weights_dir << "\n";

  // Auto-discover pre-traced ``.mlxfn`` bundles under
  // ``<weights-dir>/mlxfn/`` and set the env vars the Depthformer reads at
  // ``compile_for_inference`` time. Without this, the binary loads but
  // falls back to the capturing-lambda path (~50 % slower per chunk on
  // M3 Ultra). Existing env-var values win, so power users can still
  // point at a custom directory or disable individual bundles.
  if (args.use_mlxfn) {
    fs::path mlxfn_dir = weights_dir / "mlxfn";
    if (fs::is_directory(mlxfn_dir)) {
      const std::string suffix =
          std::string("_") + args.tag + "_" + args.dtype;
      fs::path encode_path = mlxfn_dir / ("encode" + suffix + ".mlxfn");
      const bool have_encode = fs::is_regular_file(encode_path);
      // Padded single-graph bundles (mlxfn manifest format_version 3,
      // see g29). When present, these REPLACE the per-cl tables -- one
      // file per method instead of 49 + 15.
      fs::path padded_temporal_path =
          mlxfn_dir / ("temporal_step_padded" + suffix + ".mlxfn");
      fs::path padded_depth_path =
          mlxfn_dir / ("depth_step_padded" + suffix + ".mlxfn");
      const bool have_padded_temporal =
          fs::is_regular_file(padded_temporal_path);
      const bool have_padded_depth =
          fs::is_regular_file(padded_depth_path);
      // Per-cl bundles (format_version 2) -- legacy fallback. depth_step
      // / temporal_step _cl<NN>.mlxfn presence is enough to consider the
      // bundle "available"; the Depthformer loader scans the directory.
      const bool have_depth_per_cl = fs::is_regular_file(
          mlxfn_dir / ("depth_step" + suffix + "_cl01.mlxfn"));
      const bool have_temporal_per_cl = fs::is_regular_file(
          mlxfn_dir / ("temporal_step" + suffix + "_cl01.mlxfn"));
      auto setenv_if_unset = [](const char* name, const std::string& val) {
        if (std::getenv(name) == nullptr) {
          ::setenv(name, val.c_str(), /*overwrite=*/0);
        }
      };
      if (have_encode) {
        setenv_if_unset("MRT_DEPTHFORMER_ENCODE_MLXFN", encode_path.string());
      }
      // Per-cl wins over padded when both are present. Per-cl runs SDPA
      // at the actual K seq-len; the padded variant always runs at the
      // max length, which on Apple Silicon (per-shape-fused SDPA Metal
      // kernels) is ~25% slower in mlx-stream end-to-end. The padded
      // path stays loadable as a fallback when only the smaller 3-file
      // bundle is present (e.g., size-sensitive deployments).
      const bool depth_active = have_depth_per_cl || have_padded_depth;
      const bool temporal_active = have_temporal_per_cl || have_padded_temporal;
      if (have_temporal_per_cl) {
        setenv_if_unset("MRT_DEPTHFORMER_TEMPORAL_MLXFN_DIR",
                        mlxfn_dir.string());
      } else if (have_padded_temporal) {
        setenv_if_unset("MRT_DEPTHFORMER_TEMPORAL_PADDED_MLXFN",
                        padded_temporal_path.string());
      }
      if (have_depth_per_cl) {
        setenv_if_unset("MRT_DEPTHFORMER_DEPTH_MLXFN_DIR",
                        mlxfn_dir.string());
      } else if (have_padded_depth) {
        setenv_if_unset("MRT_DEPTHFORMER_DEPTH_PADDED_MLXFN",
                        padded_depth_path.string());
      }
      const int loaded_count =
          (have_encode ? 1 : 0) + (depth_active ? 1 : 0) +
          (temporal_active ? 1 : 0);
      if (loaded_count == 0) {
        std::cerr << "mlxfn: directory exists but no bundles for tag="
                  << args.tag << " dtype=" << args.dtype
                  << " (looked for encode/depth_step/temporal_step"
                  << suffix << "*.mlxfn); falling back to capturing-lambda "
                  << "path (~50% slower on M3 Ultra)\n";
      } else {
        std::vector<std::string> bits;
        if (have_encode) bits.emplace_back("encode");
        if (have_depth_per_cl) bits.emplace_back("depth_step");
        else if (have_padded_depth) bits.emplace_back("depth_step(padded)");
        if (have_temporal_per_cl) bits.emplace_back("temporal_step");
        else if (have_padded_temporal) bits.emplace_back("temporal_step(padded)");
        std::ostringstream joined;
        for (std::size_t i = 0; i < bits.size(); ++i) {
          if (i) joined << "+";
          joined << bits[i];
        }
        std::cerr << "mlxfn: enabled " << joined.str() << " from "
                  << mlxfn_dir << "\n";
      }
    } else {
      std::cerr << "mlxfn: " << mlxfn_dir << " not found; running on the "
                << "capturing-lambda path (~50% slower). The published "
                << "bundles ship with the weights on Hugging Face -- "
                << "rerun `make ensure-weights-cache` to download them, "
                << "or pass --no-mlxfn to silence this notice.\n";
    }
  }

  std::unique_ptr<mrt::System> system;
  try {
    system = std::make_unique<mrt::System>(weights_dir, args.tag, dtype);
  } catch (const std::exception& e) {
    std::cerr << "failed to load System: " << e.what() << "\n";
    return 4;
  }

  std::cerr << "prompt: \"" << args.prompt << "\"\n";
  mx::array style_tokens_lm = system->embed_style(args.prompt);
  mx::eval(style_tokens_lm);

  mrt::SystemState state = system->empty_state();

  mrt::GenerateChunkOptions base_opts;
  base_opts.temperature = args.temperature;
  base_opts.top_k = args.top_k;
  base_opts.guidance_weight = args.guidance_weight;
  base_opts.speculative = args.speculative;
  // Cross-chunk CPU encoder pipelining: hides the ~90 ms encoder slot
  // under the previous chunk's codec phase. ``mlx-stream`` always uses
  // a single fixed prompt for the whole session, so the cache always
  // hits. Opt-out via ``MRT_DISABLE_ENCODER_PIPELINE=1`` for A/B
  // benchmarks (mirrors ``MRT_DISABLE_ASYNC_EVAL``).
  base_opts.pipeline_encoder =
      std::getenv("MRT_DISABLE_ENCODER_PIPELINE") == nullptr ||
      std::getenv("MRT_DISABLE_ENCODER_PIPELINE")[0] == '\0';

  std::cerr << "warmup: " << args.warmup_chunks << " chunk(s)\n";
  auto warm_t0 = std::chrono::steady_clock::now();
  for (int w = 0; w < args.warmup_chunks; ++w) {
    mrt::GenerateChunkOptions opts = base_opts;
    opts.seed = args.seed + static_cast<std::uint64_t>(w);
    system->generate_chunk(state, style_tokens_lm, opts);
  }
  const double warm_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - warm_t0)
                            .count();
  std::cerr << "warmup done in " << std::fixed << std::setprecision(2)
            << warm_s << " s (chunk_index=" << state.chunk_index << ")\n";

  if (args.dry_run) {
    std::vector<double> per_chunk_ms;
    per_chunk_ms.reserve(*args.max_chunks);
    std::vector<float> recorded_dry;
    const bool want_record_dry = args.record.has_value() && !args.no_record;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < *args.max_chunks; ++i) {
      auto c0 = std::chrono::steady_clock::now();
      mrt::GenerateChunkOptions opts = base_opts;
      opts.seed = args.seed + static_cast<std::uint64_t>(state.chunk_index);
      auto r = system->generate_chunk(state, style_tokens_lm, opts);
      const double cdt_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - c0)
              .count();
      per_chunk_ms.push_back(cdt_ms);
      if (want_record_dry) {
        std::vector<float> samples = copy_float(r.waveform.samples);
        const std::size_t old = recorded_dry.size();
        recorded_dry.resize(old + samples.size());
        std::memcpy(recorded_dry.data() + old, samples.data(),
                    sizeof(float) * samples.size());
      }
    }
    if (want_record_dry && !recorded_dry.empty()) {
      const fs::path& rp = *args.record;
      fs::create_directories(rp.parent_path());
      try {
        write_wav_pcm16(rp, recorded_dry, system->sample_rate(),
                        system->num_channels());
        std::cerr << "wrote " << rp << " ("
                  << recorded_dry.size() / system->num_channels() << " frames)\n";
      } catch (const std::exception& e) {
        std::cerr << "failed to write WAV: " << e.what() << "\n";
      }
    }
    const double dt =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0)
            .count();
    const double gen_s = *args.max_chunks * system->chunk_length();
    std::cerr << "dry-run: " << *args.max_chunks << " chunks in "
              << std::fixed << std::setprecision(2) << dt
              << " s -> RTF " << gen_s / dt << "x\n";

    if (!per_chunk_ms.empty()) {
      std::vector<double> sorted = per_chunk_ms;
      std::sort(sorted.begin(), sorted.end());
      auto pct = [&](double p) {
        if (sorted.size() == 1) return sorted.front();
        double idx = p * (sorted.size() - 1);
        std::size_t lo = static_cast<std::size_t>(idx);
        std::size_t hi = std::min(lo + 1, sorted.size() - 1);
        double frac = idx - lo;
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
      };
      double sum = 0.0;
      for (double v : per_chunk_ms) sum += v;
      std::cerr << "  latency ms  mean=" << std::fixed << std::setprecision(1)
                << sum / per_chunk_ms.size()
                << " p50=" << pct(0.50)
                << " p99=" << pct(0.99)
                << " min=" << sorted.front()
                << " max=" << sorted.back() << "\n";
    }
    return 0;
  }

  // Live playback.
  pb::PlaybackConfig pcfg{};
  pcfg.sample_rate = system->sample_rate();
  pcfg.num_channels = system->num_channels();
  pcfg.device_substring = args.device;

  pb::PlaybackQueue queue;
  std::unique_ptr<pb::PortAudioStream> stream;
  try {
    // Open the device but do NOT start the callback yet -- we want to
    // accumulate ``args.prebuffer_chunks`` worth of audio first so the
    // first thing the user hears is real generation, not silence, and so
    // a single slow chunk doesn't underrun the device.
    stream = std::make_unique<pb::PortAudioStream>(queue, pcfg);
  } catch (const std::exception& e) {
    std::cerr << "playback init failed: " << e.what() << "\n";
    return 5;
  }

  std::signal(SIGINT, handle_sigint);

  std::vector<float> recorded;
  fs::path record_path =
      args.record.value_or(fs::path("output") / "mlx_stream.wav");
  const bool want_record = !args.no_record;

  // ``stream->start()`` is deferred until either ``args.prebuffer_chunks``
  // chunks have been queued or we hit the chunk budget. ``stream_started``
  // tracks that one-shot transition.
  bool stream_started = false;
  auto maybe_start_stream = [&](std::size_t qsize) {
    if (stream_started) return;
    if (static_cast<int>(qsize) < args.prebuffer_chunks) return;
    try {
      stream->start();
      stream_started = true;
      const double buffered_s = static_cast<double>(qsize) *
                                 system->chunk_length();
      std::cerr << "playback: started after pre-buffering " << qsize
                << " chunk(s) (~" << std::fixed << std::setprecision(1)
                << buffered_s << " s of audio)\n";
    } catch (const std::exception& e) {
      std::cerr << "playback start failed: " << e.what() << "\n";
    }
  };

  if (args.prebuffer_chunks > 0) {
    std::cerr << "streaming (pre-buffering " << args.prebuffer_chunks
              << " chunk(s) before playback; Ctrl+C to stop)\n";
  } else {
    // Match the original behaviour when the user explicitly opts out.
    try {
      stream->start();
      stream_started = true;
    } catch (const std::exception& e) {
      std::cerr << "playback start failed: " << e.what() << "\n";
      return 5;
    }
    std::cerr << "streaming (Ctrl+C to stop)\n";
  }
  int budget = args.max_chunks.value_or(std::numeric_limits<int>::max());
  while (budget > 0 && !g_stop_requested.load()) {
    // Back-pressure: once playback is running and the queue is at the
    // configured cap, sleep in 50 ms increments until either a chunk
    // drains or the user hits Ctrl+C. This is what bounds memory and
    // latency over a long session (RTF 1.55x means we'd otherwise grow
    // the queue by ~0.7 s of audio per chunk wall time, forever).
    double throttle_ms = 0.0;
    if (args.max_queue_chunks > 0 && stream_started) {
      auto throttle_t0 = std::chrono::steady_clock::now();
      while (!g_stop_requested.load() &&
             static_cast<int>(queue.size()) >= args.max_queue_chunks) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      throttle_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - throttle_t0)
                        .count();
      if (g_stop_requested.load()) break;
    }
    auto t0 = std::chrono::steady_clock::now();
    mrt::GenerateChunkOptions opts = base_opts;
    opts.seed = args.seed + static_cast<std::uint64_t>(state.chunk_index);
    std::optional<mrt::GenerateResult> r_opt;
    try {
      r_opt = system->generate_chunk(state, style_tokens_lm, opts);
    } catch (const std::exception& e) {
      std::cerr << "generate_chunk failed: " << e.what() << "\n";
      break;
    }
    mrt::GenerateResult& r = *r_opt;
    const double dt =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0)
            .count();
    std::vector<float> samples = copy_float(r.waveform.samples);
    const int T = r.waveform.num_samples();
    const int C = r.waveform.num_channels();
    const double chunk_s = r.waveform.duration();

    if (want_record) {
      const std::size_t old = recorded.size();
      recorded.resize(old + samples.size());
      std::memcpy(recorded.data() + old, samples.data(),
                  sizeof(float) * samples.size());
    }

    pb::Chunk chunk;
    chunk.samples = std::move(samples);
    chunk.num_frames = T;
    chunk.num_channels = C;
    queue.push(std::move(chunk));
    const std::size_t qsize = queue.size();
    maybe_start_stream(qsize);

    std::cerr << "  chunk " << state.chunk_index << ": wall "
              << std::fixed << std::setprecision(0) << dt * 1000.0 << " ms "
              << "RTF " << std::setprecision(2) << chunk_s / dt << "x"
              << " queue=" << qsize
              << (stream_started ? " (playing)" : " (buffering)");
    if (throttle_ms > 5.0) {
      std::cerr << " throttled=" << std::setprecision(0) << throttle_ms
                << " ms";
    }
    std::cerr << "\n";

    if (args.max_chunks) budget -= 1;
  }

  const bool stopped_by_user = g_stop_requested.load();

  if (stopped_by_user) {
    // Ctrl+C path: stop the device *now*, don't wait for the queue to
    // drain. Whatever was already generated is still in ``recorded``
    // and will be written below, so the user gets the full session
    // captured to disk even though playback cuts off mid-chunk.
    if (stream_started) {
      stream->stop();
      stream_started = false;
    }
    queue.close();
    std::cerr << "\nstopped (Ctrl+C); writing buffer...\n";
  } else {
    // Natural exit (``--max-chunks`` budget reached or ``generate_chunk``
    // threw): if we never started the stream (e.g. ``--max-chunks`` <
    // ``--prebuffer-chunks``), start it now so the user hears the
    // result, then drain the remaining queue before stopping.
    if (!stream_started && queue.size() > 0) {
      try {
        stream->start();
        stream_started = true;
      } catch (const std::exception& e) {
        std::cerr << "playback start failed: " << e.what() << "\n";
      }
    }
    std::cerr << "\ndraining playback...\n";
    queue.close();
    if (stream_started) {
      // Each queued chunk is ``system->chunk_length()`` seconds; pad
      // by 1 s for callback latency. Ctrl+C during the drain wake-up
      // still aborts (we re-check the flag in the loop below).
      const double drain_s =
          static_cast<double>(queue.size()) * system->chunk_length() + 1.0;
      const auto deadline = std::chrono::steady_clock::now() +
                             std::chrono::milliseconds(
                                 static_cast<int>(drain_s * 1000.0));
      while (std::chrono::steady_clock::now() < deadline &&
             !g_stop_requested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      stream->stop();
    }
  }

  if (want_record && !recorded.empty()) {
    fs::create_directories(record_path.parent_path());
    try {
      write_wav_pcm16(record_path, recorded, system->sample_rate(),
                      system->num_channels());
      std::cerr << "wrote " << record_path << " ("
                << recorded.size() / system->num_channels() /
                       system->sample_rate()
                << " s)\n";
    } catch (const std::exception& e) {
      std::cerr << "failed to write wav: " << e.what() << "\n";
    }
  }

  return 0;
}

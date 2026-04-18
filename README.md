# MagentaRealtimeMLX

https://github.com/user-attachments/assets/e8771941-56ed-452b-8615-d2391a6fa520

Real-time music generation with [Magenta RealTime](https://github.com/magenta/magenta-realtime),
implemented natively against MLX in C++ for Apple Silicon. The streaming
binary holds **>1.0× real-time factor on M-series laptops** (M3 Pro and
faster), so a 16-bar prompt generates 16 bars of audio in less wall time
than it takes to play back.

The repository ships two end-to-end paths:

```bash
make mlx-stream       # C++ MLX binary -- the production path
make ref-stream       # upstream JAX/TF reference, for comparison
```

`mlx-stream` is the main path. `ref-stream` runs the upstream
`magenta_rt` baseline through an isolated Python venv (much slower; it
exists so you can A/B the audio output character against the reference
implementation).

## Quick Start

```bash
make mlx-stream                                           # the basics
make mlx-stream PROMPT="deep house"          # any free-form style prompt
make mlx-stream MLX_FLAGS="--dry-run --max-chunks 8"      # benchmark, no playback
make ref-stream PROMPT="deep house"                       # reference baseline
```

On first run, `mlx-stream` will:

1. `uv sync` the install-time helper dependencies,
2. download the MLX weights bundle from Hugging Face into
   `.weights-cache/` (via `scripts/download_weights_from_hf.py`),
3. configure and build the C++ binary under `build/`,
4. start live playback. Press Ctrl+C to stop.

`make help` lists the small public command surface.

## Measured Performance

Sustained per-chunk RTF (chunk = 2 s of audio, default `--dtype bf16`,
`.mlxfn` bundles loaded). Reproduce with
`make mlx-stream MLX_FLAGS="--dry-run --max-chunks 30"`.

| Machine                            | `mlx-stream` (Metal GPU)           | `ref-stream` (JAX/TF, CPU only)       |
| ---------------------------------- | ---------------------------------- | ------------------------------------- |
| MacBook Pro, M3 Pro, 18 GB         | ~1.03x RTF (~1.94 s / 2 s chunk)   | ~0.08x RTF (~24 s / 2 s chunk)        |

`mlx-stream` clears real time on a 14" laptop without active cooling.
The `ref-stream` column is **not an apples-to-apples comparison**: the
upstream JAX/TF stack has no production Metal backend on Apple Silicon
and falls back to CPU here, so the gap above is part "C++ vs Python
overhead" and mostly "GPU vs CPU". `ref-stream` exists for audio
side-by-side, not as a performance baseline.

## Why the C++ Port Is Faster Than Real Time on a MacBook

MagentaRT generates 16 RVQ tokens per audio frame and 50 frames per
2-second chunk -- roughly 800 sequential transformer steps in the
Depthformer LLM per chunk, plus codec and style-encoder work. Hitting
RTF >= 1.0 on a laptop GPU (M3 Pro: ~12 TFLOPS fp16) means leaving very
little overhead on the table. The C++ port reaches that target by
removing the per-step Python overhead the original implementation paid
on every one of those 800 steps, and by lowering MLX scheduling cost
into the static pre-compile phase wherever possible.

The headline wins on M3 Ultra (chunk = 2 s of audio):

| Optimization                                     | RTF impact (M3 Ultra)     |
| ------------------------------------------------ | ------------------------- |
| C++ port baseline (capturing-lambda compile)     | ~0.84× -> ~1.05×          |
| Pre-traced `.mlxfn` source-graph bundles         | ~1.05× -> ~1.55×          |
| `bf16` Depthformer compute (codec stays fp32)    | additional ~30 %          |
| Cross-chunk encoder pipelining                   | RTF 1.55× -> 1.62×        |
| Cached `Linear` weight views + weights-as-args   | a few ms / chunk          |
| Hot-path scalar reuse + `mx::compile` on FFN     | a few ms / chunk each     |

In more detail:

**1. `.mlxfn` source-graph bundles.**  MLX's `mx::compile` produces
fused Metal kernels from an op graph; the topology of that graph
matters. The first cut of the C++ port rebuilt the graph from scratch
in a `mx::compile(...)` capturing lambda on every call, and Apple's MLX
compiler turned out to fuse the same logical computation slightly
*differently* than when the graph was traced ahead of time and loaded
with `mx::core::import_function`. The pre-traced version produces
better-fused kernels. We ship one `.mlxfn` per logical entry point
(`encode`, plus `temporal_step` / `depth_step` per cache length 0..15),
download them with the weights, and load them once at startup. Without
them the binary still works -- it just falls back to the
capturing-lambda path, which is roughly 50 % slower per chunk.

**2. `bfloat16` Depthformer.**  The 24-layer transformer dominates the
chunk budget. Apple Silicon's matmul throughput at `bf16` is roughly
2× `fp32`, with no audible quality loss for this model. The codec
(SpectroStream) and the style encoder (MusicCoCa) stay at `fp32` because
they are short and quality-sensitive at the spectral edges. `--dtype`
controls the LLM precision; the codec dtype is fixed.

**3. Cross-chunk encoder pipelining.**  The Depthformer encoder runs on
the prompt-context tokens once per chunk and feeds cross-attention K/V
to the temporal stack. By queuing the next chunk's encoder work on the
GPU *before* the current chunk's codec phase, that encoder gets to
overlap with codec instead of contending with the next chunk's first
temporal_step. Net win: ~16 ms per chunk on M3 Ultra without any CPU
parallelism (the same Metal command queue just keeps the encoder warm).

**4. Static weight precomputation.**  Two micro-wins that stack:
`Linear::weight()` used to call `mx::transpose` on every access, which
was being hit O(thousands) times per chunk inside the depth loop; we
now cache the transposed view at construction. Separately, the lists of
weight `mx::array` references handed to the `.mlxfn` compiled functions
are now precomputed once at `Depthformer` construction (instead of
walked from the layer tree on every step). Each saves a few
milliseconds per chunk; together they push us from RTF 0.95× to 0.97×
on a baseline M3 Pro before the encoder pipelining lands.

**5. Hot-path scalar reuse + shapeless-compile FFN.**  The inner depth
loop runs ~800 iterations per chunk. Constructing fresh `mx::array`
literals for `-inf`, the CFG cond/uncond weights, the sampling
temperature, and the RVQ depth divisor on every iteration meant
allocating ~3,000 redundant 0-d arrays per chunk. Building them once
per chunk is a measurable win. Similarly, `gelu_approx` (used in every
FFN block, ~19 k calls per chunk) is now wrapped in
`mx::compile(..., shapeless=true)` once at first use; without that
wrapper the FFN paid for ~9 individual ops + 4 scalar literal
allocations per call.

The overall recipe is unsurprising: pre-trace what you can,
precompute what only depends on weights, reuse hot-path scalars, and
let the GPU command queue overlap independent work across chunk
boundaries. The numbers above are reproducible via
`make mlx-stream MLX_FLAGS="--dry-run --max-chunks 30"`, which prints
mean / p50 / p99 / min / max per-chunk latency at the end of the run.

## Weights

`mlx-stream` reads weights from `.weights-cache/`. The published bundle
contains exactly what the C++ binary loads at startup -- nothing else:

- `*.safetensors` for SpectroStream encoder / decoder / codebooks,
  MusicCoCa encoder / codebooks, and the Depthformer LLM.
- `musiccoca_vocab.model` for the SentencePiece tokenizer.
- `mlxfn/*.mlxfn` source-graph bundles loaded via
  `mlx::core::import_function` (1 + 49 + 15 files, optional but on by
  default; without them the binary falls back to a ~50 % slower
  capturing-lambda compile path).

The original `.pt` / `.npy` checkpoints and the bundling / `.mlxfn`
export tooling are not part of this distribution.

`make mlx-stream` runs `ensure-weights-cache` first, which calls
`scripts/download_weights_from_hf.py` to snapshot
[`rhymeswithlion/magenta-realtime-mlx-cpp`](https://huggingface.co/datasets/rhymeswithlion/magenta-realtime-mlx-cpp)
into `.weights-cache/`. If the bundle is already complete that step is
a no-op. The dataset may be private during initial release;
authenticate with `huggingface-cli login` (or `HF_TOKEN`) before the
first download. Override the source dataset with `HF_WEIGHTS_REPO=...`
for forks or mirrors.

To reuse weights from another checkout, copy its `.weights-cache/`
directory into this repo, or set `MAGENTA_RT_WEIGHTS_DIR` to that
location.

## Reference Path (`make ref-stream`)

`ref-stream` is the upstream JAX/TF MagentaRT pipeline, run through a
dedicated Python venv at `reference/.venv/` so its TensorFlow / JAX /
T5X dependencies never touch the main interpreter. It exists for
side-by-side comparison only -- it is much slower than the C++ MLX
path and may fall behind wall clock on a laptop. The runner UX matches
`mlx-stream`: warm up, start playback, generate until Ctrl+C.

`make ref-stream` provisions `reference/.venv/` automatically on first
run via `reference/setup_reference_venv.sh`.

## C++ Build (`make build`)

The C++ sources live at the top level (`src/`, `cli/`, `tests/`). A
direct invocation:

```bash
brew install cmake ninja catch2 sentencepiece portaudio
make sync             # so MLX's libmlx.dylib + headers exist under .venv/
make build            # configure + compile into build/
make test-cpp         # Catch2 unit tests
make mlx-stream MLX_FLAGS="--dry-run --max-chunks 4"
make devices-mlx      # list PortAudio devices
```

CMake auto-discovers MLX from `.venv/lib/python3.*/site-packages/mlx/`.
Override with `-DMLX_ROOT=/path/to/mlx-install` for a from-source MLX.

The binary takes the flag surface documented by `mlx-stream --help`:
`--prompt`, `--seed`, `--temperature`, `--top-k`, `--guidance-weight`,
`--dtype {fp32|fp16|bf16}`, `--tag`, `--warmup-chunks`, `--max-chunks`,
`--device`, `--list-devices`, `--dry-run`, `--record <path>`, plus
operational knobs (`--prebuffer-chunks`, `--max-queue-chunks`,
`--no-mlxfn`).

## Layout

- `Makefile` -- the main entrypoint
- `CMakeLists.txt` -- top-level C++ build (no nested `cpp/` dir)
- `src/magenta_realtime_mlx/` -- core C++ library
- `cli/` -- `mlx-stream` binary entry point
- `tests/` -- mixed: Catch2 C++ tests (`*.cpp`) and pytest suites for
  the reference-stream helpers (`*.py`)
- `scripts/` -- HF download helper + reference-stream launcher (Python)
- `reference/` -- isolated reference-Python environment + subprocess
  runner code for `ref-stream`
- `vendor/magenta-realtime/` -- upstream git submodule used by the
  reference path

## License

The code in this repository is released under the
[Apache License 2.0](LICENSE), matching the upstream
[Magenta RealTime](https://github.com/magenta/magenta-realtime) codebase.
See [`NOTICE`](NOTICE) for upstream attribution.

The MLX-ready weights distributed on Hugging Face are licensed under
[Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode),
matching the upstream model card. Their use is also subject to Google's
Magenta RealTime Terms of Use, reproduced verbatim on the
[Hugging Face dataset card](https://huggingface.co/datasets/rhymeswithlion/magenta-realtime-mlx-cpp).

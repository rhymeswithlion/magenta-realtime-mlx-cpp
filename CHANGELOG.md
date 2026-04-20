# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-04-20

Initial public release.

### Added

- C++ MLX implementation of Magenta RealTime for Apple Silicon (`mlx-stream`),
  sustaining >1.0× real-time generation on M3 Pro and faster.
- `.mlxfn` source-graph bundles for the SpectroStream encoder/decoder and the
  Depthformer temporal/depth steps, loaded via `mlx::core::import_function`;
  falls back to a capturing-lambda compile path when bundles are absent.
- `bf16` Depthformer compute path selectable via `--dtype {fp32|fp16|bf16}`
  (codec stays at `fp32` for spectral quality).
- Cross-chunk encoder pipelining and hot-path scalar / weight-view caching for
  per-chunk latency wins on the inner depth loop.
- Speculative depth decoding driver in `generate_tokens` for warm chunks.
- PortAudio playback path with prebuffering, device selection
  (`--list-devices`, `--device`), and WAV recording (`--record`).
- `scripts/download_weights_from_hf.py` to fetch the published MLX weight
  bundle from Hugging Face into `.weights-cache/`.
- `scripts/run_reference_stream.py` and `reference/` venv bootstrap for an
  isolated upstream JAX/TF baseline (`make ref-stream`) suitable for A/B audio
  comparison against `mlx-stream`.
- Catch2 unit tests for dtype handling and pytest coverage for the
  reference-stream subprocess protocol.
- Apache 2.0 licensing with `NOTICE` attribution covering both the upstream
  Magenta RealTime code and the redistributed CC-BY 4.0 weight bundle.

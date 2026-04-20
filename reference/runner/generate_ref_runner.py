#!/usr/bin/env python3

# Copyright 2026 Brian Cruz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full JAX/TF reference generation — run **only** via the reference Python executable.

Uses a private temp directory and sibling ``ref_*.py`` modules shipped in this
directory so the reference subprocess stays self-contained.

CLI (preferred)::

    python generate_ref_runner.py PROMPT --chunks N --output out.wav [...]

Environment::

    MAGENTA_RT_REF_KEEP_TMP=1 — keep the working temp directory (printed to stdout).
    MAGENTA_RT_REF_STREAM_DIR=path — after each post-warmup chunk, save float32
    stereo playback to ``playback_0000.npy``, … and emit a ``MAGENTA_RT_STREAM_V1``
    line on stdout for the parent driver (see ``scripts/run_reference_stream.py`` in the MLX tree).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from time import perf_counter

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))
if RUNNER_DIR not in sys.path:
    sys.path.insert(0, RUNNER_DIR)

PYTHON = sys.executable

STREAM_PREFIX_V1 = "MAGENTA_RT_STREAM_V1"


def run_step(script: str, args: list[str], label: str) -> None:
    cmd = [PYTHON, os.path.join(RUNNER_DIR, script), *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"FAILED ({label}): exit code {result.returncode}")
        print(result.stderr[-1000:] if result.stderr else "(no stderr)")
        sys.exit(1)
    for line in result.stdout.strip().split("\n"):
        print(line)


def _apply_jax_patches() -> None:
    import jax._src.core as _jcore
    import jax.core

    for _name in dir(_jcore):
        if not _name.startswith("_") and not hasattr(jax.core, _name):
            with contextlib.suppress(Exception):
                setattr(jax.core, _name, getattr(_jcore, _name))
    import jax.experimental.pjit as _pjit_mod
    from jax.sharding import PartitionSpec

    _pjit_mod.PartitionSpec = PartitionSpec


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("prompt", type=str)
    p.add_argument(
        "--chunks",
        type=int,
        default=3,
        help="Total RVQ/audio chunks to generate; 0 means run until terminated by the parent",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--topk", type=int, default=40)
    p.add_argument("--guidance-weight", type=float, default=5.0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--warmup-chunks",
        type=int,
        default=0,
        help="First N chunks: full pipeline but no stream export (parent playback warmup)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    num_chunks = args.chunks
    unbounded = num_chunks == 0
    if num_chunks < 0:
        raise SystemExit("--chunks must be >= 0")
    warmup_chunks = max(0, args.warmup_chunks)
    if unbounded and os.environ.get("MAGENTA_RT_REF_STREAM_DIR", "").strip() == "":
        raise SystemExit("--chunks 0 requires MAGENTA_RT_REF_STREAM_DIR for parent-driven streaming")
    if not unbounded and warmup_chunks > num_chunks:
        raise SystemExit("--warmup-chunks cannot exceed --chunks")

    stream_dir_raw = os.environ.get("MAGENTA_RT_REF_STREAM_DIR", "").strip()
    stream_dir = Path(stream_dir_raw).resolve() if stream_dir_raw else None
    if stream_dir is not None:
        stream_dir.mkdir(parents=True, exist_ok=True)

    prompt = args.prompt
    seed = args.seed
    temperature = args.temperature
    topk = args.topk
    guidance_weight = args.guidance_weight
    output_path = args.output

    tmp_dir = tempfile.mkdtemp(prefix="magenta_rt_publish_ref_")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    chunk_desc = "unbounded" if unbounded else str(num_chunks)
    print(f"Prompt: {prompt!r}, chunks={chunk_desc}, seed={seed}")
    print(f"temp={temperature}, topk={topk}, guidance={guidance_weight}")
    if warmup_chunks:
        print(f"warmup_chunks={warmup_chunks} (no stream export for these)")
    print(f"ref_workdir: {tmp_dir}")

    t0 = time.time()
    run_step("ref_tokenize.py", [prompt, tmp_dir], "tokenize")
    print(f"Tokenize: {time.time() - t0:.1f}s")

    t0 = time.time()
    run_step("ref_tf_ops.py", ["embed", tmp_dir], "embed")
    print(f"Embed: {time.time() - t0:.1f}s")

    style_tokens = np.load(os.path.join(tmp_dir, "style_tokens.npy"))

    _apply_jax_patches()

    import jax
    from magenta_rt import asset, utils
    from magenta_rt.depthformer import model
    from magenta_rt.system import MagentaRTConfiguration

    from ref_tf_decode_lib import decode_tokens_to_waveform, load_ssv2_decoder_bundle

    config = MagentaRTConfiguration()
    print(f"JAX {jax.__version__} on {jax.devices()}")
    print("Loading LLM...", flush=True)

    checkpoint_dir = asset.fetch(
        "checkpoints/llm_base_x4286_c1860k.tar", is_dir=True, extract_archive=True
    )
    task_feature_lengths, partitioner, interactive_model = model.load_pretrained_model(
        checkpoint_dir=checkpoint_dir,
        size="base",
        batch_size=2,
        num_partitions=1,
        model_parallel_submesh=None,
    )
    infer_fn = model.get_infer_fn(
        interactive_model=interactive_model,
        partitioner=partitioner,
        batch_size=2,
        task_feature_lengths=task_feature_lengths,
        default_guidance_weight=guidance_weight,
        default_temperature=temperature,
        default_topk=topk,
    )
    print("LLM ready", flush=True)

    print("Loading TF decoder / quantizer (SpectroStream)...", flush=True)
    t_tf = time.time()
    decode_bundle = load_ssv2_decoder_bundle()
    print(f"TF decode ready in {time.time() - t_tf:.1f}s", flush=True)

    from magenta_rt import audio as mrt_audio

    context_tokens = np.full(config.context_tokens_shape, -1, dtype=np.int32)
    t_total = perf_counter()
    playable_chunks: list[np.ndarray] = []
    prev_xfade = np.zeros(
        (config.crossfade_length_samples, config.codec_num_channels), dtype=np.float32
    )
    xfade_ramp = mrt_audio.crossfade_ramp(config.crossfade_length_samples, style="eqpower")[
        :, np.newaxis
    ]
    stream_seq = 0

    i = 0
    while unbounded or i < num_chunks:
        t_chunk = perf_counter()

        codec_tokens_lm = np.where(
            context_tokens >= 0,
            utils.rvq_to_llm(
                np.maximum(context_tokens, 0),
                config.codec_rvq_codebook_size,
                config.vocab_codec_offset,
            ),
            np.full_like(context_tokens, config.vocab_mask_token),
        )
        style_tokens_lm = utils.rvq_to_llm(
            style_tokens[: config.encoder_style_rvq_depth],
            config.style_rvq_codebook_size,
            config.vocab_style_offset,
        )
        enc_pos = np.concatenate(
            [
                codec_tokens_lm[:, : config.encoder_codec_rvq_depth].reshape(-1),
                style_tokens_lm,
            ],
            axis=0,
        )
        enc_neg = enc_pos.copy()
        enc_neg[-config.encoder_style_rvq_depth :] = config.vocab_mask_token
        encoder_inputs = np.stack([enc_pos, enc_neg], axis=0)

        max_frames = config.chunk_length_frames
        generated_tokens, _ = infer_fn(
            {
                "encoder_input_tokens": encoder_inputs,
                "decoder_input_tokens": np.zeros(
                    (2, max_frames * config.decoder_codec_rvq_depth), dtype=np.int32
                ),
            },
            {
                "max_decode_steps": np.array(
                    max_frames * config.decoder_codec_rvq_depth, dtype=np.int32
                ),
                "guidance_weight": guidance_weight,
                "temperature": temperature,
                "topk": topk,
            },
            jax.random.PRNGKey(seed + i),
        )

        generated_tokens = np.array(generated_tokens)[:1].reshape(
            max_frames, config.decoder_codec_rvq_depth
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gen_rvq = utils.llm_to_rvq(
                generated_tokens,
                config.codec_rvq_codebook_size,
                config.vocab_codec_offset,
                safe=False,
            )

        xfade = context_tokens[-config.crossfade_length_frames :]
        if i == 0:
            xfade = np.zeros_like(xfade)
        xfade = np.maximum(xfade, 0)
        chunk_with_xfade = np.concatenate([xfade, gen_rvq], axis=0)
        np.save(os.path.join(tmp_dir, f"chunk_{i:02d}.npy"), chunk_with_xfade)

        context_tokens = np.concatenate([context_tokens[gen_rvq.shape[0] :], gen_rvq], axis=0)

        samples = decode_tokens_to_waveform(decode_bundle, chunk_with_xfade)
        xfade_out = samples[-config.crossfade_length_samples :]
        chunk = samples[: -config.crossfade_length_samples].copy()
        chunk[: config.crossfade_length_samples] *= xfade_ramp
        chunk[: config.crossfade_length_samples] += prev_xfade * np.flip(xfade_ramp, axis=0)
        prev_xfade = xfade_out
        if not unbounded:
            playable_chunks.append(chunk)

        wall = perf_counter() - t_chunk
        audio_seconds = float(chunk.shape[0]) / 48000.0
        rtf = audio_seconds / wall if wall > 0 else 0.0
        unique = len(np.unique(gen_rvq))
        chunk_label = f"{i + 1}" if unbounded else f"{i + 1}/{num_chunks}"
        print(f"chunk {chunk_label}: wall {wall * 1000:.0f} ms RTF {rtf:.2f}x unique={unique}", flush=True)

        if stream_dir is not None and i >= warmup_chunks:
            out_npy = stream_dir / f"playback_{stream_seq:04d}.npy"
            np.save(out_npy, chunk.astype(np.float32, copy=False))
            print(
                f"{STREAM_PREFIX_V1}\t{i + 1}\t{audio_seconds:.6f}\t{wall:.6f}\t{out_npy}\n",
                end="",
                flush=True,
            )
            stream_seq += 1
        i += 1

    print(f"Total generate+decode: {perf_counter() - t_total:.1f}s")

    if unbounded:
        return

    combined = np.concatenate(playable_chunks, axis=0)
    total_rms = float(np.sqrt(np.mean(combined**2)))
    total_peak = float(np.abs(combined).max())
    duration = combined.shape[0] / 48000
    print(f"Audio: {duration:.2f}s, RMS={total_rms:.4f}, peak={total_peak:.4f}")

    import soundfile as sf

    sf.write(str(output_path), combined, 48000)
    print(f"Saved: {output_path}")

    if os.environ.get("MAGENTA_RT_REF_KEEP_TMP", "").strip() not in ("1", "true", "yes"):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f"Keeping ref_workdir (MAGENTA_RT_REF_KEEP_TMP): {tmp_dir}")


if __name__ == "__main__":
    main()

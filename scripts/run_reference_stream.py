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

"""Run the **upstream reference** pipeline (JAX/TF) with MLX-like defaults.

This script does **not** import ``magenta_rt`` in the current interpreter. It
always delegates to a **separate** Python executable where TensorFlow, JAX, and
``magenta_rt`` are installed (typically a dedicated venv).

Run from the **repository root** (the directory that contains ``src/`` and
``scripts/``), for example::

    cd /path/to/magenta-realtime-mlx-cpp
    uv run python scripts/run_reference_stream.py --ref-python … --prompt "…"

By default this mirrors ``run_mlx_stream.py``: warm up, start playback, and keep
running until interrupted. Use ``--dry-run --max-chunks N`` for a bounded smoke
test, or ``--batch`` for the old one-shot subprocess path.

Default recorded WAV path is ``output/ref_baseline.wav`` under the repository
root; override with ``--output`` or ``--record``.

The reference subprocess is CPU-bound and usually **cannot** keep up with wall
clock; queue growth illustrates why the MLX path exists. Chunk RTF lines use the
child-reported wall time for each chunk (LLM + TF decode + crossfade).

Configuration::

    MAGENTA_RT_REF_PYTHON=/path/to/venv/bin/python

The modules under ``reference/runner/`` implement the subprocess reference
pipeline (tokenize → TF embed → JAX LLM → TF decode → WAV); they are executed
only by ``MAGENTA_RT_REF_PYTHON`` (``reference/.venv`` after ``reference/setup_reference_venv.sh``).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _SCRIPT_DIR.parent
_SRC = _PACKAGE_ROOT / "src"
if _SRC.is_dir():
    s = str(_SRC)
    if s not in sys.path:
        sys.path.insert(0, s)
_scripts = str(_SCRIPT_DIR)
if _scripts not in sys.path:
    sys.path.append(_scripts)

from realtime_playback import playback_worker, resolve_sd_device  # noqa: E402

_RUNNER_DIR = _PACKAGE_ROOT / "reference" / "runner"
_RUNNER = _RUNNER_DIR / "generate_ref_runner.py"

STREAM_PREFIX_V1 = "MAGENTA_RT_STREAM_V1"


def _parse_stream_v1_line(line: str) -> tuple[int, float, float, Path] | None:
    s = line.rstrip("\n\r")
    if not s.startswith(STREAM_PREFIX_V1):
        return None
    parts = s.split("\t")
    if len(parts) != 5:
        return None
    return int(parts[1]), float(parts[2]), float(parts[3]), Path(parts[4])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--ref-python", type=str, default=None, help="Reference venv python (overrides env)"
    )
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument(
        "--chunks",
        type=int,
        default=None,
        help="Legacy finite chunk count. In streaming mode this is an alias for "
        "--max-chunks; in --batch mode it is the total chunk count (default 3).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--topk", type=int, default=40)
    p.add_argument("--guidance-weight", type=float, default=5.0)
    p.add_argument(
        "--output",
        type=Path,
        default=_PACKAGE_ROOT / "output" / "ref_baseline.wav",
        help="Output WAV path (default: <repo>/output/ref_baseline.wav)",
    )
    p.add_argument(
        "--ref-timeout",
        type=int,
        default=3600,
        metavar="SEC",
        help="Subprocess wall-clock limit in seconds (default: 3600)",
    )
    p.add_argument(
        "--stream-playback",
        action="store_true",
        help="Deprecated compatibility flag; playback is now the default behavior",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Run one blocking reference subprocess and exit instead of live playback",
    )
    p.add_argument(
        "--warmup-chunks",
        type=int,
        default=1,
        help="Reference subprocess warmup iterations (stream mode only; default 1). "
        "Ignored when not using --stream-playback.",
    )
    p.add_argument(
        "--queue-size", type=int, default=3, help="Max audio chunks buffered for playback"
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="PortAudio device name substring or index (sounddevice); default output",
    )
    p.add_argument(
        "--list-devices", action="store_true", help="Print sounddevice devices and exit"
    )
    p.add_argument(
        "--record",
        type=Path,
        default=None,
        help="Write concatenated played audio to this WAV path on exit",
    )
    p.add_argument(
        "--no-record",
        action="store_true",
        help="Do not write a WAV capture on exit (overrides --record and --output)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="No playback; run reference subprocess only (stream mode: no stream dir)",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Stop stream subprocess after this many post-warmup chunks (default: same as --chunks)",
    )
    return p.parse_args()


def _stream_chunk_budget(args: argparse.Namespace) -> int | None:
    if args.max_chunks is not None:
        return max(1, args.max_chunks)
    if args.chunks is not None and not args.batch:
        return max(1, args.chunks)
    return None


def _stream_total_chunks(args: argparse.Namespace) -> int:
    budget = _stream_chunk_budget(args)
    if budget is None:
        return 0
    return max(0, args.warmup_chunks) + budget


def _reference_record_path(args: argparse.Namespace) -> Path | None:
    if args.list_devices or args.dry_run or args.batch:
        return None
    if args.no_record:
        return None
    if args.record is not None:
        return args.record
    return args.output


def _reference_cmd(
    *,
    ref_py: Path,
    prompt: str,
    total_chunks: int,
    seed: int,
    temperature: float,
    topk: int,
    guidance_weight: float,
    output: Path,
    warmup_chunks: int,
) -> list[str]:
    return [
        str(ref_py),
        "-u",
        str(_RUNNER),
        prompt,
        "--chunks",
        str(total_chunks),
        "--seed",
        str(seed),
        "--temperature",
        str(temperature),
        "--topk",
        str(topk),
        "--guidance-weight",
        str(guidance_weight),
        "--output",
        str(output),
        "--warmup-chunks",
        str(warmup_chunks),
    ]


def main() -> None:
    args = _parse_args()
    record_path = _reference_record_path(args)
    if not _RUNNER.is_file():
        raise SystemExit(f"Missing reference runner at {_RUNNER}")

    # Local import (the ``magenta_realtime_mlx`` Python package isn't
    # shipped in this distribution; the resolver lives in ``scripts/``).
    import importlib.util as _ilu
    _ref_spec = _ilu.spec_from_file_location(
        "_ref_python", Path(__file__).resolve().parent / "_ref_python.py"
    )
    assert _ref_spec is not None and _ref_spec.loader is not None
    _ref_mod = _ilu.module_from_spec(_ref_spec)
    _ref_spec.loader.exec_module(_ref_mod)
    resolve_reference_python = _ref_mod.resolve_reference_python

    if args.list_devices:
        try:
            import sounddevice as sd
        except ImportError as e:
            raise SystemExit(
                "sounddevice is required. Re-run: make reference-venv"
            ) from e
        print(sd.query_devices())
        return

    try:
        ref_py = resolve_reference_python(cli_path=args.ref_python)
    except Exception as e:
        raise SystemExit(str(e)) from e

    if not ref_py.is_file():
        raise SystemExit(
            f"Reference Python path is not a file: {ref_py}\n"
            "Install magenta_rt + TF + JAX in that environment, or fix --ref-python / "
            "MAGENTA_RT_REF_PYTHON."
        )

    if not args.batch:
        if args.dry_run:
            total = _stream_total_chunks(args)
            warmup = max(0, args.warmup_chunks)
            if total <= 0:
                raise SystemExit("--dry-run requires --max-chunks N (or legacy --chunks N)")
            cmd = _reference_cmd(
                ref_py=ref_py,
                prompt=args.prompt,
                total_chunks=total,
                seed=args.seed,
                temperature=args.temperature,
                topk=args.topk,
                guidance_weight=args.guidance_weight,
                output=args.output,
                warmup_chunks=warmup,
            )
            print(f"Dry-run: reference subprocess, total_chunks={total} warmup={warmup}")
            print(f"Running: {' '.join(cmd[:4])} ...")
            proc = subprocess.run(cmd, text=True, timeout=args.ref_timeout)
            raise SystemExit(proc.returncode)

        try:
            import sounddevice as sd
        except ImportError as e:
            raise SystemExit(
                "sounddevice is required for playback. Re-run: make reference-venv"
            ) from e

        play_n = _stream_chunk_budget(args)
        warmup = max(0, args.warmup_chunks)
        total = _stream_total_chunks(args)

        stream_dir = Path(tempfile.mkdtemp(prefix="magenta_rt_ref_stream_"))
        env = {**os.environ, "MAGENTA_RT_REF_STREAM_DIR": str(stream_dir)}
        cmd = _reference_cmd(
            ref_py=ref_py,
            prompt=args.prompt,
            total_chunks=total,
            seed=args.seed,
            temperature=args.temperature,
            topk=args.topk,
            guidance_weight=args.guidance_weight,
            output=args.output,
            warmup_chunks=warmup,
        )

        play_desc = "unbounded" if play_n is None else str(play_n)
        total_desc = "unbounded" if total == 0 else str(total)
        print(f"Streaming reference: warmup={warmup}, play_chunks={play_desc}, total={total_desc}")
        print(f"Prompt: {args.prompt!r}")
        device = resolve_sd_device(sd, args.device)
        audio_q: queue.Queue = queue.Queue(maxsize=max(1, args.queue_size))
        recorded: list[np.ndarray] = []
        stop_event = threading.Event()
        buffer_lock = threading.Lock()
        queued_audio_seconds: list[float] = [0.0]

        player = threading.Thread(
            target=playback_worker,
            args=(audio_q, device, recorded, stop_event, buffer_lock, queued_audio_seconds),
            name="magenta-realtime-ref-playback",
            daemon=True,
        )
        player.start()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        interrupted = False
        rc = 0
        try:
            try:
                for line in proc.stdout:
                    parsed = _parse_stream_v1_line(line)
                    if parsed is None:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        continue
                    chunk_idx, audio_seconds, wall, path = parsed
                    samples = np.load(path)
                    samples = np.ascontiguousarray(samples, dtype=np.float32)
                    audio_q.put((samples, 48000))
                    rtf = audio_seconds / wall if wall > 0 else 0.0
                    with buffer_lock:
                        queued_audio_seconds[0] += audio_seconds
                        buf_s = queued_audio_seconds[0]
                    print(
                        f"  chunk {chunk_idx}: wall {wall * 1000:.0f} ms "
                        f"RTF {rtf:.2f}x queue~{audio_q.qsize()} queued~{buf_s:.2f}s"
                    )
            except KeyboardInterrupt:
                interrupted = True
                print("\nStopping…", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=30)
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10)
            rc = int(proc.returncode or 0)
            stop_event.set()
            with contextlib.suppress(queue.Full, KeyboardInterrupt):
                audio_q.put(None, timeout=30.0)
            player.join(timeout=5.0)
            shutil.rmtree(stream_dir, ignore_errors=True)

        if not interrupted and rc != 0:
            raise SystemExit(rc)

        if record_path and recorded:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            import soundfile as sf

            combo = np.concatenate(recorded, axis=0)
            sf.write(str(record_path), combo, 48000, subtype="PCM_16")
            print(f"Wrote {record_path} ({combo.shape[0] / 48000:.1f} s)")
        return

    # Batch (no stream): one blocking subprocess, no warmup unless user passes via future flag
    cmd = _reference_cmd(
        ref_py=ref_py,
        prompt=args.prompt,
        total_chunks=max(1, args.chunks or 3),
        seed=args.seed,
        temperature=args.temperature,
        topk=args.topk,
        guidance_weight=args.guidance_weight,
        output=args.output,
        warmup_chunks=0,
    )
    print(f"Running reference subprocess: {ref_py} (timeout {args.ref_timeout}s)")
    print(" ".join(cmd[:5]), "...")
    try:
        proc = subprocess.run(cmd, text=True, timeout=args.ref_timeout)
    except subprocess.TimeoutExpired:
        raise SystemExit(
            f"Reference subprocess exceeded --ref-timeout ({args.ref_timeout}s). "
            "Check the reference venv or raise the limit if a long first-run download is expected."
        ) from None
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()

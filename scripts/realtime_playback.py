"""PortAudio playback helpers shared by MLX and reference stream CLIs."""

from __future__ import annotations

import queue
import threading

import numpy as np


def resolve_sd_device(sd, device: str | None) -> int | str | None:
    if device is None:
        return None
    if device.isdigit():
        return int(device)
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if device.lower() in str(d["name"]).lower():
            return i
    raise SystemExit(f"No device matching {device!r}; use --list-devices")


def playback_worker(
    audio_q: queue.Queue,
    device: int | str | None,
    recorded: list[np.ndarray],
    stop_event: threading.Event,
    buffer_lock: threading.Lock,
    queued_audio_seconds: list[float],
) -> None:
    """Play chunks through one continuous ``OutputStream`` (same semantics as MLX demo)."""
    import sounddevice as sd

    stream: sd.OutputStream | None = None
    open_sr: int | None = None
    open_ch: int | None = None
    try:
        while not stop_event.is_set():
            try:
                item = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            samples, sample_rate = item
            chunk_dur_s = float(samples.shape[0]) / float(sample_rate)
            with buffer_lock:
                queued_audio_seconds[0] -= chunk_dur_s
            recorded.append(samples.copy())
            ch = int(samples.shape[1])
            sr_i = int(sample_rate)
            if stream is None or open_sr != sr_i or open_ch != ch:
                if stream is not None:
                    stream.stop()
                    stream.close()
                stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=ch,
                    dtype=np.float32,
                    device=device,
                )
                stream.start()
                open_sr = sr_i
                open_ch = ch
            stream.write(samples)
    finally:
        if stream is not None:
            stream.stop()
            stream.close()

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_reference_stream", _ROOT / "scripts" / "run_reference_stream.py"
)
assert _SPEC and _SPEC.loader
_rrs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_rrs)
_parse_stream_v1_line = _rrs._parse_stream_v1_line


def test_parse_stream_v1_roundtrip() -> None:
    p = Path("/tmp/playback_0000.npy")
    line = f"MAGENTA_RT_STREAM_V1\t3\t2.000000\t11.500000\t{p}\n"
    got = _parse_stream_v1_line(line)
    assert got is not None
    idx, audio_s, wall, path = got
    assert idx == 3
    assert audio_s == pytest.approx(2.0)
    assert wall == pytest.approx(11.5)
    assert path == p


def test_parse_stream_v1_rejects_garbage() -> None:
    assert _parse_stream_v1_line("chunk 1/3\n") is None
    assert _parse_stream_v1_line("MAGENTA_RT_STREAM_V1\ta\n") is None


def test_reference_stream_defaults_to_unbounded_playback(tmp_path: Path) -> None:
    assert hasattr(_rrs, "_stream_chunk_budget")
    assert hasattr(_rrs, "_stream_total_chunks")
    assert hasattr(_rrs, "_reference_record_path")
    args = SimpleNamespace(
        max_chunks=None,
        chunks=None,
        warmup_chunks=1,
        dry_run=False,
        list_devices=False,
        no_record=False,
        record=None,
        output=tmp_path / "ref.wav",
        batch=False,
    )
    assert _rrs._stream_chunk_budget(args) is None
    assert _rrs._stream_total_chunks(args) == 0
    assert _rrs._reference_record_path(args) == tmp_path / "ref.wav"


def test_reference_stream_dry_run_stays_bounded(tmp_path: Path) -> None:
    assert hasattr(_rrs, "_stream_chunk_budget")
    assert hasattr(_rrs, "_stream_total_chunks")
    assert hasattr(_rrs, "_reference_record_path")
    args = SimpleNamespace(
        max_chunks=2,
        chunks=None,
        warmup_chunks=1,
        dry_run=True,
        list_devices=False,
        no_record=False,
        record=None,
        output=tmp_path / "ref.wav",
        batch=False,
    )
    assert _rrs._stream_chunk_budget(args) == 2
    assert _rrs._stream_total_chunks(args) == 3
    assert _rrs._reference_record_path(args) is None

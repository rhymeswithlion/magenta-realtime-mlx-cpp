"""Resolve the reference (upstream ``magenta_rt`` + JAX/TF) Python executable."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


class ReferencePythonNotConfiguredError(RuntimeError):
    """Raised when no reference interpreter was provided or discovered."""


def resolve_reference_python(
    *, cli_path: str | None, environ: dict[str, str] | None = None
) -> Path:
    """Return absolute path to the reference venv's ``python``.

    Precedence: ``cli_path`` (e.g. ``--ref-python``), then ``MAGENTA_RT_REF_PYTHON``,
    then no implicit default — callers must handle absence before calling this
    by catching ``ReferencePythonNotConfiguredError``.
    """
    env = environ if environ is not None else os.environ
    choice = (cli_path or "").strip() or env.get("MAGENTA_RT_REF_PYTHON", "").strip()
    if not choice:
        raise ReferencePythonNotConfiguredError(
            "Reference Python is not configured. Pass --ref-python PATH or set "
            "MAGENTA_RT_REF_PYTHON to a Python that has magenta_rt, JAX, and TensorFlow "
            "(see this package's README / docs)."
        )
    p = Path(choice).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    # Use ``absolute()`` not ``resolve()`` so a venv ``python`` symlink is kept
    # intact; resolving would follow the symlink to the base interpreter and drop
    # the venv's site-packages.
    p = p.absolute()
    return p


def reference_python_status(path: Path) -> dict[str, str | bool]:
    """Lightweight existence check (does not import ``magenta_rt``)."""
    return {
        "path": str(path),
        "exists": path.is_file(),
        "is_file": path.is_file(),
    }


def current_mlx_invocation_summary() -> dict[str, str]:
    """Provenance for the MLX side (current process)."""
    return {
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }

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

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# ``ref_python`` lives in ``scripts/_ref_python.py`` (the
# ``magenta_realtime_mlx`` Python package isn't shipped in this
# distribution). Load it via importlib so we don't have to touch
# ``sys.path`` or pollute ``conftest.py``.
_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_ref_python", _ROOT / "scripts" / "_ref_python.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_ref_python = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ref_python
_SPEC.loader.exec_module(_ref_python)
ReferencePythonNotConfiguredError = _ref_python.ReferencePythonNotConfiguredError
resolve_reference_python = _ref_python.resolve_reference_python


def test_resolve_prefers_cli_over_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = tmp_path / "a_python"
    b = tmp_path / "b_python"
    a.write_text("")
    b.write_text("")
    a.chmod(0o644)
    b.chmod(0o644)
    monkeypatch.setenv("MAGENTA_RT_REF_PYTHON", str(a))
    got = resolve_reference_python(cli_path=str(b), environ=os.environ)
    assert got == b.resolve()


def test_resolve_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "venv" / "bin" / "python"
    p.parent.mkdir(parents=True)
    p.write_text("")
    monkeypatch.setenv("MAGENTA_RT_REF_PYTHON", str(p))
    got = resolve_reference_python(cli_path=None, environ=os.environ)
    assert got == p.resolve()


def test_resolve_relative_path_is_relative_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    rel = Path("rpython")
    rel.write_text("")
    got = resolve_reference_python(cli_path="rpython", environ={})
    assert got == (tmp_path / "rpython").resolve()


def test_not_configured_raises() -> None:
    with pytest.raises(ReferencePythonNotConfiguredError) as excinfo:
        resolve_reference_python(cli_path=None, environ={})
    assert "this package's README" in str(excinfo.value)

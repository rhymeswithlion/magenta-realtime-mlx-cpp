#!/usr/bin/env bash

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

# Install patched google-research/t5x for reference JAX generation
# (reference/runner/generate_ref_runner.py). Same flow as upstream Magenta RT
# local install: clone t5x at a pinned commit, patch, install --no-deps, patch
# seqio / partitioning for CPU.
#
# Usage (from repository root):
#   bash reference/install_reference_t5x.sh [path/to/venv]
# Default venv: reference/.venv (see reference/setup_reference_venv.sh)
#
# Env:
#   MAGENTA_RT_SKIP_T5X=1  — exit 0 without doing anything
#   MAGENTA_RT_T5X_DIR     — clone directory (default: .t5x-build/t5x)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$ROOT/reference/.venv}"
PY="$VENV/bin/python"
PATCH_DIR="$ROOT/vendor/magenta-realtime/patch"
T5X_PARENT="${MAGENTA_RT_T5X_BUILD_DIR:-$ROOT/.t5x-build}"
T5X_ROOT="${MAGENTA_RT_T5X_DIR:-$T5X_PARENT/t5x}"
T5X_COMMIT="7781d167ab421dae96281860c09d5bd785983853"
STAMP="$VENV/.t5x_reference_stack"
STAMP_LINE="t5x_commit=${T5X_COMMIT}"

if [[ "${MAGENTA_RT_SKIP_T5X:-}" == "1" ]]; then
  echo "Skipping t5x install (MAGENTA_RT_SKIP_T5X=1)"
  exit 0
fi

if [[ ! -x "$PY" ]]; then
  echo "error: no python at $PY — run reference/setup_reference_venv.sh first"
  exit 1
fi

# ``uv sync`` can remove t5x from the venv without deleting the stamp; only skip
# when the stamp matches *and* the package tree is still present.
T5X_SITE="$(find "$VENV/lib" -type d -path '*/site-packages/t5x' 2>/dev/null | head -n 1)"
if [[ -f "$STAMP" ]] && grep -qF "$STAMP_LINE" "$STAMP"; then
  if [[ -n "$T5X_SITE" && -f "$T5X_SITE/partitioning.py" ]]; then
    echo "t5x reference stack already installed ($STAMP)."
    exit 0
  fi
  echo "Stamp present but t5x missing from venv (e.g. after uv sync). Reinstalling..."
  rm -f "$STAMP"
fi

if [[ ! -f "$PATCH_DIR/t5x_setup.py.patch" ]]; then
  echo "error: missing $PATCH_DIR/t5x_setup.py.patch (init submodule?)"
  exit 1
fi

# Use ``python -m pip`` (uv-created venvs may not ship a ``pip`` entrypoint).
pip_install() {
  "$PY" -m pip install "$@"
}

echo "Cloning / updating t5x at $T5X_ROOT (commit ${T5X_COMMIT:0:7})..."
mkdir -p "$T5X_PARENT"
if [[ ! -d "$T5X_ROOT/.git" ]]; then
  git clone https://github.com/google-research/t5x.git "$T5X_ROOT"
fi
(
  cd "$T5X_ROOT"
  git fetch --depth 1 origin "$T5X_COMMIT"
  git checkout -f "$T5X_COMMIT"
  git checkout -f -- setup.py
  patch -p0 <"$PATCH_DIR/t5x_setup.py.patch"
  if grep -q "'tensorflow-cpu'" setup.py; then
    sed "/'tensorflow-cpu',/d" setup.py >setup.py.tmp && mv setup.py.tmp setup.py
  fi
)

echo "Installing t5x (no-deps) + runtime packages..."
pip_install --no-deps "$T5X_ROOT"
pip_install "git+https://github.com/google/airio" \
  "git+https://github.com/google-research/jestimator" \
  "fiddle>=0.2.5" \
  "cached-property" \
  "huggingface_hub>=0.20"

# Avoid ``import t5x`` here: some macOS builds abort in native code during import
# even though the package is installed; we only need the install tree path.
T5X_SITE="$(find "$VENV/lib" -type d -path '*/site-packages/t5x' 2>/dev/null | head -n 1)"
if [[ -z "$T5X_SITE" || ! -f "$T5X_SITE/partitioning.py" ]]; then
  echo "error: could not locate t5x under $VENV/lib (site-packages/t5x/partitioning.py)"
  exit 1
fi
echo "Patching $T5X_SITE/partitioning.py (CPU mesh)..."
patch -N "$T5X_SITE/partitioning.py" <"$PATCH_DIR/t5x_partitioning.py.patch" || {
  if grep -q "if platform == 'cpu':" "$T5X_SITE/partitioning.py"; then
    echo "  (partitioning.py already patched)"
  else
    echo "error: failed to patch t5x partitioning.py"
    exit 1
  fi
}

SEQIO_VOC="$(find "$VENV/lib" -type f -path '*/site-packages/seqio/vocabularies.py' 2>/dev/null | head -n 1)"
if [[ -z "$SEQIO_VOC" || ! -f "$SEQIO_VOC" ]]; then
  echo "error: could not locate seqio/vocabularies.py under $VENV/lib"
  exit 1
fi
if grep -q "^import tensorflow_text" "$SEQIO_VOC"; then
  echo "Patching seqio vocabularies (drop tensorflow_text import)..."
  patch -N "$SEQIO_VOC" <"$PATCH_DIR/seqio_vocabularies.py.patch" || {
    if grep -q "^import tensorflow_text" "$SEQIO_VOC"; then
      echo "error: failed to patch seqio vocabularies.py"
      exit 1
    fi
  }
else
  echo "seqio vocabularies.py already patched"
fi

if [[ "${MAGENTA_RT_REFERENCE_T5X_SMOKE:-}" == "1" ]]; then
  "$PY" -c "import t5x.models, t5x.decoding; print('t5x import OK (MAGENTA_RT_REFERENCE_T5X_SMOKE=1)')"
else
  echo "Skipping optional t5x import smoke test (set MAGENTA_RT_REFERENCE_T5X_SMOKE=1 to enable)."
fi

{
  echo "$STAMP_LINE"
  echo "installed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$STAMP"
echo "Done. Stamp written to $STAMP"

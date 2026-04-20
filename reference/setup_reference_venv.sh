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

# Create reference/.venv with upstream magenta_rt (JAX/TF) for run_reference_stream.py.
# Requires: git submodule vendor/magenta-realtime, uv on PATH.
#
# Usage (from repository root):
#   bash reference/setup_reference_venv.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB="$ROOT/vendor/magenta-realtime"

if [[ ! -f "$SUB/pyproject.toml" ]]; then
  echo "error: vendor/magenta-realtime missing or empty."
  echo "  Run: git submodule update --init vendor/magenta-realtime"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required (https://docs.astral.sh/uv/)"
  exit 1
fi

echo "Syncing reference/pyproject.toml (Python 3.12; TensorFlow has no cp313 wheels yet)..."
uv sync --directory "$ROOT/reference" --python 3.12

REF_PY="$ROOT/reference/.venv/bin/python"
if ! "$REF_PY" -m pip --version >/dev/null 2>&1; then
  echo "Bootstrapping pip into reference/.venv (uv default venv may omit pip)..."
  "$REF_PY" -m ensurepip --upgrade
fi

echo "Installing patched t5x into reference/.venv..."
bash "$ROOT/reference/install_reference_t5x.sh" "$ROOT/reference/.venv"

echo ""
echo "Done. The reference venv now hosts both the upstream magenta_rt model"
echo "*and* the run_reference_stream.py orchestrator (numpy + soundfile +"
echo "sounddevice). Drive it directly:"
echo "  $ROOT/reference/.venv/bin/python \\"
echo "    $ROOT/scripts/run_reference_stream.py --prompt \"Deep House\""
echo "  # or, easier:"
echo "  make ref-stream PROMPT=\"Deep House\""
echo "  (WAV defaults to output/ref_baseline.wav; use --output PATH to override)"

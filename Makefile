# MagentaRealtimeMLX -- common dev tasks (uv-based).
# Run from this directory: `make help`
#
# Two main targets:
#
#   * mlx-stream   - live MagentaRT music streaming via the C++ MLX binary
#   * ref-stream   - same UX backed by the upstream JAX/TF reference path
#                    (much slower; useful as a comparison baseline)
#
# Python is used only for installation-time helpers: downloading the
# weights bundle from Hugging Face and bootstrapping the isolated
# reference venv. There is no Python MLX runtime in this distribution;
# the real-time path is C++ end-to-end.

UV         ?= uv
ROOT       := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROMPT     ?= deep house

# Reference venv (default layout: make reference-venv or any ref-using target)
REFERENCE_PYTHON := $(ROOT)/reference/.venv/bin/python
REF_PYTHON ?= $(REFERENCE_PYTHON)
REF_OUT    ?= $(ROOT)/output/ref_baseline.wav

# Hugging Face dataset that hosts the MLX weights bundle.
HF_WEIGHTS_REPO ?= rhymeswithlion/magenta-realtime-mlx-cpp

# Extra CLI flags (e.g. `make mlx-stream MLX_FLAGS="--dry-run --max-chunks 2"`)
MLX_FLAGS  ?=
REF_FLAGS  ?=
WEIGHTS_TAG ?= base

.PHONY: help sync test submodule reference-venv ensure-ref-python \
	ensure-weights-cache clean \
	mlx-stream ref-stream devices-mlx devices-ref \
	build test-cpp clean-build

help:
	@echo "MagentaRealtimeMLX"
	@echo ""
	@echo "Main commands:"
	@echo "  mlx-stream        live C++ MLX playback; bootstraps deps and weights on first run"
	@echo "  ref-stream        live reference playback; bootstraps deps and reference/.venv on first run"
	@echo ""
	@echo "Utilities:"
	@echo "  build             configure and build the C++ MLX implementation"
	@echo "  test-cpp          run the C++ Catch2 test suite"
	@echo "  devices-mlx       list PortAudio devices via the C++ binary"
	@echo "  devices-ref       list PortAudio devices for reference playback"
	@echo "  ensure-weights-cache  download .safetensors + .mlxfn from Hugging Face"
	@echo "  reference-venv    set up reference/.venv (JAX / TF / magenta_rt)"
	@echo "  sync              uv sync the install-time helper deps (HF download)"
	@echo "  test              uv pytest tests/ (small set: ref-stream helpers)"
	@echo "  clean-build       remove build/"
	@echo "  clean             remove local caches, outputs, build artifacts"
	@echo ""
	@echo "Use PROMPT, MLX_FLAGS, REF_FLAGS, REF_OUT, REF_PYTHON, HF_WEIGHTS_REPO, or WEIGHTS_TAG to override defaults."

# ---------------------------------------------------------------------------
# Install-time helper (Python). The only place Python touches the
# playback path: download pre-bundled .safetensors + .mlxfn from
# Hugging Face. Weight bundling and .mlxfn export are not part of this
# distribution.
# ---------------------------------------------------------------------------

ensure-weights-cache: sync
	$(UV) run python $(ROOT)/scripts/download_weights_from_hf.py \
		--repo-id $(HF_WEIGHTS_REPO) \
		--weights-cache $(ROOT)/.weights-cache \
		--depthformer-tag $(WEIGHTS_TAG) \
		--skip-if-complete

sync:
	$(UV) sync --group dev

test: sync
	$(UV) run pytest $(ROOT)/tests/

submodule:
	git -C $(ROOT) submodule update --init vendor/magenta-realtime

# ---------------------------------------------------------------------------
# Reference (upstream JAX/TF) path. Python here, but in an isolated venv
# that this project does not import in-process.
# ---------------------------------------------------------------------------

# Real file target: rebuild when reference deps or install scripts change.
$(REFERENCE_PYTHON): $(ROOT)/reference/pyproject.toml $(ROOT)/reference/uv.lock \
		$(ROOT)/reference/setup_reference_venv.sh $(ROOT)/reference/install_reference_t5x.sh
	git -C $(ROOT) submodule update --init vendor/magenta-realtime
	bash $(ROOT)/reference/setup_reference_venv.sh

reference-venv: $(REFERENCE_PYTHON)

ensure-ref-python:
ifeq ($(REF_PYTHON),$(REFERENCE_PYTHON))
	@$(MAKE) $(REFERENCE_PYTHON)
else
	@test -x "$(REF_PYTHON)" || ( \
		echo "error: REF_PYTHON=$(REF_PYTHON) is not executable"; \
		echo "  Use default REF_PYTHON (reference/.venv) or run: make reference-venv"; \
		exit 1)
endif

ref-stream: ensure-ref-python
	"$(REF_PYTHON)" $(ROOT)/scripts/run_reference_stream.py \
		--ref-python "$(REF_PYTHON)" \
		--prompt "$(PROMPT)" \
		--output "$(REF_OUT)" \
		$(REF_FLAGS)

devices-ref: ensure-ref-python
	"$(REF_PYTHON)" $(ROOT)/scripts/run_reference_stream.py \
		--ref-python "$(REF_PYTHON)" \
		--list-devices \
		--prompt "_"

# ---------------------------------------------------------------------------
# C++ MLX path -- the main runtime. Requires `brew install cmake ninja
# catch2 sentencepiece portaudio`, plus a completed `uv sync` so MLX C++
# headers and `libmlx.dylib` from the Python wheel are on disk.
# ---------------------------------------------------------------------------

build: sync
	cmake -S $(ROOT) -B $(ROOT)/build -G Ninja
	cmake --build $(ROOT)/build

test-cpp: build
	ctest --test-dir $(ROOT)/build --output-on-failure

clean-build:
	rm -rf $(ROOT)/build

mlx-stream: ensure-weights-cache build
	$(ROOT)/build/bin/mlx-stream \
		--prompt "$(PROMPT)" \
		--weights-dir $(ROOT)/.weights-cache \
		--tag $(WEIGHTS_TAG) \
		$(MLX_FLAGS)

devices-mlx: build
	$(ROOT)/build/bin/mlx-stream --list-devices

# ---------------------------------------------------------------------------

clean:
	@echo "Cleaning build artifacts, caches, output, and weight trees under $(ROOT) (skipping .venv, reference/.venv)"
	rm -rf "$(ROOT)/build" "$(ROOT)/dist" "$(ROOT)/.pytest_cache" "$(ROOT)/.ruff_cache" \
		"$(ROOT)/.mypy_cache" "$(ROOT)/.hypothesis" "$(ROOT)/htmlcov" \
		"$(ROOT)/output" "$(ROOT)/weights" "$(ROOT)/.weights-cache" "$(ROOT)/.magenta-rt-asset-cache"
	rm -f "$(ROOT)/.coverage" "$(ROOT)"/.coverage.*
	rm -rf .venv .t5x-build reference/.venv
	cd "$(ROOT)" && find . -type d \( -path "./.venv" -o -path "./reference/.venv" \) -prune -o \
		-type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	cd "$(ROOT)" && find . -type d \( -path "./.venv" -o -path "./reference/.venv" \) -prune -o \
		-type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

#!/usr/bin/env python3
"""Download the C++-MLX weights bundle from Hugging Face into ``.weights-cache/``.

This is the only piece of Python the runtime needs; everything else is
C++ end-to-end via ``build/bin/mlx-stream``. The script snapshots a
Hugging Face dataset into the local ``.weights-cache/`` directory in
the layout the C++ binary expects:

    .weights-cache/
      spectrostream_encoder.safetensors
      spectrostream_decoder.safetensors
      spectrostream_codebooks.safetensors
      musiccoca_encoder.safetensors
      musiccoca_codebooks.safetensors
      musiccoca_vocab.model
      depthformer/
        depthformer_<tag>.safetensors
      mlxfn/
        encode_<tag>_<dtype>.mlxfn
        depth_step_<tag>_<dtype>_cl<NN>.mlxfn      (15 files)
        temporal_step_<tag>_<dtype>_cl<NN>.mlxfn   (49 files)

The ``.mlxfn`` source-graph bundles are an optional accelerator -- the
C++ binary still works without them but falls back to a ~50 % slower
capturing-lambda compile path. Default download includes them.

The upstream ``.pt`` / ``.npy`` checkpoints are *not* fetched; they are
only useful for the (Python) reference / re-export tooling, which is
not part of this distribution.

Default repo: ``rhymeswithlion/magenta-realtime-mlx-cpp``. Authenticate via
``huggingface-cli login`` or ``HF_TOKEN`` if the dataset is private.

Typical invocation::

    uv run python scripts/download_weights_from_hf.py --skip-if-complete

``make ensure-weights-cache`` calls this with ``--skip-if-complete``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Files the C++ ``mlx-stream`` binary requires at startup. Mirrors
# ``src/magenta_realtime_mlx/weights.cpp::resolve_inference_bundle``.
_REQUIRED_ROOT = (
    "spectrostream_encoder.safetensors",
    "spectrostream_decoder.safetensors",
    "spectrostream_codebooks.safetensors",
    "musiccoca_encoder.safetensors",
    "musiccoca_codebooks.safetensors",
    "musiccoca_vocab.model",
)


def _bundle_complete(root: Path, *, depthformer_tag: str) -> bool:
    """True when every C++-required file is present.

    Note: ``.mlxfn`` files are intentionally *not* required here -- the
    binary works without them (just slower), and a partial mlxfn dir
    would still let the user generate audio. We download them by
    default but don't gate completeness on them.
    """
    df = root / "depthformer" / f"depthformer_{depthformer_tag}.safetensors"
    if not df.is_file():
        return False
    return all((root / name).is_file() for name in _REQUIRED_ROOT)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--repo-id",
        type=str,
        default="rhymeswithlion/magenta-realtime-mlx-cpp",
        help="Hugging Face dataset repo id (default: rhymeswithlion/magenta-realtime-mlx-cpp)",
    )
    p.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional revision (branch, tag, or commit) to download",
    )
    p.add_argument(
        "--weights-cache",
        type=Path,
        default=_default_repo_root() / ".weights-cache",
        help="Destination directory (default: <repo>/.weights-cache)",
    )
    p.add_argument(
        "--depthformer-tag",
        type=str,
        default="base",
        help="Depthformer tag to require for completeness check (default: base)",
    )
    p.add_argument(
        "--skip-if-complete",
        action="store_true",
        help="No-op if the destination already passes the completeness check",
    )
    p.add_argument(
        "--no-mlxfn",
        action="store_true",
        help=(
            "Skip downloading the mlxfn/ source-graph bundles. The C++ "
            "binary still runs (capturing-lambda fallback path, ~50%% slower)."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cache = args.weights_cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)

    if args.skip_if_complete and _bundle_complete(cache, depthformer_tag=args.depthformer_tag):
        print(f"Weights cache already complete at {cache}; nothing to do.")
        return 0

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as e:
        print(
            "huggingface_hub is required. Install with: uv sync",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    required = list(_REQUIRED_ROOT) + [
        f"depthformer/depthformer_{args.depthformer_tag}.safetensors",
    ]

    # Pre-flight: list what's actually on the dataset so a partially-published
    # repo (e.g. only upstream .pt / .npy artefacts uploaded, no converted
    # .safetensors yet) fails loudly with an actionable diff instead of
    # silently "downloading" zero bytes.
    try:
        server_files = set(HfApi().list_repo_files(
            args.repo_id, repo_type="dataset", revision=args.revision,
        ))
    except Exception as e:  # network / auth / 404
        print(
            f"Could not list {args.repo_id}: {type(e).__name__}: {e}\n"
            f"If the dataset is private, authenticate with `huggingface-cli "
            f"login` or set HF_TOKEN.",
            file=sys.stderr,
        )
        return 2
    missing_on_server = [f for f in required if f not in server_files]
    if missing_on_server:
        print(
            f"Dataset {args.repo_id} is missing {len(missing_on_server)} of "
            f"{len(required)} required C++ artefacts:\n"
            + "\n".join(f"  - {f}" for f in missing_on_server)
            + f"\n\nThe C++ binary loads .safetensors directly; upstream .pt / "
            f".npy checkpoints are not consumed.\nDataset URL: "
            f"https://huggingface.co/datasets/{args.repo_id}",
            file=sys.stderr,
        )
        return 2

    allow_patterns = [
        "*.safetensors",
        "depthformer/*.safetensors",
        "*.model",
    ]
    if not args.no_mlxfn:
        allow_patterns.append("mlxfn/*.mlxfn")

    print(f"Downloading {args.repo_id} -> {cache}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(cache),
        allow_patterns=allow_patterns,
    )

    if not _bundle_complete(cache, depthformer_tag=args.depthformer_tag):
        # Server had the files (per pre-flight) but they didn't land --
        # likely a partial download or pattern mismatch. Re-check locally.
        local_missing = [f for f in required if not (cache / f).is_file()]
        print(
            f"Download finished but bundle is still incomplete at {cache}.\n"
            f"Missing locally:\n"
            + "\n".join(f"  - {f}" for f in local_missing),
            file=sys.stderr,
        )
        return 2

    print(f"Weights cache ready at {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

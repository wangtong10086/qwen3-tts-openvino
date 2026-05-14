#!/usr/bin/env python3
"""Download OpenVINO IR artifacts from a Hugging Face model repo."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from qwen3_tts_ov.model_download import DEFAULT_RELEASE_MODEL_REPO, DEFAULT_RELEASE_MODEL_SUBDIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_RELEASE_MODEL_REPO)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help=f"Allowed file pattern. Can be repeated. Defaults to {DEFAULT_RELEASE_MODEL_SUBDIR}/**.",
    )
    args = parser.parse_args()

    patterns = args.allow_pattern or [f"{DEFAULT_RELEASE_MODEL_SUBDIR}/**"]
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=str(local_dir),
        allow_patterns=patterns,
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()

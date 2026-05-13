#!/usr/bin/env python3
"""Download OpenVINO IR artifacts from a Hugging Face model repo."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Allowed file pattern. Can be repeated. Defaults to openvino_realtime/**.",
    )
    args = parser.parse_args()

    patterns = args.allow_pattern or ["openvino_realtime/**"]
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

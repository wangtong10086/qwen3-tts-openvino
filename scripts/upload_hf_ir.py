#!/usr/bin/env python3
"""Upload exported OpenVINO IR directories to a Hugging Face model repo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


MODE_DIR = {
    "voice_design": "voice_design",
    "custom_voice": "custom_voice",
    "voice_clone": "base",
    "base": "base",
}


def normalize_mode(value: str) -> str:
    key = value.strip().replace("-", "_")
    if key == "all":
        return key
    if key == "base":
        return "voice_clone"
    if key not in {"voice_design", "custom_voice", "voice_clone"}:
        raise ValueError("mode must be voice_design, custom_voice, voice_clone, base, or all")
    return key


def parse_modes(value: str) -> list[str]:
    result = []
    for item in value.split(","):
        mode = normalize_mode(item)
        if mode == "all":
            result.extend(["voice_design", "custom_voice", "voice_clone"])
        else:
            result.append(mode)
    deduped = []
    for mode in result:
        if mode not in deduped:
            deduped.append(mode)
    return deduped


def load_manifest(ir_dir: Path) -> dict:
    manifest_path = ir_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def supports_mode(manifest: dict, mode: str) -> bool:
    model_type = str(manifest.get("tts_model_type") or "").replace("-", "_").lower()
    if mode == "voice_design":
        return model_type in {"", "voice_design"}
    if mode == "custom_voice":
        return model_type == "custom_voice"
    if mode == "voice_clone":
        return model_type in {"base", "voice_clone"}
    return False


def upload_folder(api: HfApi, *, repo_id: str, folder: Path, path_in_repo: str, token: str | None, commit_message: str) -> None:
    kwargs = {
        "repo_id": repo_id,
        "repo_type": "model",
        "folder_path": str(folder),
        "path_in_repo": path_in_repo,
        "commit_message": commit_message,
        "token": token,
        "ignore_patterns": [
            "**/ov_cache/**",
            "**/__pycache__/**",
            "**/.DS_Store",
            "**/*.wav",
            "**/*.tmp",
        ],
        "delete_patterns": [f"{path_in_repo}/**"],
    }
    try:
        api.upload_folder(**kwargs)
    except TypeError:
        kwargs.pop("delete_patterns", None)
        api.upload_folder(**kwargs)


def write_repo_readme(api: HfApi, *, repo_id: str, repo_subdir: str, token: str | None, modes: list[str]) -> None:
    mode_lines = []
    for mode in modes:
        mode_dir = MODE_DIR[mode]
        label = "VoiceClone/Base" if mode == "voice_clone" else mode
        mode_lines.append(f"- `{repo_subdir}/{mode_dir}/`: {label} OpenVINO IR")
    content = "\n".join(
        [
            "---",
            "license: apache-2.0",
            "tags:",
            "- qwen3-tts",
            "- openvino",
            "- text-to-speech",
            "---",
            "",
            "# Qwen3-TTS OpenVINO IR",
            "",
            "This repository stores exported OpenVINO IR artifacts for `qwen3-tts-openvino` runtime downloads.",
            "",
            "## Layout",
            "",
            *mode_lines,
            "",
            "The runtime sidecar downloads these files automatically when a mode is missing locally.",
            "",
        ]
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo="README.md",
        commit_message="Update OpenVINO IR model card",
        token=token,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="waston10086/qwen3-tts-openvino-voice-design")
    parser.add_argument("--repo-subdir", default="openvino_realtime")
    parser.add_argument("--ir-root", default="openvino")
    parser.add_argument("--modes", default="all", help="Comma-separated: voice_design,custom_voice,voice_clone,base,all")
    parser.add_argument("--token", default=None, help="Defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN.")
    parser.add_argument("--create-repo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--update-readme", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    modes = parse_modes(args.modes)
    ir_root = Path(args.ir_root)
    repo_subdir = args.repo_subdir.strip().strip("/\\")
    uploads = []
    for mode in modes:
        mode_dir = MODE_DIR[mode]
        ir_dir = ir_root / mode_dir
        manifest = load_manifest(ir_dir)
        if not supports_mode(manifest, mode):
            raise ValueError(f"{ir_dir}/manifest.json tts_model_type does not support mode={mode}")
        uploads.append((mode, mode_dir, ir_dir, f"{repo_subdir}/{mode_dir}"))

    summary = {
        "repo_id": args.repo_id,
        "repo_subdir": repo_subdir,
        "uploads": [
            {"mode": mode, "source": str(ir_dir), "path_in_repo": path_in_repo}
            for mode, _, ir_dir, path_in_repo in uploads
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    if not token:
        raise RuntimeError("HF token is required. Set HF_TOKEN or pass --token.")

    api = HfApi(token=token)
    if args.create_repo:
        api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True, token=token)
    for mode, _, ir_dir, path_in_repo in uploads:
        upload_folder(
            api,
            repo_id=args.repo_id,
            folder=ir_dir,
            path_in_repo=path_in_repo,
            token=token,
            commit_message=f"Upload {mode} OpenVINO IR",
        )
    if args.update_readme:
        write_repo_readme(api, repo_id=args.repo_id, repo_subdir=repo_subdir, token=token, modes=modes)


if __name__ == "__main__":
    main()

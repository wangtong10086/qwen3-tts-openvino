#!/usr/bin/env python3
"""Upload exported OpenVINO IR directories to a Hugging Face model repo."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationDelete, HfApi


MODE_DIR = {
    "voice_design": "voice_design",
    "custom_voice": "custom_voice",
    "voice_clone": "base",
    "base": "base",
}


def load_package_ir_module():
    path = Path(__file__).with_name("package_ir.py")
    spec = importlib.util.spec_from_file_location("qwen3_tts_ov_package_ir", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def model_type_for_mode(mode: str) -> str:
    return "base" if mode == "voice_clone" else mode


def supports_mode(manifest: dict, mode: str) -> bool:
    model_type = str(manifest.get("tts_model_type") or "").replace("-", "_").lower()
    if mode == "voice_design":
        return model_type in {"", "voice_design"}
    if mode == "custom_voice":
        return model_type == "custom_voice"
    if mode == "voice_clone":
        return model_type in {"base", "voice_clone"}
    return False


def prepare_upload_folder(
    *,
    ir_dir: Path,
    mode: str,
    profile: str,
    staging_root: Path,
    materialize: bool = True,
) -> tuple[Path, dict]:
    source_manifest = load_manifest(ir_dir)
    if not supports_mode(source_manifest, mode):
        raise ValueError(f"{ir_dir}/manifest.json tts_model_type does not support mode={mode}")
    model_type = model_type_for_mode(mode)
    if profile == "full":
        return ir_dir, {
            "profile": "full",
            "file_count": len([p for p in ir_dir.rglob("*") if p.is_file()]),
            "model_type": model_type,
        }

    package_ir = load_package_ir_module()
    manifest = package_ir.manifest_for_profile(source_manifest, "runtime-minimal", model_type)
    files = package_ir.manifest_referenced_files(ir_dir, manifest)
    tokenizer_sources = package_ir.tokenizer_file_sources(ir_dir, source_manifest)
    total_bytes = 0
    copied = 1
    for source in files:
        if source.name != "manifest.json":
            total_bytes += source.stat().st_size
            copied += 1
    for source in tokenizer_sources.values():
        total_bytes += source.stat().st_size
        copied += 1
    xml_files = sorted(source.name for source in files if source.suffix == ".xml")
    if not materialize:
        return ir_dir, {
            "profile": "runtime-minimal",
            "file_count": copied,
            "total_bytes": total_bytes,
            "model_type": model_type,
            "xml_files": xml_files,
        }

    target = staging_root / MODE_DIR[mode]
    target.mkdir(parents=True, exist_ok=True)
    for source in files:
        if source.name == "manifest.json":
            continue
        rel = source.relative_to(ir_dir)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    for rel, source in tokenizer_sources.items():
        dest = target / rel
        shutil.copy2(source, dest)
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total_bytes += manifest_path.stat().st_size
    return target, {
        "profile": "runtime-minimal",
        "file_count": copied,
        "total_bytes": total_bytes,
        "model_type": model_type,
        "xml_files": xml_files,
    }


def folder_relative_files(folder: Path) -> set[str]:
    return {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}


def prune_remote_folder(
    api: HfApi,
    *,
    repo_id: str,
    folder: Path,
    path_in_repo: str,
    token: str | None,
    commit_message: str,
) -> list[str]:
    keep = {f"{path_in_repo.rstrip('/')}/{rel}" for rel in folder_relative_files(folder)}
    prefix = path_in_repo.rstrip("/") + "/"
    remote_files = set(api.list_repo_files(repo_id, repo_type="model", token=token))
    extras = sorted(path for path in remote_files if path.startswith(prefix) and path not in keep)
    if not extras:
        return []
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message=commit_message,
        operations=[CommitOperationDelete(path_in_repo=path) for path in extras],
    )
    return extras


def upload_folder(
    api: HfApi,
    *,
    repo_id: str,
    folder: Path,
    path_in_repo: str,
    token: str | None,
    commit_message: str,
    clean_remote: bool,
) -> None:
    ignore_patterns = [
        "**/ov_cache/**",
        "**/__pycache__/**",
        "**/.DS_Store",
        "**/*.wav",
        "**/*.tmp",
    ]
    try:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(folder),
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            token=token,
            ignore_patterns=ignore_patterns,
            delete_patterns=[f"{path_in_repo}/**"] if clean_remote else None,
        )
    except TypeError:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(folder),
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            token=token,
            ignore_patterns=ignore_patterns,
        )
    if clean_remote:
        deleted = prune_remote_folder(
            api,
            repo_id=repo_id,
            folder=folder,
            path_in_repo=path_in_repo,
            token=token,
            commit_message=f"Prune stale files under {path_in_repo}",
        )
        if deleted:
            print(
                json.dumps(
                    {"pruned": path_in_repo, "deleted": deleted},
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )


def write_repo_readme(api: HfApi, *, repo_id: str, repo_subdir: str, token: str | None, modes: list[str], profile: str) -> None:
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
            "## Profile",
            "",
            f"Current public profile: `{profile}`.",
            "",
            "For `runtime-minimal`, each mode keeps only the validated low-memory production graph set:",
            "",
            "- text and codec embedding graphs",
            "- `int8_sym_batch_fused_gqa` paged-KV batch talker seed graph",
            "- cached standalone subcode graph, executed row-by-row by the online scheduler",
            "- streaming decoders `c0_t8`, `c25_t12`, and `c25_t24`",
            "- VoiceClone/Base additionally keeps `speech_encoder`, `speaker_encoder`, and `code_frame_embedding`",
            "",
            "Legacy full decoders, alternate streaming chunks, no-cache talker graphs, batch-fused decode graphs, sampled batch subcode graphs, fixed-bucket/unroll diagnostic graphs, and OpenVINO compile caches are intentionally not published here.",
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
    parser.add_argument(
        "--profile",
        default="runtime-minimal",
        choices=["runtime-minimal", "full"],
        help="Upload runtime-minimal by default. Use full only for private diagnostics.",
    )
    parser.add_argument("--token", default=None, help="Defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN.")
    parser.add_argument("--create-repo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--update-readme", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--clean-remote",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete the target remote mode directory before upload so stale graphs are removed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    modes = parse_modes(args.modes)
    ir_root = Path(args.ir_root)
    repo_subdir = args.repo_subdir.strip().strip("/\\")
    uploads = []
    with tempfile.TemporaryDirectory() as tmp:
        staging_root = Path(tmp) / "openvino_realtime"
        for mode in modes:
            mode_dir = MODE_DIR[mode]
            ir_dir = ir_root / mode_dir
            folder, info = prepare_upload_folder(
                ir_dir=ir_dir,
                mode=mode,
                profile=args.profile,
                staging_root=staging_root,
                materialize=not args.dry_run,
            )
            uploads.append((mode, mode_dir, ir_dir, folder, f"{repo_subdir}/{mode_dir}", info))

        summary = {
            "repo_id": args.repo_id,
            "repo_subdir": repo_subdir,
            "profile": args.profile,
            "clean_remote": bool(args.clean_remote),
            "uploads": [
                {
                    "mode": mode,
                    "source": str(ir_dir),
                    "upload_folder": str(folder),
                    "path_in_repo": path_in_repo,
                    **info,
                }
                for mode, _, ir_dir, folder, path_in_repo, info in uploads
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
        for mode, _, _, folder, path_in_repo, _ in uploads:
            upload_folder(
                api,
                repo_id=args.repo_id,
                folder=folder,
                path_in_repo=path_in_repo,
                token=token,
                commit_message=f"Upload {mode} {args.profile} OpenVINO IR",
                clean_remote=args.clean_remote,
            )
    if args.update_readme:
        write_repo_readme(
            api,
            repo_id=args.repo_id,
            repo_subdir=repo_subdir,
            token=token,
            modes=modes,
            profile=args.profile,
        )


if __name__ == "__main__":
    main()

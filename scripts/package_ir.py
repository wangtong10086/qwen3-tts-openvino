#!/usr/bin/env python3
"""Package an exported OpenVINO IR directory as a user-distributable artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable


def walk_manifest_strings(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_manifest_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_manifest_strings(item)


def manifest_referenced_files(ir_dir: Path, manifest: dict) -> list[Path]:
    names = {"manifest.json"}
    for value in walk_manifest_strings(manifest):
        if value.endswith((".xml", ".bin", ".json")) and not Path(value).is_absolute():
            names.add(value)
            if value.endswith(".xml"):
                names.add(str(Path(value).with_suffix(".bin")))
    paths = []
    missing = []
    for name in sorted(names):
        path = ir_dir / name
        if path.exists():
            paths.append(path)
        elif name.endswith((".xml", ".bin")):
            missing.append(name)
    if missing:
        raise FileNotFoundError("IR manifest references missing files: " + ", ".join(missing[:20]))
    return paths


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        rel = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_archive(staging_root: Path, output: Path, archive_format: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if archive_format == "zip":
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(p for p in staging_root.rglob("*") if p.is_file()):
                zf.write(path, path.relative_to(staging_root.parent).as_posix())
        return output
    if archive_format == "tar.gz":
        with tarfile.open(output, "w:gz") as tf:
            tf.add(staging_root, arcname=staging_root.name)
        return output
    if archive_format == "tar.zst":
        try:
            import zstandard as zstd
        except Exception as exc:  # pragma: no cover - depends on optional env
            raise SystemExit("tar.zst packaging requires `uv pip install -e .[release]`") from exc
        with tempfile.NamedTemporaryFile(suffix=".tar") as raw:
            with tarfile.open(raw.name, "w") as tf:
                tf.add(staging_root, arcname=staging_root.name)
            raw.seek(0)
            cctx = zstd.ZstdCompressor(level=10)
            with output.open("wb") as dst:
                cctx.copy_stream(raw, dst)
        return output
    raise ValueError(f"unsupported archive format: {archive_format}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--model-type", default="voice_design", choices=["voice_design", "custom_voice", "base"])
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--out-dir", default="dist/release")
    parser.add_argument("--format", default="auto", choices=["auto", "zip", "tar.gz", "tar.zst"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ir_dir = Path(args.ir_dir).resolve()
    manifest_path = ir_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest_referenced_files(ir_dir, manifest)
    archive_format = "tar.zst" if args.format == "auto" else args.format
    suffix = {"zip": ".zip", "tar.gz": ".tar.gz", "tar.zst": ".tar.zst"}[archive_format]
    package_name = f"qwen3-tts-openvino-ir-{args.model_type}-{args.version}"
    output = Path(args.out_dir) / f"{package_name}{suffix}"
    print(json.dumps({"package": str(output), "file_count": len(files), "format": archive_format}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    with tempfile.TemporaryDirectory() as tmp:
        staging_root = Path(tmp) / package_name
        target_ir = staging_root / "openvino" / args.model_type
        target_ir.mkdir(parents=True)
        for source in files:
            rel = source.relative_to(ir_dir)
            dest = target_ir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
        readme = staging_root / "README_RELEASE_IR.md"
        readme.write_text(
            "\n".join(
                [
                    f"# Qwen3-TTS OpenVINO IR {args.model_type} {args.version}",
                    "",
                    "Extract this package next to the app package so the server can find `openvino/`.",
                    "",
                    "Example:",
                    "",
                    "```bash",
                    "./qwen3-tts-ov-server --model-root openvino --device GPU",
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        write_checksums(staging_root)
        make_archive(staging_root, output, archive_format)
    print(output, flush=True)


if __name__ == "__main__":
    main()

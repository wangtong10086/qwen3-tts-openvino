#!/usr/bin/env python3
"""Build a user-facing Qwen3-TTS OpenVINO sidecar release package."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "qwen3-tts-ov-server"
EXCLUDED_DEV_MODULES = (
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "nncf",
    "onnx",
    "onnxruntime",
    "modelscope",
    "qwen_tts",
    "pytest",
    "pandas",
    "triton",
    "tensorflow",
    "tensorboard",
)
EXCLUDED_RUNTIME_MINIMAL_MODULES = (
    "librosa",
    "scipy",
    "sklearn",
    "scikit_learn",
    "numba",
    "llvmlite",
    "joblib",
    "audioread",
    "pooch",
    "resampy",
)
RELEASE_PROFILES = ("full", "runtime-minimal")


def current_target() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = "x64" if machine in {"x86_64", "amd64"} else machine
    if system == "windows":
        return f"windows-{arch}"
    if system == "linux":
        return f"linux-{arch}"
    return f"{system}-{arch}"


def native_library_name(target: str) -> str:
    return "qwen3_tts_ov_genai.dll" if target.startswith("windows") else "libqwen3_tts_ov_genai.so"


def find_native_library(target: str, override: str | None = None) -> Path:
    if override:
        path = Path(override).resolve()
        if not path.exists():
            raise FileNotFoundError(f"native library not found: {path}")
        return path
    name = native_library_name(target)
    candidates = [
        REPO_ROOT / "native" / "build" / name,
        REPO_ROOT / "native" / "build" / "Release" / name,
        REPO_ROOT / "native" / "build" / "Debug" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"native library not found; build it first with `uv run python scripts/build_native_codegen.py`")


def add_data_arg(source: Path, dest: str) -> str:
    sep = ";" if os.name == "nt" else ":"
    return f"{source}{sep}{dest}"


def write_release_readme(bundle_dir: Path, version: str, profile: str) -> None:
    text = f"""# Qwen3-TTS OpenVINO Server {version}

This package contains the user-facing sidecar server only. It does not include OpenVINO IR models.

Release profile: `{profile}`.

On first start, the server automatically downloads the default public OpenVINO IR
from Hugging Face to the user cache if no local model is found.

Start the server:

Linux:

```bash
./{APP_NAME} --device GPU
```

Windows:

```powershell
.\\{APP_NAME}.exe --device GPU
```

Open the web demo at `http://127.0.0.1:17860/`.

For offline use, download the IR manually and pass `--model-root <openvino_realtime>`.
Use `--no-auto-download-model` to disable network access.

The `runtime-minimal` profile supports common libsndfile-readable reference audio formats such as WAV/FLAC/OGG.
Use the `full` profile for broader optional codec support.
"""
    (bundle_dir / "README_RELEASE.md").write_text(text, encoding="utf-8")


def ensure_native_library_in_bundle(bundle_dir: Path, native_lib: Path) -> None:
    targets = [
        bundle_dir / "native" / "build" / native_lib.name,
        bundle_dir / "_internal" / "native" / "build" / native_lib.name,
        bundle_dir / "_internal" / native_lib.name,
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != native_lib.stat().st_size:
            shutil.copy2(native_lib, target)


def make_archive(bundle_dir: Path, output: Path, archive_format: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if archive_format == "zip":
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
                zf.write(path, path.relative_to(bundle_dir.parent).as_posix())
        return output
    if archive_format == "tar.gz":
        with tarfile.open(output, "w:gz") as tf:
            tf.add(bundle_dir, arcname=bundle_dir.name)
        return output
    if archive_format == "tar.zst":
        try:
            import zstandard as zstd
        except Exception as exc:  # pragma: no cover - optional dependency
            raise SystemExit("tar.zst packaging requires `uv pip install -e .[release]`") from exc
        with tempfile.NamedTemporaryFile(suffix=".tar") as raw:
            with tarfile.open(raw.name, "w") as tf:
                tf.add(bundle_dir, arcname=bundle_dir.name)
            raw.seek(0)
            with output.open("wb") as dst:
                zstd.ZstdCompressor(level=10).copy_stream(raw, dst)
        return output
    raise ValueError(f"unsupported archive format: {archive_format}")


def build_pyinstaller_command(args, target: str, native_lib: Path, entry_script: Path) -> list[str]:
    dist_root = Path(args.work_dir) / "pyinstaller" / "dist"
    build_root = Path(args.work_dir) / "pyinstaller" / "build"
    spec_root = Path(args.work_dir) / "pyinstaller" / "spec"
    pyinstaller = shutil.which("pyinstaller")
    pyinstaller_cmd = [pyinstaller] if pyinstaller else [sys.executable, "-m", "PyInstaller"]
    cmd = [
        *pyinstaller_cmd,
        "--noconfirm",
        "--clean",
        "--onedir",
        "--console",
        "--name",
        APP_NAME,
        "--distpath",
        str(dist_root),
        "--workpath",
        str(build_root),
        "--specpath",
        str(spec_root),
        "--paths",
        str(REPO_ROOT),
        "--collect-all",
        "openvino",
        "--collect-all",
        "openvino_genai",
        "--collect-all",
        "openvino_tokenizers",
        "--collect-all",
        "huggingface_hub",
        "--collect-all",
        "soundfile",
        "--collect-all",
        "soxr",
        "--hidden-import",
        "uvicorn.logging",
        "--hidden-import",
        "uvicorn.loops.auto",
        "--hidden-import",
        "uvicorn.protocols.http.auto",
        "--hidden-import",
        "uvicorn.protocols.websockets.auto",
        "--add-binary",
        add_data_arg(native_lib, "native/build"),
        str(entry_script),
    ]
    excluded_modules = list(EXCLUDED_DEV_MODULES)
    if args.profile == "runtime-minimal":
        excluded_modules.extend(EXCLUDED_RUNTIME_MINIMAL_MODULES)
    for module in excluded_modules:
        cmd[-1:-1] = ["--exclude-module", module]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=current_target(), choices=["linux-x64", "windows-x64"])
    parser.add_argument("--version", default="0.1.1")
    parser.add_argument("--out-dir", default="dist/release")
    parser.add_argument("--work-dir", default="build/release")
    parser.add_argument("--native-lib", default=None)
    parser.add_argument("--format", default="auto", choices=["auto", "zip", "tar.gz", "tar.zst"])
    parser.add_argument("--profile", default="runtime-minimal", choices=RELEASE_PROFILES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.target != current_target():
        raise SystemExit(
            f"cannot build {args.target} on {current_target()}; use a native runner for that platform"
        )

    native_lib = find_native_library(args.target, args.native_lib)
    work_dir = Path(args.work_dir)
    entry_dir = work_dir / "entry"
    entry_dir.mkdir(parents=True, exist_ok=True)
    entry_script = entry_dir / "qwen3_tts_ov_server_entry.py"
    entry_script.write_text(
        "from qwen3_tts_ov.release_server import main\n\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )
    cmd = build_pyinstaller_command(args, args.target, native_lib, entry_script)
    archive_format = "zip" if args.target.startswith("windows") else "tar.zst"
    if args.format != "auto":
        archive_format = args.format
    suffix = {"zip": ".zip", "tar.gz": ".tar.gz", "tar.zst": ".tar.zst"}[archive_format]
    profile_suffix = "" if args.profile == "full" else f"-{args.profile}"
    package_name = f"qwen3-tts-ov-server-{args.target}-{args.version}{profile_suffix}"
    output = Path(args.out_dir) / f"{package_name}{suffix}"
    print(
        json.dumps(
            {"target": args.target, "profile": args.profile, "native_lib": str(native_lib), "output": str(output), "cmd": cmd},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return
    if shutil.which("pyinstaller") is None and importlib.util.find_spec("PyInstaller") is None:
        raise SystemExit("PyInstaller is not installed. Run `uv sync --extra release` or `uv run --extra release ...`.")

    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    bundle_dir = Path(args.work_dir) / "pyinstaller" / "dist" / APP_NAME
    if not bundle_dir.exists():
        raise FileNotFoundError(f"PyInstaller bundle not found: {bundle_dir}")
    release_bundle = Path(args.out_dir) / package_name
    if release_bundle.exists():
        shutil.rmtree(release_bundle)
    shutil.copytree(bundle_dir, release_bundle)
    ensure_native_library_in_bundle(release_bundle, native_lib)
    write_release_readme(release_bundle, args.version, args.profile)
    make_archive(release_bundle, output, archive_format)
    print(output, flush=True)


if __name__ == "__main__":
    main()

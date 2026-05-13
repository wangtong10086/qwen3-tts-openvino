#!/usr/bin/env python3
"""Smoke-test a packaged Qwen3-TTS OpenVINO sidecar release."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


APP_NAME = "qwen3-tts-ov-server"


def extract_archive(archive: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    name = archive.name
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(destination)
    elif name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(destination, filter="data")
    elif name.endswith(".tar.zst"):
        try:
            import zstandard as zstd
        except Exception as exc:  # pragma: no cover - depends on optional release env
            raise SystemExit("tar.zst smoke requires the release extra: `uv sync --extra release`") from exc
        with tempfile.NamedTemporaryFile(suffix=".tar") as raw:
            with archive.open("rb") as src:
                zstd.ZstdDecompressor().copy_stream(src, raw)
            raw.flush()
            with tarfile.open(raw.name, "r:") as tf:
                tf.extractall(destination, filter="data")
    else:
        raise ValueError(f"unsupported release archive: {archive}")

    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"expected one extracted bundle root in {destination}, found {len(roots)}")
    return roots[0]


def executable_name() -> str:
    return f"{APP_NAME}.exe" if sys.platform.startswith("win") else APP_NAME


def find_executable(bundle_root: Path) -> Path:
    direct = bundle_root / executable_name()
    if direct.exists():
        return direct
    matches = sorted(bundle_root.rglob(executable_name()))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"release executable not found under {bundle_root}")


def read_health(url: str, timeout: float) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except (OSError, urllib.error.URLError):
        return None
    return json.loads(body)


def wait_for_health(url: str, deadline: float) -> dict:
    last_payload = None
    while time.time() < deadline:
        payload = read_health(url, timeout=2.0)
        if payload is not None:
            last_payload = payload
            if payload.get("ok") is True:
                return payload
        time.sleep(0.5)
    raise TimeoutError(f"server did not become healthy at {url}; last_payload={last_payload!r}")


def terminate_process(process: subprocess.Popen, timeout: float = 10.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def tail(path: Path, max_chars: int = 8000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-max_chars:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, help="Release archive to extract and smoke-test.")
    parser.add_argument("--work-dir", default="build/release-smoke")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17960)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--exe-path-out", default=None, help="Optional file that receives the extracted executable path.")
    args = parser.parse_args()

    archive = Path(args.archive).resolve()
    if not archive.exists():
        raise FileNotFoundError(f"release archive not found: {archive}")

    work_dir = Path(args.work_dir).resolve()
    extract_dir = work_dir / "extracted"
    bundle_root = extract_archive(archive, extract_dir)
    exe = find_executable(bundle_root)
    if args.exe_path_out:
        out = Path(args.exe_path_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(str(exe), encoding="utf-8")

    help_result = subprocess.run([str(exe), "--help"], text=True, capture_output=True, timeout=30)
    if help_result.returncode != 0:
        raise RuntimeError(f"`{exe} --help` failed:\n{help_result.stderr}\n{help_result.stdout}")

    empty_model_root = work_dir / "empty-model-root"
    empty_model_root.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "server.log"
    cmd = [
        str(exe),
        "--model-root",
        str(empty_model_root),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--device",
        args.device,
        "--no-warmup",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        payload = wait_for_health(f"http://{args.host}:{args.port}/health", deadline=time.time() + args.timeout)
        print(json.dumps({"executable": str(exe), "health": payload}, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        print("--- server log tail ---", file=sys.stderr)
        print(tail(log_path), file=sys.stderr)
        raise
    finally:
        terminate_process(process)


if __name__ == "__main__":
    main()

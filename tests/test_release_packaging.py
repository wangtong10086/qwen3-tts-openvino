from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

from qwen3_tts_ov import native_codegen
from qwen3_tts_ov.release_server import default_model_root


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, REPO_ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_release_server_default_model_root_prefers_cwd_openvino(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "openvino").mkdir()

    assert default_model_root() == tmp_path / "openvino"


def test_native_library_name_is_platform_specific():
    name = native_codegen.native_library_name()
    if sys.platform.startswith("win"):
        assert name.endswith(".dll")
    elif sys.platform == "darwin":
        assert name.endswith(".dylib")
    else:
        assert name.endswith(".so")


def test_package_ir_collects_manifest_referenced_files(tmp_path):
    package_ir = load_script("scripts/package_ir.py")
    ir_dir = tmp_path / "ir"
    ir_dir.mkdir()
    manifest = {
        "graphs": {
            "text_embedding": "text_embedding.xml",
            "nested": {"decoder": "decoder/speech_decoder.xml"},
        }
    }
    (ir_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for rel in ("text_embedding.xml", "text_embedding.bin", "decoder/speech_decoder.xml", "decoder/speech_decoder.bin"):
        path = ir_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rel, encoding="utf-8")

    files = {path.relative_to(ir_dir).as_posix() for path in package_ir.manifest_referenced_files(ir_dir, manifest)}

    assert files == {
        "manifest.json",
        "text_embedding.xml",
        "text_embedding.bin",
        "decoder/speech_decoder.xml",
        "decoder/speech_decoder.bin",
    }


def test_package_ir_dry_run(tmp_path):
    ir_dir = tmp_path / "openvino" / "voice_design"
    ir_dir.mkdir(parents=True)
    manifest = {"graphs": {"text_embedding": "text_embedding.xml"}}
    (ir_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (ir_dir / "text_embedding.xml").write_text("<xml/>", encoding="utf-8")
    (ir_dir / "text_embedding.bin").write_bytes(b"bin")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/package_ir.py",
            "--ir-dir",
            str(ir_dir),
            "--model-type",
            "voice_design",
            "--version",
            "test",
            "--out-dir",
            str(tmp_path / "dist"),
            "--format",
            "zip",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "qwen3-tts-openvino-ir-voice_design-test.zip" in result.stdout


def test_package_release_dry_run_uses_server_entry_and_native_lib(tmp_path):
    target = "windows-x64" if platform.system().lower() == "windows" else "linux-x64"
    native_name = "qwen3_tts_ov_genai.dll" if target.startswith("windows") else "libqwen3_tts_ov_genai.so"
    native_lib = tmp_path / native_name
    native_lib.write_bytes(b"fake")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/package_release.py",
            "--target",
            target,
            "--version",
            "test",
            "--native-lib",
            str(native_lib),
            "--work-dir",
            str(tmp_path / "work"),
            "--out-dir",
            str(tmp_path / "dist"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["target"] == target
    assert payload["native_lib"] == str(native_lib)
    assert "qwen3_tts_ov_server_entry.py" in " ".join(payload["cmd"])

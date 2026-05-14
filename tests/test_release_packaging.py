from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

from qwen3_tts_ov import native_codegen
from qwen3_tts_ov import model_download
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


def test_release_model_download_uses_local_manifest(tmp_path):
    model_root = tmp_path / "openvino_realtime"
    voice_design = model_root / "voice_design"
    voice_design.mkdir(parents=True)
    (voice_design / "manifest.json").write_text("{}", encoding="utf-8")

    result = model_download.ensure_release_model_root(model_root, auto_download=True)

    assert result.status == "local"
    assert result.model_root == model_root


def test_release_model_download_fetches_missing_model_root(tmp_path, monkeypatch):
    def fake_snapshot_download(*, repo_id, revision, local_dir, subdir):
        target = local_dir / subdir / "voice_design"
        target.mkdir(parents=True)
        (target / "manifest.json").write_text("{}", encoding="utf-8")
        return local_dir

    monkeypatch.setattr(model_download, "_snapshot_download", fake_snapshot_download)

    result = model_download.ensure_release_model_root(
        tmp_path / "missing",
        auto_download=True,
        repo_id="owner/repo",
        revision="main",
        subdir="openvino_realtime",
        cache_dir=tmp_path / "cache",
    )

    assert result.status == "downloaded"
    assert result.model_root == tmp_path / "cache" / "owner--repo--main" / "openvino_realtime"
    assert (result.model_root / "voice_design" / "manifest.json").exists()


def test_release_model_download_fetches_missing_mode_into_model_root(tmp_path, monkeypatch):
    def fake_snapshot_download(*, repo_id, revision, local_dir, subdir, allow_patterns=None):
        target = local_dir / subdir / "base"
        target.mkdir(parents=True)
        (target / "manifest.json").write_text(json.dumps({"tts_model_type": "base"}), encoding="utf-8")
        return local_dir

    monkeypatch.setattr(model_download, "_snapshot_download", fake_snapshot_download)

    result = model_download.download_mode_ir(
        tmp_path / "openvino",
        "voice_clone",
        repo_id="owner/repo",
        revision="main",
        subdir="openvino_realtime",
        cache_dir=tmp_path / "cache",
    )

    assert result.status == "downloaded"
    assert result.target_dir == tmp_path / "openvino" / "base"
    assert (tmp_path / "openvino" / "base" / "manifest.json").exists()


def test_release_model_download_can_be_disabled(tmp_path):
    result = model_download.ensure_release_model_root(
        tmp_path / "missing",
        auto_download=False,
        cache_dir=tmp_path / "cache",
    )

    assert result.status == "missing"
    assert result.model_root == tmp_path / "missing"


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
    (ir_dir / "vocab.json").write_text("{}", encoding="utf-8")
    (ir_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (ir_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

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
            "--profile",
            "full",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "qwen3-tts-openvino-ir-voice_design-test.zip" in result.stdout


def test_package_ir_runtime_minimal_keeps_only_long_ar_graphs(tmp_path):
    package_ir = load_script("scripts/package_ir.py")
    ir_dir = tmp_path / "ir"
    model_dir = tmp_path / "model"
    ir_dir.mkdir()
    model_dir.mkdir()
    manifest = {
        "tts_model_type": "voice_design",
        "model_dir": str(model_dir),
        "tokenizer_ir": {"tokenizer": "openvino_tokenizer.xml", "detokenizer": "openvino_detokenizer.xml"},
        "graphs": {
            "text_embedding": "text_embedding.xml",
            "codec_embedding": "codec_embedding.xml",
            "talker": "talker_no_cache.xml",
            "paged_kv_seed": {"talker_stateful_gqa": "fp16_paged.xml"},
            "subcode_greedy": "subcode_greedy.xml",
            "subcode_greedy_cached": "subcode_greedy_cached.xml",
            "speech_decoder": {"256": "speech_decoder_t256.xml"},
            "streaming_decoder": {"12": "speech_decoder_stream_c25_t12.xml", "24": "speech_decoder_stream_c25_t24.xml"},
        },
        "graph_variants": {
            "int8_sym_paged_talker_split": {
                "precision": "int8_sym_weights",
                "graphs": {"paged_kv_seed": {"talker_stateful_gqa": "talker_int8.xml"}},
            }
        },
        "streaming_decoder": {
            "contexts": {
                "0": {"8": "speech_decoder_stream_c0_t8.xml", "12": "speech_decoder_stream_c0_t12.xml"},
                "25": {"12": "speech_decoder_stream_c25_t12.xml", "24": "speech_decoder_stream_c25_t24.xml"},
            },
            "output_format": "pcm_f32",
        },
    }
    (ir_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for rel in (
        "text_embedding.xml",
        "text_embedding.bin",
        "codec_embedding.xml",
        "codec_embedding.bin",
        "talker_int8.xml",
        "talker_int8.bin",
        "subcode_greedy_cached.xml",
        "subcode_greedy_cached.bin",
        "speech_decoder_stream_c0_t8.xml",
        "speech_decoder_stream_c0_t8.bin",
        "speech_decoder_stream_c25_t24.xml",
        "speech_decoder_stream_c25_t24.bin",
    ):
        (ir_dir / rel).write_text(rel, encoding="utf-8")
    for rel in ("vocab.json", "merges.txt", "tokenizer_config.json"):
        (model_dir / rel).write_text(rel, encoding="utf-8")

    minimal = package_ir.manifest_for_profile(manifest, "runtime-minimal", "voice_design")
    files = {path.relative_to(ir_dir).as_posix() for path in package_ir.manifest_referenced_files(ir_dir, minimal)}
    tokenizer_sources = package_ir.tokenizer_file_sources(ir_dir, manifest)

    assert minimal["model_dir"] == "."
    assert "tokenizer_ir" not in minimal
    assert files == {
        "manifest.json",
        "text_embedding.xml",
        "text_embedding.bin",
        "codec_embedding.xml",
        "codec_embedding.bin",
        "talker_int8.xml",
        "talker_int8.bin",
        "subcode_greedy_cached.xml",
        "subcode_greedy_cached.bin",
        "speech_decoder_stream_c0_t8.xml",
        "speech_decoder_stream_c0_t8.bin",
        "speech_decoder_stream_c25_t24.xml",
        "speech_decoder_stream_c25_t24.bin",
    }
    assert set(tokenizer_sources) == {"vocab.json", "merges.txt", "tokenizer_config.json"}
    assert minimal["streaming_decoder"]["contexts"] == {
        "0": {"8": "speech_decoder_stream_c0_t8.xml"},
        "25": {"24": "speech_decoder_stream_c25_t24.xml"},
    }


def test_package_ir_runtime_minimal_base_requires_clone_graphs(tmp_path):
    package_ir = load_script("scripts/package_ir.py")
    manifest = {
        "tts_model_type": "base",
        "graphs": {
            "text_embedding": "text_embedding.xml",
            "codec_embedding": "codec_embedding.xml",
            "subcode_greedy_cached": "subcode_greedy_cached.xml",
        },
        "graph_variants": {
            "int8_sym_paged_talker_split": {
                "graphs": {"paged_kv_seed": {"talker_stateful_gqa": "talker_int8.xml"}}
            }
        },
        "streaming_decoder": {"contexts": {"0": {"8": "c0.xml"}, "25": {"24": "c25.xml"}}},
    }

    try:
        package_ir.manifest_for_profile(manifest, "runtime-minimal", "base")
    except ValueError as exc:
        assert "code_frame_embedding" in str(exc)
    else:
        raise AssertionError("runtime-minimal base packaging should require clone prompt graphs")


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
    assert payload["profile"] == "runtime-minimal"
    assert payload["native_lib"] == str(native_lib)
    assert "qwen3_tts_ov_server_entry.py" in " ".join(payload["cmd"])
    assert "huggingface_hub" in " ".join(payload["cmd"])
    assert "librosa" in " ".join(payload["cmd"])
    assert "scipy" in " ".join(payload["cmd"])


def test_package_release_full_profile_does_not_exclude_audio_full_modules(tmp_path):
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
            "--profile",
            "full",
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
    cmd = " ".join(payload["cmd"])
    assert payload["profile"] == "full"
    assert "librosa" not in cmd
    assert payload["output"].endswith(f"qwen3-tts-ov-server-{target}-test.{ 'zip' if target.startswith('windows') else 'tar.zst' }")


def test_package_release_places_native_library_in_frozen_search_paths(tmp_path):
    package_release = load_script("scripts/package_release.py")
    native_lib = tmp_path / "qwen3_tts_ov_genai.dll"
    native_lib.write_bytes(b"fake dll")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    package_release.ensure_native_library_in_bundle(bundle_dir, native_lib)

    assert (bundle_dir / "native" / "build" / native_lib.name).read_bytes() == b"fake dll"
    assert (bundle_dir / "_internal" / "native" / "build" / native_lib.name).read_bytes() == b"fake dll"
    assert (bundle_dir / "_internal" / native_lib.name).read_bytes() == b"fake dll"


def test_build_native_codegen_parses_dumpbin_exports():
    build_native = load_script("scripts/build_native_codegen.py")
    output = """
          ordinal hint RVA      name

                1    0 00001000 ?Tokenizer@genai@ov@@QEAA@XZ
                2    1 00002000 plain_c_symbol
                3    2 00003000 [NONAME]
                4    3 00004000 ?Tokenizer@genai@ov@@QEAA@XZ
    """

    assert build_native.parse_dumpbin_exports(output) == [
        "?Tokenizer@genai@ov@@QEAA@XZ",
        "plain_c_symbol",
    ]

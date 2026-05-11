import json

import pytest

from qwen3_tts_ov.cache import build_ov_cache_config, default_ov_cache_root, normalize_ov_cache_mode, resolve_ov_cache_dir
from qwen3_tts_ov.cache_warmup import collect_warmup_tasks
from qwen3_tts_ov.manifest import resolve_ir_dir


def test_default_ov_cache_root_uses_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.delenv("QWEN3_TTS_OV_CACHE_DIR", raising=False)

    assert default_ov_cache_root() == tmp_path / "xdg-cache" / "qwen3-tts-ov" / "openvino-cache"


def test_build_ov_cache_config_can_disable_cache(tmp_path):
    assert build_ov_cache_config(tmp_path, ov_cache_mode="optimize_size") == {
        "CACHE_DIR": str(tmp_path),
        "CACHE_MODE": "OPTIMIZE_SIZE",
    }
    assert build_ov_cache_config(tmp_path, disable_ov_cache=True) == {}


def test_resolve_cache_dir_is_namespaced(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN3_TTS_OV_CACHE_DIR", str(tmp_path / "cache-root"))
    manifest = {"tts_model_type": "voice_design", "graphs": {"text_embedding": "text_embedding.xml"}}

    first = resolve_ov_cache_dir(
        tmp_path / "ir",
        manifest,
        device="GPU",
        decoder_device="GPU",
        mode="cache",
        cache_kernel="exact",
        cache_step="fused",
        graph_variant="fp16",
        precision_hint="f16",
        compile_config={},
    )
    second = resolve_ov_cache_dir(
        tmp_path / "ir",
        manifest,
        device="GPU",
        decoder_device="CPU",
        mode="cache",
        cache_kernel="exact",
        cache_step="fused",
        graph_variant="fp16",
        precision_hint="f16",
        compile_config={},
    )

    assert str(first).startswith(str(tmp_path / "cache-root"))
    assert first != second


def test_normalize_ov_cache_mode_accepts_cli_values():
    assert normalize_ov_cache_mode("optimize_speed") == "OPTIMIZE_SPEED"
    assert normalize_ov_cache_mode("optimize-size") == "OPTIMIZE_SIZE"


def test_collect_warmup_tasks_uses_strategy_stream_decoders(tmp_path):
    ir_dir = tmp_path / "ir"
    ir_dir.mkdir()
    manifest = {
        "tts_model_type": "voice_design",
        "graphs": {
            "text_embedding": "text_embedding.xml",
            "codec_embedding": "codec_embedding.xml",
            "code_frame_embedding": "code_frame_embedding.xml",
            "fused_cache_step_buckets": {"exact": {"128": "fused_cache_step_exact_cache128.xml"}},
            "speech_decoder": {"64": "speech_decoder_t64.xml"},
        },
        "streaming_decoder": {
            "left_context_frames": 25,
            "strategies": {
                "low_latency": {
                    "initial_chunk_frames": 8,
                    "chunk_frames": 12,
                    "left_context_frames": 25,
                }
            },
            "contexts": {
                "0": {"8": "speech_decoder_stream_c0_t8.xml"},
                "25": {"12": "speech_decoder_stream_c25_t12.xml"},
            },
        },
    }
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    tasks, _ = collect_warmup_tasks(ir_dir, graphs="core,stream,buckets", preload_buckets="warmup")

    labels = [task.label for task in tasks]
    assert "core:text_embedding" in labels
    assert "stream:c0_t8" in labels
    assert "stream:c25_t12" in labels
    assert "bucket:fused_cache_step_buckets:128" in labels


def test_collect_warmup_tasks_reports_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="OpenVINO IR manifest not found"):
        collect_warmup_tasks(tmp_path / "missing-ir")


def test_default_voice_design_ir_falls_back_to_legacy_openvino_full(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "openvino_full"
    legacy.mkdir()
    manifest = {
        "tts_model_type": "voice_design",
        "graphs": {
            "text_embedding": "text_embedding.xml",
            "codec_embedding": "codec_embedding.xml",
            "code_frame_embedding": "code_frame_embedding.xml",
        },
    }
    with open(legacy / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    assert resolve_ir_dir("openvino/voice_design", fallback_to_local_voice_design=True) == legacy.relative_to(tmp_path)
    assert resolve_ir_dir("auto", fallback_to_local_voice_design=True) == legacy.relative_to(tmp_path)
    tasks, _ = collect_warmup_tasks("openvino/voice_design", graphs="core")
    assert [task.label for task in tasks] == [
        "core:text_embedding",
        "core:codec_embedding",
        "core:code_frame_embedding",
    ]


def test_cache_warmup_default_auto_uses_local_voice_design(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    exported = tmp_path / "openvino" / "voice_design"
    exported.mkdir(parents=True)
    manifest = {
        "tts_model_type": "voice_design",
        "graphs": {"text_embedding": "text_embedding.xml"},
    }
    with open(exported / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    tasks, _ = collect_warmup_tasks("auto", graphs="core")

    assert [task.label for task in tasks] == ["core:text_embedding"]

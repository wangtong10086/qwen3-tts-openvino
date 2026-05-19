import json
import os
from types import SimpleNamespace

import pytest

import qwen3_tts_ov.cache as cache_module
from qwen3_tts_ov.cache import build_ov_cache_config, default_ov_cache_root, normalize_ov_cache_mode, resolve_ov_cache_dir
from qwen3_tts_ov.cache_warmup import collect_warmup_tasks, select_buckets, subprocess_base_args
from qwen3_tts_ov.cli import apply_native_env, apply_profile_defaults
from qwen3_tts_ov.manifest import resolve_ir_dir
from qwen3_tts_ov.profiles import (
    effective_codegen_unroll,
    kv_cache_profile_from_options,
    kv_cache_profile_options,
    kv_cache_precision_bytes,
    effective_runtime_options,
    missing_graph_variant_message,
    scheduled_codegen_unrolls,
)
from qwen3_tts_ov.runtime import OpenVINOQwen3TTS


def test_default_ov_cache_root_uses_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_module.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.delenv("QWEN3_TTS_OV_CACHE_DIR", raising=False)

    assert default_ov_cache_root() == tmp_path / "xdg-cache" / "qwen3-tts-ov" / "openvino-cache"


def test_build_ov_cache_config_can_disable_cache(tmp_path):
    assert build_ov_cache_config(tmp_path, ov_cache_mode="optimize_size") == {
        "CACHE_DIR": str(tmp_path),
        "CACHE_MODE": "OPTIMIZE_SIZE",
    }
    assert build_ov_cache_config(tmp_path, disable_ov_cache=True) == {}


def test_kv_cache_profiles_map_to_runtime_precision():
    assert kv_cache_profile_options("auto") == {}
    assert kv_cache_profile_options("u8") == {
        "native_paged_kv_precision": "u8",
        "native_paged_kv_cache_input_precision": "f32",
        "native_paged_kv_block_size": 16,
    }
    assert kv_cache_profile_options("u8-all")["native_paged_kv_cache_input_precision"] == "u8"
    assert kv_cache_profile_from_options("u8", "f32", 16) == "u8"
    assert kv_cache_profile_from_options("f16", "u8", 16) == "u8-input"
    assert kv_cache_precision_bytes("u8") == 1
    assert kv_cache_precision_bytes("f16") == 2


def test_fastest_profile_defaults_to_u8_kv_cache():
    args = SimpleNamespace(
        realtime_profile="fastest",
        native_paged_kv=None,
        native_pipeline=None,
        native_paged_kv_precision=None,
        native_paged_kv_cache_input_precision=None,
        native_paged_kv_block_size=None,
        kv_cache_profile="auto",
    )

    apply_profile_defaults(args)

    assert args.native_paged_kv == "require"
    assert args.native_paged_kv_precision == "u8"
    assert args.native_paged_kv_cache_input_precision == "f32"
    assert args.native_paged_kv_block_size == 16


def test_fastest_profile_respects_kv_cache_profile_override():
    args = SimpleNamespace(
        realtime_profile="fastest",
        native_paged_kv=None,
        native_pipeline=None,
        native_paged_kv_precision=None,
        native_paged_kv_cache_input_precision=None,
        native_paged_kv_block_size=None,
        kv_cache_profile="u8",
    )

    apply_profile_defaults(args)

    assert args.native_paged_kv == "require"
    assert args.native_paged_kv_precision == "u8"
    assert args.native_paged_kv_cache_input_precision == "f32"
    assert args.native_paged_kv_block_size == 16


def test_fastest_profile_preserves_explicit_native_kv_cache_precision():
    args = SimpleNamespace(
        realtime_profile="fastest",
        native_paged_kv=None,
        native_pipeline=None,
        native_paged_kv_precision="u8",
        native_paged_kv_cache_input_precision="u8",
        native_paged_kv_block_size=8,
        kv_cache_profile="auto",
    )

    apply_profile_defaults(args)

    assert args.native_paged_kv_precision == "u8"
    assert args.native_paged_kv_cache_input_precision == "u8"
    assert args.native_paged_kv_block_size == 8


def test_native_async_decode_off_sets_disable_env(monkeypatch):
    monkeypatch.delenv("QWEN3_TTS_OV_NATIVE_ASYNC_DECODE", raising=False)

    apply_native_env(SimpleNamespace(native_async_decode="off"))

    assert os.environ["QWEN3_TTS_OV_NATIVE_ASYNC_DECODE"] == "0"


def test_cache_warmup_prefers_common_low_latency_bucket():
    available = {80: "cache80.xml", 96: "cache96.xml", 112: "cache112.xml", 128: "cache128.xml"}

    assert select_buckets(available, "warmup") == {112: "cache112.xml"}


def test_cache_warmup_subprocess_preserves_npu_offload():
    args = SimpleNamespace(
        ir_dir="openvino/voice_design",
        device="GPU",
        decoder_device="NPU",
        encoder_device="NPU",
        prompt_device="NPU",
        npu_offload="all",
        mode="no-cache",
        cache_kernel="exact",
        cache_step="fused",
        graph_variant="int8_sym_paged_talker_split",
        codegen_unroll=1,
        codegen_schedule="current",
        preferred_cache_bucket=0,
        precision_hint="f16",
        ov_cache_mode="optimize_speed",
        ov_cache_dir=None,
        disable_ov_cache=False,
        allow_cpu_fallback=False,
    )

    cmd = subprocess_base_args(args, {})

    assert cmd[cmd.index("--decoder-device") + 1] == "NPU"
    assert cmd[cmd.index("--encoder-device") + 1] == "NPU"
    assert cmd[cmd.index("--prompt-device") + 1] == "NPU"
    assert cmd[cmd.index("--npu-offload") + 1] == "all"


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
    third = resolve_ov_cache_dir(
        tmp_path / "ir",
        manifest,
        device="GPU",
        decoder_device="GPU",
        prompt_device="NPU",
        mode="cache",
        cache_kernel="exact",
        cache_step="fused",
        graph_variant="fp16",
        precision_hint="f16",
        compile_config={},
    )
    assert third != first


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
    assert next(task for task in tasks if task.label == "core:text_embedding").device_role == "prompt"
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


def test_fastest_profile_expands_to_native_paged_kv_defaults():
    assert effective_runtime_options("fastest", "sdpa", "split", "fp16") == (
        "no-cache",
        "exact",
        "fused",
        "int8_sym_paged_talker_split",
    )
    assert effective_codegen_unroll("fastest", "int8_sym_paged_talker_split", "profile") == 1
    assert effective_codegen_unroll("fastest", "int8_sym_paged_talker_split", "12") == 12
    assert scheduled_codegen_unrolls("current", 1) == (1,)


def test_missing_fastest_variant_message_reports_production_hint():
    hint = missing_graph_variant_message("int8_sym_paged_talker_split", "none")
    assert "--preset fastest" in hint


def test_runtime_bucket_loader_prefers_int8_fused_variant():
    runtime = object.__new__(OpenVINOQwen3TTS)
    graphs = {"fused_cache_step_buckets": {"exact": {"128": "fp16.xml"}}}
    variant_graphs = {"fused_cache_step_buckets": {"exact": {"128": "int8.xml"}}}

    assert runtime._load_fused_cache_bucket_graphs(graphs, "exact", variant_graphs) == {128: "int8.xml"}


def test_runtime_bucket_selector_reuses_compiled_larger_bucket():
    available = {80: "cache80.xml", 112: "cache112.xml", 128: "cache128.xml"}

    assert OpenVINOQwen3TTS.select_runtime_bucket(available, 97, [128]) == 128
    assert OpenVINOQwen3TTS.select_runtime_bucket(available, 97, []) == 112
    assert OpenVINOQwen3TTS.select_runtime_bucket(available, 72, [], preferred_min_bucket=80) == 80
    assert OpenVINOQwen3TTS.select_runtime_bucket(available, 72, [], preferred_min_bucket=None) == 80


def test_cache_warmup_select_buckets_uses_preferred_cache_bucket():
    available = {80: "cache80.xml", 96: "cache96.xml", 112: "cache112.xml"}

    assert select_buckets(available, "warmup", preferred_cache_bucket=96) == {96: "cache96.xml"}
    assert select_buckets(available, "warmup", preferred_cache_bucket="none") == {80: "cache80.xml"}


def test_runtime_codegen_schedule_selects_low_latency_then_steady_unroll():
    runtime = object.__new__(OpenVINOQwen3TTS)
    runtime.codegen_schedule = "current"
    runtime.codegen_unroll = 1
    runtime.fused_cache_unroll_bucket_graphs_by_step = {
        1: {128: "u1.xml"},
    }

    assert runtime.select_codegen_unroll_for_step(0) == 1
    assert runtime.select_codegen_unroll_for_step(8) == 1


def test_runtime_unroll_bucket_loader_prefers_variant():
    runtime = object.__new__(OpenVINOQwen3TTS)
    graphs = {
        "fused_cache_step_unroll_buckets": {
            "exact": {"4": {"128": "fp16.xml"}}
        }
    }
    variant_graphs = {
        "fused_cache_step_unroll_buckets": {
            "exact": {"4": {"128": "int8-sym.xml"}}
        }
    }

    assert runtime._load_fused_cache_unroll_bucket_graphs_by_step(graphs, "exact", variant_graphs) == {
        4: {128: "int8-sym.xml"}
    }


def test_runtime_decode_unroll_bucket_loader_prefers_variant():
    runtime = object.__new__(OpenVINOQwen3TTS)
    graphs = {
        "fused_cache_decode_unroll_stateful_mask_buckets": {
            "exact": {"4": {"96": "fp16.xml", "128": "fp16-128.xml"}}
        }
    }
    variant_graphs = {
        "fused_cache_decode_unroll_stateful_mask_buckets": {
            "exact": {"4": {"96": "int8-sym.xml"}}
        }
    }

    assert runtime._load_fused_cache_decode_unroll_bucket_graphs_by_step(
        graphs,
        "exact",
        variant_graphs,
        "fused_cache_decode_unroll_stateful_mask_buckets",
    ) == {
        4: {96: "int8-sym.xml", 128: "fp16-128.xml"}
    }

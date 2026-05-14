import json
from types import SimpleNamespace

import numpy as np
import pytest

from qwen3_tts_ov import server


def test_auto_segment_requires_explicit_allow(monkeypatch):
    monkeypatch.delenv(server.ENABLE_AUTO_SEGMENT_ENV, raising=False)
    request = {
        "mode": "voice_design",
        "text": "这是一段明显超过自动分段阈值的中文测试文本，用于确认默认不会再切分。",
        "auto_segment_text": True,
        "auto_segment_units": 4,
    }

    assert server.request_will_auto_segment(request) is False

    request["allow_auto_segment_text"] = True
    assert server.request_will_auto_segment(request) is True


def test_continuous_long_output_metadata_defaults_to_full_ar(monkeypatch):
    monkeypatch.delenv(server.ENABLE_PAGED_LONG_AR_ENV, raising=False)
    monkeypatch.setenv("QWEN3_TTS_OV_NATIVE_PAGED_KV", "require")

    metadata = server.continuous_long_output_metadata(True)

    assert metadata["long_text_mode"] == "full_ar"
    assert metadata["segmented"] is False
    assert metadata["continuous_backend"] == "single_prompt_full_ar_reference"
    assert metadata["paged_kv"] is False


def test_paged_long_ar_requires_explicit_enable(monkeypatch):
    monkeypatch.setenv("QWEN3_TTS_OV_NATIVE_PAGED_KV", "require")
    monkeypatch.setenv(server.ENABLE_PAGED_LONG_AR_ENV, "1")

    metadata = server.continuous_long_output_metadata(True)

    assert metadata["continuous_backend"] == "native_paged_attention"
    assert metadata["paged_kv"] is True


def test_explicit_long_ar_paged_sample_profile_sets_native_runtime():
    profile = server.explicit_long_text_profile("paged-sample-int8")

    assert profile["profile"] == "long_paged_split_sample_int8_sym"
    assert profile["runtime"]["native_pipeline"] == "require"
    assert profile["runtime"]["native_paged_kv"] == "require"
    assert profile["runtime"]["native_paged_kv_split_subcode"] == "1"
    assert profile["profile_env"]["native_paged_kv_split_subcode_mode"] == "cached_exact"


def test_builtin_long_ar_profile_uses_int8_paged_split_when_manifest_supports_it():
    manifest = {
        "graphs": {
            "subcode_greedy_cached": "subcode_greedy_cached.xml",
            "paged_kv_seed": {"talker_stateful_gqa": "talker_stateful_sdpa_paged_gqa_seed.xml"},
        },
        "graph_variants": {
            "int8_sym_paged_talker_split": {
                "graphs": {
                    "paged_kv_seed": {
                        "talker_stateful_gqa": "talker_stateful_sdpa_paged_gqa_seed_int8_sym_paged_talker_split.xml"
                    }
                }
            }
        },
    }

    profile = server.builtin_long_text_profile_from_manifest(manifest)

    assert profile["profile"] == "long_paged_split_sample_int8_sym"
    assert profile["source"] == "builtin_manifest"
    assert profile["runtime"]["graph_variant"] == "int8_sym_paged_talker_split"
    assert profile["runtime"]["native_paged_kv"] == "require"
    assert profile["runtime"]["native_paged_kv_split_subcode"] == "1"
    assert profile["profile_env"]["native_paged_kv_split_subcode_mode"] == "cached"
    assert profile["split_subcode_mode_fallback"]["requested"] == "cached_exact"
    assert profile["split_subcode_mode_fallback"]["effective"] == "cached"


def test_builtin_long_ar_profile_keeps_cached_exact_when_manifest_supports_it():
    manifest = {
        "graphs": {
            "subcode_greedy_cached": "subcode_greedy_cached.xml",
            "subcode_greedy_cached_exact": "subcode_greedy_cached_exact.xml",
            "paged_kv_seed": {"talker_stateful_gqa": "talker_stateful_sdpa_paged_gqa_seed.xml"},
        },
        "graph_variants": {
            "int8_sym_paged_talker_split": {
                "graphs": {
                    "paged_kv_seed": {
                        "talker_stateful_gqa": "talker_stateful_sdpa_paged_gqa_seed_int8_sym_paged_talker_split.xml"
                    }
                }
            }
        },
    }

    profile = server.builtin_long_text_profile_from_manifest(manifest)

    assert profile["profile_env"]["native_paged_kv_split_subcode_mode"] == "cached_exact"
    assert "split_subcode_mode_fallback" not in profile


def test_long_ar_profile_uses_cpu_codegen_when_server_device_has_no_gpu():
    profile = server.explicit_long_text_profile("paged-sample-int8")

    normalized = server.normalize_long_text_profile_for_devices(profile, ["CPU"])

    assert normalized["profile_env"]["native_codegen_device"] == "CPU"
    assert normalized["native_codegen_device_fallback"]["requested"] == "GPU"


def test_long_ar_profile_keeps_gpu_codegen_when_server_uses_gpu():
    profile = server.explicit_long_text_profile("paged-sample-int8")

    normalized = server.normalize_long_text_profile_for_devices(profile, ["GPU"])

    assert normalized["profile_env"]["native_codegen_device"] == "GPU"
    assert "native_codegen_device_fallback" not in normalized


def test_builtin_long_ar_profile_falls_back_to_fp16_paged_seed_without_int8_variant():
    manifest = {
        "graphs": {
            "subcode_greedy_cached": "subcode_greedy_cached.xml",
            "paged_kv_seed": {"talker_stateful_gqa": "talker_stateful_sdpa_paged_gqa_seed.xml"},
        },
        "graph_variants": {},
    }

    profile = server.builtin_long_text_profile_from_manifest(manifest)

    assert profile["profile"] == "long_paged_split_sample_fp16"
    assert profile["runtime"]["graph_variant"] == "fp16"


def test_builtin_long_ar_profile_requires_paged_seed_and_cached_subcode():
    assert server.builtin_long_text_profile_from_manifest({"graphs": {}, "graph_variants": {}}) is None


def test_sidecar_long_text_uses_builtin_paged_sample_profile_without_outputs_summary(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    monkeypatch.chdir(tmp_path)
    ir_dir = tmp_path / "openvino" / "voice_design"
    ir_dir.mkdir(parents=True)
    manifest = {
        "tts_model_type": "voice_design",
        "graphs": {
            "subcode_greedy_cached": "subcode_greedy_cached.xml",
            "paged_kv_seed": {"talker_stateful_gqa": "talker_stateful_sdpa_paged_gqa_seed.xml"},
        },
        "graph_variants": {
            "int8_sym_paged_talker_split": {
                "graphs": {
                    "paged_kv_seed": {
                        "talker_stateful_gqa": "talker_stateful_sdpa_paged_gqa_seed_int8_sym_paged_talker_split.xml"
                    }
                }
            }
        },
    }
    (ir_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    runtime_kwargs = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            runtime_kwargs.append(kwargs)

        def stream_voice_design(self, **kwargs):
            yield SimpleNamespace(
                audio=np.zeros(16, dtype=np.float32),
                sample_rate=24000,
                codes=np.zeros((1, 16), dtype=np.int64),
                is_final=True,
                timings={"stream_rtf": 0.9},
                index=0,
            )

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    app = server.create_app(model_root=tmp_path / "openvino", warmup=False)
    client = fastapi_testclient.TestClient(app)

    with client.websocket_connect("/v1/tts/stream") as websocket:
        websocket.send_json(
            {
                "mode": "voice_design",
                "text": "这是一段长文本，用来验证整理仓库之后不会因为 outputs 目录被清理而退回慢速 reference 路径。",
                "generation": {"max_new_tokens": 160},
                "stream": {"include_chunk_metadata": True},
            }
        )
        assert websocket.receive_json()["type"] == "metadata"
        assert websocket.receive_json()["type"] == "audio"
        websocket.receive_bytes()
        assert websocket.receive_json()["type"] == "final"

    assert runtime_kwargs[0]["graph_variant"] == "int8_sym_paged_talker_split"
    assert runtime_kwargs[0]["mode"] == "no-cache"
    assert runtime_kwargs[0]["native_pipeline"] == "require"
    assert runtime_kwargs[0]["native_paged_kv"] == "require"
    assert runtime_kwargs[0]["native_paged_kv_split_subcode"] == "1"


def test_auto_long_text_profile_only_applies_sampled_paged_winners(monkeypatch):
    assert server.should_auto_apply_long_text_profile({"profile": "long_paged_split_sample_int8_sym"})
    assert not server.should_auto_apply_long_text_profile({"profile": "long_reference_no_cache_fp16_sample"})
    monkeypatch.setenv(server.USE_LONG_TEXT_QUALITY_PROFILE_ENV, "1")
    assert server.should_auto_apply_long_text_profile({"profile": "long_reference_no_cache_fp16_sample"})


def test_long_voice_design_defaults_to_sampling():
    request = {
        "mode": "voice_design",
        "text": "这是一段比较长的中文文本，用来确认长文本默认跟随上游采样式生成，而不是贪心生成。",
        "generation": {"max_new_tokens": 128},
    }

    kwargs = server.generation_kwargs(request)

    assert kwargs["do_sample"] is True
    assert kwargs["repetition_penalty"] == 1.05


def test_full_context_long_text_uses_conservative_frame_budget():
    text = "据央视网昨日消息，白宫公布了将随特朗普一同访华的商界领袖名单。" * 4
    request = {
        "mode": "voice_design",
        "text": text,
        "full_context_text": True,
        "generation": {"max_new_tokens": 48},
    }

    kwargs = server.generation_kwargs(request)

    assert kwargs["max_new_tokens"] >= server.speech_text_unit_count(text) * 4
    assert kwargs["max_new_tokens"] <= 2048


def test_explicit_do_sample_false_is_respected_for_long_text():
    request = {
        "mode": "voice_design",
        "text": "这是一段比较长的中文文本，用来确认用户仍然可以显式关闭采样。",
        "generation": {"max_new_tokens": 128, "do_sample": False},
    }

    assert server.generation_kwargs(request)["do_sample"] is False

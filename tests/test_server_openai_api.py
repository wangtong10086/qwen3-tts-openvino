import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from qwen3_tts_ov.server import (
    custom_voice_online_batch_supported,
    generation_kwargs,
    include_chunk_metadata,
    needs_continuous_long_output,
    online_batch_prompt_family_supported,
    openai_speech_to_tts_request,
    playback_buffer_for_stream,
    request_voice_clone_prompt_is_xvector_only,
    request_x_vector_only,
    select_continuous_long_output_variant,
    effective_forced_stream_strategy,
    stream_metadata,
    stream_kwargs,
)


def test_openai_speech_maps_custom_voice_request():
    internal, response_format, stream_enabled = openai_speech_to_tts_request(
        {
            "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "voice": "Vivian",
            "input": "Hello",
            "language": "English",
            "instructions": "Speak brightly.",
            "stream": True,
            "response_format": "pcm",
            "chunk_strategy": "stable",
            "max_new_tokens": 32,
        }
    )

    assert response_format == "pcm"
    assert stream_enabled is True
    assert internal["mode"] == "custom_voice"
    assert internal["speaker"] == "Vivian"
    assert internal["instruct"] == "Speak brightly."
    assert internal["generation"]["max_new_tokens"] == 32
    assert internal["stream"]["chunk_strategy"] == "stable"


def test_voice_clone_x_vector_only_defaults_false_and_parses_false_strings():
    internal, _, _ = openai_speech_to_tts_request(
        {
            "model": "Qwen/Qwen3-TTS-12Hz-1.7B",
            "input": "Hello",
            "task_type": "voice_clone",
            "ref_audio": "reference.wav",
            "ref_text": "Reference text.",
        }
    )

    assert internal["mode"] == "voice_clone"
    assert internal["x_vector_only"] is False
    assert request_x_vector_only({"x_vector_only": "false"}) is False
    assert request_x_vector_only({"x_vector_only_mode": "0"}) is False
    assert request_x_vector_only({"x_vector_only": "true"}) is True


def test_voice_clone_prompt_without_ref_code_is_treated_as_xvector_only_for_online_gate():
    assert request_voice_clone_prompt_is_xvector_only({"voice_clone_prompt": {"ref_spk_embedding": [[0.0, 1.0]]}})
    assert request_voice_clone_prompt_is_xvector_only(
        {"voice_clone_prompt": {"ref_spk_embedding": [[0.0, 1.0]], "x_vector_only_mode": [True]}}
    )
    assert request_voice_clone_prompt_is_xvector_only(
        {"voice_clone_prompt": {"ref_spk_embedding": [0.0, 1.0], "x_vector_only_mode": False, "icl_mode": True}}
    )
    assert request_voice_clone_prompt_is_xvector_only(
        {"voice_clone_prompt": {"ref_spk_embedding": [[0.0, 1.0]], "icl_mode": [False]}}
    )
    assert not request_voice_clone_prompt_is_xvector_only(
        {"voice_clone_prompt": {"ref_spk_embedding": [[0.0, 1.0]], "ref_code": [[[1] * 16]]}}
    )
    assert not request_voice_clone_prompt_is_xvector_only(
        {
            "voice_clone_prompt": {
                "ref_spk_embedding": [0.0, 1.0],
                "ref_code": [[1] * 16, [2] * 16],
                "x_vector_only_mode": False,
                "icl_mode": True,
            }
        }
    )


def test_stream_metadata_uses_strategy_defaults():
    metadata = stream_metadata({"stream": {"chunk_strategy": "low_latency"}})

    assert metadata["chunk_strategy"] == "low_latency"
    assert metadata["initial_chunk_frames"] == 8
    assert metadata["chunk_frames"] == 12
    assert metadata["left_context_frames"] == 25


def test_smooth_stream_metadata_and_playback_buffer_floor():
    metadata = stream_metadata({"stream": {"chunk_strategy": "smooth"}})

    assert metadata["chunk_strategy"] == "smooth"
    assert metadata["initial_chunk_frames"] == 8
    assert metadata["chunk_frames"] == 24
    assert playback_buffer_for_stream(metadata, 250) == 1900


def test_realtime_stream_metadata_uses_low_latency_chunks_with_larger_buffer():
    metadata = stream_metadata({"stream": {"chunk_strategy": "realtime"}})

    assert metadata["chunk_strategy"] == "realtime"
    assert metadata["initial_chunk_frames"] == 8
    assert metadata["chunk_frames"] == 12
    assert playback_buffer_for_stream(metadata, 250) == 1900


def test_auto_stream_strategy_uses_short_compute_for_short_requests():
    request = {"text": "你好，短句。", "stream": {"chunk_strategy": "auto"}}

    metadata = stream_metadata(request)
    kwargs = stream_kwargs(request)

    assert metadata["chunk_strategy"] == "short_compute"
    assert metadata["chunk_strategy_requested"] == "auto"
    assert metadata["auto_chunk_strategy"] is True
    assert metadata["initial_chunk_frames"] == 12
    assert metadata["chunk_frames"] == 24
    assert kwargs["chunk_strategy"] == "short_compute"
    assert kwargs["initial_chunk_frames"] == 12
    assert kwargs["chunk_frames"] == 24


def test_auto_stream_strategy_preserves_full_context_long_requests():
    request = {"text": "这是一段较长文本。" * 10, "full_context_text": True, "stream": {"chunk_strategy": "auto"}}

    metadata = stream_metadata(request)

    assert metadata["chunk_strategy"] == "stable"
    assert metadata["chunk_strategy_requested"] == "auto"
    assert metadata["auto_chunk_strategy"] is True


def test_explicit_chunk_strategy_overrides_fastest_forced_default():
    assert effective_forced_stream_strategy({"stream": {"chunk_strategy": "auto"}}, "smooth") is None
    assert effective_forced_stream_strategy({"stream": {"format": "pcm_s16le"}}, "smooth") == "smooth"


def test_forced_stream_metadata_ignores_requested_realtime_strategy():
    metadata = stream_metadata(
        {
            "stream": {
                "chunk_strategy": "realtime",
                "initial_chunk_frames": 8,
                "chunk_frames": 12,
                "left_context_frames": 25,
            }
        },
        default_strategy="smooth",
        forced_strategy="smooth",
    )

    assert metadata["chunk_strategy"] == "smooth"
    assert metadata["initial_chunk_frames"] == 8
    assert metadata["chunk_frames"] == 24
    assert metadata["left_context_frames"] == 25
    assert metadata["forced_chunk_strategy"] is True


def test_fastest_long_text_uses_continuous_single_prompt_policy():
    text = "你好，这是第一段较长的测试文本，用来验证自动切分。这里还有第二段内容，确保不会因为最大生成帧数太小而截断。"

    assert needs_continuous_long_output(text, max_new_tokens=48) is True
    assert needs_continuous_long_output("你好", max_new_tokens=48) is False
    assert needs_continuous_long_output("你好", max_new_tokens=160) is True


def test_long_custom_voice_defaults_to_stronger_repetition_penalty():
    request = {
        "mode": "custom_voice",
        "text": "这是一段较长的自定义音色测试文本，用来验证长输出时不会进入重复或者静音循环。",
        "generation": {"max_new_tokens": 960},
    }

    kwargs = generation_kwargs(request, default_repetition_penalty=1.0, allow_sampled_defaults=False)

    assert kwargs["do_sample"] is True
    assert kwargs["repetition_penalty"] == 1.2

    request["generation"]["repetition_penalty"] = 1.05
    explicit_kwargs = generation_kwargs(request, default_repetition_penalty=1.0, allow_sampled_defaults=False)
    assert explicit_kwargs["repetition_penalty"] == 1.05


def test_sampled_custom_voice_medium_english_defaults_to_stronger_repetition_penalty():
    request = {
        "mode": "custom_voice",
        "text": (
            "This custom voice validation sentence should be spoken completely in English "
            "with a stable Ryan style and without falling into a repeated low-energy loop."
        ),
        "generation": {"max_new_tokens": 220, "do_sample": True},
    }

    kwargs = generation_kwargs(request, default_repetition_penalty=1.05, allow_sampled_defaults=True)

    assert kwargs["do_sample"] is True
    assert kwargs["repetition_penalty"] == 1.2

    request["generation"]["repetition_penalty"] = 1.05
    explicit_kwargs = generation_kwargs(request, default_repetition_penalty=1.05, allow_sampled_defaults=True)
    assert explicit_kwargs["repetition_penalty"] == 1.05


def test_custom_voice_online_batch_gate_is_validated_subset_only():
    assert custom_voice_online_batch_supported({"speaker": "Vivian", "language": "Chinese"}) == (
        True,
        "custom_voice_online_batch_validated",
    )

    ok, reason = custom_voice_online_batch_supported({"speaker": "Ryan", "language": "English"})
    assert ok is False
    assert reason == "custom_voice_online_batch_requires_repetition_penalty:ryan:english:1.2"

    assert custom_voice_online_batch_supported(
        {"speaker": "Ryan", "language": "English"},
        {"do_sample": True, "repetition_penalty": 1.2},
    ) == (
        True,
        "custom_voice_online_batch_validated_rp12",
    )


def test_online_batch_prompt_family_gate_allows_public_modes_without_quality_subset_fallback():
    assert online_batch_prompt_family_supported(
        {"mode": "voice_design", "text": "short"},
        "voice_design",
    ) == (True, "vllm_like_voice_design_supported")

    assert online_batch_prompt_family_supported(
        {"speaker": "Vivian", "language": "Chinese"},
        "custom_voice",
    ) == (True, "vllm_like_custom_voice_supported")

    assert online_batch_prompt_family_supported(
        {"speaker": "Ryan", "language": "English"},
        "custom_voice",
    ) == (True, "vllm_like_custom_voice_supported")

    assert online_batch_prompt_family_supported(
        {"speaker": "Ryan", "language": "English"},
        "custom_voice",
        gen_kwargs={"do_sample": True, "repetition_penalty": 1.2},
    ) == (True, "vllm_like_custom_voice_supported")

    assert online_batch_prompt_family_supported(
        {"x_vector_only": True},
        "voice_clone",
        voice_clone_xvector_env=None,
    ) == (True, "vllm_like_voice_clone_xvector_supported")

    assert online_batch_prompt_family_supported(
        {"x_vector_only": True},
        "voice_clone",
        voice_clone_xvector_env="1",
    ) == (True, "vllm_like_voice_clone_xvector_supported")
    assert online_batch_prompt_family_supported(
        {"voice_clone_prompt": {"ref_spk_embedding": [[0.0, 1.0]]}},
        "voice_clone",
        voice_clone_xvector_env=None,
    ) == (True, "vllm_like_voice_clone_xvector_supported")

    assert online_batch_prompt_family_supported(
        {"full_context_text": True},
        "voice_design",
    ) == (True, "vllm_like_voice_design_supported")
    assert online_batch_prompt_family_supported(
        {"_prefix_codes": np.zeros((2, 16), dtype=np.int64)},
        "voice_design",
    ) == (False, "prefix_codes_not_supported")


def test_continuous_long_output_prefers_cachedsub_variant_when_bucket_available():
    manifest = {
        "graph_variants": {
            "int8_sym_fused": {"graphs": {"fused_cache_step_buckets": {"exact": {"384": "fallback.xml"}}}},
            "int8_sym_fused_cachedsub": {
                "graphs": {"fused_cache_step_buckets": {"exact": {"384": "cachedsub.xml"}}}
            },
        }
    }

    assert select_continuous_long_output_variant(manifest) == "int8_sym_fused_cachedsub"


def test_include_chunk_metadata_accepts_stream_flag():
    assert include_chunk_metadata({"stream": {"include_chunk_metadata": True}}) is True
    assert include_chunk_metadata({"include_chunk_metadata": True}) is True
    assert include_chunk_metadata({"stream": {}}) is False


def test_server_default_model_root_falls_back_to_openvino_full(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from qwen3_tts_ov import server

    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "openvino_full"
    legacy.mkdir()
    with open(legacy / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"tts_model_type": "voice_design"}, handle)

    seen = {}

    class FakeRuntime:
        def __init__(self, ir_dir, *args, **kwargs):
            seen["ir_dir"] = str(ir_dir)

        def generate_voice_design(self, **kwargs):
            return [np.zeros(16, dtype=np.float32)], 24000

        def stream_voice_design(self, **kwargs):
            yield SimpleNamespace(
                audio=np.zeros(16, dtype=np.float32),
                sample_rate=24000,
                codes=np.zeros((1, 16), dtype=np.int64),
                is_final=True,
                timings={},
                index=0,
            )

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    app = server.create_app(
        model_root="openvino",
        warmup=False,
    )
    client = fastapi_testclient.TestClient(app)

    response = client.post("/v1/tts", json={"mode": "voice_design", "text": "hello", "online_batching": False})

    assert response.status_code == 400
    assert "online batching is the only sidecar generation backend" in response.json()["detail"]
    assert seen["ir_dir"] == "openvino_full"


def test_server_auto_model_root_resolves_voice_design_legacy_dir(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from qwen3_tts_ov import server

    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "openvino_full"
    legacy.mkdir()
    with open(legacy / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"tts_model_type": "voice_design"}, handle)

    seen = {}

    class FakeRuntime:
        def __init__(self, ir_dir, *args, **kwargs):
            seen["ir_dir"] = str(ir_dir)

        def generate_voice_design(self, **kwargs):
            return [np.zeros(16, dtype=np.float32)], 24000

        def stream_voice_design(self, **kwargs):
            yield SimpleNamespace(
                audio=np.zeros(16, dtype=np.float32),
                sample_rate=24000,
                codes=np.zeros((1, 16), dtype=np.int64),
                is_final=True,
                timings={},
                index=0,
            )

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    app = server.create_app(
        model_root="auto",
        warmup=False,
    )
    client = fastapi_testclient.TestClient(app)

    response = client.post("/v1/tts", json={"mode": "voice_design", "text": "hello", "online_batching": False})

    assert response.status_code == 400
    assert "online batching is the only sidecar generation backend" in response.json()["detail"]
    assert seen["ir_dir"] == "openvino_full"


def test_websocket_can_send_optional_chunk_metadata(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from qwen3_tts_ov import server

    ir_dir = tmp_path / "openvino" / "voice_design"
    ir_dir.mkdir(parents=True)
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"tts_model_type": "voice_design"}, handle)

    class FakeRuntime:
        def __init__(self, ir_dir, *args, **kwargs):
            self.ir_dir = ir_dir
            self.manifest = {"tts_model_type": "voice_design"}
            self.device = "GPU"
            self.cache_dir = ""
            self.disable_ov_cache = True
            self.compile_config = {}

        def build_prompt(self, **kwargs):
            return np.zeros((1, 4, 8), dtype=np.float32), np.zeros((1, 1, 8), dtype=np.float32)

        def stream_decode_codes(self, codes, **kwargs):
            for index, code in enumerate(codes):
                yield SimpleNamespace(
                    audio=np.zeros(16, dtype=np.float32),
                    sample_rate=24000,
                    codes=np.asarray([code], dtype=np.int64),
                    is_final=True,
                    timings={"rtf": 1.2, "stream_rtf": 0.8},
                    index=index,
                )

    class FakeScheduler:
        def __init__(self, runtime, config=None):
            pass

        def ensure_ready(self):
            pass

        def close(self):
            pass

        def submit(self, *args, **kwargs):
            return iter([np.arange(16, dtype=np.int64)])

        def stats(self):
            return {"active": 1, "pending": 0}

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    monkeypatch.setattr(server, "OnlineBatchScheduler", FakeScheduler)
    monkeypatch.setattr(
        server,
        "online_batch_graph_capability",
        lambda *args, **kwargs: {"ok": True, "reason": "ready", "sampled_batch_subcode_effective": "on"},
    )
    app = server.create_app(
        model_root=tmp_path / "openvino",
        warmup=False,
    )
    client = fastapi_testclient.TestClient(app)

    with client.websocket_connect("/v1/tts/stream") as websocket:
        websocket.send_json(
            {
                "mode": "voice_design",
                "text": "hello",
                "stream": {"format": "pcm_s16le", "include_chunk_metadata": True},
            }
        )
        metadata = websocket.receive_json()
        assert metadata["type"] == "metadata"
        assert metadata["realtime_profile"] == "fastest"
        assert metadata["production_profile"] == "minimal-online-paged-kv"
        assert metadata["graph_variant"] == "int8_sym_paged_talker_split"
        audio_meta = websocket.receive_json()
        assert audio_meta["type"] == "audio"
        assert audio_meta["timings"]["stream_rtf"] == 0.8
        assert audio_meta["timings"]["online_batching"] is True
        assert len(websocket.receive_bytes()) == 32
        assert websocket.receive_json()["type"] == "final"


def test_websocket_default_online_batching_hard_errors_without_graphs(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from qwen3_tts_ov import server

    ir_dir = tmp_path / "openvino" / "voice_design"
    ir_dir.mkdir(parents=True)
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"tts_model_type": "voice_design", "graphs": {}, "graph_variants": {}}, handle)

    class FakeRuntime:
        def __init__(self, ir_dir, *args, **kwargs):
            self.ir_dir = ir_dir
            self.manifest = {"tts_model_type": "voice_design", "graphs": {}, "graph_variants": {}}

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    app = server.create_app(model_root=tmp_path / "openvino", warmup=False)
    client = fastapi_testclient.TestClient(app)

    with client.websocket_connect("/v1/tts/stream") as websocket:
        websocket.send_json(
            {
                "mode": "voice_design",
                "text": "hello",
                "stream": {"format": "pcm_s16le", "include_chunk_metadata": True},
            }
        )
        metadata = websocket.receive_json()
        assert metadata["type"] == "metadata"
        assert metadata["online_batching"] == "on"
        assert metadata["generation_fallback_allowed"] is False
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert "online_batch_graphs_unavailable" in error["message"]


def test_create_app_rejects_legacy_online_batch_scheduler(tmp_path):
    from qwen3_tts_ov import server

    with pytest.raises(ValueError, match="fixed to layered"):
        server.create_app(
            model_root=tmp_path / "openvino",
            warmup=False,
            online_batch_scheduler="legacy",
        )


def test_websocket_custom_voice_rejects_disabled_online_batching(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from qwen3_tts_ov import server

    monkeypatch.setenv("QWEN3_TTS_OV_ONLINE_BATCHING_QUALITY", "1")
    ir_dir = tmp_path / "openvino" / "custom_voice"
    ir_dir.mkdir(parents=True)
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"tts_model_type": "custom_voice"}, handle)

    calls = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            pass

        def stream_custom_voice(self, **kwargs):
            calls.append(kwargs)
            yield SimpleNamespace(
                audio=np.zeros(16, dtype=np.float32),
                sample_rate=24000,
                codes=np.zeros((1, 16), dtype=np.int64),
                is_final=True,
                timings={},
                index=0,
            )

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    app = server.create_app(
        model_root=tmp_path / "openvino",
        warmup=False,
    )
    client = fastapi_testclient.TestClient(app)

    with client.websocket_connect("/v1/tts/stream") as websocket:
        websocket.send_json(
            {
                "mode": "custom_voice",
                "text": "This custom voice prompt must stay on the online scheduler.",
                "speaker": "Ryan",
                "language": "English",
                "online_batching": False,
                "generation": {"repetition_penalty": 1.05},
                "stream": {"format": "pcm_s16le", "include_chunk_metadata": True},
            }
        )
        assert websocket.receive_json()["type"] == "metadata"
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert "online batching is the only sidecar generation backend" in error["message"]

    assert calls == []


def test_websocket_custom_voice_ryan_english_rp12_uses_online_batching(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from qwen3_tts_ov import server

    monkeypatch.setenv("QWEN3_TTS_OV_ONLINE_BATCHING_QUALITY", "1")
    ir_dir = tmp_path / "openvino" / "custom_voice"
    ir_dir.mkdir(parents=True)
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"tts_model_type": "custom_voice"}, handle)

    observed = {}

    class FakeRuntime:
        def __init__(self, ir_dir, *args, **kwargs):
            self.ir_dir = ir_dir
            self.manifest = {"tts_model_type": "custom_voice"}
            self.device = "GPU"
            self.cache_dir = ""
            self.disable_ov_cache = True
            self.compile_config = {}
            self.num_code_groups = 16
            self.ids = {"vocab_size": 1024, "codec_eos_token_id": 0}

        def build_prompt(self, **kwargs):
            observed["build_prompt"] = kwargs
            return np.zeros((1, 4, 8), dtype=np.float32), np.zeros((1, 1, 8), dtype=np.float32)

        def stream_decode_codes(self, codes, **kwargs):
            observed["prefix_codes"] = kwargs["prefix_codes"]
            for index, code in enumerate(codes):
                yield SimpleNamespace(
                    audio=np.zeros(16, dtype=np.float32),
                    sample_rate=24000,
                    codes=np.asarray([code], dtype=np.int64),
                    is_final=True,
                    timings={},
                    index=index,
                )

    class FakeScheduler:
        def __init__(self, runtime, config=None):
            observed["scheduler_config"] = config

        def ensure_ready(self):
            observed["scheduler_ready"] = True

        def close(self):
            observed["scheduler_closed"] = True

        def submit(self, *args, **kwargs):
            observed["submit_kwargs"] = kwargs
            return iter([np.arange(16, dtype=np.int64)])

        def stats(self):
            return {"active": 1, "pending": 0}

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    monkeypatch.setattr(server, "OnlineBatchScheduler", FakeScheduler)
    monkeypatch.setattr(
        server,
        "online_batch_graph_capability",
        lambda *args, **kwargs: {
            "ok": True,
            "reason": "ready",
            "sampled_batch_subcode_effective": "on",
        },
    )
    app = server.create_app(model_root=tmp_path / "openvino", warmup=False, online_batching="auto")
    client = fastapi_testclient.TestClient(app)

    with client.websocket_connect("/v1/tts/stream") as websocket:
        websocket.send_json(
            {
                "mode": "custom_voice",
                "text": "This custom voice validation sentence should stay in Ryan's English voice.",
                "speaker": "Ryan",
                "language": "English",
                "generation": {"repetition_penalty": 1.2, "do_sample": True, "max_new_tokens": 4},
                "stream": {"format": "pcm_s16le", "include_chunk_metadata": True},
            }
        )
        assert websocket.receive_json()["type"] == "metadata"
        audio_meta = websocket.receive_json()
        assert audio_meta["type"] == "audio"
        assert audio_meta["timings"]["online_batching"] is True
        assert audio_meta["timings"]["online_batching_reason"].startswith("enabled:off:ready")
        assert audio_meta["timings"]["online_batch_continuous_subcode"] is False
        assert len(websocket.receive_bytes()) == 32
        assert websocket.receive_json()["type"] == "final"

    assert observed["build_prompt"]["speaker"] == "Ryan"
    assert observed["build_prompt"]["language"] == "English"
    assert observed["prefix_codes"] is None
    assert observed["submit_kwargs"]["repetition_penalty"] == 1.2
    assert observed["scheduler_config"].continuous_batch_subcode is False


def test_websocket_voice_clone_xvector_rejects_disabled_online_batching(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from qwen3_tts_ov import server

    monkeypatch.setenv("QWEN3_TTS_OV_ONLINE_BATCHING_QUALITY", "1")
    monkeypatch.delenv("QWEN3_TTS_OV_ENABLE_VOICE_CLONE_XVECTOR_ONLINE_BATCHING", raising=False)
    ir_dir = tmp_path / "openvino" / "base"
    ir_dir.mkdir(parents=True)
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"tts_model_type": "base"}, handle)

    calls = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            pass

        def stream_voice_clone(self, **kwargs):
            calls.append(kwargs)
            yield SimpleNamespace(
                audio=np.zeros(16, dtype=np.float32),
                sample_rate=24000,
                codes=np.zeros((1, 16), dtype=np.int64),
                is_final=True,
                timings={},
                index=0,
            )

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    app = server.create_app(
        model_root=tmp_path / "openvino",
        warmup=False,
    )
    client = fastapi_testclient.TestClient(app)

    with client.websocket_connect("/v1/tts/stream") as websocket:
        websocket.send_json(
            {
                "mode": "voice_clone",
                "text": "Voice clone x-vector should stay on the online scheduler.",
                "language": "English",
                "ref_audio": "reference.wav",
                "x_vector_only": True,
                "online_batching": False,
                "stream": {"format": "pcm_s16le", "include_chunk_metadata": True},
            }
        )
        assert websocket.receive_json()["type"] == "metadata"
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert "online batching is the only sidecar generation backend" in error["message"]

    assert calls == []


def test_websocket_voice_clone_prompt_rejects_disabled_online_batching(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from qwen3_tts_ov import server

    monkeypatch.setenv("QWEN3_TTS_OV_ONLINE_BATCHING_QUALITY", "1")
    ir_dir = tmp_path / "openvino" / "base"
    ir_dir.mkdir(parents=True)
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"tts_model_type": "base"}, handle)

    prompt = {"ref_spk_embedding": [[0.0, 1.0]], "x_vector_only_mode": [True], "icl_mode": [False]}
    calls = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            pass

        def stream_voice_clone(self, **kwargs):
            calls.append(kwargs)
            yield SimpleNamespace(
                audio=np.zeros(16, dtype=np.float32),
                sample_rate=24000,
                codes=np.zeros((1, 16), dtype=np.int64),
                is_final=True,
                timings={},
                index=0,
            )

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    app = server.create_app(
        model_root=tmp_path / "openvino",
        warmup=False,
    )
    client = fastapi_testclient.TestClient(app)

    with client.websocket_connect("/v1/tts/stream") as websocket:
        websocket.send_json(
            {
                "mode": "voice_clone",
                "text": "Prompt reuse should not require ref_audio.",
                "language": "English",
                "voice_clone_prompt": prompt,
                "online_batching": False,
                "stream": {"format": "pcm_s16le", "include_chunk_metadata": True},
            }
        )
        assert websocket.receive_json()["type"] == "metadata"
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert "online batching is the only sidecar generation backend" in error["message"]

    assert calls == []


def test_websocket_voice_clone_prompt_natural_json_is_forwarded_and_normalizable(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from qwen3_tts_ov import server
    from qwen3_tts_ov.runtime import OpenVINOQwen3TTS

    ir_dir = tmp_path / "openvino" / "base"
    ir_dir.mkdir(parents=True)
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"tts_model_type": "base"}, handle)

    prompt = {
        "ref_spk_embedding": [0.0, 1.0],
        "ref_code": [[1] * 16, [2] * 16],
        "x_vector_only_mode": False,
        "icl_mode": True,
        "ref_text": "Reference text.",
    }
    normalized = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            pass

        def stream_voice_clone(self, **kwargs):
            item = OpenVINOQwen3TTS._normalize_voice_clone_prompt(
                SimpleNamespace(num_code_groups=16),
                kwargs["voice_clone_prompt"],
                1,
            )[0]
            normalized.append(item)
            assert kwargs["ref_audio"] is None
            yield SimpleNamespace(
                audio=np.zeros(16, dtype=np.float32),
                sample_rate=24000,
                codes=np.zeros((1, 16), dtype=np.int64),
                is_final=True,
                timings={},
                index=0,
            )

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    app = server.create_app(
        model_root=tmp_path / "openvino",
        warmup=False,
    )
    client = fastapi_testclient.TestClient(app)

    with client.websocket_connect("/v1/tts/stream") as websocket:
        websocket.send_json(
            {
                "mode": "voice_clone",
                "text": "Natural JSON prompt reuse should keep ICL fields.",
                "language": "English",
                "voice_clone_prompt": prompt,
                "online_batching": False,
                "stream": {"format": "pcm_s16le", "include_chunk_metadata": True},
            }
        )
        assert websocket.receive_json()["type"] == "metadata"
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert "online batching is the only sidecar generation backend" in error["message"]

    assert normalized == []


def test_websocket_voice_clone_prompt_natural_json_online_batch_uses_ref_code_prefix(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from qwen3_tts_ov import server
    from qwen3_tts_ov.runtime import OpenVINOQwen3TTS

    monkeypatch.setenv("QWEN3_TTS_OV_ONLINE_BATCHING_QUALITY", "1")
    ir_dir = tmp_path / "openvino" / "base"
    ir_dir.mkdir(parents=True)
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"tts_model_type": "base"}, handle)

    prompt = {
        "ref_spk_embedding": [0.0, 1.0],
        "ref_code": [[1] * 16, [2] * 16],
        "x_vector_only_mode": False,
        "icl_mode": True,
        "ref_text": "Reference text.",
    }
    observed = {}

    class FakeRuntime:
        def __init__(self, ir_dir, *args, **kwargs):
            self.ir_dir = ir_dir
            self.manifest = {"tts_model_type": "base"}
            self.device = "GPU"
            self.cache_dir = ""
            self.disable_ov_cache = True
            self.compile_config = {}
            self.num_code_groups = 16

        def _normalize_voice_clone_prompt(self, prompt, text_count):
            return OpenVINOQwen3TTS._normalize_voice_clone_prompt(self, prompt, text_count)

        def build_prompt(self, **kwargs):
            observed["build_prompt"] = kwargs
            assert kwargs["voice_clone_prompt"].ref_text == "Reference text."
            assert kwargs["voice_clone_prompt"].ref_code.shape == (2, 16)
            assert kwargs["ref_text"] == "Reference text."
            return np.zeros((1, 4, 8), dtype=np.float32), np.zeros((1, 1, 8), dtype=np.float32)

        def stream_decode_codes(self, codes, **kwargs):
            observed["prefix_codes"] = kwargs["prefix_codes"]
            for index, code in enumerate(codes):
                yield SimpleNamespace(
                    audio=np.zeros(16, dtype=np.float32),
                    sample_rate=24000,
                    codes=np.asarray([code], dtype=np.int64),
                    is_final=True,
                    timings={},
                    index=index,
                )

    class FakeScheduler:
        def __init__(self, runtime, config=None):
            observed["scheduler_config"] = config

        def ensure_ready(self):
            observed["scheduler_ready"] = True

        def close(self):
            observed["scheduler_closed"] = True

        def submit(self, *args, **kwargs):
            observed["submit_kwargs"] = kwargs
            return iter([np.arange(16, dtype=np.int64)])

        def stats(self):
            return {"active": 1, "pending": 0}

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    monkeypatch.setattr(server, "OnlineBatchScheduler", FakeScheduler)
    monkeypatch.setattr(
        server,
        "online_batch_graph_capability",
        lambda *args, **kwargs: {
            "ok": True,
            "reason": "ready",
            "sampled_batch_subcode_effective": "on",
        },
    )
    app = server.create_app(
        model_root=tmp_path / "openvino",
        warmup=False,
        online_batching="auto",
        sampled_batch_subcode="auto",
        online_batch_continuous_subcode="auto",
    )
    client = fastapi_testclient.TestClient(app)

    with client.websocket_connect("/v1/tts/stream") as websocket:
        websocket.send_json(
            {
                "mode": "voice_clone",
                "text": "Natural JSON prompt should use online ICL prefix.",
                "language": "English",
                "voice_clone_prompt": prompt,
                "stream": {"format": "pcm_s16le", "include_chunk_metadata": True},
                "generation": {"max_new_tokens": 4, "do_sample": True},
            }
        )
        assert websocket.receive_json()["type"] == "metadata"
        audio_meta = websocket.receive_json()
        assert audio_meta["type"] == "audio"
        assert audio_meta["timings"]["online_batching"] is True
        assert audio_meta["timings"]["online_batch_continuous_subcode"] is True
        assert audio_meta["timings"]["online_batch_continuous_subcode_reason"] == "auto:sampled_batch_subcode_on"
        assert len(websocket.receive_bytes()) == 32
        assert websocket.receive_json()["type"] == "final"

    assert observed["prefix_codes"].shape == (2, 16)
    assert observed["prefix_codes"][0, 0] == 1
    assert observed["prefix_codes"][1, 0] == 2
    assert observed["submit_kwargs"]["do_sample"] is True
    assert observed["scheduler_config"].continuous_batch_subcode is True
    assert observed["scheduler_config"].max_num_batched_tokens == 32


def test_http_voice_clone_prompt_rejects_disabled_online_batching(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from qwen3_tts_ov import server

    ir_dir = tmp_path / "openvino" / "base"
    ir_dir.mkdir(parents=True)
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"tts_model_type": "base"}, handle)

    prompt = {"ref_spk_embedding": [[0.0, 1.0]], "x_vector_only_mode": [True], "icl_mode": [False]}
    calls = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            pass

        def generate_voice_clone(self, **kwargs):
            calls.append(kwargs)
            return [np.zeros(16, dtype=np.float32)], 24000

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    app = server.create_app(
        model_root=tmp_path / "openvino",
        warmup=False,
    )
    client = fastapi_testclient.TestClient(app)

    response = client.post(
        "/v1/tts",
        json={
            "mode": "voice_clone",
            "text": "Prompt reuse should work for full-audio HTTP too.",
            "language": "English",
            "voice_clone_prompt": prompt,
            "online_batching": False,
        },
    )

    assert response.status_code == 400
    assert "online batching is the only sidecar generation backend" in response.json()["detail"]
    assert calls == []


def test_fastest_websocket_long_output_rejects_disabled_online_batching(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from qwen3_tts_ov import server

    ir_dir = tmp_path / "openvino" / "voice_design"
    ir_dir.mkdir(parents=True)
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"tts_model_type": "voice_design"}, handle)

    calls = []
    runtime_kwargs = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            runtime_kwargs.append(kwargs)

        def stream_voice_design(self, **kwargs):
            calls.append(kwargs)
            yield SimpleNamespace(
                audio=np.ones(16, dtype=np.float32),
                sample_rate=24000,
                codes=np.zeros((1, 16), dtype=np.int64),
                is_final=True,
                timings={"stream_rtf": 0.8},
                index=0,
            )

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    app = server.create_app(
        model_root=tmp_path / "openvino",
        warmup=False,
    )
    client = fastapi_testclient.TestClient(app)
    text = "你好，这是第一段较长的测试文本，用来验证自动切分。这里还有第二段内容，确保不会因为最大生成帧数太小而截断。"

    with client.websocket_connect("/v1/tts/stream") as websocket:
        websocket.send_json(
            {
                "mode": "voice_design",
                "text": text,
                "online_batching": False,
                "stream": {"format": "pcm_s16le", "include_chunk_metadata": True},
                "generation": {"max_new_tokens": 160},
            }
        )
        metadata = websocket.receive_json()
        assert metadata["forced_chunk_strategy"] is True
        assert metadata["continuous_long_output"] is True
        assert metadata["continuous_backend"] == "vllm_like_online_scheduler"
        assert metadata["continuous_bucket"] == 384
        assert metadata["paged_kv"] is False

        audio_messages = 0
        while True:
            message = websocket.receive_json()
            assert message["type"] == "error"
            assert "online batching is the only sidecar generation backend" in message["message"]
            break

    assert audio_messages == 0
    assert calls == []
    assert runtime_kwargs[0]["graph_variant"] == "fp16"
    assert runtime_kwargs[0]["codegen_unroll"] == 1
    assert runtime_kwargs[0]["preferred_cache_bucket"] == 0
    assert runtime_kwargs[0]["native_pipeline"] == "off"


def test_server_fastest_profile_reaches_runtime_and_health(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from qwen3_tts_ov import server

    ir_dir = tmp_path / "openvino" / "voice_design"
    ir_dir.mkdir(parents=True)
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"tts_model_type": "voice_design"}, handle)

    seen = {}

    class FakeRuntime:
        def __init__(self, ir_dir, *args, **kwargs):
            seen["kwargs"] = kwargs
            self.mode = kwargs["mode"]
            self.requested_mode = kwargs["mode"]
            self.cache_kernel = kwargs["cache_kernel"]
            self.cache_step = kwargs["cache_step"]
            self.graph_variant = kwargs["graph_variant"]
            self.codegen_unroll = kwargs["codegen_unroll"]
            self.codegen_schedule = kwargs.get("codegen_schedule", "current")
            self.codegen_unroll_fallback = False
            self.variant_graphs = {"paged_kv_seed": {"talker_stateful_gqa": "talker_stateful.xml"}}
            self.cache_dir = None
            self.ov_cache_mode = "OPTIMIZE_SPEED"
            self.disable_ov_cache = False
            self.streaming_decoder_graphs_by_context = {}
            self.streaming_decoders = {}
            self.fused_cache_step_by_bucket = {}
            self.fused_cache_bucket_graphs = {}
            self.fused_cache_unroll_bucket_graphs = {}
            self.talker_stateful_by_bucket = {}

        def generate_voice_design(self, **kwargs):
            return [np.zeros(16, dtype=np.float32)], 24000

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    app = server.create_app(
        model_root=tmp_path / "openvino",
        warmup=False,
        realtime_profile="fastest",
    )
    client = fastapi_testclient.TestClient(app)

    assert client.post(
        "/v1/tts",
        json={
            "mode": "voice_design",
            "text": "hello",
            "online_batching": False,
            "generation": {"max_new_tokens": 48, "do_sample": False},
        },
    ).status_code == 400
    health = client.get("/health").json()
    runtime_status = next(iter(health["runtimes"].values()))

    assert seen["kwargs"]["mode"] == "no-cache"
    assert seen["kwargs"]["cache_kernel"] == "exact"
    assert seen["kwargs"]["cache_step"] == "fused"
    assert seen["kwargs"]["graph_variant"] == "int8_sym_paged_talker_split"
    assert seen["kwargs"]["codegen_unroll"] == 1
    assert health["warmup"]["realtime_profile"] == "fastest"
    assert runtime_status["graph_variant"] == "int8_sym_paged_talker_split"
    assert runtime_status["codegen_unroll"] == 1
    assert runtime_status["preferred_cache_bucket"] == 0


def test_server_auto_realtime_profile_selects_best_stable_benchmark(tmp_path):
    from qwen3_tts_ov import server

    report = {
        "summaries": [
            {
                "profile": "fastest",
                "accepted": True,
                "p90_stream_rtf": 0.97,
            },
            {
                "profile": "removed_profile",
                "accepted": False,
                "p90_stream_rtf": 0.90,
            },
        ],
        "runs": [
            {"profile": "fastest", "status": "ok", "worker_exit_code": 0, "stream_compute_rtf": 1.08},
            {"profile": "removed_profile", "status": "ok", "worker_exit_code": 0, "stream_compute_rtf": 0.92},
        ]
    }
    path = tmp_path / "bench.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle)

    selected = server.select_auto_realtime_profile(path)

    assert selected["profile"] == "fastest"
    assert selected["codegen_decode_unroll"] == "off"
    assert selected["summary_metric"] == "p90_stream_rtf"


def test_fastest_profile_uses_cpu_native_device_for_cpu_server(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from qwen3_tts_ov import server

    monkeypatch.setenv("QWEN3_TTS_OV_NATIVE_GPU_LARGE_ALLOCATIONS", "1")
    app = server.create_app(
        model_root=tmp_path / "openvino",
        warmup=False,
        realtime_profile="fastest",
        device="CPU",
    )
    client = fastapi_testclient.TestClient(app)

    health = client.get("/health").json()

    assert health["warmup"]["native_codegen_device"] == "CPU"
    assert "QWEN3_TTS_OV_NATIVE_GPU_LARGE_ALLOCATIONS" not in os.environ


def test_fastest_server_reports_u8_kv_cache_profile(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from qwen3_tts_ov import server

    monkeypatch.delenv("QWEN3_TTS_OV_NATIVE_PAGED_KV_PRECISION", raising=False)
    monkeypatch.delenv("QWEN3_TTS_OV_NATIVE_PAGED_KV_CACHE_INPUT_PRECISION", raising=False)
    monkeypatch.delenv("QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE", raising=False)
    app = server.create_app(
        model_root=tmp_path / "openvino",
        warmup=False,
        realtime_profile="fastest",
        device="GPU",
        kv_cache_profile="u8",
    )
    client = fastapi_testclient.TestClient(app)

    health = client.get("/health").json()

    assert health["warmup"]["kv_cache_profile"] == "u8"
    assert health["warmup"]["native_paged_kv_precision"] == "u8"
    assert health["warmup"]["native_paged_kv_cache_input_precision"] == "f32"
    assert health["warmup"]["kv_cache_bytes_per_element"] == 1
    assert health["warmup"]["kv_cache_relative_to_fp16"] == 0.5
    assert health["memory"]["kv_cache_profile"] == "u8"

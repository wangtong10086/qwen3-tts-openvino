import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from qwen3_tts_ov.server import (
    include_chunk_metadata,
    needs_continuous_long_output,
    openai_speech_to_tts_request,
    playback_buffer_for_stream,
    select_continuous_long_output_variant,
    stream_metadata,
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
    app = server.create_app(model_root="openvino", warmup=False)
    client = fastapi_testclient.TestClient(app)

    response = client.post("/v1/tts", json={"mode": "voice_design", "text": "hello"})

    assert response.status_code == 200
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

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    app = server.create_app(model_root="auto", warmup=False)
    client = fastapi_testclient.TestClient(app)

    response = client.post("/v1/tts", json={"mode": "voice_design", "text": "hello"})

    assert response.status_code == 200
    assert seen["ir_dir"] == "openvino_full"


def test_websocket_can_send_optional_chunk_metadata(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from qwen3_tts_ov import server

    ir_dir = tmp_path / "openvino" / "voice_design"
    ir_dir.mkdir(parents=True)
    with open(ir_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"tts_model_type": "voice_design"}, handle)

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            pass

        def stream_voice_design(self, **kwargs):
            yield SimpleNamespace(
                audio=np.zeros(16, dtype=np.float32),
                sample_rate=24000,
                codes=np.zeros((1, 16), dtype=np.int64),
                is_final=True,
                timings={"rtf": 1.2, "stream_rtf": 0.8},
                index=0,
            )

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
        assert metadata["realtime_profile"] == "fastest"
        assert metadata["graph_variant"] == "int8_sym_paged_talker_split"
        audio_meta = websocket.receive_json()
        assert audio_meta["type"] == "audio"
        assert audio_meta["timings"]["stream_rtf"] == 0.8
        assert len(websocket.receive_bytes()) == 32
        assert websocket.receive_json()["type"] == "final"


def test_fastest_websocket_uses_single_prompt_continuous_long_output(monkeypatch, tmp_path):
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
    app = server.create_app(model_root=tmp_path / "openvino", warmup=False)
    client = fastapi_testclient.TestClient(app)
    text = "你好，这是第一段较长的测试文本，用来验证自动切分。这里还有第二段内容，确保不会因为最大生成帧数太小而截断。"

    with client.websocket_connect("/v1/tts/stream") as websocket:
        websocket.send_json(
            {
                "mode": "voice_design",
                "text": text,
                "stream": {"format": "pcm_s16le", "include_chunk_metadata": True},
                "generation": {"max_new_tokens": 160},
            }
        )
        metadata = websocket.receive_json()
        assert metadata["forced_chunk_strategy"] is True
        assert metadata["continuous_long_output"] is True
        assert metadata["continuous_backend"] == "single_prompt_full_ar_reference"
        assert metadata["continuous_bucket"] == 384
        assert metadata["paged_kv"] is False

        audio_messages = 0
        while True:
            message = websocket.receive_json()
            if message["type"] == "final":
                break
            assert message["type"] == "audio"
            assert message["timings"]["continuous_long_output"] is True
            assert message["timings"]["continuous_backend"] == "single_prompt_full_ar_reference"
            assert message["timings"]["paged_kv"] is False
            websocket.receive_bytes()
            audio_messages += 1

    assert audio_messages == 1
    assert len(calls) == 1
    assert calls[0]["text"] == text
    assert calls[0]["max_new_tokens"] == 160
    assert runtime_kwargs[0]["graph_variant"] == "fp16"
    assert runtime_kwargs[0]["codegen_unroll"] == 1
    assert runtime_kwargs[0]["preferred_cache_bucket"] == 0
    assert runtime_kwargs[0]["native_pipeline"] == "off"


def test_server_realtime_profile_int8_reaches_runtime_and_health(monkeypatch, tmp_path):
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
            self.variant_graphs = {
                "fused_cache_step_buckets": {
                    "exact": {"128": "fused_cache_step_exact_cache128_int8_fused.xml"}
                }
            }
            self.cache_dir = None
            self.ov_cache_mode = "OPTIMIZE_SPEED"
            self.disable_ov_cache = False
            self.streaming_decoder_graphs_by_context = {}
            self.streaming_decoders = {}
            self.fused_cache_step_by_bucket = {}
            self.fused_cache_bucket_graphs = {128: "fused_cache_step_exact_cache128_int8_fused.xml"}
            self.fused_cache_unroll_bucket_graphs = {}
            self.talker_stateful_by_bucket = {}

        def generate_voice_design(self, **kwargs):
            return [np.zeros(16, dtype=np.float32)], 24000

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    app = server.create_app(model_root=tmp_path / "openvino", warmup=False, realtime_profile="int8")
    client = fastapi_testclient.TestClient(app)

    assert client.post(
        "/v1/tts",
        json={"mode": "voice_design", "text": "hello", "generation": {"max_new_tokens": 48}},
    ).status_code == 200
    health = client.get("/health").json()
    runtime_status = next(iter(health["runtimes"].values()))

    assert seen["kwargs"]["mode"] == "cache"
    assert seen["kwargs"]["cache_kernel"] == "exact"
    assert seen["kwargs"]["cache_step"] == "fused"
    assert seen["kwargs"]["graph_variant"] == "int8_fused"
    assert seen["kwargs"]["codegen_unroll"] == 1
    assert health["warmup"]["realtime_profile"] == "int8"
    assert runtime_status["graph_variant"] == "int8_fused"
    assert runtime_status["codegen_unroll"] == 1
    assert runtime_status["preferred_cache_bucket"] == 112
    assert runtime_status["fused_cache_variant_active"] is True


def test_server_realtime_profile_int8_sym_sets_unroll4(monkeypatch, tmp_path):
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
            self.codegen_unroll_fallback = False
            self.variant_graphs = {
                "fused_cache_step_buckets": {
                    "exact": {"128": "fused_cache_step_exact_cache128_int8_sym_fused.xml"}
                },
                "fused_cache_step_unroll_buckets": {
                    "exact": {"4": {"128": "fused_cache_step_unroll4_exact_cache128_int8_sym_fused.xml"}}
                },
            }
            self.cache_dir = None
            self.ov_cache_mode = "OPTIMIZE_SPEED"
            self.disable_ov_cache = False
            self.streaming_decoder_graphs_by_context = {}
            self.streaming_decoders = {}
            self.fused_cache_step_by_bucket = {}
            self.fused_cache_bucket_graphs = {128: "fused_cache_step_exact_cache128_int8_sym_fused.xml"}
            self.fused_cache_unroll_bucket_graphs = {128: "fused_cache_step_unroll4_exact_cache128_int8_sym_fused.xml"}
            self.talker_stateful_by_bucket = {}

        def generate_voice_design(self, **kwargs):
            return [np.zeros(16, dtype=np.float32)], 24000

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    app = server.create_app(model_root=tmp_path / "openvino", warmup=False, realtime_profile="int8-sym")
    client = fastapi_testclient.TestClient(app)

    assert client.post(
        "/v1/tts",
        json={"mode": "voice_design", "text": "hello", "generation": {"max_new_tokens": 48}},
    ).status_code == 200
    health = client.get("/health").json()
    runtime_status = next(iter(health["runtimes"].values()))

    assert seen["kwargs"]["mode"] == "cache"
    assert seen["kwargs"]["cache_kernel"] == "exact"
    assert seen["kwargs"]["cache_step"] == "fused"
    assert seen["kwargs"]["graph_variant"] == "int8_sym_fused"
    assert seen["kwargs"]["codegen_unroll"] == 4
    assert seen["kwargs"]["codegen_schedule"] == "current"
    assert health["warmup"]["realtime_profile"] == "int8-sym"
    assert runtime_status["graph_variant"] == "int8_sym_fused"
    assert runtime_status["codegen_unroll"] == 4
    assert runtime_status["codegen_schedule"] == "current"
    assert runtime_status["preferred_cache_bucket"] == 112
    assert runtime_status["unroll_available"] is True


def test_server_realtime_profile_int8_sym_norepeat_defaults_penalty(monkeypatch, tmp_path):
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
            self.codegen_decode_unroll = kwargs.get("codegen_decode_unroll", "off")
            self.codegen_unroll_fallback = False
            self.variant_graphs = {
                "fused_cache_step_buckets": {
                    "exact": {"96": "fused_cache_step_exact_cache96_int8_sym_fused.xml"}
                },
                "fused_cache_step_unroll_norepeat_buckets": {
                    "exact": {"4": {"96": "fused_cache_step_unroll4_exact_norepeat_cache96_int8_sym_fused.xml"}}
                },
            }
            self.cache_dir = None
            self.ov_cache_mode = "OPTIMIZE_SPEED"
            self.disable_ov_cache = False
            self.streaming_decoder_graphs_by_context = {}
            self.streaming_decoders = {}
            self.fused_cache_step_by_bucket = {}
            self.fused_cache_bucket_graphs = {96: "fused_cache_step_exact_cache96_int8_sym_fused.xml"}
            self.fused_cache_unroll_bucket_graphs = {
                96: "fused_cache_step_unroll4_exact_norepeat_cache96_int8_sym_fused.xml"
            }
            self.talker_stateful_by_bucket = {}

        def generate_voice_design(self, **kwargs):
            seen["generate_kwargs"] = kwargs
            return [np.zeros(16, dtype=np.float32)], 24000

    monkeypatch.setattr(server, "OpenVINOQwen3TTS", FakeRuntime)
    app = server.create_app(model_root=tmp_path / "openvino", warmup=False, realtime_profile="int8-sym-norepeat")
    client = fastapi_testclient.TestClient(app)

    assert client.post(
        "/v1/tts",
        json={"mode": "voice_design", "text": "hello", "generation": {"max_new_tokens": 48}},
    ).status_code == 200
    health = client.get("/health").json()
    runtime_status = next(iter(health["runtimes"].values()))

    assert seen["kwargs"]["mode"] == "cache"
    assert seen["kwargs"]["graph_variant"] == "int8_sym_fused"
    assert seen["kwargs"]["codegen_unroll"] == 4
    assert seen["generate_kwargs"]["repetition_penalty"] == 1.0
    assert health["warmup"]["realtime_profile"] == "int8-sym-norepeat"
    assert health["warmup"]["default_repetition_penalty"] == 1.0
    assert runtime_status["graph_variant"] == "int8_sym_fused"
    assert runtime_status["unroll_available"] is True


def test_server_auto_realtime_profile_selects_best_stable_benchmark(tmp_path):
    from qwen3_tts_ov import server

    report = {
        "summaries": [
            {
                "profile": "native_int8_sym_fused_cachedsub_norepeat_bucket96",
                "accepted": True,
                "p90_stream_rtf": 0.97,
            },
            {
                "profile": "int8_sym_ll_v2",
                "accepted": False,
                "p90_stream_rtf": 0.90,
            },
        ],
        "runs": [
            {"profile": "int8_sym_unroll4", "status": "ok", "worker_exit_code": 0, "stream_compute_rtf": 1.08},
            {"profile": "int8_sym_ll_v2", "status": "ok", "worker_exit_code": 0, "stream_compute_rtf": 0.92},
            {"profile": "int8_sym_balanced_v2", "status": "ok", "worker_exit_code": 139, "stream_compute_rtf": 0.80},
        ]
    }
    path = tmp_path / "bench.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle)

    selected = server.select_auto_realtime_profile(path)

    assert selected["profile"] == "native_int8_sym_fused_cachedsub_norepeat_bucket96"
    assert selected["codegen_decode_unroll"] == "auto"
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

import json
from types import SimpleNamespace

import numpy as np
import pytest

from qwen3_tts_ov.server import openai_speech_to_tts_request, stream_metadata


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

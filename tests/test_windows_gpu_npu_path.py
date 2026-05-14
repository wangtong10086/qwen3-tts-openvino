from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from qwen3_tts_ov import server


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    scripts_dir = REPO_ROOT / "scripts"
    previous = list(sys.path)
    sys.path.insert(0, str(scripts_dir))
    try:
        spec = importlib.util.spec_from_file_location(Path(name).stem, scripts_dir / name)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous


def test_smoke_device_matching_accepts_indexed_devices():
    smoke = load_script("smoke_release_tts.py")

    assert smoke.missing_devices(["CPU", "GPU.0", "NPU"], ["GPU", "NPU"]) == []
    assert smoke.missing_devices(["CPU", "GPU.0"], ["GPU", "NPU"]) == ["NPU"]


def test_smoke_expected_device_checks_stream_metadata_and_health():
    smoke = load_script("smoke_release_tts.py")
    stream = {"metadata": {"decoder_device": "NPU"}}
    health = {"warmup": {"native_codegen_device": "GPU"}, "runtimes": {}}

    smoke.assert_expected_device(
        label="decoder_device",
        expected="NPU",
        stream=stream,
        health=health,
        metadata_key="decoder_device",
    )
    smoke.assert_expected_device(
        label="native_codegen_device",
        expected="GPU",
        stream={"metadata": {}},
        health=health,
        metadata_key="native_codegen_device",
    )

    with pytest.raises(RuntimeError):
        smoke.assert_expected_device(
            label="decoder_device",
            expected="NPU",
            stream={"metadata": {"decoder_device": "GPU"}},
            health={"warmup": {}, "runtimes": {}},
            metadata_key="decoder_device",
        )


def test_probe_selects_runtime_minimal_stream_decoders():
    probe = load_script("probe_windows_gpu_npu.py")
    manifest = {
        "streaming_decoder": {
            "contexts": {
                "0": {"8": "speech_decoder_stream_c0_t8.xml", "12": "speech_decoder_stream_c0_t12.xml"},
                "25": {"12": "speech_decoder_stream_c25_t12.xml", "24": "speech_decoder_stream_c25_t24.xml"},
            }
        }
    }

    assert probe.select_stream_decoder_graphs(manifest) == (
        "speech_decoder_stream_c0_t8.xml",
        "speech_decoder_stream_c25_t24.xml",
    )


def test_probe_selects_optional_audio_encoder_graphs():
    probe = load_script("probe_windows_gpu_npu.py")
    manifest = {
        "graphs": {
            "speech_encoder": "speech_encoder.xml",
            "speaker_encoder": "speaker_encoder.xml",
            "text_embedding": "text_embedding.xml",
        }
    }

    assert probe.select_audio_encoder_graphs(manifest) == [
        ("speech_encoder", "speech_encoder.xml"),
        ("speaker_encoder", "speaker_encoder.xml"),
    ]


def test_probe_zero_copy_reports_api_visibility_without_requiring_it():
    probe = load_script("probe_windows_gpu_npu.py")

    class FakeCore:
        pass

    result = probe.zero_copy_probe(FakeCore(), ["GPU", "NPU"])

    assert result["status"] == "info_only"
    assert result["python_api"]["core_create_context"] is False
    assert result["contexts"]["GPU"]["status"] == "unavailable"
    assert result["contexts"]["NPU"]["status"] == "unavailable"


def test_windows_gpu_npu_benchmark_builds_scenario_commands(tmp_path):
    benchmark = load_script("benchmark_windows_gpu_npu_release.py")

    gpu_cmd = benchmark.build_server_command(
        exe=tmp_path / "qwen3-tts-ov-server.exe",
        model_root=tmp_path / "openvino",
        host="127.0.0.1",
        port=17990,
        device="GPU",
        ov_cache_dir=tmp_path / "cache-gpu",
        npu_offload="off",
    )
    npu_cmd = benchmark.build_server_command(
        exe=tmp_path / "qwen3-tts-ov-server.exe",
        model_root=tmp_path / "openvino",
        host="127.0.0.1",
        port=17991,
        device="GPU",
        ov_cache_dir=tmp_path / "cache-npu",
        npu_offload="audio",
    )

    assert "--npu-offload" in gpu_cmd
    assert gpu_cmd[gpu_cmd.index("--npu-offload") + 1] == "off"
    assert npu_cmd[npu_cmd.index("--npu-offload") + 1] == "audio"
    assert "--decoder-device" not in npu_cmd


def test_windows_gpu_npu_benchmark_parses_default_scenarios():
    benchmark = load_script("benchmark_windows_gpu_npu_release.py")

    assert benchmark.parse_scenarios(None) == ["gpu_only", "npu_decoder", "npu_audio"]
    assert benchmark.parse_scenarios("npu_audio") == ["gpu_only", "npu_audio"]
    with pytest.raises(ValueError, match="unknown scenarios"):
        benchmark.parse_scenarios("gpu_only,invalid")


def test_windows_gpu_npu_benchmark_expected_offload_by_scenario():
    benchmark = load_script("benchmark_windows_gpu_npu_release.py")

    assert benchmark.expected_offload_for_scenario("gpu_only", "off") == "off"
    assert benchmark.expected_offload_for_scenario("npu_decoder", "decoder") == "decoder"
    assert benchmark.expected_offload_for_scenario("npu_audio", "audio") == "audio"


def test_windows_gpu_npu_benchmark_metric_uses_audio_duration():
    benchmark = load_script("benchmark_windows_gpu_npu_release.py")
    stream = {
        "audio_bytes": 48000,
        "metadata": {"sample_rate": 24000, "decoder_device": "NPU"},
        "final": {"elapsed": 0.5, "timings": {"stream_rtf": 0.4}},
    }
    health = {"warmup": {"native_codegen_device": "GPU", "npu_offload_effective": "decoder"}, "runtimes": {}}

    metric = benchmark.metric_from_stream(stream, health, wall_elapsed=0.7)

    assert metric["audio_seconds"] == 1.0
    assert metric["computed_rtf"] == 0.5
    assert metric["server_rtf"] == 0.4
    assert metric["decoder_device"] == "NPU"
    assert metric["speech_encoder_device"] is None
    assert metric["speaker_encoder_device"] is None
    assert metric["native_codegen_device"] == "GPU"
    assert metric["npu_offload_effective"] == "decoder"


def test_server_health_reports_device_and_decoder_device(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    monkeypatch.delenv("QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE", raising=False)
    app = server.create_app(
        model_root=tmp_path / "openvino",
        warmup=False,
        realtime_profile="fastest",
        device="GPU",
        decoder_device="NPU",
    )
    client = fastapi_testclient.TestClient(app)

    health = client.get("/health").json()

    assert health["warmup"]["device"] == "GPU"
    assert health["warmup"]["decoder_device"] == "NPU"


def test_server_auto_npu_offload_selects_npu_decoder(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    monkeypatch.delenv("QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE", raising=False)
    monkeypatch.setattr(server, "openvino_available_devices", lambda: (["CPU", "GPU.0", "NPU"], None))
    app = server.create_app(
        model_root=tmp_path / "openvino",
        warmup=False,
        realtime_profile="fastest",
        device="GPU",
        npu_offload="auto",
    )
    client = fastapi_testclient.TestClient(app)

    health = client.get("/health").json()

    assert health["warmup"]["decoder_device"] == "NPU"
    assert health["warmup"]["npu_offload_requested"] == "auto"
    assert health["warmup"]["npu_offload_effective"] == "decoder"
    assert health["warmup"]["npu_offload_reason"] == "auto_selected_npu_decoder"


def test_server_audio_npu_offload_selects_npu_audio_devices(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    monkeypatch.setattr(server, "openvino_available_devices", lambda: (["CPU", "GPU.0", "NPU"], None))
    app = server.create_app(
        model_root=tmp_path / "openvino",
        warmup=False,
        realtime_profile="fastest",
        device="GPU",
        npu_offload="audio",
    )
    client = fastapi_testclient.TestClient(app)

    health = client.get("/health").json()

    assert health["warmup"]["decoder_device"] == "NPU"
    assert health["warmup"]["encoder_device"] == "NPU"
    assert health["warmup"]["speech_encoder_device"] == "NPU"
    assert health["warmup"]["speaker_encoder_device"] == "NPU"
    assert health["warmup"]["npu_offload_effective"] == "audio"


def test_server_auto_npu_offload_falls_back_without_npu(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    monkeypatch.setattr(server, "openvino_available_devices", lambda: (["CPU", "GPU.0"], None))
    app = server.create_app(
        model_root=tmp_path / "openvino",
        warmup=False,
        realtime_profile="fastest",
        device="GPU",
        npu_offload="auto",
    )
    client = fastapi_testclient.TestClient(app)

    health = client.get("/health").json()

    assert health["warmup"]["decoder_device"] == "GPU"
    assert health["warmup"]["npu_offload_effective"] == "off"
    assert health["warmup"]["npu_offload_reason"] == "missing_npu"


def test_server_strict_npu_offload_requires_npu(monkeypatch, tmp_path):
    pytest.importorskip("fastapi.testclient")

    monkeypatch.setattr(server, "openvino_available_devices", lambda: (["CPU", "GPU.0"], None))
    with pytest.raises(ValueError, match="requires OpenVINO NPU device"):
        server.create_app(
            model_root=tmp_path / "openvino",
            warmup=False,
            realtime_profile="fastest",
            device="GPU",
            npu_offload="decoder",
        )


def test_server_npu_offload_rejects_conflicting_decoder_device(monkeypatch, tmp_path):
    pytest.importorskip("fastapi.testclient")

    monkeypatch.setattr(server, "openvino_available_devices", lambda: (["CPU", "GPU.0", "NPU"], None))
    with pytest.raises(ValueError, match="decoder-device is not NPU"):
        server.create_app(
            model_root=tmp_path / "openvino",
            warmup=False,
            realtime_profile="fastest",
            device="GPU",
            decoder_device="GPU",
            npu_offload="decoder",
        )


def test_server_audio_npu_offload_rejects_conflicting_encoder_device(monkeypatch, tmp_path):
    pytest.importorskip("fastapi.testclient")

    monkeypatch.setattr(server, "openvino_available_devices", lambda: (["CPU", "GPU.0", "NPU"], None))
    with pytest.raises(ValueError, match="encoder-device is not NPU"):
        server.create_app(
            model_root=tmp_path / "openvino",
            warmup=False,
            realtime_profile="fastest",
            device="GPU",
            encoder_device="GPU",
            npu_offload="audio",
        )

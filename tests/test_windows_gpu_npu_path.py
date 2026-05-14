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


def test_probe_zero_copy_reports_api_visibility_without_requiring_it():
    probe = load_script("probe_windows_gpu_npu.py")

    class FakeCore:
        pass

    result = probe.zero_copy_probe(FakeCore(), ["GPU", "NPU"])

    assert result["status"] == "info_only"
    assert result["python_api"]["core_create_context"] is False
    assert result["contexts"]["GPU"]["status"] == "unavailable"
    assert result["contexts"]["NPU"]["status"] == "unavailable"


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

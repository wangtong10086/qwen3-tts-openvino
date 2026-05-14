import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_long_text_quality.py"
SPEC = importlib.util.spec_from_file_location("evaluate_long_text_quality", SCRIPT_PATH)
quality = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(quality)


def test_load_aliyun_env_accepts_lowercase_keys_without_leaking_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "aliyun_api_key=secret-token\n"
        "aliyun_model_name=qwen-omni-test\n"
        "aliyun_base_url=https://example.test/compatible-mode/v1\n",
        encoding="utf-8",
    )
    for key in ("ALIYUN_API_KEY", "aliyun_api_key", "ALIYUN_MODEL_NAME", "aliyun_model_name"):
        monkeypatch.delenv(key, raising=False)

    config = quality.load_aliyun_env(env_file)
    status = quality.redacted_env_status(config)

    assert config["api_key"] == "secret-token"
    assert config["model"] == "qwen-omni-test"
    assert status == {"api_key": "set", "model": "set", "base_url_host": "example.test"}
    assert "secret-token" not in json.dumps(status)


def test_extract_json_object_accepts_markdown_fenced_json():
    parsed = quality.extract_json_object(
        '```json\n{"verdict":"pass","intelligibility":4,"noise":5}\n```'
    )

    assert parsed["verdict"] == "pass"
    assert parsed["noise"] == 5


def test_build_omni_messages_uses_input_audio_data_url():
    messages = quality.build_omni_messages(
        "hello",
        "data:audio/wav;base64,AAAA",
        segment_label="head",
    )

    content = messages[0]["content"]
    assert content[1]["type"] == "input_audio"
    assert content[1]["input_audio"] == {"data": "data:audio/wav;base64,AAAA", "format": "wav"}
    assert "head" in content[0]["text"]


def test_objective_gate_passes_valid_sine_and_fails_silence(tmp_path):
    sample_rate = 24_000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    sine = 0.08 * np.sin(2.0 * np.pi * 440.0 * t)
    sine_path = tmp_path / "sine.wav"
    sf.write(sine_path, sine, sample_rate)
    sine_metrics = quality.objective_audio_metrics(sine_path)
    codes = np.stack([np.arange(64) % 17, np.arange(64) % 23], axis=1)

    assert quality.objective_gate(sine_metrics, quality.code_metrics(codes))["pass"]

    silence_path = tmp_path / "silence.wav"
    sf.write(silence_path, np.zeros(sample_rate, dtype=np.float32), sample_rate)
    silence_gate = quality.objective_gate(quality.objective_audio_metrics(silence_path), quality.code_metrics(codes))

    assert not silence_gate["pass"]
    assert "rms_too_low" in silence_gate["failures"]


def test_code_metrics_detects_codec_collapse():
    collapsed = np.zeros((64, 2), dtype=np.int64)
    metrics = quality.code_metrics(collapsed)
    gate = quality.objective_gate(
        {
            "sample_rate": 24000,
            "duration_sec": 1.0,
            "finite": True,
            "rms": 0.05,
            "peak": 0.2,
            "clip_ratio": 0.0,
            "silence_ratio": 0.1,
            "low_rms_window_ratio": 0.1,
            "zero_crossing_rate": 0.05,
            "flatline_ratio": 0.0,
        },
        metrics,
    )

    assert not gate["pass"]
    assert "codec_frame_collapse" in gate["failures"]


def test_select_winner_filters_quality_failures_and_chooses_lowest_median_rtf():
    results = [
        {
            "ok": True,
            "profile": "bad_fast",
            "stream_rtf": 0.5,
            "first_audio_ms": 700,
            "objective_gate": {"pass": False},
            "omni": {"pass": True},
        },
        {
            "ok": True,
            "profile": "good_slow",
            "stream_rtf": 1.2,
            "first_audio_ms": 600,
            "objective_gate": {"pass": True},
            "omni": {"pass": True},
        },
        {
            "ok": True,
            "profile": "good_fast",
            "stream_rtf": 0.9,
            "first_audio_ms": 800,
            "objective_gate": {"pass": True},
            "omni": {"pass": True},
        },
    ]

    winner = quality.select_winner(results)

    assert winner["profile"] == "good_fast"
    assert winner["runtime"] is None


def test_summary_selected_profile_tracks_winner():
    winner = {"profile": "good_fast"}

    summary = {
        "selected_profile": winner.get("profile") if winner else None,
        "winner": winner,
    }

    assert summary["selected_profile"] == "good_fast"


def test_experimental_split_subcode_profile_is_not_default_safe():
    assert quality.LONG_TEXT_PROFILES["long_reference_no_cache_fp16_sample"]["default_safe"]
    assert "long_reference_no_cache_fp16_sample" in quality.parse_profiles("quality")
    assert quality.LONG_TEXT_PROFILES["long_paged_split_sample_fp16"]["default_safe"]
    assert quality.LONG_TEXT_PROFILES["long_paged_split_sample_int8_sym"]["default_safe"]
    assert "long_paged_split_sample_fp16" in quality.parse_profiles("quality")
    assert "long_paged_split_sample_int8_sym" in quality.parse_profiles("quality")
    assert quality.LONG_TEXT_PROFILES["long_paged_split_sample_int8_sym"]["runtime"]["do_sample"] is True
    assert not quality.LONG_TEXT_PROFILES["long_quality_paged_fullhead_fp16"]["default_safe"]
    assert "long_quality_paged_fullhead_fp16" not in quality.parse_profiles("quality")
    assert not quality.LONG_TEXT_PROFILES["long_experimental_split_subcode"]["default_safe"]
    assert "long_experimental_split_subcode" in quality.parse_profiles("all")
    assert "long_experimental_split_subcode" not in quality.parse_profiles("quality")


def test_server_can_select_quality_summary_winner(tmp_path):
    from qwen3_tts_ov.server import select_long_text_quality_profile

    ir_dir = tmp_path / "openvino_full"
    ir_dir.mkdir()
    summary = {
        "winner": {
            "profile": "long_quality_paged_fullhead_fp16",
            "median_stream_rtf": 0.8,
        },
        "results": [
            {
                "ok": True,
                "profile": "long_quality_paged_fullhead_fp16",
                "ir_dir": str(ir_dir),
                "stream_rtf": 0.8,
                "objective_gate": {"pass": True},
                "omni": {"pass": True},
                "runtime": {"mode": "no-cache", "graph_variant": "fp16"},
                "profile_env": {"native_paged_kv_gqa": "0"},
            }
        ],
    }
    summary_path = tmp_path / "quality_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    selected = select_long_text_quality_profile(ir_dir, summary_path)

    assert selected["profile"] == "long_quality_paged_fullhead_fp16"
    assert selected["runtime"] == {"mode": "no-cache", "graph_variant": "fp16"}
    assert selected["profile_env"] == {"native_paged_kv_gqa": "0"}
    assert select_long_text_quality_profile(tmp_path / "other_ir", summary_path) is None

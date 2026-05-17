import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_script_module(path: str):
    script = Path(path)
    spec = importlib.util.spec_from_file_location(script.stem, script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prompt_batch_matrix_dry_run_writes_current_architecture_grid(tmp_path, capsys):
    module = load_script_module("scripts/benchmark_prompt_batch_matrix.py")
    output = tmp_path / "matrix.json"

    module.main(
        [
            "--dry-run",
            "--batch-sizes",
            "1,16",
            "--prompt-lengths",
            "short,128",
            "--scenarios",
            "offline",
            "--runs",
            "1",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "prompt_batch_matrix"
    assert payload["batch_sizes"] == [1, 16]
    assert payload["prompt_lengths"] == ["short", "128"]
    assert payload["optimization_profiles"] == ["baseline"]
    assert len(payload["runs"]) == 4
    assert all(item["dry_run"] is True for item in payload["runs"])
    assert all("benchmark_online_continuous_batch.py" in " ".join(item["command"]) for item in payload["runs"])
    assert all("--optimization-profile baseline" in " ".join(item["command"]) for item in payload["runs"])
    captured = capsys.readouterr()
    assert "dry-run profile=baseline prompt=short batch=1 scenario=offline run=0" in captured.out


def test_prompt_batch_matrix_rejects_removed_experimental_profile_set(tmp_path):
    module = load_script_module("scripts/benchmark_prompt_batch_matrix.py")
    with pytest.raises(ValueError, match="unknown --profile-set"):
        module.main(
            [
                "--dry-run",
                "--profile-set",
                "low-volume-kernels",
                "--output",
                str(tmp_path / "matrix.json"),
            ]
        )


def test_prompt_batch_matrix_summary_skips_failed_worker_records():
    module = load_script_module("scripts/benchmark_prompt_batch_matrix.py")

    summary = module.summarize_runs(
        [
            {
                "optimization_profile": "baseline",
                "prompt_kind": "short",
                "batch_size": 1,
                "scenario": "offline",
                "aggregate_tps": 10.0,
                "ttft_ms_p50": 1.0,
                "ttft_ms_p90": 2.0,
                "per_user_tps_p50": 9.0,
                "hard_metrics": {"scheduler_step_count": 1},
            },
            {
                "optimization_profile": "baseline",
                "prompt_kind": "long",
                "worker_exit_code": -11,
                "stderr_tail": "segfault",
            },
        ]
    )

    assert len(summary) == 1
    assert summary[0]["optimization_profile"] == "baseline"


def test_online_batch_benchmark_is_layered_only():
    module = load_script_module("scripts/benchmark_online_continuous_batch.py")
    assert module.OPTIMIZATION_PROFILES == {"baseline": {}}

    source = Path("scripts/benchmark_online_continuous_batch.py").read_text(encoding="utf-8")
    assert 'choices=["layered"]' in source
    assert 'choices=["layered_vllm"]' in source
    assert "raw" + "_fused_v2" not in source
    assert "adaptive" + "_v2" not in source


def test_native_build_exposes_vllm_like_online_pipeline():
    build_source = Path("scripts/build_native_codegen.py").read_text(encoding="utf-8")
    cli_source = Path("native/qwen3_tts_ov_genai/qwen3_tts_cli.cpp").read_text(encoding="utf-8")
    header_source = Path("native/qwen3_tts_ov_genai/qwen3_tts_codegen.h").read_text(encoding="utf-8")
    cpp_source = Path("native/qwen3_tts_ov_genai/qwen3_tts_codegen.cpp").read_text(encoding="utf-8")

    assert "qwen3_tts_ov_native_cli" in build_source
    assert "qwen3_tts_codegen_run_voice_design_audio_stream" in cli_source
    assert "qwen3_tts_codegen_online_batch_step" in header_source
    assert "QWEN3_TTS_OV_NATIVE_SCHEDULER" in cpp_source
    assert "QWEN3_TTS_OV_NATIVE_PREFILL_SEQ_BUCKETS" in cpp_source
    assert "QWEN3_TTS_OV_NATIVE_DECODE_BATCH_BUCKETS" in cpp_source
    assert "QWEN3_TTS_OV_NATIVE_MAX_NUM_BATCHED_TOKENS" in cpp_source
    assert "layered_vllm" in cpp_source


def test_compression_script_keeps_production_variants():
    source = Path("scripts/compress_openvino_weights.py").read_text(encoding="utf-8")

    assert "fastest" in source
    assert "minimal-online-gqa" in source
    assert "int8_sym_batch_fused_gqa" in source


def test_single_arch_gate_builds_full_context_requests_for_all_public_modes():
    module = load_script_module("scripts/evaluate_single_arch_gate.py")
    args = SimpleNamespace(
        language="Chinese",
        max_vram_ratio=80.0,
        long_max_new_tokens=512,
        max_new_tokens=96,
        min_new_tokens=2,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        chunk_strategy="smooth",
        chunk_frames=24,
        left_context_frames=25,
        initial_chunk_frames=8,
        instruct="自然朗读。",
        speaker="Vivian",
        ref_audio="ref.wav",
        ref_text="参考文本。",
        x_vector_only=False,
        seed=0,
    )

    for mode in ("voice_design", "custom_voice", "voice_clone"):
        payload = module.build_request(args, mode=mode, text_kind="long", index=0, text="长文本测试。")

        assert payload["full_context_text"] is True
        assert payload["auto_segment_text"] is False
        assert payload["allow_auto_segment_text"] is False
        assert payload["force_auto_segment_text"] is False
        assert payload["stream"]["include_chunk_metadata"] is True
        assert payload["generation"]["max_new_tokens"] == 512
        if mode == "voice_clone":
            assert payload["x_vector_only"] is False
            assert payload["ref_audio"] == "ref.wav"


def test_single_arch_gate_rejects_fallback_and_segmented_metadata():
    module = load_script_module("scripts/evaluate_single_arch_gate.py")
    record = {
        "text_kind": "long",
        "metadata": {
            "generation_fallback_allowed": False,
            "online_batching": "on",
            "online_batch_scheduler": "layered",
            "continuous_backend": "vllm_like_online_scheduler",
            "segmented": True,
            "full_context_text": True,
        },
        "chunk_timings": [
            {
                "online_batching": True,
                "fallback": False,
                "sampled_batch_subcode_fallback_count": 1,
            }
        ],
        "final": {"timings": {"online_batch_scheduler": "layered"}},
    }

    failures = module.architecture_failures(record)

    assert "segmented_or_auto_segmented" in failures
    assert "chunk0.sampled_batch_subcode_fallback_count=1" in failures


def test_single_arch_gate_rtf_threshold_only_applies_to_concurrency_one():
    module = load_script_module("scripts/evaluate_single_arch_gate.py")
    args = SimpleNamespace(
        modes="voice_design",
        short_rtf_metric="stream_compute_rtf",
        long_rtf_metric="computed_rtf",
        max_rtf_p90=1.0,
        max_ttft_p90_ms=10_000.0,
        baseline_json=None,
        max_aggregate_rtf_regression=0.05,
        server_url="http://127.0.0.1:17860",
    )
    scenarios = [
        {
            "mode": "voice_design",
            "text_kind": "short",
            "concurrency": 1,
            "aggregate_rtf": 1.5,
            "requests": [
                {
                    "ok": True,
                    "quality_passed": True,
                    "text_kind": "short",
                    "first_audio_ms": 100.0,
                    "computed_rtf": 1.5,
                    "performance": {"stream_compute_rtf": 0.8},
                    "metadata": {
                        "generation_fallback_allowed": False,
                        "online_batching": "on",
                        "online_batch_scheduler": "layered",
                        "continuous_backend": "vllm_like_online_scheduler",
                    },
                    "chunk_timings": [{"online_batching": True}],
                    "final": {"timings": {"online_batch_scheduler": "layered"}},
                }
            ],
        },
        {
            "mode": "voice_design",
            "text_kind": "short",
            "concurrency": 4,
            "aggregate_rtf": 0.7,
            "requests": [
                {
                    "ok": True,
                    "quality_passed": True,
                    "text_kind": "short",
                    "first_audio_ms": 100.0,
                    "computed_rtf": 3.0,
                    "performance": {"stream_compute_rtf": 2.5},
                    "metadata": {
                        "generation_fallback_allowed": False,
                        "online_batching": "on",
                        "online_batch_scheduler": "layered",
                        "continuous_backend": "vllm_like_online_scheduler",
                    },
                    "chunk_timings": [{"online_batching": True}],
                    "final": {"timings": {"online_batch_scheduler": "layered"}},
                }
            ],
        },
    ]

    summary = module.summarize_gate(args, scenarios, health={})

    assert summary["modes"]["voice_design"]["performance_passed"] is True
    assert summary["modes"]["voice_design"]["gate_rtf_p90"] == 0.8


def test_web_client_long_text_uses_full_context_for_all_modes():
    source = Path("qwen3_tts_ov/web_static/web_demo.js").read_text(encoding="utf-8")

    assert 'const fullContext = textUnits > autoSegmentUnits;' in source
    assert 'const fullContext = mode === "voice_design" && textUnits > autoSegmentUnits;' not in source
    assert "auto_segment_text: false" in source


def test_prefill_quality_omni_normalizer_allows_high_score_pitch_accent_soft_fail():
    module = load_script_module("scripts/evaluate_prefill_quality.py")

    result = module.normalize_pair_omni(
        {
            "verdict": "fail",
            "text_match": 4,
            "speaker_style_similarity": 5,
            "naturalness": 5,
            "continuity": 5,
            "noise": 5,
            "failure_reason": "Audio B is intelligible but has a pitch accent and pause difference in the name.",
            "actionable_notes": ["Prosody differs slightly."],
        }
    )

    assert result["pass"] is True
    assert result["verdict"] == "pass"
    assert result["omni_verdict_overridden"] is True


def test_prefill_quality_cache_key_supports_mixed_batch_values():
    module = load_script_module("scripts/evaluate_prefill_quality.py")
    args = SimpleNamespace(
        mode="custom_voice",
        qwen3_tts_repo="/tmp/qwen3",
        python_model="/tmp/model",
        python_executable="auto",
        python_device="gpu",
        python_dtype="auto",
        python_attn_implementation="auto",
        language=["Chinese", "English"],
        instruct=["自然朗读。", "Read naturally."],
        speaker=["Vivian", "Ryan"],
        ref_audio=None,
        ref_text=None,
        x_vector_only=False,
        max_new_tokens=96,
        min_new_tokens=2,
        do_sample=True,
        top_k=20,
        top_p=0.8,
        temperature=0.7,
        repetition_penalty=1.05,
        seed=0,
    )

    payload = module.reference_cache_payload(args, ["中文测试。", "English test."])

    assert payload["languages"] == ["Chinese", "English"]
    assert payload["instructs"] == ["自然朗读。", "Read naturally."]
    assert payload["speakers"] == ["Vivian", "Ryan"]


def test_prefill_quality_evaluate_mode_records_aggregate_rtf(tmp_path):
    module = load_script_module("scripts/evaluate_prefill_quality.py")
    wav_paths = []
    code_paths = []
    for index in range(2):
        wav_path = tmp_path / f"sample_{index}.wav"
        code_path = tmp_path / f"sample_{index}.codes.npy"
        wav_path.write_bytes(b"RIFF....WAVE")
        np.save(code_path, np.ones((10, 16), dtype=np.int64))
        wav_paths.append(wav_path)
        code_paths.append(code_path)

    class FakeQuality:
        @staticmethod
        def objective_audio_metrics(path):
            return {"duration_sec": 1.0}

        @staticmethod
        def code_metrics(codes):
            return {"frames": int(codes.shape[0])}

        @staticmethod
        def objective_gate(audio, codes):
            return {"pass": True}

    args = SimpleNamespace(skip_omni=True, objective_only=False, omni_max_audio_mb=1.0, max_new_tokens=100)
    candidate_items = [
        {
            "index": 0,
            "text": "a",
            "wav_path": str(wav_paths[0]),
            "codes_path": str(code_paths[0]),
            "generated_frames": 10,
            "audio_sec": 4.0,
            "elapsed_ms": 6000.0,
            "first_audio_ms": 100.0,
            "stream_rtf": 1.5,
        },
        {
            "index": 1,
            "text": "b",
            "wav_path": str(wav_paths[1]),
            "codes_path": str(code_paths[1]),
            "generated_frames": 10,
            "audio_sec": 6.0,
            "elapsed_ms": 7000.0,
            "first_audio_ms": 200.0,
            "stream_rtf": 1.1667,
        },
    ]
    reference_items = [{"index": 0, "wav_path": str(wav_paths[0])}, {"index": 1, "wav_path": str(wav_paths[1])}]

    result = module.evaluate_mode(
        args,
        mode="dynamic_ragged",
        reference_items=reference_items,
        candidate_items=candidate_items,
        scheduler_stats={},
        quality=FakeQuality,
        omni_config={},
    )

    assert result["passed"] is True
    assert result["aggregate_audio_sec"] == 10.0
    assert result["elapsed_sec_max"] == 7.0
    assert result["aggregate_rtf"] == pytest.approx(0.7)

import ast
from pathlib import Path


def test_streaming_benchmark_parent_does_not_top_level_import_runtime():
    source = Path("scripts/benchmark_streaming_realtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "qwen3_tts_ov.runtime"
        for node in top_level_imports
    )


def test_streaming_benchmark_default_is_fastest_gate():
    source = Path("scripts/benchmark_streaming_realtime.py").read_text(encoding="utf-8")

    assert '"fastest-gate": "fastest"' in source
    assert '"mode": "fastest"' in source
    assert '"native_pipeline": "require"' in source
    assert '"native_buffer_reuse": "1"' in source
    assert '"preferred_cache_bucket": "96"' in source
    assert '"repetition_penalty": "1.0"' in source


def test_streaming_benchmark_keeps_experimental_profiles_in_devtools():
    source = Path("scripts/benchmark_streaming_realtime.py").read_text(encoding="utf-8")
    dev_source = Path("devtools/bench/benchmark_streaming_realtime_experiments.py").read_text(encoding="utf-8")

    assert '"int8_sym_ll_v2"' not in source
    assert '"native_int8_sym_unroll8_decode"' not in source
    assert '"int8_sym_ll_v2"' in dev_source
    assert '"native_int8_sym_unroll8_decode"' in dev_source
    assert '"codegen_no_repeat": bool(audio_timings.get("codegen_no_repeat", False))' in source
    assert '"native_remote_embed": audio_timings.get("native_remote_embed")' in source
    assert "audio_timings = chunk_timings[-1] if chunk_timings else final_timings" in source
    assert "--warmup-generations" in source
    assert "--native-ov-profile" in source
    assert "QWEN3_TTS_OV_NATIVE_PERF_COUNT" in source
    assert "QWEN3_TTS_OV_NATIVE_LATENCY_HIGH" in source
    assert "NATIVE_COMPILE_ENV_KEYS" in source
    assert '"native_ov_profile": final_timings.get("native_ov_profile")' in source
    assert "--worker-timeout-sec" in source
    assert "subprocess.TimeoutExpired" in source
    assert "--preferred-cache-bucket" in source
    assert "--profile-set" in source
    assert "--max-new-tokens-set" in source
    assert "fastest-gate" in source
    assert '"accepted": accepted' in source
    assert '"host_prepare_ms": native_timing.get("host_prepare_ms")' in source
    assert '"chunk_timings"' in source


def test_streaming_benchmark_production_script_uses_unroll4_only():
    source = Path("scripts/benchmark_streaming_realtime.py").read_text(encoding="utf-8")

    assert '"codegen_unroll": "4"' in source
    assert '"codegen_unroll": "8"' not in source
    assert '"codegen_unroll": "12"' not in source
    assert '"codegen_decode_unroll": "auto"' in source


def test_native_build_exposes_standalone_cpp_cli():
    build_source = Path("scripts/build_native_codegen.py").read_text(encoding="utf-8")
    cli_source = Path("native/qwen3_tts_ov_genai/qwen3_tts_cli.cpp").read_text(encoding="utf-8")

    assert "qwen3_tts_ov_native_cli" in build_source
    assert "qwen3_tts_cli.cpp" in build_source
    assert "qwen3_tts_codegen_run_voice_design_audio_stream" in cli_source
    assert "write_wav" in cli_source
    assert "--profile-json" in cli_source
    assert "--warmup-generations" in cli_source
    assert "--ov-profile" in cli_source
    assert "first_audio_ms" in cli_source
    assert "generate_rtf" in cli_source
    assert "native_ov_profile" in cli_source
    assert "qwen3_tts_codegen_get_profile_json" in cli_source
    assert "qwen3_tts_codegen_reset_profile" in Path("native/qwen3_tts_ov_genai/qwen3_tts_codegen.h").read_text(encoding="utf-8")
    assert "qwen3_tts_codegen_get_last_timing_json" in Path("native/qwen3_tts_ov_genai/qwen3_tts_codegen.h").read_text(encoding="utf-8")
    assert "QWEN3_TTS_OV_NATIVE_BUFFER_REUSE" in Path("native/qwen3_tts_ov_genai/qwen3_tts_codegen.cpp").read_text(encoding="utf-8")


def test_python_cli_exposes_native_ov_profile_switch():
    cli_source = Path("qwen3_tts_ov/cli.py").read_text(encoding="utf-8")
    server_source = Path("qwen3_tts_ov/server.py").read_text(encoding="utf-8")
    profiles_source = Path("qwen3_tts_ov/profiles.py").read_text(encoding="utf-8")

    assert "--native-ov-profile" in cli_source
    assert "--native-buffer-reuse" in cli_source
    assert "QWEN3_TTS_OV_NATIVE_PERF_COUNT" in cli_source
    assert "QWEN3_TTS_OV_NATIVE_BUFFER_REUSE" in cli_source
    assert '"native_ov_profile"' in server_source
    assert "realtime-int8-sym-norepeat" in profiles_source
    assert "int8-sym-norepeat" in profiles_source
    assert "FASTEST_PROFILE_NAME" in server_source
    assert "FASTEST_NATIVE_PIPELINE" in server_source
    assert "FASTEST_CHUNK_STRATEGY" in server_source
    assert "auto_profile and \"repetition_penalty\" in auto_profile" in server_source
    assert "is_fastest_or_norepeat_mode" in cli_source
    assert "preferred_cache_bucket = auto_profile.get(\"preferred_cache_bucket\", preferred_cache_bucket)" in server_source
    assert "FASTEST_MODE" in profiles_source
    assert '"fastest"' in profiles_source
    assert "native_int8_sym_fused_cachedsub_norepeat_bucket96" in profiles_source
    assert "native_int8_sym_fused_cachedsub_norepeat_unroll8_bucket96" in profiles_source
    assert '"realtime_profile": "int8-sym-fused-cachedsub-norepeat"' in profiles_source
    assert "native_int8_sym_fused_cachedsub_rms_norepeat_bucket96" in profiles_source


def test_benchmark_summary_accepts_only_stable_p90_profiles():
    namespace = {"__name__": "not_main"}
    source = Path("scripts/benchmark_streaming_realtime.py").read_text(encoding="utf-8")
    exec(compile(source, "benchmark_streaming_realtime.py", "exec"), namespace)
    summarize_results = namespace["summarize_results"]

    accepted = [
        {
            "profile": "fast",
            "status": "ok",
            "max_new_tokens": 32,
            "stream_rtf": 0.9,
            "stream_compute_rtf": 0.88,
            "first_audio_ms": 700,
            "fallback": False,
            "unroll_fallback": False,
            "worker_exit_code": 0,
            "underrun_risk": False,
        },
        {
            "profile": "fast",
            "status": "ok",
            "max_new_tokens": 32,
            "stream_rtf": 0.98,
            "stream_compute_rtf": 0.95,
            "first_audio_ms": 720,
            "fallback": False,
            "unroll_fallback": False,
            "worker_exit_code": 0,
            "underrun_risk": False,
        },
    ]
    rejected = {
        "profile": "slow",
        "status": "ok",
        "max_new_tokens": 32,
        "stream_rtf": 1.01,
        "stream_compute_rtf": 1.0,
        "first_audio_ms": 900,
        "fallback": False,
        "unroll_fallback": False,
        "worker_exit_code": 0,
        "underrun_risk": True,
    }

    summaries = {item["profile"]: item for item in summarize_results([*accepted, rejected])}

    assert summaries["fast"]["accepted"] is True
    assert summaries["slow"]["accepted"] is False
    assert "p90_stream_rtf" in summaries["slow"]["acceptance_reason"]


def test_compression_script_can_use_source_graph_variant():
    source = Path("scripts/compress_openvino_weights.py").read_text(encoding="utf-8")

    assert "--source-variant" in source
    assert "merge_nested_graphs(graphs" in source


def test_exporter_exposes_rms_variant_export_mode():
    source = Path("qwen3_tts_ov/exporter.py").read_text(encoding="utf-8")

    assert "--rms-export-mode" in source
    assert "fp16_fused_rms" in source
    assert "canonical_rms_norm" in source
    assert "--fused-subcode-mode" in source
    assert "SubcodeGreedyCachedWrapper if normalize_subcode_export_mode" in source
    assert 'features.append("cachedsub")' in source

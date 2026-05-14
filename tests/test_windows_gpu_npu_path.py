from __future__ import annotations

import importlib.util
import json
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


def test_smoke_expected_value_checks_offload_metadata_and_health():
    smoke = load_script("smoke_release_tts.py")
    stream = {"metadata": {"npu_offload_effective": "audio"}}
    health = {"warmup": {"npu_offload_effective": "decoder"}, "runtimes": {}}

    smoke.assert_expected_value(
        label="npu_offload_effective",
        expected="audio",
        stream=stream,
        health=health,
        metadata_key="npu_offload_effective",
    )
    smoke.assert_expected_value(
        label="npu_offload_effective",
        expected="decoder",
        stream={"metadata": {}},
        health=health,
        metadata_key="npu_offload_effective",
    )
    with pytest.raises(RuntimeError):
        smoke.assert_expected_value(
            label="npu_offload_effective",
            expected="all",
            stream=stream,
            health=health,
            metadata_key="npu_offload_effective",
        )


def test_smoke_script_supports_all_npu_offload_and_device_expectations():
    script = (REPO_ROOT / "scripts" / "smoke_release_tts.py").read_text(encoding="utf-8")

    assert '"all"' in script
    assert '"voice_clone"' in script
    assert "--ref-audio" in script
    assert "--expect-encoder-device" in script
    assert "--expect-prompt-device" in script
    assert "--expect-npu-offload-effective" in script


def test_smoke_stage_coverage_distinguishes_voice_design_and_clone():
    smoke = load_script("smoke_release_tts.py")

    voice_design = smoke.npu_offload_coverage("audio", smoke.exercised_runtime_stages("voice_design"))
    voice_clone = smoke.npu_offload_coverage("audio", smoke.exercised_runtime_stages("voice_clone"))
    x_vector = smoke.npu_offload_coverage(
        "audio",
        smoke.exercised_runtime_stages("voice_clone", x_vector_only=True),
    )

    assert voice_design["unexercised_npu_stages"] == ["speech_encoder", "speaker_encoder"]
    assert voice_clone["unexercised_npu_stages"] == []
    assert x_vector["unexercised_npu_stages"] == ["speech_encoder"]


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


def test_probe_selects_prompt_graphs():
    probe = load_script("probe_windows_gpu_npu.py")
    manifest = {
        "graphs": {
            "text_embedding": "text_embedding.xml",
            "codec_embedding": "codec_embedding.xml",
            "talker": "talker.xml",
        }
    }

    assert probe.select_prompt_graphs(manifest) == [
        ("text_embedding", "text_embedding.xml"),
        ("codec_embedding", "codec_embedding.xml"),
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
    assert benchmark.parse_scenarios("npu_all") == ["gpu_only", "npu_all"]
    with pytest.raises(ValueError, match="unknown scenarios"):
        benchmark.parse_scenarios("gpu_only,invalid")


def test_windows_gpu_npu_benchmark_powershell_runs_probe_and_analyzer():
    script = (REPO_ROOT / "scripts" / "windows_gpu_npu_benchmark.ps1").read_text(encoding="utf-8")

    assert "function Invoke-Checked" in script
    assert "probe_windows_gpu_npu.py" in script
    assert "analyze_windows_gpu_npu_results.py" in script
    assert "--benchmark-summary" in script
    assert "--require-scenarios" in script
    assert "RequireExercisedNpuStages" in script
    assert "$benchmarkSummary.status -eq \"skipped\"" in script
    assert "RequirePromptCompile" in script
    assert "RequireAudioCompile" in script


def test_windows_gpu_npu_workflow_builds_runtime_without_npu_validation():
    workflow = (REPO_ROOT / ".github" / "workflows" / "windows-gpu-npu.yml").read_text(encoding="utf-8")

    assert "windows-gpu-npu-runtime" in workflow
    assert "linux-x64" in workflow
    assert "windows-x64" in workflow
    assert "--profile runtime-minimal" in workflow
    assert "scripts/smoke_release_package.py" in workflow
    assert "actions/upload-artifact" in workflow
    assert "self-hosted" not in workflow
    assert "runner_labels" not in workflow
    assert "scripts/probe_windows_gpu_npu.py" not in workflow
    assert "scripts/smoke_release_tts.py" not in workflow
    assert "scripts/benchmark_windows_gpu_npu_release.py" not in workflow
    assert "scripts/analyze_windows_gpu_npu_results.py" not in workflow
    assert "--require-exercised-npu-stages" not in workflow


def test_native_pipeline_filters_gpu_only_properties_for_npu_decoders():
    source = (REPO_ROOT / "native" / "qwen3_tts_ov_genai" / "qwen3_tts_codegen.cpp").read_text(encoding="utf-8")

    assert "compile_config_for_device" in source
    assert 'config.erase("GPU_ENABLE_LARGE_ALLOCATIONS")' in source
    assert 'config.erase("GPU_QUEUE_PRIORITY")' in source
    assert 'config.erase("GPU_HOST_TASK_PRIORITY")' in source
    assert 'config.erase("GPU_QUEUE_THROTTLE")' in source
    assert "m_runner.first_stream_decoder_model = m_runner.core.compile_model(first_decoder_xml, device, config)" in source


def test_windows_gpu_npu_smoke_powershell_asserts_audio_and_prompt_devices():
    script = (REPO_ROOT / "scripts" / "windows_gpu_npu_smoke.ps1").read_text(encoding="utf-8")

    assert "expect-npu-offload-effective" in script
    assert "expect-encoder-device" in script
    assert "expect-prompt-device" in script
    assert "expectedNpuOffload" in script
    assert "Mode voice_clone requires -RefAudio" in script
    assert "$Mode -eq \"voice_clone\"" in script


def test_windows_gpu_npu_powershell_entrypoints_check_native_exit_codes():
    for name in ("windows_gpu_npu_smoke.ps1", "windows_gpu_npu_benchmark.ps1"):
        script = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "function Invoke-Checked" in script
        assert "if ($LASTEXITCODE -ne 0)" in script
        assert "exit code ${LASTEXITCODE}" in script


def test_windows_gpu_npu_benchmark_expected_offload_by_scenario():
    benchmark = load_script("benchmark_windows_gpu_npu_release.py")

    assert benchmark.expected_offload_for_scenario("gpu_only", "off") == "off"
    assert benchmark.expected_offload_for_scenario("npu_decoder", "decoder") == "decoder"
    assert benchmark.expected_offload_for_scenario("npu_audio", "audio") == "audio"
    assert benchmark.expected_offload_for_scenario("npu_all", "all") == "all"


def test_windows_gpu_npu_benchmark_marks_exercised_runtime_stages():
    benchmark = load_script("benchmark_windows_gpu_npu_release.py")

    assert benchmark.exercised_runtime_stages("voice_design") == ["prompt", "text_embedding", "stream_decoder"]
    assert benchmark.exercised_runtime_stages("voice_clone", x_vector_only=True) == [
        "prompt",
        "text_embedding",
        "stream_decoder",
        "speaker_encoder",
    ]
    coverage = benchmark.npu_offload_coverage("audio", benchmark.exercised_runtime_stages("voice_design"))

    assert coverage["exercised_npu_stages"] == ["stream_decoder"]
    assert coverage["unexercised_npu_stages"] == ["speech_encoder", "speaker_encoder"]


def test_windows_gpu_npu_benchmark_acceptance_checks_speedup_and_regression():
    benchmark = load_script("benchmark_windows_gpu_npu_release.py")
    results = [
        {
            "name": "gpu_only",
            "summary": {
                "median_computed_rtf": 1.0,
                "accelerator_counters": {"gpu": {"utilization_average": 80.0}},
            },
        },
        {
            "name": "npu_decoder",
            "summary": {
                "median_computed_rtf": 0.8,
                "accelerator_counters": {
                    "gpu": {"utilization_average": 60.0},
                    "npu": {"utilization_average": 25.0},
                },
            },
        },
        {
            "name": "npu_audio",
            "summary": {
                "median_computed_rtf": 1.05,
                "accelerator_counters": {"gpu": {"utilization_average": 79.0}},
            },
        },
    ]

    comparison = benchmark.compare_to_gpu_baseline(results)

    assert comparison["npu_decoder"]["computed_rtf_speedup"] == 1.25
    assert comparison["npu_decoder"]["gpu_utilization_reduction"] == 0.25
    assert comparison["npu_decoder"]["npu_utilization_average"] == 25.0
    failures = benchmark.check_acceptance(
        comparison,
        min_speedup=1.0,
        max_rtf_regression=0.03,
        min_gpu_utilization_reduction=0.05,
    )
    assert any("npu_audio" in item for item in failures)


def test_windows_gpu_npu_benchmark_recommends_balanced_npu_offload():
    benchmark = load_script("benchmark_windows_gpu_npu_release.py")
    results = [
        {
            "name": "gpu_only",
            "summary": {
                "median_computed_rtf": 1.0,
                "npu_offload_effective": "off",
                "accelerator_counters": {"gpu": {"utilization_average": 80.0}},
            },
        },
        {
            "name": "npu_decoder",
            "summary": {
                "median_computed_rtf": 0.9,
                "npu_offload_effective": "decoder",
                "accelerator_counters": {"gpu": {"utilization_average": 72.0}},
            },
        },
        {
            "name": "npu_audio",
            "summary": {
                "median_computed_rtf": 1.02,
                "npu_offload_effective": "audio",
                "accelerator_counters": {"gpu": {"utilization_average": 56.0}},
            },
        },
    ]

    comparison = benchmark.compare_to_gpu_baseline(results)
    recommendation = benchmark.recommend_offload(results, comparison, max_rtf_regression=0.03)

    assert recommendation["fastest"]["scenario"] == "npu_decoder"
    assert recommendation["lowest_gpu_utilization"]["scenario"] == "npu_audio"
    assert recommendation["balanced"]["scenario"] == "npu_audio"
    assert recommendation["recommended_npu_offload"] == "audio"


def test_release_server_loads_windows_npu_offload_summary(monkeypatch, tmp_path):
    from qwen3_tts_ov import release_server
    from qwen3_tts_ov.model_download import ModelDownloadResult

    summary_path = tmp_path / "benchmark-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "recommendation": {
                    "recommended_scenario": "npu_decoder",
                    "recommended_npu_offload": "decoder",
                    "balanced": {"scenario": "npu_audio", "npu_offload": "audio"},
                    "fastest": {"scenario": "npu_decoder", "npu_offload": "decoder"},
                    "lowest_gpu_utilization": {"scenario": "npu_all", "npu_offload": "all"},
                },
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    monkeypatch.setattr(release_server, "configure_native_library_env", lambda: None)
    monkeypatch.setattr(
        release_server,
        "ensure_release_model_root",
        lambda model_root, **_: ModelDownloadResult(
            model_root=Path(model_root),
            status="local",
            repo_id="repo",
            revision="main",
            subdir="openvino_realtime",
            cache_dir=tmp_path / "cache",
            message="local",
        ),
    )
    monkeypatch.setattr(release_server, "serve", lambda **kwargs: captured.update(kwargs))

    release_server.main(
        [
            "--model-root",
            str(tmp_path / "openvino"),
            "--no-auto-download-model",
            "--npu-offload-summary",
            str(summary_path),
            "--npu-offload-policy",
            "lowest-gpu",
            "--no-warmup",
        ]
    )

    assert captured["npu_offload"] == "all"


def test_cli_applies_windows_npu_offload_summary(tmp_path):
    from qwen3_tts_ov import cli

    summary_path = tmp_path / "benchmark-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "recommendation": {
                    "recommended_scenario": "npu_audio",
                    "recommended_npu_offload": "audio",
                    "fastest": {"scenario": "npu_decoder", "npu_offload": "decoder"},
                },
            }
        ),
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "npu_offload": "off",
            "npu_offload_summary": str(summary_path),
            "npu_offload_policy": "fastest",
        },
    )()

    profile = cli.apply_npu_offload_profile_summary(args)

    assert args.npu_offload == "decoder"
    assert profile["scenario"] == "npu_decoder"


def test_windows_gpu_npu_benchmark_counter_sampler_targets_server_pid(tmp_path):
    benchmark = load_script("benchmark_windows_gpu_npu_release.py")

    cmd = benchmark.build_counter_sampler_command(
        powershell="pwsh",
        output_json=tmp_path / "counters.json",
        stop_file=tmp_path / "stop",
        interval_ms=50,
        process_id=1234,
        counter_scope="server",
    )

    assert "-ProcessId" in cmd
    assert cmd[cmd.index("-ProcessId") + 1] == "1234"
    assert "-CounterScope" in cmd
    assert cmd[cmd.index("-CounterScope") + 1] == "server"
    assert cmd[cmd.index("-IntervalMs") + 1] == "100"


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


def test_windows_gpu_npu_result_analyzer_accepts_valid_artifacts(tmp_path):
    analyzer = load_script("analyze_windows_gpu_npu_results.py")
    benchmark_summary = {
        "status": "ok",
        "results": [
            {
                "name": "gpu_only",
                "summary": {
                    "median_computed_rtf": 1.0,
                    "npu_offload_effective": "off",
                    "decoder_device": "GPU",
                    "accelerator_counters": {
                        "status": "ok",
                        "gpu": {"utilization_average": 80.0},
                        "npu": {"utilization_average": 0.0},
                    },
                },
            },
            {
                "name": "npu_all",
                "summary": {
                    "median_computed_rtf": 0.8,
                    "npu_offload_effective": "all",
                    "decoder_device": "NPU",
                    "encoder_device": "NPU",
                    "prompt_device": "NPU",
                    "text_embedding_device": "NPU",
                    "accelerator_counters": {
                        "status": "ok",
                        "gpu": {"utilization_average": 60.0},
                        "npu": {"utilization_average": 25.0},
                    },
                    "npu_offload_coverage": {
                        "expected_npu_stages": ["stream_decoder"],
                        "exercised_npu_stages": ["stream_decoder"],
                        "unexercised_npu_stages": [],
                    },
                },
            },
        ],
    }
    probe = {
        "status": "ok",
        "decoder_compile": [{"label": "stream_decoder:c0_t8", "device": "NPU"}],
        "prompt_compile": {"status": "ok", "graphs": [{"label": "text_embedding", "device": "NPU"}]},
        "audio_encoder_compile": {"status": "ok", "graphs": [{"label": "speech_encoder", "device": "NPU"}]},
    }
    benchmark_path = tmp_path / "benchmark.json"
    probe_path = tmp_path / "probe.json"
    benchmark_path.write_text(json.dumps(benchmark_summary), encoding="utf-8")
    probe_path.write_text(json.dumps(probe), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "benchmark_summary": str(benchmark_path),
            "probe_json": str(probe_path),
            "require_scenarios": "gpu_only,npu_all",
            "min_speedup": 1.0,
            "max_rtf_regression": 0.0,
            "min_gpu_utilization_reduction": 0.05,
            "require_counters": True,
            "require_probe_ok": True,
            "require_prompt_compile": True,
            "require_audio_compile": True,
        },
    )()

    report = analyzer.analyze(args)

    assert report["status"] == "ok"
    assert report["failures"] == []


def test_windows_gpu_npu_result_analyzer_rejects_missing_npu_device(tmp_path):
    analyzer = load_script("analyze_windows_gpu_npu_results.py")
    benchmark_summary = {
        "status": "ok",
        "results": [
            {
                "name": "gpu_only",
                "summary": {"median_computed_rtf": 1.0, "npu_offload_effective": "off", "decoder_device": "GPU"},
            },
            {
                "name": "npu_decoder",
                "summary": {
                    "median_computed_rtf": 0.9,
                    "npu_offload_effective": "decoder",
                    "decoder_device": "GPU",
                },
            },
        ],
    }
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark_summary), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "benchmark_summary": str(benchmark_path),
            "probe_json": None,
            "require_scenarios": "gpu_only,npu_decoder",
            "min_speedup": None,
            "max_rtf_regression": None,
            "min_gpu_utilization_reduction": None,
            "require_counters": False,
            "require_probe_ok": False,
            "require_prompt_compile": False,
            "require_audio_compile": False,
        },
    )()

    report = analyzer.analyze(args)

    assert report["status"] == "failed"
    assert any("decoder_device" in item for item in report["failures"])


def test_windows_gpu_npu_result_analyzer_warns_unexercised_audio_stage(tmp_path):
    analyzer = load_script("analyze_windows_gpu_npu_results.py")
    benchmark_summary = {
        "status": "ok",
        "results": [
            {
                "name": "gpu_only",
                "summary": {"median_computed_rtf": 1.0, "npu_offload_effective": "off", "decoder_device": "GPU"},
            },
            {
                "name": "npu_audio",
                "summary": {
                    "median_computed_rtf": 0.9,
                    "npu_offload_effective": "audio",
                    "decoder_device": "NPU",
                    "encoder_device": "NPU",
                    "npu_offload_coverage": {
                        "expected_npu_stages": ["stream_decoder", "speech_encoder", "speaker_encoder"],
                        "exercised_npu_stages": ["stream_decoder"],
                        "unexercised_npu_stages": ["speech_encoder", "speaker_encoder"],
                    },
                },
            },
        ],
    }
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark_summary), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "benchmark_summary": str(benchmark_path),
            "probe_json": None,
            "require_scenarios": "gpu_only,npu_audio",
            "min_speedup": None,
            "max_rtf_regression": None,
            "min_gpu_utilization_reduction": None,
            "require_counters": False,
            "require_probe_ok": False,
            "require_prompt_compile": False,
            "require_audio_compile": False,
        },
    )()

    report = analyzer.analyze(args)

    assert report["status"] == "ok"
    assert any("not exercised" in item for item in report["warnings"])


def test_windows_gpu_npu_result_analyzer_can_require_exercised_npu_stages(tmp_path):
    analyzer = load_script("analyze_windows_gpu_npu_results.py")
    benchmark_summary = {
        "status": "ok",
        "results": [
            {
                "name": "gpu_only",
                "summary": {"median_computed_rtf": 1.0, "npu_offload_effective": "off", "decoder_device": "GPU"},
            },
            {
                "name": "npu_audio",
                "summary": {
                    "median_computed_rtf": 0.9,
                    "npu_offload_effective": "audio",
                    "decoder_device": "NPU",
                    "encoder_device": "NPU",
                    "npu_offload_coverage": {
                        "expected_npu_stages": ["stream_decoder", "speech_encoder", "speaker_encoder"],
                        "exercised_npu_stages": ["stream_decoder"],
                        "unexercised_npu_stages": ["speech_encoder", "speaker_encoder"],
                    },
                },
            },
        ],
    }
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark_summary), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "benchmark_summary": str(benchmark_path),
            "probe_json": None,
            "require_scenarios": "gpu_only,npu_audio",
            "min_speedup": None,
            "max_rtf_regression": None,
            "min_gpu_utilization_reduction": None,
            "require_counters": False,
            "require_exercised_npu_stages": True,
            "require_probe_ok": False,
            "require_prompt_compile": False,
            "require_audio_compile": False,
        },
    )()

    report = analyzer.analyze(args)

    assert report["status"] == "failed"
    assert any("not exercised" in item for item in report["failures"])


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


def test_server_all_npu_offload_selects_prompt_and_audio_devices(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    monkeypatch.setattr(server, "openvino_available_devices", lambda: (["CPU", "GPU.0", "NPU"], None))
    app = server.create_app(
        model_root=tmp_path / "openvino",
        warmup=False,
        realtime_profile="fastest",
        device="GPU",
        npu_offload="all",
    )
    client = fastapi_testclient.TestClient(app)

    health = client.get("/health").json()

    assert health["warmup"]["decoder_device"] == "NPU"
    assert health["warmup"]["encoder_device"] == "NPU"
    assert health["warmup"]["prompt_device"] == "NPU"
    assert health["warmup"]["npu_offload_effective"] == "all"


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


def test_server_auto_npu_offload_falls_back_when_npu_decoder_probe_fails(monkeypatch, tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    ir_dir = tmp_path / "voice_design"
    ir_dir.mkdir()

    monkeypatch.setattr(server, "openvino_available_devices", lambda: (["CPU", "GPU.0", "NPU"], None))
    monkeypatch.setattr(server, "resolve_budget_ir_dir", lambda model_root, mode_name: ir_dir)
    monkeypatch.setattr(
        server,
        "load_manifest",
        lambda path: {
            "tts_model_type": "voice_design",
            "streaming_decoder": {"contexts": {"0": {"8": "speech_decoder_stream_c0_t8.xml"}, "25": {"24": "speech_decoder_stream_c25_t24.xml"}}},
        },
    )
    monkeypatch.setattr(server, "probe_stream_decoders_on_npu", lambda *args, **kwargs: (False, "negative shape bound"))
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
    assert health["warmup"]["npu_offload_reason"].startswith("auto_npu_decoder_compile_failed")


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

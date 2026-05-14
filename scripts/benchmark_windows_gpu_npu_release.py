#!/usr/bin/env python3
"""Compare packaged Windows release performance across GPU-only and GPU+NPU paths."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from smoke_release_package import extract_archive, find_executable, read_health, tail, terminate_process
from smoke_release_tts import (
    first_stream_or_health_value,
    missing_devices,
    query_openvino_devices,
    run_stream_request,
    write_summary,
)


SCENARIOS = {
    "gpu_only": {"npu_offload": "off"},
    "npu_decoder": {"npu_offload": "decoder"},
    "npu_audio": {"npu_offload": "audio"},
    "npu_all": {"npu_offload": "all"},
}


def parse_scenarios(value: str | None) -> list[str]:
    raw = value or "gpu_only,npu_decoder,npu_audio"
    result = [item.strip() for item in str(raw).split(",") if item.strip()]
    unknown = [item for item in result if item not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown scenarios: {', '.join(unknown)}; available: {', '.join(SCENARIOS)}")
    if "gpu_only" not in result:
        result.insert(0, "gpu_only")
    return result


def build_server_command(
    *,
    exe: Path,
    model_root: Path,
    host: str,
    port: int,
    device: str,
    ov_cache_dir: Path,
    npu_offload: str = "off",
    decoder_device: str | None = None,
    no_warmup: bool = False,
) -> list[str]:
    cmd = [
        str(exe),
        "--model-root",
        str(model_root),
        "--host",
        host,
        "--port",
        str(port),
        "--device",
        device,
        "--realtime-profile",
        "fastest",
        "--ov-cache-dir",
        str(ov_cache_dir),
        "--npu-offload",
        npu_offload,
    ]
    if decoder_device:
        cmd.extend(["--decoder-device", decoder_device])
    if no_warmup:
        cmd.append("--no-warmup")
    return cmd


def wait_for_health(url: str, deadline: float) -> dict:
    last_payload = None
    while time.time() < deadline:
        payload = read_health(url, timeout=2.0)
        if payload is not None:
            last_payload = payload
            warmup = payload.get("warmup")
            warmup_status = warmup.get("status") if isinstance(warmup, dict) else None
            if payload.get("ok") is True and warmup_status in {None, "ready", "ready_with_errors", "disabled"}:
                return payload
        time.sleep(0.5)
    raise TimeoutError(f"server did not become healthy at {url}; last_payload={last_payload!r}")


def metric_from_stream(stream: dict, health: dict, wall_elapsed: float) -> dict:
    metadata = stream.get("metadata") or {}
    final = stream.get("final") or {}
    timings = final.get("timings") if isinstance(final, dict) else {}
    if not isinstance(timings, dict):
        timings = {}
    sample_rate = int(metadata.get("sample_rate") or 24000)
    audio_bytes = int(stream.get("audio_bytes") or 0)
    audio_seconds = audio_bytes / float(sample_rate * 2) if sample_rate > 0 else 0.0
    elapsed = float(final.get("elapsed") or wall_elapsed)
    computed_rtf = elapsed / audio_seconds if audio_seconds > 0 else None
    server_rtf = timings.get("stream_rtf") or timings.get("rtf")
    try:
        server_rtf = None if server_rtf is None else float(server_rtf)
    except (TypeError, ValueError):
        server_rtf = None
    return {
        "audio_bytes": audio_bytes,
        "audio_seconds": audio_seconds,
        "elapsed": elapsed,
        "wall_elapsed": wall_elapsed,
        "computed_rtf": computed_rtf,
        "server_rtf": server_rtf,
        "first_audio_ms": timings.get("first_audio_ms") or timings.get("first_chunk_ms"),
        "stream_compute_rtf": timings.get("stream_compute_rtf"),
        "decode_path": timings.get("decode_path") or metadata.get("decode_path"),
        "device": first_stream_or_health_value(stream, health, "device", None),
        "decoder_device": first_stream_or_health_value(stream, health, "decoder_device", None),
        "encoder_device": first_stream_or_health_value(stream, health, "encoder_device", None),
        "prompt_device": first_stream_or_health_value(stream, health, "prompt_device", None),
        "text_embedding_device": first_stream_or_health_value(stream, health, "text_embedding_device", None),
        "speech_encoder_device": first_stream_or_health_value(stream, health, "speech_encoder_device", None),
        "speaker_encoder_device": first_stream_or_health_value(stream, health, "speaker_encoder_device", None),
        "native_codegen_device": first_stream_or_health_value(stream, health, "native_codegen_device", None),
        "npu_offload_effective": first_stream_or_health_value(stream, health, "npu_offload_effective", None),
        "npu_offload_reason": first_stream_or_health_value(stream, health, "npu_offload_reason", None),
        "timings": timings,
    }


def aggregate_metrics(metrics: list[dict]) -> dict:
    def values(key: str) -> list[float]:
        result = []
        for metric in metrics:
            value = metric.get(key)
            if isinstance(value, (int, float)):
                result.append(float(value))
        return result

    computed_rtf = values("computed_rtf")
    server_rtf = values("server_rtf")
    elapsed = values("elapsed")
    return {
        "runs": metrics,
        "median_computed_rtf": statistics.median(computed_rtf) if computed_rtf else None,
        "median_server_rtf": statistics.median(server_rtf) if server_rtf else None,
        "median_elapsed": statistics.median(elapsed) if elapsed else None,
        "decoder_device": metrics[-1].get("decoder_device") if metrics else None,
        "encoder_device": metrics[-1].get("encoder_device") if metrics else None,
        "prompt_device": metrics[-1].get("prompt_device") if metrics else None,
        "text_embedding_device": metrics[-1].get("text_embedding_device") if metrics else None,
        "speech_encoder_device": metrics[-1].get("speech_encoder_device") if metrics else None,
        "speaker_encoder_device": metrics[-1].get("speaker_encoder_device") if metrics else None,
        "native_codegen_device": metrics[-1].get("native_codegen_device") if metrics else None,
        "npu_offload_effective": metrics[-1].get("npu_offload_effective") if metrics else None,
        "npu_offload_reason": metrics[-1].get("npu_offload_reason") if metrics else None,
    }


def find_powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def read_json_file(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def build_counter_sampler_command(
    *,
    powershell: str,
    output_json: Path,
    stop_file: Path,
    interval_ms: int,
    process_id: int | None = None,
    counter_scope: str = "server",
) -> list[str]:
    script = Path(__file__).resolve().with_name("collect_windows_accelerator_counters.ps1")
    cmd = [
        powershell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-OutputJson",
        str(output_json),
        "-StopFile",
        str(stop_file),
        "-IntervalMs",
        str(max(100, int(interval_ms))),
    ]
    if process_id:
        cmd.extend(["-ProcessId", str(int(process_id))])
    if counter_scope:
        cmd.extend(["-CounterScope", str(counter_scope)])
    return cmd


def start_counter_sampler(
    *,
    enabled: bool,
    scenario_dir: Path,
    interval_ms: int,
    process_id: int | None,
    counter_scope: str,
) -> dict | None:
    if not enabled:
        return None
    powershell = find_powershell()
    if not powershell:
        return {
            "status": "unavailable",
            "error": "PowerShell executable not found; accelerator counters were not collected.",
        }
    output_json = scenario_dir / "accelerator-counters.json"
    stop_file = scenario_dir / "accelerator-counters.stop"
    log_path = scenario_dir / "accelerator-counters.log"
    if stop_file.exists():
        stop_file.unlink()
    cmd = build_counter_sampler_command(
        powershell=powershell,
        output_json=output_json,
        stop_file=stop_file,
        interval_ms=interval_ms,
        process_id=process_id,
        counter_scope=counter_scope,
    )
    try:
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    except Exception as exc:
        return {
            "status": "failed_to_start",
            "cmd": cmd,
            "error": str(exc),
        }
    return {
        "process": process,
        "log": log,
        "cmd": cmd,
        "output_json": output_json,
        "stop_file": stop_file,
        "log_path": log_path,
    }


def stop_counter_sampler(sampler: dict | None, timeout: float = 10.0) -> dict | None:
    if sampler is None:
        return None
    process = sampler.get("process")
    if process is None:
        return sampler
    stop_file = Path(sampler["stop_file"])
    try:
        stop_file.write_text("stop\n", encoding="utf-8")
    except Exception:
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_process(process)
    log = sampler.get("log")
    if log:
        log.close()
    output_json = Path(sampler["output_json"])
    summary = read_json_file(output_json)
    if summary is None:
        summary = {
            "status": "missing_summary",
            "cmd": sampler.get("cmd"),
            "log_tail": tail(Path(sampler["log_path"])),
        }
    else:
        summary["cmd"] = sampler.get("cmd")
        summary["log_path"] = str(sampler.get("log_path"))
    return summary


def expected_offload_for_scenario(name: str, npu_offload: str) -> str | None:
    if name == "gpu_only":
        return "off"
    if npu_offload in {"decoder", "require", "auto"}:
        return "decoder"
    if npu_offload == "audio":
        return "audio"
    if npu_offload == "all":
        return "all"
    return None


def compare_to_gpu_baseline(results: list[dict]) -> dict:
    by_name = {item["name"]: item for item in results}
    gpu_rtf = ((by_name.get("gpu_only") or {}).get("summary") or {}).get("median_computed_rtf")
    gpu_util = accelerator_utilization_average(by_name.get("gpu_only"), "gpu")
    comparison = {}
    for name, result in by_name.items():
        if name == "gpu_only":
            continue
        npu_rtf = (result.get("summary") or {}).get("median_computed_rtf")
        scenario_gpu_util = accelerator_utilization_average(result, "gpu")
        scenario_npu_util = accelerator_utilization_average(result, "npu")
        comparison[name] = {
            "computed_rtf_delta": None if gpu_rtf is None or npu_rtf is None else npu_rtf - gpu_rtf,
            "computed_rtf_speedup": None if gpu_rtf is None or not npu_rtf else gpu_rtf / npu_rtf,
            "gpu_utilization_average": scenario_gpu_util,
            "npu_utilization_average": scenario_npu_util,
            "gpu_utilization_delta": (
                None if gpu_util is None or scenario_gpu_util is None else scenario_gpu_util - gpu_util
            ),
            "gpu_utilization_reduction": (
                None
                if gpu_util is None or scenario_gpu_util is None or gpu_util == 0
                else (gpu_util - scenario_gpu_util) / gpu_util
            ),
        }
    return comparison


def accelerator_utilization_average(result: dict | None, category: str) -> float | None:
    if not result:
        return None
    counters = ((result.get("summary") or {}).get("accelerator_counters") or {})
    category_summary = counters.get(category) if isinstance(counters, dict) else None
    if not isinstance(category_summary, dict):
        return None
    value = category_summary.get("utilization_average")
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def check_acceptance(
    comparison: dict,
    *,
    min_speedup: float | None,
    max_rtf_regression: float | None,
    min_gpu_utilization_reduction: float | None = None,
) -> list[str]:
    failures = []
    for name, metrics in comparison.items():
        speedup = metrics.get("computed_rtf_speedup")
        delta = metrics.get("computed_rtf_delta")
        gpu_reduction = metrics.get("gpu_utilization_reduction")
        if min_speedup is not None and (speedup is None or float(speedup) < float(min_speedup)):
            failures.append(f"{name}: computed_rtf_speedup={speedup} < {min_speedup}")
        if max_rtf_regression is not None and (
            delta is None or float(delta) > float(max_rtf_regression)
        ):
            failures.append(f"{name}: computed_rtf_delta={delta} > {max_rtf_regression}")
        if min_gpu_utilization_reduction is not None and (
            gpu_reduction is None or float(gpu_reduction) < float(min_gpu_utilization_reduction)
        ):
            failures.append(
                f"{name}: gpu_utilization_reduction={gpu_reduction} < {min_gpu_utilization_reduction}"
            )
    return failures


def run_scenario(
    *,
    name: str,
    exe: Path,
    model_root: Path,
    work_dir: Path,
    host: str,
    port: int,
    device: str,
    npu_offload: str,
    decoder_device: str | None,
    no_warmup: bool,
    request_payload: dict,
    timeout: float,
    runs: int,
    collect_accelerator_counters: bool,
    counter_interval_ms: int,
    counter_scope: str,
) -> dict:
    scenario_dir = work_dir / name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    log_path = scenario_dir / "server.log"
    cmd = build_server_command(
        exe=exe,
        model_root=model_root,
        host=host,
        port=port,
        device=device,
        ov_cache_dir=scenario_dir / "ov-cache",
        npu_offload=npu_offload,
        decoder_device=decoder_device,
        no_warmup=no_warmup,
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        health = wait_for_health(f"http://{host}:{port}/health", deadline=time.time() + timeout)
        metrics = []
        counter_sampler = start_counter_sampler(
            enabled=collect_accelerator_counters,
            scenario_dir=scenario_dir,
            interval_ms=counter_interval_ms,
            process_id=process.pid,
            counter_scope=counter_scope,
        )
        try:
            for run_index in range(max(1, int(runs))):
                started = time.time()
                stream = run_stream_request(f"http://{host}:{port}/v1/tts/stream", request_payload, timeout=timeout)
                metrics.append(
                    {
                        "run_index": run_index,
                        **metric_from_stream(stream, health, time.time() - started),
                    }
                )
        finally:
            counter_summary = stop_counter_sampler(counter_sampler)
        summary = aggregate_metrics(metrics)
        if counter_summary is not None:
            summary["accelerator_counters"] = counter_summary
        expected_offload = expected_offload_for_scenario(name, npu_offload)
        if expected_offload and summary.get("npu_offload_effective") != expected_offload:
            return {
                "name": name,
                "cmd": cmd,
                "error": (
                    f"expected npu_offload_effective={expected_offload}, "
                    f"got {summary.get('npu_offload_effective')!r}"
                ),
                "summary": summary,
                "health": {"ok": health.get("ok"), "warmup": health.get("warmup", {})},
                "server_log_tail": tail(log_path),
            }
        return {
            "name": name,
            "cmd": cmd,
            "health": {"ok": health.get("ok"), "warmup": health.get("warmup", {})},
            "summary": summary,
        }
    except Exception as exc:
        return {
            "name": name,
            "cmd": cmd,
            "error": str(exc),
            "server_log_tail": tail(log_path),
        }
    finally:
        terminate_process(process)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--work-dir", default="build/windows-gpu-npu-benchmark")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=17990)
    parser.add_argument("--device", default="GPU")
    parser.add_argument(
        "--scenarios",
        default="gpu_only,npu_decoder,npu_audio",
        help="Comma-separated benchmark scenarios: gpu_only,npu_decoder,npu_audio,npu_all.",
    )
    parser.add_argument("--require-devices", default="GPU,NPU")
    parser.add_argument("--skip-if-missing-devices", action="store_true")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--mode", default="voice_design", choices=("voice_design", "voice_clone"))
    parser.add_argument("--text", default="你好，这是 Windows GPU 加 NPU 推理性能对比测试。")
    parser.add_argument("--instruct", default="A calm young female voice.")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--x-vector-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--chunk-strategy", default="smooth")
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=None,
        help="Fail if any NPU scenario has computed_rtf_speedup below this value.",
    )
    parser.add_argument(
        "--max-rtf-regression",
        type=float,
        default=None,
        help="Fail if any NPU scenario has median computed RTF worse than GPU-only by more than this value.",
    )
    parser.add_argument(
        "--collect-accelerator-counters",
        action="store_true",
        help="Collect Windows GPU/NPU utilization counters during each scenario.",
    )
    parser.add_argument(
        "--counter-interval-ms",
        type=int,
        default=500,
        help="Sampling interval for --collect-accelerator-counters.",
    )
    parser.add_argument(
        "--counter-scope",
        default="server",
        choices=("server", "system"),
        help=(
            "Counter scope for --collect-accelerator-counters. "
            "server filters GPU/NPU engine counters to the release server PID when available."
        ),
    )
    parser.add_argument(
        "--min-gpu-utilization-reduction",
        type=float,
        default=None,
        help="Fail if any NPU scenario does not reduce average GPU utilization by this fraction.",
    )
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args()

    available_devices, device_error = query_openvino_devices()
    required_devices = [item.strip() for item in str(args.require_devices).split(",") if item.strip()]
    missing = missing_devices(available_devices, required_devices)
    if missing:
        summary = {
            "status": "skipped" if args.skip_if_missing_devices else "failed",
            "skip_reason": f"missing required OpenVINO devices: {', '.join(missing)}",
            "required_devices": required_devices,
            "available_devices": available_devices,
            "openvino_error": device_error,
        }
        write_summary(summary, args.summary_out)
        if args.skip_if_missing_devices:
            return
        raise RuntimeError(summary["skip_reason"])

    work_dir = Path(args.work_dir).resolve()
    bundle_root = extract_archive(Path(args.archive).resolve(), work_dir / "extracted")
    exe = find_executable(bundle_root)
    model_root = Path(args.model_root).resolve()
    manifest = model_root / "voice_design" / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"voice_design manifest not found under model root: {manifest}")
    scenario_names = parse_scenarios(args.scenarios)
    if args.mode == "voice_clone" and not args.ref_audio:
        raise ValueError("--mode voice_clone requires --ref-audio")

    request_payload = {
        "mode": args.mode,
        "text": args.text,
        "language": args.language,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": 1,
            "do_sample": False,
        },
        "stream": {
            "chunk_strategy": args.chunk_strategy,
            "format": "pcm_s16le",
        },
    }
    if args.mode == "voice_design":
        request_payload["instruct"] = args.instruct
    else:
        request_payload["ref_audio"] = args.ref_audio
        request_payload["ref_text"] = args.ref_text
        request_payload["x_vector_only"] = bool(args.x_vector_only)
    scenarios = []
    for offset, scenario_name in enumerate(scenario_names):
        scenario = SCENARIOS[scenario_name]
        scenarios.append(
            {
                "name": scenario_name,
                "port": args.base_port + offset,
                "npu_offload": scenario["npu_offload"],
                "decoder_device": None,
            }
        )
    results = []
    for scenario in scenarios:
        results.append(
            run_scenario(
                name=scenario["name"],
                exe=exe,
                model_root=model_root,
                work_dir=work_dir,
                host=args.host,
                port=scenario["port"],
                device=args.device,
                npu_offload=scenario["npu_offload"],
                decoder_device=scenario["decoder_device"],
                no_warmup=args.no_warmup,
                request_payload=request_payload,
                timeout=args.timeout,
                runs=args.runs,
                collect_accelerator_counters=args.collect_accelerator_counters,
                counter_interval_ms=args.counter_interval_ms,
                counter_scope=args.counter_scope,
            )
        )

    comparison = compare_to_gpu_baseline(results)
    acceptance_failures = check_acceptance(
        comparison,
        min_speedup=args.min_speedup,
        max_rtf_regression=args.max_rtf_regression,
        min_gpu_utilization_reduction=args.min_gpu_utilization_reduction,
    )
    status = "ok" if all("error" not in item for item in results) else "failed"
    if acceptance_failures:
        status = "failed"
    summary = {
        "status": status,
        "executable": str(exe),
        "model_root": str(model_root),
        "available_devices": available_devices,
        "request": {
            "text": args.text,
            "mode": args.mode,
            "max_new_tokens": args.max_new_tokens,
            "chunk_strategy": args.chunk_strategy,
            "runs": args.runs,
            "scenarios": scenario_names,
            "collect_accelerator_counters": args.collect_accelerator_counters,
            "counter_interval_ms": args.counter_interval_ms,
            "counter_scope": args.counter_scope,
        },
        "results": results,
        "comparison": comparison,
        "acceptance": {
            "min_speedup": args.min_speedup,
            "max_rtf_regression": args.max_rtf_regression,
            "min_gpu_utilization_reduction": args.min_gpu_utilization_reduction,
            "failures": acceptance_failures,
        },
    }
    write_summary(summary, args.summary_out)
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

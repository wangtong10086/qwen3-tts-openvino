#!/usr/bin/env python3
"""Compare packaged Windows release performance with GPU-only and GPU+NPU decoder paths."""

from __future__ import annotations

import argparse
import json
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
        "speech_encoder_device": metrics[-1].get("speech_encoder_device") if metrics else None,
        "speaker_encoder_device": metrics[-1].get("speaker_encoder_device") if metrics else None,
        "native_codegen_device": metrics[-1].get("native_codegen_device") if metrics else None,
        "npu_offload_effective": metrics[-1].get("npu_offload_effective") if metrics else None,
        "npu_offload_reason": metrics[-1].get("npu_offload_reason") if metrics else None,
    }


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
        for run_index in range(max(1, int(runs))):
            started = time.time()
            stream = run_stream_request(f"http://{host}:{port}/v1/tts/stream", request_payload, timeout=timeout)
            metrics.append(
                {
                    "run_index": run_index,
                    **metric_from_stream(stream, health, time.time() - started),
                }
            )
        summary = aggregate_metrics(metrics)
        expected_offload = {
            "off": "off",
            "auto": "decoder",
            "decoder": "decoder",
            "require": "decoder",
            "audio": "audio",
        }.get(npu_offload)
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
    parser.add_argument("--npu-offload", default="audio", choices=("auto", "decoder", "audio", "require"))
    parser.add_argument("--require-devices", default="GPU,NPU")
    parser.add_argument("--skip-if-missing-devices", action="store_true")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--text", default="你好，这是 Windows GPU 加 NPU 推理性能对比测试。")
    parser.add_argument("--instruct", default="A calm young female voice.")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--chunk-strategy", default="smooth")
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

    request_payload = {
        "mode": "voice_design",
        "text": args.text,
        "language": args.language,
        "instruct": args.instruct,
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
    scenarios = [
        {
            "name": "gpu_only",
            "port": args.base_port,
            "npu_offload": "off",
            "decoder_device": None,
        },
        {
            "name": "gpu_npu_audio" if args.npu_offload == "audio" else "gpu_npu_decoder",
            "port": args.base_port + 1,
            "npu_offload": args.npu_offload,
            "decoder_device": None,
        },
    ]
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
            )
        )

    by_name = {item["name"]: item for item in results}
    gpu_rtf = ((by_name.get("gpu_only") or {}).get("summary") or {}).get("median_computed_rtf")
    npu_result = by_name.get("gpu_npu_audio") or by_name.get("gpu_npu_decoder") or {}
    npu_rtf = (npu_result.get("summary") or {}).get("median_computed_rtf")
    comparison = {
        "computed_rtf_delta": None if gpu_rtf is None or npu_rtf is None else npu_rtf - gpu_rtf,
        "computed_rtf_speedup": None if gpu_rtf is None or not npu_rtf else gpu_rtf / npu_rtf,
    }
    status = "ok" if all("error" not in item for item in results) else "failed"
    summary = {
        "status": status,
        "executable": str(exe),
        "model_root": str(model_root),
        "available_devices": available_devices,
        "request": {
            "text": args.text,
            "max_new_tokens": args.max_new_tokens,
            "chunk_strategy": args.chunk_strategy,
            "runs": args.runs,
        },
        "results": results,
        "comparison": comparison,
    }
    write_summary(summary, args.summary_out)
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

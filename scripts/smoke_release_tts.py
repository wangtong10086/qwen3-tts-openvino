#!/usr/bin/env python3
"""Run a real streaming TTS smoke test against a packaged release."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from smoke_release_package import extract_archive, find_executable, read_health, tail, terminate_process


def parse_device_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def normalize_device_name(value: str | None) -> str:
    return str(value or "").strip().upper()


def device_available(available_devices: list[str], required: str) -> bool:
    required_name = normalize_device_name(required)
    if not required_name:
        return True
    for item in available_devices:
        name = normalize_device_name(item)
        if name == required_name or name.startswith(f"{required_name}."):
            return True
    return False


def missing_devices(available_devices: list[str], required_devices: list[str]) -> list[str]:
    return [device for device in required_devices if not device_available(available_devices, device)]


def query_openvino_devices() -> tuple[list[str], str | None]:
    try:
        import openvino as ov

        return [str(item) for item in ov.Core().available_devices], None
    except Exception as exc:  # pragma: no cover - depends on local OpenVINO install
        return [], str(exc)


def write_summary(summary: dict, summary_out: str | None) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary_out:
        summary_path = Path(summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def wait_for_health(url: str, deadline: float) -> dict:
    last_payload = None
    while time.time() < deadline:
        payload = read_health(url, timeout=2.0)
        if payload is not None:
            last_payload = payload
            if payload.get("ok") is True:
                return payload
        time.sleep(0.5)
    raise TimeoutError(f"server did not become healthy at {url}; last_payload={last_payload!r}")


def run_stream_request(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    audio_bytes = 0
    event_types: list[str] = []
    metadata_event = None
    final_event = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            if not raw_line.strip():
                continue
            event = json.loads(raw_line.decode("utf-8"))
            event_type = event.get("type")
            event_types.append(str(event_type))
            if event_type == "error":
                raise RuntimeError(str(event.get("message")))
            if event_type == "metadata":
                metadata_event = event
            if event_type == "audio":
                audio_bytes += len(base64.b64decode(event["audio"]))
            if event_type == "final":
                final_event = event
                break
    if audio_bytes <= 0:
        raise RuntimeError(f"stream produced no audio; event_types={event_types}")
    if final_event is None:
        raise RuntimeError(f"stream did not produce a final event; event_types={event_types}")
    return {"audio_bytes": audio_bytes, "event_types": event_types, "metadata": metadata_event or {}, "final": final_event}


def health_runtime_values(health: dict, key: str) -> list[str]:
    values: list[str] = []
    warmup = health.get("warmup")
    if isinstance(warmup, dict) and warmup.get(key) is not None:
        values.append(str(warmup[key]))
    runtimes = health.get("runtimes")
    if isinstance(runtimes, dict):
        for runtime in runtimes.values():
            if isinstance(runtime, dict) and runtime.get(key) is not None:
                values.append(str(runtime[key]))
    return values


def assert_expected_device(
    *,
    label: str,
    expected: str | None,
    stream: dict,
    health: dict,
    metadata_key: str,
    health_key: str | None = None,
) -> None:
    if not expected:
        return
    expected_name = normalize_device_name(expected)
    candidates = []
    metadata = stream.get("metadata") if isinstance(stream, dict) else None
    if isinstance(metadata, dict) and metadata.get(metadata_key) is not None:
        candidates.append(str(metadata[metadata_key]))
    candidates.extend(health_runtime_values(health, health_key or metadata_key))
    if not any(normalize_device_name(value) == expected_name for value in candidates):
        raise RuntimeError(f"expected {label}={expected_name}, got candidates={candidates!r}")


def first_stream_or_health_value(stream: dict, health: dict, key: str, default: str | None = None) -> str | None:
    metadata = stream.get("metadata") if isinstance(stream, dict) else None
    if isinstance(metadata, dict) and metadata.get(key) is not None:
        return str(metadata[key])
    values = health_runtime_values(health, key)
    if values:
        return values[0]
    return default


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--work-dir", default="build/release-real-tts")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17970)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--decoder-device", default=None)
    parser.add_argument("--npu-offload", default=None, choices=("off", "auto", "decoder", "require"))
    parser.add_argument("--require-devices", default="")
    parser.add_argument("--skip-if-missing-devices", action="store_true")
    parser.add_argument("--expect-native-codegen-device", default=None)
    parser.add_argument("--expect-decoder-device", default=None)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--text", default="你好。")
    parser.add_argument("--instruct", default="A calm young female voice.")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--do-sample",
        choices=("false", "true", "auto"),
        default="false",
        help="Sampling flag for the request. 'auto' omits do_sample and uses the server default.",
    )
    parser.add_argument("--chunk-strategy", default="smooth")
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args()

    available_devices, device_error = query_openvino_devices()
    required_devices = parse_device_list(args.require_devices)
    missing = missing_devices(available_devices, required_devices)
    if missing:
        summary = {
            "status": "skipped" if args.skip_if_missing_devices else "failed",
            "skip_reason": f"missing required OpenVINO devices: {', '.join(missing)}",
            "required_devices": required_devices,
            "available_devices": available_devices,
            "openvino_error": device_error,
            "device": args.device,
            "decoder_device": args.decoder_device or args.device,
            "npu_offload": args.npu_offload,
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

    log_path = work_dir / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(exe),
        "--model-root",
        str(model_root),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--device",
        args.device,
        "--no-warmup",
        "--realtime-profile",
        "fastest",
        "--ov-cache-dir",
        str(work_dir / "ov-cache"),
    ]
    if args.decoder_device:
        cmd.extend(["--decoder-device", args.decoder_device])
    if args.npu_offload:
        cmd.extend(["--npu-offload", args.npu_offload])
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        health = wait_for_health(f"http://{args.host}:{args.port}/health", deadline=time.time() + 90)
        generation = {
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": 1,
        }
        if args.do_sample != "auto":
            generation["do_sample"] = args.do_sample == "true"

        request_payload = {
            "mode": "voice_design",
            "text": args.text,
            "language": args.language,
            "instruct": args.instruct,
            "generation": generation,
            "stream": {
                "chunk_strategy": args.chunk_strategy,
                "format": "pcm_s16le",
            },
        }
        stream = run_stream_request(
            f"http://{args.host}:{args.port}/v1/tts/stream",
            request_payload,
            timeout=args.timeout,
        )
        assert_expected_device(
            label="native_codegen_device",
            expected=args.expect_native_codegen_device,
            stream=stream,
            health=health,
            metadata_key="native_codegen_device",
        )
        assert_expected_device(
            label="decoder_device",
            expected=args.expect_decoder_device,
            stream=stream,
            health=health,
            metadata_key="decoder_device",
        )
        effective_decoder_device = first_stream_or_health_value(
            stream,
            health,
            "decoder_device",
            args.decoder_device or args.device,
        )
        summary = {
            "status": "ok",
            "executable": str(exe),
            "model_root": str(model_root),
            "device": args.device,
            "decoder_device": effective_decoder_device,
            "npu_offload": args.npu_offload,
            "npu_offload_effective": first_stream_or_health_value(stream, health, "npu_offload_effective", None),
            "required_devices": required_devices,
            "available_devices": available_devices,
            "request": {
                "text": args.text,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "chunk_strategy": args.chunk_strategy,
            },
            "health": {"ok": health.get("ok"), "warmup": health.get("warmup", {})},
            "stream": stream,
        }
        write_summary(summary, args.summary_out)
    except Exception:
        print("--- server log tail ---", file=sys.stderr)
        print(tail(log_path), file=sys.stderr)
        raise
    finally:
        terminate_process(process)


if __name__ == "__main__":
    main()

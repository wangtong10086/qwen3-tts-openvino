#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TEXTS = {
    "voice_design": {
        "short": "你好，这是单一在线调度架构的短文本测试。",
        "long": (
            "这是一段用于验证完整上下文自回归语音合成的长文本。"
            "它必须在同一个 online scheduler 请求中完成，不允许切分成多个独立 prompt。"
            "请保持稳定、自然、清晰的语气连续朗读完整内容。"
        ),
    },
    "custom_voice": {
        "short": "你好，这是 CustomVoice 单一在线调度架构的短文本测试。",
        "long": (
            "这是一段用于验证 CustomVoice 长文本完整上下文生成的测试内容。"
            "它需要保持同一个发言人的音色、语气和节奏，不允许通过自动切段来规避长自回归。"
            "请完整、稳定、自然地朗读。"
        ),
    },
    "voice_clone": {
        "short": "你好，这是 VoiceClone 单一在线调度架构的短文本测试。",
        "long": (
            "这是一段用于验证 VoiceClone 长文本完整上下文生成的测试内容。"
            "它需要参考音频中的音色和说话风格，并在同一条自回归链路中完成目标文本生成。"
            "请保持声音风格一致，不要出现噪音、截断或重复。"
        ),
    },
}

DEFAULT_INSTRUCT = "用自然、清晰、稳定的中文女声朗读。"
EXPECTED_PRODUCTION_PROFILE = "minimal-online-paged-kv"
FALLBACK_COUNTER_SUFFIXES = ("fallback", "fallback_count", "retry_count")
FAIL_FAST_TIMING_KEYS = {
    "fallback",
    "generation_fallback",
    "unroll_fallback",
}


@dataclass(frozen=True)
class Scenario:
    mode: str
    text_kind: str
    concurrency: int
    run: int


def parse_csv_strings(value: str | None, *, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    items = [item.strip().replace("-", "_") for item in str(value).split(",") if item.strip()]
    return items or list(default)


def parse_csv_ints(value: str | None, *, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    items: list[int] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if raw:
            items.append(int(raw))
    return items or list(default)


def percentile(values: list[float], q: float) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return math.inf
    index = min(len(clean) - 1, max(0, int(round((len(clean) - 1) * q))))
    return float(clean[index])


def http_json(url: str, *, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_health(
    base_url: str,
    *,
    timeout_sec: float,
    wait_warmup: bool = False,
    allow_warmup_errors: bool = False,
) -> dict[str, Any]:
    deadline = time.time() + float(timeout_sec)
    last_error: str | None = None
    while time.time() < deadline:
        try:
            health = http_json(base_url.rstrip("/") + "/health", timeout=3.0)
            if not wait_warmup:
                return health
            warmup = health.get("warmup") or {}
            status = str(warmup.get("status") or "")
            if status in {"disabled", "ready"}:
                return health
            if status == "ready_with_errors":
                if allow_warmup_errors:
                    return health
                raise RuntimeError(
                    "server warmup completed with errors: "
                    + json.dumps(warmup.get("errors") or {}, ensure_ascii=False)
                )
            last_error = f"warmup status={status or 'missing'}"
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        except RuntimeError:
            raise
        time.sleep(0.5)
    raise TimeoutError(f"server did not become healthy within {timeout_sec:.1f}s: {last_error or 'unknown error'}")


def ws_url_from_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://") :] + "/v1/tts/stream"
    if base.startswith("http://"):
        return "ws://" + base[len("http://") :] + "/v1/tts/stream"
    if base.startswith(("ws://", "wss://")):
        return base + "/v1/tts/stream" if not base.endswith("/v1/tts/stream") else base
    return "ws://" + base + "/v1/tts/stream"


def load_quality_module():
    script = Path(__file__).resolve().with_name("evaluate_long_text_quality.py")
    spec = importlib.util.spec_from_file_location("evaluate_long_text_quality", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_texts(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    texts = json.loads(json.dumps(DEFAULT_TEXTS))
    if args.text:
        for mode in texts:
            texts[mode]["short"] = str(args.text)
    if args.text_file:
        content = Path(args.text_file).read_text(encoding="utf-8").strip()
        if content:
            for mode in texts:
                texts[mode]["long"] = content
    return texts


def read_ref_text(args: argparse.Namespace) -> str:
    ref_text_file = getattr(args, "ref_text_file", None)
    if ref_text_file:
        return Path(ref_text_file).read_text(encoding="utf-8").strip()
    return str(getattr(args, "ref_text", "") or "")


def build_request(args: argparse.Namespace, *, mode: str, text_kind: str, index: int, text: str) -> dict[str, Any]:
    long_text = text_kind == "long"
    max_new_tokens = int(args.long_max_new_tokens if long_text else args.max_new_tokens)
    payload: dict[str, Any] = {
        "mode": mode,
        "text": text,
        "language": args.language,
        "full_context_text": bool(long_text),
        "auto_segment_text": False,
        "allow_auto_segment_text": False,
        "force_auto_segment_text": False,
        "max_vram_ratio": float(args.max_vram_ratio),
        "generation": {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": int(args.min_new_tokens),
            "do_sample": bool(args.do_sample),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "temperature": float(args.temperature),
            "repetition_penalty": float(args.repetition_penalty),
        },
        "stream": {
            "format": "pcm_s16le",
            "include_chunk_metadata": True,
            "chunk_strategy": args.chunk_strategy,
        },
    }
    if args.chunk_frames > 0:
        payload["stream"]["chunk_frames"] = int(args.chunk_frames)
    if args.left_context_frames >= 0:
        payload["stream"]["left_context_frames"] = int(args.left_context_frames)
    if args.initial_chunk_frames > 0:
        payload["stream"]["initial_chunk_frames"] = int(args.initial_chunk_frames)
    if mode == "voice_design":
        payload["instruct"] = args.instruct
    elif mode == "custom_voice":
        payload["speaker"] = args.speaker
        payload["instruct"] = args.instruct
    elif mode == "voice_clone":
        if not args.ref_audio:
            raise ValueError("voice_clone gate requires --ref-audio")
        payload["ref_audio"] = args.ref_audio
        payload["ref_text"] = read_ref_text(args)
        payload["x_vector_only"] = bool(args.x_vector_only)
        payload["generation"]["seed"] = int(args.seed) + index
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return payload


def write_pcm_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm)


def timing_fallback_failures(source: dict[str, Any], *, prefix: str) -> list[str]:
    failures: list[str] = []
    for key, value in source.items():
        normalized = str(key)
        if normalized in FAIL_FAST_TIMING_KEYS and bool(value):
            failures.append(f"{prefix}.{normalized}={value}")
            continue
        if normalized.endswith(FALLBACK_COUNTER_SUFFIXES):
            if isinstance(value, bool):
                if value:
                    failures.append(f"{prefix}.{normalized}=true")
            elif isinstance(value, (dict, list, tuple, str)):
                if value:
                    failures.append(f"{prefix}.{normalized}={value}")
            else:
                try:
                    if float(value) > 0:
                        failures.append(f"{prefix}.{normalized}={value}")
                except (TypeError, ValueError):
                    pass
    return failures


def architecture_failures(record: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    metadata = record.get("metadata") or {}
    final = record.get("final") or {}
    final_timings = final.get("timings") or record.get("final_timings") or {}
    if metadata.get("generation_fallback_allowed") is not False:
        failures.append("metadata.generation_fallback_allowed_not_false")
    production_profile = metadata.get("production_profile")
    if production_profile not in {None, EXPECTED_PRODUCTION_PROFILE}:
        failures.append(f"metadata.production_profile={production_profile!r}")
    if str(metadata.get("online_batching") or "").lower() != "on":
        failures.append(f"metadata.online_batching={metadata.get('online_batching')!r}")
    scheduler = metadata.get("online_batch_scheduler") or final_timings.get("online_batch_scheduler")
    if str(scheduler or "").replace("-", "_") != "layered":
        failures.append(f"online_batch_scheduler={scheduler!r}")
    backend_candidates = [
        metadata.get("continuous_backend"),
        metadata.get("inference_backend"),
        final_timings.get("inference_backend"),
    ]
    if not any("vllm_like_online_scheduler" in str(item or "") for item in backend_candidates):
        failures.append(f"backend={backend_candidates!r}")
    if bool(metadata.get("segmented")) or bool(metadata.get("auto_segment_text")):
        failures.append("segmented_or_auto_segmented")
    if record.get("text_kind") == "long" and metadata.get("full_context_text") is not True:
        failures.append("long_request_not_full_context")
    failures.extend(timing_fallback_failures(metadata, prefix="metadata"))
    for index, timings in enumerate(record.get("chunk_timings") or []):
        if timings.get("online_batching") is not True:
            failures.append(f"chunk{index}.online_batching_not_true")
        failures.extend(timing_fallback_failures(timings, prefix=f"chunk{index}"))
    failures.extend(timing_fallback_failures(final_timings, prefix="final"))
    return failures


PERFORMANCE_KEYS = (
    "stream_rtf",
    "stream_compute_rtf",
    "decode_ms",
    "codegen_ms",
    "native_pipeline_ms",
    "native_ttft_ms",
    "codegen_infer_ms",
    "codegen_prefill_infer_ms",
    "codegen_decode_infer_ms",
    "codegen_subcode_infer_ms",
    "host_prepare_ms",
    "host_copy_ms",
    "tensor_bind_ms",
    "subcode_bind_ms",
    "subcode_output_read_ms",
    "subcode_next_embed_ms",
    "sampling_ms",
    "prompt_ms",
    "ref_audio_ms",
)


def numeric_value(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def collect_performance(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    final = record.get("final") or {}
    final_timings = final.get("timings") or {}
    chunk_timings = [item for item in (record.get("chunk_timings") or []) if isinstance(item, dict)]
    audio_sec = numeric_value(record.get("audio_sec")) or 0.0
    server_elapsed = numeric_value(final.get("elapsed"))
    perf: dict[str, Any] = {
        "first_audio_ms": record.get("first_audio_ms"),
        "end_to_end_rtf": record.get("computed_rtf"),
        "audio_sec": record.get("audio_sec"),
        "decode_path": final_timings.get("decode_path") or metadata.get("decode_path"),
        "inference_backend": final_timings.get("inference_backend") or metadata.get("inference_backend"),
    }
    if server_elapsed is not None:
        perf["server_elapsed_ms"] = server_elapsed * 1000.0
        perf["server_rtf"] = server_elapsed / max(1.0e-9, audio_sec)
    for key in PERFORMANCE_KEYS:
        value = numeric_value(final_timings.get(key))
        if value is None:
            value = numeric_value(metadata.get(key))
        if value is None and chunk_timings:
            clean = [item for item in (numeric_value(chunk.get(key)) for chunk in chunk_timings) if item is not None]
            if clean:
                value = clean[-1] if key.endswith("rtf") else sum(clean)
        if value is not None:
            perf[key] = value
    for key in (
        "sampled_batch_subcode_fallback_count",
        "subcode_host_copy_fallback_count",
        "split_subcode_hidden_bind_fallback_count",
        "split_subcode_remote_next_embed_fallback_count",
        "prompt_component_cache_hits",
        "prompt_component_cache_misses",
        "voice_clone_prompt_cache_hits",
        "voice_clone_prompt_cache_misses",
    ):
        value = final_timings.get(key, metadata.get(key))
        if value is None and chunk_timings:
            value = chunk_timings[-1].get(key)
        parsed = numeric_value(value)
        if parsed is not None:
            perf[key] = int(parsed)
    return perf


RTF_METRIC_CHOICES = {
    "computed_rtf",
    "end_to_end_rtf",
    "server_rtf",
    "stream_rtf",
    "stream_compute_rtf",
}


def rtf_metric_value(record: dict[str, Any], metric: str) -> float:
    normalized = str(metric or "computed_rtf").strip().lower()
    if normalized in {"computed_rtf", "end_to_end_rtf"}:
        return float(record.get("computed_rtf", math.inf))
    if normalized == "server_rtf":
        perf = record.get("performance") or {}
        value = perf.get("server_rtf")
        return math.inf if value is None else float(value)
    perf = record.get("performance") or {}
    value = perf.get(normalized)
    if value is None:
        return math.inf
    return float(value)


def gate_rtf_metric_name(args: argparse.Namespace, text_kind: str) -> str:
    return str(args.long_rtf_metric if text_kind == "long" else args.short_rtf_metric)


async def run_ws_request(
    ws_url: str,
    payload: dict[str, Any],
    *,
    timeout_sec: float,
    wav_path: Path,
    scenario: Scenario,
    request_index: int,
    arrival_delay_ms: float,
) -> dict[str, Any]:
    import websockets

    await asyncio.sleep(max(0.0, arrival_delay_ms) / 1000.0)
    started = time.perf_counter()
    metadata: dict[str, Any] = {}
    final: dict[str, Any] = {}
    chunk_timings: list[dict[str, Any]] = []
    pcm_parts: list[bytes] = []
    first_audio_at: float | None = None
    sample_rate = 24000
    error: str | None = None
    try:
        async with websockets.connect(ws_url, max_size=None, open_timeout=timeout_sec) as ws:
            await ws.send(json.dumps(payload, ensure_ascii=False))
            while True:
                message = await asyncio.wait_for(ws.recv(), timeout=timeout_sec)
                now = time.perf_counter()
                if isinstance(message, bytes):
                    if first_audio_at is None:
                        first_audio_at = now
                    pcm_parts.append(message)
                    continue
                item = json.loads(message)
                item_type = item.get("type")
                if item_type == "metadata":
                    metadata = item
                    sample_rate = int(item.get("sample_rate") or sample_rate)
                elif item_type == "audio":
                    sample_rate = int(item.get("sample_rate") or sample_rate)
                    chunk_timings.append(dict(item.get("timings") or {}))
                elif item_type == "final":
                    final = item
                    break
                elif item_type == "error":
                    error = str(item.get("message") or item)
                    break
    except Exception as exc:
        error = str(exc)
    finished = time.perf_counter()
    pcm = b"".join(pcm_parts)
    if pcm:
        write_pcm_wav(wav_path, pcm, sample_rate)
    audio_sec = len(pcm) / 2.0 / float(sample_rate) if sample_rate > 0 else 0.0
    elapsed_sec = max(0.0, finished - started)
    record = {
        "mode": scenario.mode,
        "text_kind": scenario.text_kind,
        "concurrency": scenario.concurrency,
        "run": scenario.run,
        "index": request_index,
        "text": str(payload.get("text") or ""),
        "ok": error is None and bool(pcm),
        "error": error,
        "wav_path": str(wav_path) if pcm else None,
        "sample_rate": int(sample_rate),
        "audio_bytes": int(len(pcm)),
        "audio_sec": float(audio_sec),
        "elapsed_ms": float(elapsed_sec * 1000.0),
        "first_audio_ms": float(((first_audio_at or finished) - started) * 1000.0),
        "computed_rtf": float(elapsed_sec / max(1.0e-9, audio_sec)) if audio_sec > 0 else math.inf,
        "metadata": metadata,
        "chunk_timings": chunk_timings,
        "final": final,
    }
    record["performance"] = collect_performance(record)
    failures = architecture_failures(record)
    if failures:
        record["ok"] = False
        record["architecture_failures"] = failures
    return record


async def run_scenario(args: argparse.Namespace, ws_url: str, scenario: Scenario, texts: dict[str, dict[str, str]], out_dir: Path) -> dict[str, Any]:
    text = texts[scenario.mode][scenario.text_kind]
    tasks = []
    started = time.perf_counter()
    for index in range(scenario.concurrency):
        payload = build_request(args, mode=scenario.mode, text_kind=scenario.text_kind, index=index, text=text)
        wav_path = out_dir / scenario.mode / scenario.text_kind / f"run{scenario.run:02d}_req{index:02d}.wav"
        tasks.append(
            run_ws_request(
                ws_url,
                payload,
                timeout_sec=float(args.request_timeout_sec),
                wav_path=wav_path,
                scenario=scenario,
                request_index=index,
                arrival_delay_ms=float(args.arrival_gap_ms) * index,
            )
        )
    results = await asyncio.gather(*tasks)
    elapsed_sec = time.perf_counter() - started
    aggregate_audio_sec = sum(float(item.get("audio_sec") or 0.0) for item in results)
    ttft_values = [float(item.get("first_audio_ms", math.inf)) for item in results]
    rtf_values = [float(item.get("computed_rtf", math.inf)) for item in results]
    stream_rtf_values = [
        float((item.get("performance") or {}).get("stream_rtf", math.inf))
        for item in results
        if math.isfinite(float((item.get("performance") or {}).get("stream_rtf", math.inf)))
    ]
    stream_compute_rtf_values = [
        float((item.get("performance") or {}).get("stream_compute_rtf", math.inf))
        for item in results
        if math.isfinite(float((item.get("performance") or {}).get("stream_compute_rtf", math.inf)))
    ]
    gate_metric = gate_rtf_metric_name(args, scenario.text_kind)
    gate_rtf_values = [rtf_metric_value(item, gate_metric) for item in results]
    summary = {
        "mode": scenario.mode,
        "text_kind": scenario.text_kind,
        "concurrency": scenario.concurrency,
        "run": scenario.run,
        "elapsed_ms": float(elapsed_sec * 1000.0),
        "aggregate_audio_sec": float(aggregate_audio_sec),
        "aggregate_rtf": float(elapsed_sec / max(1.0e-9, aggregate_audio_sec)),
        "ttft_ms_p90": percentile(ttft_values, 0.90),
        "computed_rtf_p90": percentile(rtf_values, 0.90),
        "stream_rtf_p90": percentile(stream_rtf_values, 0.90),
        "stream_compute_rtf_p90": percentile(stream_compute_rtf_values, 0.90),
        "gate_rtf_metric": gate_metric,
        "gate_rtf_p90": percentile(gate_rtf_values, 0.90),
        "requests": results,
    }
    return summary


def judge_quality(args: argparse.Namespace, scenario_summaries: list[dict[str, Any]]) -> None:
    quality = load_quality_module()
    omni_enabled = bool(args.require_omni and not args.skip_omni)
    omni_config = quality.load_aliyun_env(args.env_file) if omni_enabled else {"api_key": None, "model": None, "base_url": None}
    if omni_enabled and (not omni_config.get("api_key") or not omni_config.get("model")):
        status = quality.redacted_env_status(omni_config)
        raise SystemExit(
            "Omni judge is required but credentials are incomplete. "
            f"status={json.dumps(status, ensure_ascii=False)}"
        )
    max_base64_bytes = int(float(args.omni_max_audio_mb) * 1024 * 1024)
    for scenario in scenario_summaries:
        for item in scenario["requests"]:
            wav_path = item.get("wav_path")
            if not wav_path or not Path(wav_path).exists():
                item["quality_passed"] = False
                item["objective_gate"] = {"pass": False, "failures": ["missing_wav"]}
                continue
            audio_metrics = quality.objective_audio_metrics(wav_path)
            gate = quality.objective_gate(audio_metrics, None)
            item["objective_audio"] = audio_metrics
            item["objective_gate"] = gate
            if omni_enabled and gate.get("pass"):
                item["omni"] = quality.judge_wav_with_omni(
                    wav_path,
                    text=str(item.get("text") or ""),
                    config=omni_config,
                    max_base64_bytes=max_base64_bytes,
                    segment_seconds=float(args.segment_seconds),
                )
            elif omni_enabled:
                item["omni"] = {"pass": False, "skipped": "objective_gate_failed"}
            item["quality_passed"] = bool(gate.get("pass") and (not omni_enabled or item.get("omni", {}).get("pass")))


def summarize_gate(args: argparse.Namespace, scenarios: list[dict[str, Any]], health: dict[str, Any]) -> dict[str, Any]:
    mode_summaries: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    baseline_regressions = baseline_aggregate_regressions(
        scenarios,
        baseline_json=getattr(args, "baseline_json", None),
        max_regression=float(getattr(args, "max_aggregate_rtf_regression", 0.05)),
    )
    for mode in parse_csv_strings(args.modes, default=list(DEFAULT_TEXTS)):
        mode_records = [
            request
            for scenario in scenarios
            if scenario["mode"] == mode
            for request in scenario.get("requests", [])
        ]
        single_records = [
            request
            for scenario in scenarios
            if scenario["mode"] == mode and int(scenario.get("concurrency") or 0) == 1
            for request in scenario.get("requests", [])
        ]
        rtf = [float(item.get("computed_rtf", math.inf)) for item in mode_records]
        gate_rtf = [
            rtf_metric_value(item, gate_rtf_metric_name(args, str(item.get("text_kind") or "")))
            for item in single_records
        ]
        ttft = [float(item.get("first_audio_ms", math.inf)) for item in single_records]
        perf_records = [item.get("performance") or {} for item in mode_records]
        stream_rtf = [
            float(perf["stream_rtf"])
            for perf in perf_records
            if perf.get("stream_rtf") is not None and math.isfinite(float(perf["stream_rtf"]))
        ]
        stream_compute_rtf = [
            float(perf["stream_compute_rtf"])
            for perf in perf_records
            if perf.get("stream_compute_rtf") is not None and math.isfinite(float(perf["stream_compute_rtf"]))
        ]
        native_ttft = [
            float(perf["native_ttft_ms"])
            for perf in perf_records
            if perf.get("native_ttft_ms") is not None and math.isfinite(float(perf["native_ttft_ms"]))
        ]
        quality_ok = all(bool(item.get("quality_passed")) for item in mode_records) if mode_records else False
        architecture_ok = all(not item.get("architecture_failures") for item in mode_records) if mode_records else False
        request_ok = all(bool(item.get("ok")) for item in mode_records) if mode_records else False
        rtf_p90 = percentile(rtf, 0.90)
        gate_rtf_p90 = percentile(gate_rtf, 0.90)
        ttft_p90 = percentile(ttft, 0.90)
        mode_baseline_regressions = [item for item in baseline_regressions if item.get("mode") == mode]
        performance_ok = (
            gate_rtf_p90 < float(args.max_rtf_p90)
            and ttft_p90 <= float(args.max_ttft_p90_ms)
            and not mode_baseline_regressions
        )
        mode_passed = bool(request_ok and architecture_ok and quality_ok and performance_ok)
        if not mode_passed:
            failures.append(mode)
        mode_summaries[mode] = {
            "passed": mode_passed,
            "request_ok": request_ok,
            "architecture_ok": architecture_ok,
            "quality_passed": quality_ok,
            "performance_passed": performance_ok,
            "computed_rtf_p90": rtf_p90,
            "gate_rtf_p90": gate_rtf_p90,
            "gate_rtf_metric": {
                "short": args.short_rtf_metric,
                "long": args.long_rtf_metric,
            },
            "ttft_ms_p90": ttft_p90,
            "stream_rtf_p90": percentile(stream_rtf, 0.90),
            "stream_compute_rtf_p90": percentile(stream_compute_rtf, 0.90),
            "native_ttft_ms_p90": percentile(native_ttft, 0.90),
            "baseline_regressions": mode_baseline_regressions,
            "requests": len(mode_records),
            "single_concurrency_requests": len(single_records),
        }
    return {
        "feature": "single_arch_sidecar_gate",
        "passed": not failures,
        "created_at_unix": time.time(),
        "thresholds": {
            "max_rtf_p90": float(args.max_rtf_p90),
            "max_ttft_p90_ms": float(args.max_ttft_p90_ms),
            "short_rtf_metric": args.short_rtf_metric,
            "long_rtf_metric": args.long_rtf_metric,
            "max_aggregate_rtf_regression": float(args.max_aggregate_rtf_regression),
        },
        "server": {
            "base_url": args.server_url,
            "health_online_batching": (health.get("online_batching") or {}),
            "warmup": health.get("warmup"),
        },
        "modes": mode_summaries,
        "scenarios": scenarios,
        "baseline_regressions": baseline_regressions,
        "failures": failures,
    }


def scenario_key(scenario: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(scenario.get("mode") or ""),
        str(scenario.get("text_kind") or ""),
        int(scenario.get("concurrency") or 0),
    )


def baseline_aggregate_regressions(
    scenarios: list[dict[str, Any]],
    *,
    baseline_json: str | None,
    max_regression: float,
) -> list[dict[str, Any]]:
    if not baseline_json:
        return []
    path = Path(baseline_json)
    if not path.exists():
        raise FileNotFoundError(f"baseline JSON not found: {path}")
    baseline = json.loads(path.read_text(encoding="utf-8"))
    baseline_by_key: dict[tuple[str, str, int], float] = {}
    for item in baseline.get("scenarios", []):
        value = numeric_value(item.get("aggregate_rtf"))
        if value is not None:
            baseline_by_key[scenario_key(item)] = value
    regressions: list[dict[str, Any]] = []
    for scenario in scenarios:
        key = scenario_key(scenario)
        baseline_value = baseline_by_key.get(key)
        current_value = numeric_value(scenario.get("aggregate_rtf"))
        if baseline_value is None or current_value is None or baseline_value <= 0:
            continue
        ratio = current_value / baseline_value
        if ratio > 1.0 + max_regression:
            regressions.append(
                {
                    "mode": key[0],
                    "text_kind": key[1],
                    "concurrency": key[2],
                    "baseline_aggregate_rtf": baseline_value,
                    "current_aggregate_rtf": current_value,
                    "regression_ratio": ratio,
                }
            )
    return regressions


def start_server(args: argparse.Namespace) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "qwen3_tts_ov",
        "serve",
        "--model-root",
        args.model_root,
        "--device",
        args.device,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--realtime-profile",
        args.realtime_profile,
        "--online-batching",
        "on",
        "--online-batch-scheduler",
        "layered",
        "--online-batch-max-size",
        str(max(parse_csv_ints(args.concurrency, default=[1]))),
        "--online-batch-max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--online-batch-fused-decode",
        args.online_batch_fused_decode,
        "--sampled-batch-subcode",
        args.sampled_batch_subcode,
        "--online-batch-continuous-subcode",
        args.online_batch_continuous_subcode,
        "--max-concurrent-tts",
        str(max(parse_csv_ints(args.concurrency, default=[1]))),
        "--preload-modes",
        args.preload_modes,
        "--preload-buckets",
        args.preload_buckets,
        "--warmup-strategy",
        args.warmup_strategy,
        "--warmup-speaker",
        args.speaker,
        "--runtime-residency",
        args.runtime_residency,
    ]
    if args.ref_audio:
        cmd.extend(["--warmup-ref-audio", args.ref_audio])
    if args.ref_text_file:
        cmd.extend(["--warmup-ref-text-file", args.ref_text_file])
    elif args.ref_text:
        cmd.extend(["--warmup-ref-text", args.ref_text])
    if args.no_warmup:
        cmd.append("--no-warmup")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(cmd, cwd=str(Path.cwd()), env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end gate for the single sidecar inference architecture.")
    parser.add_argument("--server-url", default="http://127.0.0.1:17860")
    parser.add_argument("--ws-url", default=None)
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17860)
    parser.add_argument("--model-root", default="openvino")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--realtime-profile", default="fastest")
    parser.add_argument("--preload-modes", default="voice_design,custom_voice,voice_clone")
    parser.add_argument("--preload-buckets", default="warmup")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--warmup-strategy", default="smooth")
    parser.add_argument("--no-wait-warmup", action="store_true")
    parser.add_argument("--allow-warmup-errors", action="store_true")
    parser.add_argument("--serve-timeout-sec", type=float, default=180.0)
    parser.add_argument("--request-timeout-sec", type=float, default=240.0)
    parser.add_argument("--modes", default="voice_design,custom_voice,voice_clone")
    parser.add_argument("--text-kinds", default="short,long")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--arrival-gap-ms", type=float, default=20.0)
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--ref-audio", default=None)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--ref-text-file", default=None)
    parser.add_argument("--x-vector-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--long-max-new-tokens", type=int, default=0)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--do-sample", action="store_true", default=True)
    parser.add_argument("--greedy", dest="do_sample", action="store_false")
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-strategy", default="auto")
    parser.add_argument("--initial-chunk-frames", type=int, default=0)
    parser.add_argument("--chunk-frames", type=int, default=0)
    parser.add_argument("--left-context-frames", type=int, default=25)
    parser.add_argument("--max-vram-ratio", type=float, default=80.0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32)
    parser.add_argument("--online-batch-fused-decode", default="off", choices=["auto", "on", "off"])
    parser.add_argument("--sampled-batch-subcode", default="off", choices=["auto", "off", "verify", "on"])
    parser.add_argument("--online-batch-continuous-subcode", default="off", choices=["auto", "off", "on"])
    parser.add_argument("--runtime-residency", default="lazy", choices=["lazy", "all"])
    parser.add_argument("--short-rtf-metric", default="stream_compute_rtf", choices=sorted(RTF_METRIC_CHOICES))
    parser.add_argument("--long-rtf-metric", default="server_rtf", choices=sorted(RTF_METRIC_CHOICES))
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--max-aggregate-rtf-regression", type=float, default=0.05)
    parser.add_argument("--max-rtf-p90", type=float, default=1.0)
    parser.add_argument("--max-ttft-p90-ms", type=float, default=1500.0)
    parser.add_argument("--require-omni", action="store_true")
    parser.add_argument("--skip-omni", action="store_true")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--omni-max-audio-mb", type=float, default=8.0)
    parser.add_argument("--segment-seconds", type=float, default=8.0)
    parser.add_argument("--out-dir", default="outputs/single_arch_gate")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.start_server and args.server_url == parser.get_default("server_url"):
        args.server_url = f"http://{args.host}:{args.port}"
    if int(args.long_max_new_tokens) <= 0:
        args.long_max_new_tokens = max(512, int(args.max_new_tokens))
    modes = parse_csv_strings(args.modes, default=list(DEFAULT_TEXTS))
    text_kinds = parse_csv_strings(args.text_kinds, default=["short", "long"])
    concurrency = parse_csv_ints(args.concurrency, default=[1, 2, 4, 8])
    scenarios = [
        Scenario(mode=mode, text_kind=text_kind, concurrency=batch, run=run)
        for mode in modes
        for text_kind in text_kinds
        for batch in concurrency
        for run in range(int(args.runs))
    ]
    texts = read_texts(args)
    if args.dry_run:
        print(json.dumps({"scenarios": [scenario.__dict__ for scenario in scenarios]}, ensure_ascii=False, indent=2))
        return 0
    proc = start_server(args) if args.start_server else None
    try:
        health = wait_for_health(
            args.server_url,
            timeout_sec=float(args.serve_timeout_sec),
            wait_warmup=bool(args.start_server and not args.no_warmup and not args.no_wait_warmup),
            allow_warmup_errors=bool(args.allow_warmup_errors),
        )
        ws_url = args.ws_url or ws_url_from_base(args.server_url)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        scenario_results = [
            asyncio.run(run_scenario(args, ws_url, scenario, texts, out_dir))
            for scenario in scenarios
        ]
        judge_quality(args, scenario_results)
        summary = summarize_gate(args, scenario_results, health)
        output = Path(args.output_json) if args.output_json else out_dir / "quality_summary.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            "passed={passed} modes={modes} wrote {output}".format(
                passed=summary["passed"],
                modes=",".join(f"{key}:{'pass' if value['passed'] else 'fail'}" for key, value in summary["modes"].items()),
                output=output,
            ),
            flush=True,
        )
        return 0 if summary["passed"] else 1
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())

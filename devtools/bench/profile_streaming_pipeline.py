import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import openvino as ov


DEFAULT_TEXT = "你好，这是一次用于系统分析实时流式合成性能的 OpenVINO 语音生成。"
DEFAULT_INSTRUCT = "用自然、清晰的中文女声朗读。"

PROFILES = {
    "int8_fused": {
        "mode": "realtime-int8",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8_fused",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
    },
    "int8_sym_fused": {
        "mode": "realtime-int8-sym",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8_sym_fused",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
    },
    "int8_sym_unroll4": {
        "mode": "realtime-int8-sym",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8_sym_fused",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
    },
    "int8_sym_ll_v2": {
        "mode": "realtime-int8-sym",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8_sym_fused",
        "codegen_unroll": "4",
        "codegen_schedule": "ll-v2",
        "codegen_decode_unroll": "off",
    },
    "int8_sym_balanced_v2": {
        "mode": "realtime-int8-sym",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8_sym_fused",
        "codegen_unroll": "4",
        "codegen_schedule": "balanced-v2",
        "codegen_decode_unroll": "off",
    },
    "int8_sym_unroll4_decode": {
        "mode": "realtime-int8-sym",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "int8_sym_fused",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "auto",
    },
    "fp16_fused": {
        "mode": "cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "1",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
    },
    "fp16_unroll4": {
        "mode": "cache",
        "cache_kernel": "exact",
        "cache_step": "fused",
        "graph_variant": "fp16",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "off",
    },
}

DEVICE_KEYS = (
    "FULL_DEVICE_NAME",
    "OPTIMIZATION_CAPABILITIES",
    "RANGE_FOR_STREAMS",
    "RANGE_FOR_ASYNC_INFER_REQUESTS",
    "NUM_STREAMS",
    "PERFORMANCE_HINT",
    "PERFORMANCE_HINT_NUM_REQUESTS",
    "INFERENCE_PRECISION_HINT",
    "GPU_EXECUTION_UNITS_COUNT",
    "GPU_DEVICE_TOTAL_MEM_SIZE",
    "GPU_DEVICE_MAX_ALLOC_MEM_SIZE",
    "COMPILATION_NUM_THREADS",
)


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def pct(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100.0) * (len(ordered) - 1))))
    return float(ordered[index])


def median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def safe_property(core: ov.Core, device: str, key: str) -> Any:
    try:
        value = core.get_property(device, key)
    except Exception as exc:
        return {"error": str(exc).splitlines()[0]}
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def device_report() -> dict:
    core = ov.Core()
    report = {"available_devices": list(core.available_devices), "devices": {}}
    for device in core.available_devices:
        report["devices"][device] = {key: safe_property(core, device, key) for key in DEVICE_KEYS}
    return report


def flatten_graphs(prefix: str, value: Any) -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(prefix, value)]
    if isinstance(value, dict):
        items = []
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(flatten_graphs(child_prefix, child))
        return items
    return []


def graph_inventory(ir_dir: Path, manifest: dict) -> dict:
    graphs = []
    for name, graph in flatten_graphs("graphs", manifest.get("graphs", {})):
        path = ir_dir / graph
        graphs.append(
            {
                "name": name,
                "graph": graph,
                "exists": path.exists(),
                "xml_bytes": path.stat().st_size if path.exists() else None,
                "bin_bytes": path.with_suffix(".bin").stat().st_size if path.with_suffix(".bin").exists() else None,
            }
        )
    variants = {}
    for variant, entry in sorted((manifest.get("graph_variants") or {}).items()):
        variant_graphs = []
        for name, graph in flatten_graphs("graphs", entry.get("graphs", {})):
            path = ir_dir / graph
            variant_graphs.append(
                {
                    "name": name,
                    "graph": graph,
                    "exists": path.exists(),
                    "xml_bytes": path.stat().st_size if path.exists() else None,
                    "bin_bytes": path.with_suffix(".bin").stat().st_size if path.with_suffix(".bin").exists() else None,
                }
            )
        variants[variant] = {"precision": entry.get("precision"), "graphs": variant_graphs}
    return {
        "ir_dir": str(ir_dir),
        "tts_model_type": manifest.get("tts_model_type"),
        "sample_rate": manifest.get("sample_rate"),
        "num_code_groups": manifest.get("num_code_groups"),
        "graphs": graphs,
        "graph_variants": variants,
        "streaming_decoder": manifest.get("streaming_decoder", {}),
    }


def parse_compile_config(items: list[str]) -> dict:
    config = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--compile-config entries must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        value = value.strip()
        if value.lower() == "true":
            parsed: Any = True
        elif value.lower() == "false":
            parsed = False
        else:
            parsed = value
        config[key.strip()] = parsed
    return config


def simulate_playback(events: list[dict], playback_buffer_ms: float) -> dict:
    audio_events = [event for event in events if event.get("audio_ms", 0.0) > 0.0]
    if not audio_events:
        return {"playback_buffer_ms": playback_buffer_ms, "underrun_count": 0, "empty_ms": 0.0, "min_queue_ms": 0.0}

    start_play_ms = float(audio_events[0]["arrival_ms"]) + float(playback_buffer_ms)
    queue_ms = 0.0
    last_ms = float(audio_events[0]["arrival_ms"])
    underrun_count = 0
    empty_ms = 0.0
    min_queue_ms = float("inf")
    for event in audio_events:
        arrival_ms = float(event["arrival_ms"])
        if arrival_ms > start_play_ms:
            consume_start = max(last_ms, start_play_ms)
            consumed = max(0.0, arrival_ms - consume_start)
            if consumed > queue_ms:
                if queue_ms > 0.0 or consumed > 0.0:
                    underrun_count += 1
                empty_ms += consumed - queue_ms
                queue_ms = 0.0
            else:
                queue_ms -= consumed
        queue_ms += float(event.get("audio_ms", 0.0))
        min_queue_ms = min(min_queue_ms, queue_ms)
        last_ms = arrival_ms
    return {
        "playback_buffer_ms": float(playback_buffer_ms),
        "underrun_count": int(underrun_count),
        "empty_ms": float(empty_ms),
        "min_queue_ms": 0.0 if min_queue_ms == float("inf") else float(min_queue_ms),
        "final_queue_ms": float(queue_ms),
    }


def reset_runtime_counters(runtime) -> None:
    runtime.timings.values = {}
    runtime.ov_profiler.ops = {}


def profile_hot_run(runtime, args: argparse.Namespace, run_index: int) -> dict:
    reset_runtime_counters(runtime)
    started = time.time()
    first_audio_ms = None
    audio_samples = 0
    chunk_events = []
    final_timings = {}
    for chunk in runtime.stream_voice_design(
        text=args.text,
        instruct=args.instruct,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        repetition_penalty=args.repetition_penalty,
        max_prompt_tokens=args.max_prompt_tokens,
        progress_interval=0,
        chunk_strategy=args.chunk_strategy,
    ):
        arrival_ms = (time.time() - started) * 1000.0
        timings = dict(chunk.timings)
        audio_ms = (float(chunk.audio.shape[0]) / float(chunk.sample_rate) * 1000.0) if chunk.audio.size else 0.0
        if chunk.audio.size and first_audio_ms is None:
            first_audio_ms = arrival_ms
        audio_samples += int(chunk.audio.shape[0])
        final_timings = timings
        chunk_events.append(
            {
                "index": int(chunk.index),
                "phase": "final" if chunk.is_final else ("first" if chunk.index == 0 else "steady"),
                "arrival_ms": arrival_ms,
                "audio_ms": audio_ms,
                "samples": int(chunk.audio.shape[0]),
                "codes": int(chunk.codes.shape[0]) if getattr(chunk, "codes", None) is not None else 0,
                "selected_bucket": timings.get("selected_bucket"),
                "selected_codegen_graph": timings.get("selected_codegen_graph"),
                "codegen_graph_kind": timings.get("codegen_graph_kind"),
                "active_codegen_unroll": timings.get("active_codegen_unroll"),
                "codegen_schedule": timings.get("codegen_schedule"),
                "is_final": bool(chunk.is_final),
                "timings": timings,
            }
        )
        if chunk.is_final:
            break

    elapsed_ms = (time.time() - started) * 1000.0
    sample_rate = int(runtime.sample_rate)
    audio_ms = (audio_samples / sample_rate * 1000.0) if audio_samples else 0.0
    codegen_sum_ms = sum(float(event["timings"].get("codegen_ms", 0.0)) for event in chunk_events if event["audio_ms"] > 0)
    decode_sum_ms = sum(float(event["timings"].get("decode_ms", 0.0)) for event in chunk_events if event["audio_ms"] > 0)
    effective_sum_ms = sum(float(event["timings"].get("chunk_compute_ms", 0.0)) for event in chunk_events if event["audio_ms"] > 0)
    audio_events = [event for event in chunk_events if event["audio_ms"] > 0]
    audio_timings = audio_events[-1]["timings"] if audio_events else final_timings
    emitted_frames = int(final_timings.get("emitted_frames", 0) or 0)
    return {
        "run": int(run_index),
        "first_audio_ms": first_audio_ms,
        "elapsed_ms": elapsed_ms,
        "audio_ms": audio_ms,
        "chunks": len([event for event in chunk_events if event["audio_ms"] > 0]),
        "emitted_frames": emitted_frames,
        "tokens_per_second": (emitted_frames / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0,
        "stream_rtf": final_timings.get("stream_rtf"),
        "stream_compute_rtf": final_timings.get("stream_compute_rtf"),
        "decode_path": audio_timings.get("decode_path"),
        "prompt_len": audio_timings.get("prompt_len"),
        "required_cache_len": audio_timings.get("required_cache_len"),
        "unroll_required_cache_len": audio_timings.get("unroll_required_cache_len"),
        "selected_bucket": audio_timings.get("selected_bucket"),
        "selected_codegen_graph": audio_timings.get("selected_codegen_graph"),
        "codegen_graph_kind": audio_timings.get("codegen_graph_kind"),
        "active_codegen_unroll": audio_timings.get("active_codegen_unroll"),
        "codegen_schedule": audio_timings.get("codegen_schedule"),
        "decode_unroll_available": bool(audio_timings.get("decode_unroll_available", False)),
        "decode_unroll_stateful_mask": bool(audio_timings.get("decode_unroll_stateful_mask", False)),
        "fallback": bool(audio_timings.get("fallback", False)),
        "codegen_unroll": audio_timings.get("codegen_unroll"),
        "unroll_fallback": bool(audio_timings.get("unroll_fallback", False)),
        "chunk_events": chunk_events,
        "stage_totals_ms": {
            "codegen_sum_ms": codegen_sum_ms,
            "decode_sum_ms": decode_sum_ms,
            "pipeline_effective_sum_ms": effective_sum_ms,
            "overlap_saved_ms": max(0.0, codegen_sum_ms + decode_sum_ms - effective_sum_ms),
        },
        "playback_simulation": simulate_playback(chunk_events, args.playback_buffer_ms),
        "audio_timings": audio_timings,
        "final_timings": final_timings,
        "ov_profile_by_label": runtime.ov_profiler.aggregate("label"),
        "ov_profile_by_type": runtime.ov_profiler.aggregate("node_type"),
        "ov_profile_top": runtime.ov_profiler.top(args.ov_profile_top),
    }


def summarize_profile_runs(profile_name: str, runs: list[dict]) -> dict:
    values = {
        "first_audio_ms": [run["first_audio_ms"] for run in runs if run.get("first_audio_ms") is not None],
        "stream_rtf": [run["stream_rtf"] for run in runs if run.get("stream_rtf") is not None],
        "stream_compute_rtf": [run["stream_compute_rtf"] for run in runs if run.get("stream_compute_rtf") is not None],
        "tokens_per_second": [run["tokens_per_second"] for run in runs if run.get("tokens_per_second") is not None],
    }
    stage_totals = {
        key: [run["stage_totals_ms"][key] for run in runs if run.get("stage_totals_ms")]
        for key in ("codegen_sum_ms", "decode_sum_ms", "pipeline_effective_sum_ms", "overlap_saved_ms")
    }
    phase_codegen = {}
    for phase in ("first", "steady", "final"):
        values_for_phase = []
        for run in runs:
            for event in run.get("chunk_events", []):
                if event.get("phase") == phase and event.get("audio_ms", 0.0) > 0:
                    values_for_phase.append(float(event.get("timings", {}).get("codegen_ms", 0.0)))
        phase_codegen[phase] = median(values_for_phase)
    return {
        "profile": profile_name,
        "runs": len(runs),
        "p50_first_audio_ms": median(values["first_audio_ms"]),
        "p90_first_audio_ms": pct(values["first_audio_ms"], 90),
        "p50_stream_rtf": median(values["stream_rtf"]),
        "p90_stream_rtf": pct(values["stream_rtf"], 90),
        "p50_stream_compute_rtf": median(values["stream_compute_rtf"]),
        "p50_tokens_per_second": median(values["tokens_per_second"]),
        "avg_codegen_sum_ms": statistics.mean(stage_totals["codegen_sum_ms"]) if stage_totals["codegen_sum_ms"] else None,
        "avg_decode_sum_ms": statistics.mean(stage_totals["decode_sum_ms"]) if stage_totals["decode_sum_ms"] else None,
        "avg_effective_sum_ms": statistics.mean(stage_totals["pipeline_effective_sum_ms"]) if stage_totals["pipeline_effective_sum_ms"] else None,
        "avg_overlap_saved_ms": statistics.mean(stage_totals["overlap_saved_ms"]) if stage_totals["overlap_saved_ms"] else None,
        "p50_codegen_first_ms": phase_codegen["first"],
        "p50_codegen_steady_ms": phase_codegen["steady"],
        "p50_codegen_final_ms": phase_codegen["final"],
        "selected_buckets": sorted({str(run.get("selected_bucket")) for run in runs}),
        "codegen_graph_kinds": sorted({str(run.get("codegen_graph_kind")) for run in runs}),
        "codegen_schedules": sorted({str(run.get("codegen_schedule")) for run in runs}),
        "active_codegen_unrolls": sorted({str(run.get("active_codegen_unroll")) for run in runs}),
        "underrun_counts": [run.get("playback_simulation", {}).get("underrun_count") for run in runs],
        "decode_paths": sorted({str(run.get("decode_path")) for run in runs}),
        "unroll_fallbacks": [bool(run.get("unroll_fallback")) for run in runs],
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown(report: dict) -> str:
    summaries = report["summaries"]
    summary_rows = []
    for item in summaries:
        summary_rows.append(
            [
                item["profile"],
                fmt(item["p50_first_audio_ms"], 1),
                fmt(item["p50_stream_rtf"], 3),
                fmt(item["p50_stream_compute_rtf"], 3),
                fmt(item["p50_tokens_per_second"], 2),
                fmt(item["avg_codegen_sum_ms"], 1),
                fmt(item["avg_decode_sum_ms"], 1),
                fmt(item["p50_codegen_steady_ms"], 1),
                fmt(item["avg_overlap_saved_ms"], 1),
                ",".join(item["selected_buckets"]),
                ",".join(item["codegen_schedules"]),
                ",".join(item["active_codegen_unrolls"]),
                ",".join(item["codegen_graph_kinds"]),
                ",".join(str(v) for v in item["underrun_counts"]),
            ]
        )

    lines = [
        "# Qwen3-TTS OpenVINO Streaming Performance Report",
        "",
        "## Run Configuration",
        "",
        f"- IR: `{report['ir']['ir_dir']}`",
        f"- Device: `{report['config']['device']}` / decoder `{report['config']['decoder_device'] or report['config']['device']}`",
        f"- Text chars: `{len(report['config']['text'])}`",
        f"- max_new_tokens: `{report['config']['max_new_tokens']}`",
        f"- chunk_strategy: `{report['config']['chunk_strategy']}`",
        f"- playback_buffer_ms: `{report['config']['playback_buffer_ms']}`",
        f"- ov_profile: `{report['config']['ov_profile']}`",
        "",
        "## Profile Summary",
        "",
        markdown_table(
            [
                "profile",
                "p50 first audio ms",
                "p50 stream rtf",
                "p50 compute rtf",
                "p50 tokens/s",
                "avg codegen ms",
                "avg decode ms",
                "p50 steady codegen ms",
                "avg overlap saved ms",
                "buckets",
                "schedules",
                "active unrolls",
                "graph kinds",
                "underruns",
            ],
            summary_rows,
        ),
        "",
        "## Hotspots",
        "",
    ]

    best_compute = min((item for item in summaries if item["p50_stream_compute_rtf"] is not None), key=lambda item: item["p50_stream_compute_rtf"], default=None)
    if best_compute:
        lines.append(
            f"- Best compute profile: `{best_compute['profile']}` with p50 compute RTF `{fmt(best_compute['p50_stream_compute_rtf'])}`."
        )
    for profile in report["profiles"]:
        if not profile["runs"]:
            continue
        last = profile["runs"][-1]
        stage = last["stage_totals_ms"]
        total_stage = max(stage["codegen_sum_ms"] + stage["decode_sum_ms"], 1e-9)
        lines.append(
            f"- `{profile['profile']}` last run stage split: codegen `{stage['codegen_sum_ms']:.1f}ms` "
            f"({stage['codegen_sum_ms'] / total_stage * 100:.1f}%), decode `{stage['decode_sum_ms']:.1f}ms` "
            f"({stage['decode_sum_ms'] / total_stage * 100:.1f}%), overlap saved `{stage['overlap_saved_ms']:.1f}ms`."
        )
        if last.get("ov_profile_by_label"):
            top_labels = ", ".join(
                f"{item['name']}={item['real_time'] * 1000:.1f}ms" for item in last["ov_profile_by_label"][:5]
            )
            lines.append(f"- `{profile['profile']}` OV label time: {top_labels}.")
        if last.get("ov_profile_by_type"):
            top_types = ", ".join(
                f"{item['name']}={item['real_time'] * 1000:.1f}ms" for item in last["ov_profile_by_type"][:8]
            )
            lines.append(f"- `{profile['profile']}` OV op types: {top_types}.")

    lines.extend(["", "## Device", "", "```json", json.dumps(report["device"], indent=2, ensure_ascii=False), "```", ""])
    return "\n".join(lines)


def run_profile(profile_name: str, profile_config: dict, args: argparse.Namespace) -> dict:
    from qwen3_tts_ov.runtime import OpenVINOQwen3TTS

    compile_config = parse_compile_config(args.compile_config)
    started = time.time()
    runtime = OpenVINOQwen3TTS(
        args.ir_dir,
        args.device,
        decoder_device=args.decoder_device,
        allow_cpu_fallback=args.allow_cpu_fallback,
        mode=profile_config["mode"],
        cache_kernel=profile_config["cache_kernel"],
        cache_step=profile_config["cache_step"],
        graph_variant=profile_config["graph_variant"],
        codegen_unroll=profile_config["codegen_unroll"],
        codegen_schedule=profile_config.get("codegen_schedule", "current"),
        codegen_decode_unroll=profile_config.get("codegen_decode_unroll", "off"),
        ov_cache_dir=args.ov_cache_dir,
        ov_cache_mode=args.ov_cache_mode,
        disable_ov_cache=args.disable_ov_cache,
        compile_config=compile_config,
        profile=True,
        ov_profile=args.ov_profile,
    )
    construct_ms = (time.time() - started) * 1000.0
    warmup = {}
    if not args.no_warmup:
        warmup = runtime.prewarm_streaming(
            text=args.warmup_text,
            instruct=args.instruct,
            language=args.language,
            chunk_strategy=args.chunk_strategy,
            max_new_tokens=args.warmup_tokens,
            preload_buckets=args.preload_buckets,
            run_generation=True,
        )
    runs = [profile_hot_run(runtime, args, index + 1) for index in range(args.runs)]
    return {
        "profile": profile_name,
        "config": profile_config,
        "compile_config": compile_config,
        "construct_ms": construct_ms,
        "warmup": warmup,
        "runtime": {
            "ir_dir": str(runtime.ir_dir),
            "mode": runtime.mode,
            "cache_kernel": runtime.cache_kernel,
            "cache_step": runtime.cache_step,
            "graph_variant": runtime.graph_variant,
            "codegen_unroll": runtime.codegen_unroll,
            "unroll_available": bool(runtime.fused_cache_unroll_bucket_graphs),
            "streaming_decoder_contexts": {
                str(context): sorted(chunks) for context, chunks in runtime.streaming_decoder_graphs_by_context.items()
            },
            "cache_dir": str(runtime.cache_dir) if runtime.cache_dir else None,
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="auto")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--decoder-device", default=None)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--ov-cache-dir", default=None)
    parser.add_argument("--ov-cache-mode", default="optimize_speed", choices=["optimize_speed", "optimize_size"])
    parser.add_argument("--disable-ov-cache", action="store_true")
    parser.add_argument("--profiles", default="int8_fused,int8_sym_fused,int8_sym_unroll4,int8_sym_ll_v2")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--chunk-strategy", default="low_latency", choices=["realtime", "low_latency", "smooth", "balanced", "stable"])
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--min-new-tokens", type=int, default=12)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--warmup-text", default="你好，这是一次流式预热。")
    parser.add_argument("--warmup-tokens", type=int, default=20)
    parser.add_argument("--preload-buckets", default="warmup")
    parser.add_argument("--playback-buffer-ms", type=float, default=250.0)
    parser.add_argument("--ov-profile", action="store_true")
    parser.add_argument("--ov-profile-top", type=int, default=30)
    parser.add_argument("--compile-config", action="append", default=[])
    parser.add_argument("--output-json", default="outputs/perf_analysis/streaming_profile.json")
    parser.add_argument("--output-md", default="outputs/perf_analysis/streaming_profile.md")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    selected = parse_csv(args.profiles)
    unknown = [profile for profile in selected if profile not in PROFILES]
    if unknown:
        raise ValueError(f"unknown profiles: {', '.join(unknown)}; available: {', '.join(sorted(PROFILES))}")

    from qwen3_tts_ov.manifest import load_manifest, resolve_ir_dir

    resolved_ir = resolve_ir_dir(args.ir_dir, fallback_to_local_voice_design=True, warn=True)
    manifest = load_manifest(resolved_ir)
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "ir_dir": args.ir_dir,
            "device": args.device,
            "decoder_device": args.decoder_device,
            "profiles": selected,
            "runs": args.runs,
            "text": args.text,
            "language": args.language,
            "chunk_strategy": args.chunk_strategy,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "playback_buffer_ms": args.playback_buffer_ms,
            "ov_profile": args.ov_profile,
            "compile_config": parse_compile_config(args.compile_config),
        },
        "device": device_report(),
        "ir": graph_inventory(resolved_ir, manifest),
        "profiles": [],
        "summaries": [],
    }

    for profile_name in selected:
        profile_result = run_profile(profile_name, PROFILES[profile_name], args)
        report["profiles"].append(profile_result)
        report["summaries"].append(summarize_profile_runs(profile_name, profile_result["runs"]))
        print(json.dumps(report["summaries"][-1], ensure_ascii=False), flush=True)

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown(report), encoding="utf-8")
    print(f"wrote {output_json}", flush=True)
    print(f"wrote {output_md}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

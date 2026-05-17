from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import threading
import time
from pathlib import Path

import numpy as np

from qwen3_tts_ov.online_batch import OnlineBatchConfig, OnlineBatchScheduler
from qwen3_tts_ov.runtime import OpenVINOQwen3TTS


DEFAULT_TEXTS = [
    "第一条在线 continuous batching 请求，用于测量动态加入时的首 token 延迟。",
    "第二条请求稍后进入同一个调度器，观察是否可以和已有请求共同解码。",
    "第三条请求长度略有变化，用于避免只测试完全相同的 prompt。",
    "第四条请求模拟桌面应用中用户连续触发多个合成任务。",
    "第五条请求继续复用同一个 native paged KV session。",
    "第六条请求用于观察并发提升后的总体吞吐。",
    "第七条请求保持普通中文朗读内容，避免 tokenizer 输入过于极端。",
    "第八条请求补齐更大的在线并发测试。",
]
DEFAULT_INSTRUCT = "用自然、清晰的中文女声朗读。"

OPTIMIZATION_PROFILES = {
    "baseline": {},
}

HARD_METRIC_KEYS = (
    "scheduler_step_count",
    "batch_prefill_step_count",
    "batch_fused_decode_step_count",
    "batch_fused_decode_token_count",
    "batch_single_decode_step_count",
    "batch_single_decode_token_count",
    "batch_fused_decode_active1_bypass_count",
    "batch_fused_decode_logits_bypass_count",
    "batch_subcode_used_count",
    "sampled_batch_subcode_used_count",
    "sampled_batch_subcode_verified_count",
    "sampled_batch_subcode_fallback_count",
    "sampled_batch_subcode_mismatch_count",
    "sampled_batch_subcode_code_mismatch_count",
    "sampled_batch_subcode_embed_mismatch_count",
    "subcode_host_copy_bytes",
    "subcode_host_copy_fallback_count",
    "split_subcode_hidden_direct_bind_count",
    "split_subcode_hidden_bind_fallback_count",
    "split_subcode_hidden_copy_bytes",
    "split_subcode_remote_next_embed_fallback_count",
    "split_subcode_next_embed_host_read_count",
)

HARD_TIMING_KEYS = (
    "host_prepare_ms",
    "tensor_bind_ms",
    "codegen_infer_ms",
    "codegen_prefill_infer_ms",
    "codegen_decode_infer_ms",
    "codegen_subcode_infer_ms",
    "subcode_bind_ms",
    "subcode_output_read_ms",
    "subcode_next_embed_ms",
    "sampling_ms",
    "host_copy_ms",
    "decode_step_prebind_update_ms",
    "total_step_elapsed_ms",
)


def parse_csv_ints(value: str) -> list[int]:
    items = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one integer")
    return items


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def normalize_profile_name(value: str) -> str:
    return str(value or "baseline").strip().replace("-", "_").lower()


def apply_optimization_profile(args: argparse.Namespace) -> dict:
    profile = normalize_profile_name(args.optimization_profile)
    if profile not in OPTIMIZATION_PROFILES:
        supported = ", ".join(sorted(OPTIMIZATION_PROFILES))
        raise ValueError(f"unsupported --optimization-profile={args.optimization_profile!r}; supported: {supported}")
    config = dict(OPTIMIZATION_PROFILES[profile])
    if "sampled_batch_subcode" in config:
        args.sampled_batch_subcode = str(config["sampled_batch_subcode"])
    if "continuous_batch_subcode" in config:
        args.continuous_batch_subcode = bool(config["continuous_batch_subcode"])
    if "enable_fused_batch_decode" in config:
        args.enable_fused_batch_decode = bool(config["enable_fused_batch_decode"])
    if "prefill_mode" in config:
        args.prefill_mode = str(config["prefill_mode"])
    if "batch_prefill" in config:
        args.batch_prefill = bool(config["batch_prefill"])
    args.optimization_profile = profile
    return config


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines or ([text.strip()] if text.strip() else list(DEFAULT_TEXTS))
    if args.text:
        return [args.text]
    return list(DEFAULT_TEXTS)


def expand_texts(texts: list[str], batch_size: int, repeat_prompt: bool) -> list[str]:
    if repeat_prompt:
        return [texts[0]] * batch_size
    texts = list(texts)
    while len(texts) < batch_size:
        texts.extend(texts)
    return texts[:batch_size]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def make_arrival_offsets_ms(args: argparse.Namespace, *, scenario: str, batch_size: int, run_index: int) -> list[float]:
    if scenario == "offline":
        return [0.0] * batch_size
    if scenario != "online":
        raise ValueError(f"unsupported scenario: {scenario}")
    rng = random.Random(int(args.arrival_seed) + run_index * 1_000_003 + batch_size * 9_176)
    pattern = str(args.arrival_pattern).strip().lower()
    if pattern == "fixed":
        return [max(0.0, float(args.arrival_gap_ms)) * index for index in range(batch_size)]
    if pattern == "uniform":
        offsets = [0.0]
        offsets.extend(rng.uniform(0.0, max(0.0, float(args.arrival_window_ms))) for _ in range(batch_size - 1))
        return sorted(offsets)
    if pattern == "poisson":
        mean_ms = max(1.0, float(args.arrival_gap_ms))
        elapsed = 0.0
        offsets = [0.0]
        for _ in range(batch_size - 1):
            elapsed += rng.expovariate(1.0 / mean_ms)
            offsets.append(min(elapsed, max(0.0, float(args.arrival_window_ms))))
        return sorted(offsets)
    raise ValueError(f"unsupported arrival pattern: {args.arrival_pattern}")


def summarize(values: list[float]) -> dict:
    return {
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": max(values) if values else 0.0,
        "min": min(values) if values else 0.0,
    }


def metric_number(value: object) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except Exception:
            return 0.0
    return 0.0


def extract_hard_metrics(scheduler_stats: dict) -> dict:
    metrics: dict[str, object] = {
        key: metric_number(scheduler_stats.get(key)) for key in (*HARD_METRIC_KEYS, *HARD_TIMING_KEYS)
    }
    for key in ("prefill_modes_seen", "prefill_batch_buckets_seen", "prefill_seq_buckets_seen", "decode_batch_buckets_seen"):
        value = scheduler_stats.get(key)
        metrics[key] = value if isinstance(value, list) else []
    active_hist = scheduler_stats.get("active_batch_histogram")
    metrics["active_batch_histogram"] = active_hist if isinstance(active_hist, list) else []
    metrics["batch_subcode_compiled"] = bool(scheduler_stats.get("batch_subcode_compiled"))
    metrics["paged_fused_batch_decode_enabled"] = bool(scheduler_stats.get("paged_fused_batch_decode_enabled"))
    metrics["sampled_batch_subcode_policy_seen"] = (
        scheduler_stats.get("sampled_batch_subcode_policy_seen")
        if isinstance(scheduler_stats.get("sampled_batch_subcode_policy_seen"), list)
        else []
    )
    metrics["sampled_batch_subcode_fallback_reasons"] = (
        scheduler_stats.get("sampled_batch_subcode_fallback_reasons")
        if isinstance(scheduler_stats.get("sampled_batch_subcode_fallback_reasons"), list)
        else []
    )
    return metrics


def merge_histograms(items: list[list]) -> list[int]:
    width = max((len(item) for item in items if isinstance(item, list)), default=0)
    merged = [0] * width
    for item in items:
        if not isinstance(item, list):
            continue
        for index, value in enumerate(item):
            merged[index] += safe_int(value)
    return merged


def merge_unique_lists(items: list[list]) -> list:
    values = []
    for item in items:
        if not isinstance(item, list):
            continue
        for value in item:
            if value in (None, "", 0):
                continue
            if value not in values:
                values.append(value)
    return values


def safe_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str):
        try:
            return int(value)
        except Exception:
            return default
    return default


def summarize_scheduler_samples(samples: list[dict]) -> dict:
    if not samples:
        return {}
    active = [safe_int(item.get("active")) for item in samples]
    pending = [safe_int(item.get("pending")) for item in samples]
    tracked = [safe_int(item.get("python_tracked_requests")) for item in samples]
    hist_max_len = 0
    for item in samples:
        hist = item.get("last_active_batch_histogram")
        if isinstance(hist, list):
            hist_max_len = max(hist_max_len, len(hist))
    merged_hist = [0] * hist_max_len
    for item in samples:
        hist = item.get("last_active_batch_histogram")
        if not isinstance(hist, list):
            continue
        for index, value in enumerate(hist):
            merged_hist[index] += safe_int(value)
    return {
        "samples": len(samples),
        "max_active": max(active) if active else 0,
        "max_pending": max(pending) if pending else 0,
        "max_python_tracked_requests": max(tracked) if tracked else 0,
        "batch_subcode_compiled_any": any(bool(item.get("batch_subcode_compiled")) for item in samples),
        "batch_subcode_used_any": any(bool(item.get("last_batch_subcode_used")) for item in samples),
        "sampled_batch_subcode_used_any": any(bool(item.get("sampled_batch_subcode_used")) for item in samples),
        "sampled_batch_subcode_verified_any": any(bool(item.get("sampled_batch_subcode_verified")) for item in samples),
        "sampled_batch_subcode_fallback_count": sum(safe_int(item.get("sampled_batch_subcode_fallback_count")) for item in samples),
        "sampled_batch_subcode_mismatch_count": sum(safe_int(item.get("sampled_batch_subcode_mismatch_count")) for item in samples),
        "sampled_batch_subcode_code_mismatch_count": sum(
            safe_int(item.get("sampled_batch_subcode_code_mismatch_count")) for item in samples
        ),
        "sampled_batch_subcode_embed_mismatch_count": sum(
            safe_int(item.get("sampled_batch_subcode_embed_mismatch_count")) for item in samples
        ),
        "sampled_batch_subcode_max_abs_diff": max(
            float(item.get("sampled_batch_subcode_max_abs_diff", 0.0) or 0.0) for item in samples
        ),
        "sampled_subcode_parallel_rows_any": any(bool(item.get("sampled_subcode_parallel_rows")) for item in samples),
        "sampled_subcode_parallel_row_count": sum(safe_int(item.get("sampled_subcode_parallel_row_count")) for item in samples),
        "no_repeat_fast_path_any": any(bool(item.get("last_no_repeat_fast_path")) for item in samples),
        "batch_fused_decode_step_count": sum(safe_int(item.get("last_batch_fused_decode_step_count")) for item in samples),
        "batch_fused_decode_token_count": sum(safe_int(item.get("last_batch_fused_decode_token_count")) for item in samples),
        "batch_single_decode_step_count": sum(safe_int(item.get("last_batch_single_decode_step_count")) for item in samples),
        "batch_single_decode_token_count": sum(safe_int(item.get("last_batch_single_decode_token_count")) for item in samples),
        "batch_fused_decode_active1_bypass_count": sum(
            safe_int(item.get("last_batch_fused_decode_active1_bypass_count")) for item in samples
        ),
        "batch_fused_decode_logits_bypass_count": sum(
            safe_int(item.get("last_batch_fused_decode_logits_bypass_count")) for item in samples
        ),
        "scheduler_layered_any": any(str(item.get("online_scheduler") or "") == "layered" for item in samples),
        "max_num_batched_tokens_max": max(safe_int(item.get("max_num_batched_tokens")) for item in samples),
        "prefill_seq_buckets_seen": sorted(
            {
                safe_int(item.get("last_prefill_seq_bucket"))
                for item in samples
                if safe_int(item.get("last_prefill_seq_bucket")) > 0
            }
        ),
        "prefill_batch_buckets_seen": sorted(
            {
                safe_int(item.get("last_prefill_batch_bucket"))
                for item in samples
                if safe_int(item.get("last_prefill_batch_bucket")) > 0
            }
        ),
        "decode_batch_buckets_seen": sorted(
            {
                safe_int(item.get("last_decode_batch_bucket"))
                for item in samples
                if safe_int(item.get("last_decode_batch_bucket")) > 0
            }
        ),
        "active_batch_histogram_merged": merged_hist,
    }


def summarize_runs(runs: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for run in runs:
        groups.setdefault(
            (
                str(run.get("optimization_profile") or "baseline"),
                str(run.get("scenario")),
                int(run.get("batch_size", 0)),
            ),
            [],
        ).append(run)
    summary = []
    for (profile, scenario, batch_size), items in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        requests = [request for item in items for request in item.get("requests", []) if isinstance(request, dict)]
        ttft = [float(item.get("ttft_ms", 0.0)) for item in requests if "ttft_ms" in item]
        scheduler_ttft = [
            float(item.get("scheduler_ttft_ms", 0.0)) for item in requests if "scheduler_ttft_ms" in item
        ]
        prompt_ms = [float(item.get("prompt_ms", 0.0)) for item in requests if "prompt_ms" in item]
        per_user_tps = [float(item.get("tps", 0.0)) for item in requests if "tps" in item]
        aggregate_tps = [float(item.get("aggregate_tps", 0.0)) for item in items]
        elapsed_ms = [float(item.get("elapsed_ms", 0.0)) for item in items]
        hard_metrics = [item.get("hard_metrics", {}) for item in items if isinstance(item.get("hard_metrics"), dict)]
        merged_hard: dict[str, object] = {
            key: sum(metric_number(item.get(key)) for item in hard_metrics)
            for key in (*HARD_METRIC_KEYS, *HARD_TIMING_KEYS)
        }
        merged_hard.update(
            {
                "active_batch_histogram": merge_histograms(
                    [item.get("active_batch_histogram", []) for item in hard_metrics]
                ),
                "prefill_modes_seen": merge_unique_lists([item.get("prefill_modes_seen", []) for item in hard_metrics]),
                "prefill_batch_buckets_seen": merge_unique_lists(
                    [item.get("prefill_batch_buckets_seen", []) for item in hard_metrics]
                ),
                "prefill_seq_buckets_seen": merge_unique_lists(
                    [item.get("prefill_seq_buckets_seen", []) for item in hard_metrics]
                ),
                "decode_batch_buckets_seen": merge_unique_lists(
                    [item.get("decode_batch_buckets_seen", []) for item in hard_metrics]
                ),
                "sampled_batch_subcode_policy_seen": merge_unique_lists(
                    [item.get("sampled_batch_subcode_policy_seen", []) for item in hard_metrics]
                ),
                "sampled_batch_subcode_fallback_reasons": merge_unique_lists(
                    [item.get("sampled_batch_subcode_fallback_reasons", []) for item in hard_metrics]
                ),
                "batch_subcode_compiled_any": any(bool(item.get("batch_subcode_compiled")) for item in hard_metrics),
                "paged_fused_batch_decode_enabled_any": any(
                    bool(item.get("paged_fused_batch_decode_enabled")) for item in hard_metrics
                ),
            }
        )
        summary.append(
            {
                "optimization_profile": profile,
                "scenario": scenario,
                "batch_size": batch_size,
                "runs": len(items),
                "requests": len(requests),
                "generated_tokens_total": sum(int(item.get("generated_tokens_total", 0)) for item in items),
                "elapsed_ms_mean": statistics.mean(elapsed_ms) if elapsed_ms else 0.0,
                "aggregate_tps_mean": statistics.mean(aggregate_tps) if aggregate_tps else 0.0,
                "aggregate_tps_p50": percentile(aggregate_tps, 0.50),
                "aggregate_tps_max": max(aggregate_tps) if aggregate_tps else 0.0,
                "ttft_ms": summarize(ttft),
                "scheduler_ttft_ms": summarize(scheduler_ttft),
                "prompt_ms": summarize(prompt_ms),
                "per_user_tps": summarize(per_user_tps),
                "hard_metrics": merged_hard,
            }
        )
    return summary


def run_scenario(args: argparse.Namespace, batch_size: int, run_index: int, scenario: str) -> dict:
    if args.prompt_cache != "auto":
        os.environ["QWEN3_TTS_OV_PROMPT_COMPONENT_CACHE"] = "1" if args.prompt_cache == "on" else "0"
    runtime = OpenVINOQwen3TTS(
        args.ir_dir,
        device=args.device,
        mode=args.runtime_mode,
        graph_variant=args.runtime_graph_variant,
        ov_cache_dir=args.ov_cache_dir,
        disable_ov_cache=args.disable_ov_cache,
        native_codegen="off",
        native_pipeline="off",
    )
    scheduler = OnlineBatchScheduler(
        runtime,
        OnlineBatchConfig(
            max_batch_size=batch_size,
            wait_ms=args.wait_ms,
            max_cache_blocks=args.max_cache_blocks,
            scheduler=args.scheduler,
            max_num_batched_tokens=args.max_num_batched_tokens,
            prefill_mode=args.prefill_mode,
            prefill_seq_buckets=args.prefill_seq_buckets,
            prefill_batch_buckets=args.prefill_batch_buckets,
            decode_batch_buckets=args.decode_batch_buckets,
            graph_variant=args.online_graph_variant,
            subcode_mode=args.subcode_mode,
            sampled_batch_subcode=args.sampled_batch_subcode,
            sampled_subcode_parallel_rows=bool(args.sampled_subcode_parallel_rows),
            kv_precision=args.kv_precision,
            block_size=args.block_size,
            continuous_policy=args.continuous_policy,
            batch_prefill=bool(args.batch_prefill),
            disable_fused_decode=not bool(args.enable_fused_batch_decode),
            continuous_batch_subcode=bool(args.continuous_batch_subcode),
        ),
    )
    if not args.no_prewarm:
        scheduler.ensure_ready()
    results: list[dict] = [{} for _ in range(batch_size)]
    errors: list[str] = []
    texts = expand_texts(load_texts(args), batch_size, repeat_prompt=bool(args.repeat_prompt))
    arrival_offsets_ms = make_arrival_offsets_ms(args, scenario=scenario, batch_size=batch_size, run_index=run_index)
    prebuilt_prompts: list[tuple[np.ndarray, np.ndarray] | None] = [None] * batch_size
    if args.prebuild_prompts:
        for index, text in enumerate(texts):
            prebuilt_prompts[index] = runtime.build_prompt(
                text=text,
                instruct=args.instruct,
                language=args.language,
                max_prompt_tokens=args.max_prompt_tokens,
            )
    warmup_summary = None
    if int(args.scheduler_warmup_requests) > 0:
        warmup_prompt = prebuilt_prompts[0]
        if warmup_prompt is None:
            warmup_prompt = runtime.build_prompt(
                text=texts[0],
                instruct=args.instruct,
                language=args.language,
                max_prompt_tokens=args.max_prompt_tokens,
            )
        warmup_sequence, warmup_tts_pad_embed = warmup_prompt
        warmup_summary = scheduler.warmup(
            warmup_sequence,
            warmup_tts_pad_embed,
            batch_size=int(args.scheduler_warmup_requests),
            max_new_tokens=int(args.scheduler_warmup_tokens),
            min_new_tokens=0,
            repetition_penalty=float(args.repetition_penalty),
            do_sample=bool(args.scheduler_warmup_sample),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            temperature=float(args.temperature),
            seed=int(args.seed) + 1_000_000 + run_index,
        )
    prompt_lock = threading.Lock()
    start_barrier = threading.Barrier(batch_size + 1)
    stats_samples: list[dict] = []
    stats_lock = threading.Lock()
    stats_stop = threading.Event()
    started = 0.0

    def monitor_scheduler() -> None:
        while not stats_stop.is_set():
            try:
                sample = scheduler.stats()
                sample["sample_ms"] = (time.time() - started) * 1000.0 if started else 0.0
                with stats_lock:
                    stats_samples.append(sample)
            except Exception:
                pass
            time.sleep(max(0.001, float(args.stats_interval_ms) / 1000.0))

    def worker(index: int, text: str) -> None:
        try:
            start_barrier.wait()
            scheduled_arrival_at = started + max(0.0, arrival_offsets_ms[index]) / 1000.0
            time.sleep(max(0.0, scheduled_arrival_at - time.time()))
            arrival_at = time.time()
            prompt_start = time.time()
            prebuilt_prompt = prebuilt_prompts[index]
            if prebuilt_prompt is not None:
                sequence, tts_pad_embed = prebuilt_prompt
            else:
                with prompt_lock:
                    sequence, tts_pad_embed = runtime.build_prompt(
                        text=text,
                        instruct=args.instruct,
                        language=args.language,
                        max_prompt_tokens=args.max_prompt_tokens,
                    )
            prompt_end = time.time()
            enqueue_at = time.time()
            first_at = None
            count = 0
            for _code in scheduler.submit(
                sequence,
                tts_pad_embed,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                seed=args.seed + index,
            ):
                count += 1
                if first_at is None:
                    first_at = time.time()
            finished_at = time.time()
            first_or_finished = first_at or finished_at
            decode_seconds = max(1e-9, finished_at - first_or_finished)
            results[index] = {
                "index": index,
                "scenario": scenario,
                "arrival_offset_ms": float(arrival_offsets_ms[index]),
                "arrival_lag_ms": max(0.0, (arrival_at - scheduled_arrival_at) * 1000.0),
                "prompt_tokens": int(sequence.shape[1]),
                "generated_tokens": int(count),
                "submit_ms": (arrival_at - started) * 1000.0,
                "enqueue_ms": (enqueue_at - started) * 1000.0,
                "prompt_ms": (prompt_end - prompt_start) * 1000.0,
                "ttft_ms": (first_or_finished - arrival_at) * 1000.0,
                "scheduler_ttft_ms": (first_or_finished - enqueue_at) * 1000.0,
                "elapsed_ms": (finished_at - arrival_at) * 1000.0,
                "scheduler_elapsed_ms": (finished_at - enqueue_at) * 1000.0,
                "tps": max(0, count - 1) / decode_seconds,
            }
        except BaseException as exc:
            results[index] = {"index": index, "error": str(exc)}
            errors.append(f"request {index}: {exc}")

    threads = [threading.Thread(target=worker, args=(index, text), daemon=True) for index, text in enumerate(texts)]
    for thread in threads:
        thread.start()
    monitor = None
    if args.sample_scheduler_stats:
        monitor = threading.Thread(target=monitor_scheduler, name="qwen3-tts-online-batch-benchmark-monitor", daemon=True)
        monitor.start()
    started = time.time()
    start_barrier.wait()
    for thread in threads:
        thread.join()
    if monitor is not None:
        stats_stop.set()
        monitor.join(timeout=1.0)
    elapsed_ms = (time.time() - started) * 1000.0
    scheduler_stats = scheduler.stats()
    scheduler.close()
    if errors:
        raise RuntimeError("; ".join(errors))
    hard_metrics = extract_hard_metrics(scheduler_stats)

    ttft = [float(item["ttft_ms"]) for item in results if item]
    scheduler_ttft = [float(item["scheduler_ttft_ms"]) for item in results if item]
    prompt_ms = [float(item["prompt_ms"]) for item in results if item]
    tps = [float(item["tps"]) for item in results if item]
    total_tokens = sum(int(item["generated_tokens"]) for item in results if item)
    return {
        "run": run_index,
        "optimization_profile": args.optimization_profile,
        "optimization_profile_config": getattr(args, "optimization_profile_config", {}),
        "scenario": scenario,
        "batch_size": batch_size,
        "arrival_pattern": "none" if scenario == "offline" else args.arrival_pattern,
        "arrival_gap_ms": args.arrival_gap_ms,
        "arrival_window_ms": 0.0 if scenario == "offline" else args.arrival_window_ms,
        "arrival_offsets_ms": arrival_offsets_ms,
        "do_sample": bool(args.do_sample),
        "repetition_penalty": float(args.repetition_penalty),
        "sampled_batch_subcode": args.sampled_batch_subcode,
        "sampled_subcode_parallel_rows": bool(args.sampled_subcode_parallel_rows),
        "continuous_policy": args.continuous_policy,
        "scheduler_mode": args.scheduler,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "prefill_seq_buckets": args.prefill_seq_buckets,
        "prefill_batch_buckets": args.prefill_batch_buckets,
        "decode_batch_buckets": args.decode_batch_buckets,
        "prefill_mode": args.prefill_mode,
        "batch_prefill": bool(args.batch_prefill),
        "enable_fused_batch_decode": bool(args.enable_fused_batch_decode),
        "continuous_batch_subcode": bool(args.continuous_batch_subcode),
        "prompt_cache": args.prompt_cache,
        "scheduler_warmup": warmup_summary,
        "elapsed_ms": elapsed_ms,
        "generated_tokens_total": total_tokens,
        "aggregate_tps": total_tokens / max(1e-9, elapsed_ms / 1000.0),
        "ttft_ms_p50": percentile(ttft, 0.50),
        "ttft_ms_p90": percentile(ttft, 0.90),
        "ttft_ms_max": max(ttft) if ttft else 0.0,
        "scheduler_ttft_ms_p50": percentile(scheduler_ttft, 0.50),
        "scheduler_ttft_ms_p90": percentile(scheduler_ttft, 0.90),
        "prompt_ms_p50": percentile(prompt_ms, 0.50),
        "prompt_ms_p90": percentile(prompt_ms, 0.90),
        "per_user_tps_p50": statistics.median(tps) if tps else 0.0,
        "per_user_tps_p90": percentile(tps, 0.90),
        "per_user_tps_min": min(tps) if tps else 0.0,
        "per_user_tps": summarize(tps),
        "requests": results,
        "scheduler": scheduler_stats,
        "hard_metrics": hard_metrics,
        "scheduler_samples_summary": summarize_scheduler_samples(stats_samples),
        "scheduler_samples": stats_samples if args.keep_scheduler_samples else [],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark sampled native online continuous batching with offline and random online arrivals."
    )
    parser.add_argument("--ir-dir", default="openvino/voice_design")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--runtime-mode", default="cache")
    parser.add_argument("--runtime-graph-variant", default="int8_sym_paged_talker_split_cachedsub")
    parser.add_argument("--online-graph-variant", default="int8_sym_batch_fused_gqa")
    parser.add_argument(
        "--optimization-profile",
        default="baseline",
        choices=sorted(OPTIMIZATION_PROFILES),
        help="Apply a low-model-size online batching optimization profile.",
    )
    parser.add_argument("--subcode-mode", default="cached")
    parser.add_argument("--sampled-batch-subcode", default="off", choices=["off", "verify", "on"])
    parser.add_argument("--sampled-subcode-parallel-rows", action="store_true")
    parser.add_argument("--kv-precision", default="u8", choices=["f16", "bf16", "u8"])
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--scheduler", default="layered", choices=["layered"])
    parser.add_argument("--max-num-batched-tokens", type=int, default=16)
    parser.add_argument("--prefill-seq-buckets", default="128,256,512,1024")
    parser.add_argument("--prefill-batch-buckets", default="1,2,4,8")
    parser.add_argument("--decode-batch-buckets", default="1,2,4,8,16")
    parser.add_argument(
        "--prefill-mode",
        default="serial",
        choices=["serial", "dynamic-ragged", "bucketed-padded", "auto"],
        help="Native online batching prefill strategy. Production uses serial prefill.",
    )
    parser.add_argument(
        "--continuous-policy",
        default="layered_vllm",
        choices=["layered_vllm"],
    )
    parser.add_argument(
        "--enable-fused-batch-decode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--batch-prefill",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-batch-prefill", dest="batch_prefill", action="store_false")
    parser.add_argument(
        "--continuous-batch-subcode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-cache-blocks", type=int, default=2048)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--scenarios", default="offline,online", help="Comma-separated: offline,online")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--arrival-pattern", default="uniform", choices=["fixed", "uniform", "poisson"])
    parser.add_argument("--arrival-gap-ms", type=float, default=20.0)
    parser.add_argument("--arrival-window-ms", type=float, default=1500.0)
    parser.add_argument("--arrival-seed", type=int, default=1234)
    parser.add_argument("--wait-ms", type=float, default=2.0)
    parser.add_argument("--sample-scheduler-stats", action="store_true")
    parser.add_argument("--stats-interval-ms", type=float, default=10.0)
    parser.add_argument("--keep-scheduler-samples", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--repeat-prompt", action="store_true")
    parser.add_argument("--prebuild-prompts", action="store_true")
    parser.add_argument("--scheduler-warmup-requests", type=int, default=0)
    parser.add_argument("--scheduler-warmup-tokens", type=int, default=4)
    parser.add_argument("--scheduler-warmup-sample", action="store_true")
    parser.add_argument("--prompt-cache", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    parser.add_argument("--ov-cache-dir", default=None)
    parser.add_argument("--disable-ov-cache", action="store_true")
    parser.add_argument("--no-prewarm", action="store_true")
    parser.add_argument("--output", default="outputs/online_batch/benchmark.json")
    args = parser.parse_args(argv)
    args.optimization_profile_config = apply_optimization_profile(args)

    payload = {
        "benchmark": "native_online_continuous_batch",
        "created_at_unix": time.time(),
        "device": args.device,
        "optimization_profile": args.optimization_profile,
        "optimization_profile_config": args.optimization_profile_config,
        "scenarios": parse_csv_strings(args.scenarios),
        "runs": [],
    }
    scenarios = parse_csv_strings(args.scenarios)
    if not scenarios:
        raise ValueError("expected at least one scenario")
    for batch_size in parse_csv_ints(args.batch_sizes):
        for scenario in scenarios:
            for run_index in range(args.runs):
                result = run_scenario(args, batch_size=batch_size, run_index=run_index, scenario=scenario)
                payload["runs"].append(result)
                payload["summary"] = summarize_runs(payload["runs"])
                print(
                    "scenario={scenario} batch={batch_size} run={run} "
                    "ttft_p50={ttft_ms_p50:.1f}ms scheduler_ttft_p50={scheduler_ttft_ms_p50:.1f}ms "
                    "agg_tps={aggregate_tps:.2f} per_user_tps_p50={per_user_tps_p50:.2f}".format(**result),
                    flush=True,
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()

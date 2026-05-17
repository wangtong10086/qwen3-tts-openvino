from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


BENCHMARK_ONLINE = Path(__file__).with_name("benchmark_online_continuous_batch.py")
BASE_SENTENCE = "这是一段用于推理性能分析的中文文本，包含自然停顿、上下文信息和稳定的朗读节奏。"
PROMPT_PRESETS = {
    "short": BASE_SENTENCE,
    "medium": BASE_SENTENCE * 4,
    "long": BASE_SENTENCE * 12,
    "xlong": BASE_SENTENCE * 24,
}
OPTIMIZATION_PROFILE_SETS = {
    "baseline": ("baseline",),
}
KNOWN_OPTIMIZATION_PROFILES = sorted(
    {profile for profiles in OPTIMIZATION_PROFILE_SETS.values() for profile in profiles}
)
HARD_METRIC_KEYS = (
    "scheduler_step_count",
    "batch_prefill_step_count",
    "batch_fused_decode_step_count",
    "batch_fused_decode_token_count",
    "batch_single_decode_step_count",
    "batch_single_decode_token_count",
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
)


def parse_csv_strings(value: str) -> list[str]:
    items = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one comma-separated item")
    return items


def parse_csv_ints(value: str) -> list[int]:
    items = [int(item) for item in parse_csv_strings(value)]
    if any(item <= 0 for item in items):
        raise ValueError("batch sizes must be positive")
    return items


def normalize_profile_name(value: str) -> str:
    return str(value or "baseline").strip().replace("-", "_").lower()


def parse_optimization_profiles(profile_set: str, profiles: str | None) -> list[str]:
    if profiles:
        selected = [normalize_profile_name(item) for item in parse_csv_strings(profiles)]
    else:
        key = str(profile_set or "baseline").strip()
        if "," in key:
            selected = [normalize_profile_name(item) for item in parse_csv_strings(key)]
        else:
            if key not in OPTIMIZATION_PROFILE_SETS:
                supported = ", ".join(sorted(OPTIMIZATION_PROFILE_SETS))
                raise ValueError(f"unknown --profile-set={profile_set!r}; supported: {supported}")
            selected = list(OPTIMIZATION_PROFILE_SETS[key])
    unknown = [profile for profile in selected if profile not in KNOWN_OPTIMIZATION_PROFILES]
    if unknown:
        supported = ", ".join(KNOWN_OPTIMIZATION_PROFILES)
        raise ValueError(f"unknown optimization profile(s): {', '.join(unknown)}; supported: {supported}")
    return selected


def prompt_for_kind(kind: str) -> str:
    normalized = str(kind).strip().lower()
    if normalized in PROMPT_PRESETS:
        return PROMPT_PRESETS[normalized]
    if normalized.isdigit():
        target_units = max(1, int(normalized))
        repeated = []
        units = 0
        while units < target_units:
            repeated.append(BASE_SENTENCE)
            units += len(BASE_SENTENCE)
        return "".join(repeated)[: target_units + len(BASE_SENTENCE)]
    path = Path(kind)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    raise ValueError(f"unknown prompt length preset or file: {kind}")


def safe_filename_part(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return safe.strip("._") or "item"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def metric_number(value: object) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except Exception:
            return 0.0
    return 0.0


def summarize_runs(runs: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, int, str], list[dict]] = {}
    for item in runs:
        if item.get("dry_run"):
            continue
        if "batch_size" not in item or "scenario" not in item:
            continue
        key = (
            str(item.get("optimization_profile") or "baseline"),
            str(item["prompt_kind"]),
            int(item["batch_size"]),
            str(item["scenario"]),
        )
        groups.setdefault(key, []).append(item)
    summary = []
    for (optimization_profile, prompt_kind, batch_size, scenario), items in sorted(groups.items()):
        aggregate_tps = [float(item.get("aggregate_tps", 0.0) or 0.0) for item in items]
        ttft_p50 = [float(item.get("ttft_ms_p50", 0.0) or 0.0) for item in items]
        ttft_p90 = [float(item.get("ttft_ms_p90", 0.0) or 0.0) for item in items]
        per_user_tps = [float(item.get("per_user_tps_p50", 0.0) or 0.0) for item in items]
        hard_metrics = [
            item.get("hard_metrics", {})
            for item in items
            if isinstance(item.get("hard_metrics"), dict)
        ]
        merged_hard_metrics = {
            key: sum(metric_number(item.get(key)) for item in hard_metrics)
            for key in HARD_METRIC_KEYS
        }
        summary.append(
            {
                "optimization_profile": optimization_profile,
                "prompt_kind": prompt_kind,
                "batch_size": int(batch_size),
                "scenario": scenario,
                "runs": len(items),
                "aggregate_tps_mean": statistics.mean(aggregate_tps) if aggregate_tps else 0.0,
                "aggregate_tps_p90": percentile(aggregate_tps, 0.90),
                "per_user_tps_p50_mean": statistics.mean(per_user_tps) if per_user_tps else 0.0,
                "ttft_ms_p50_mean": statistics.mean(ttft_p50) if ttft_p50 else 0.0,
                "ttft_ms_p90_max": max(ttft_p90) if ttft_p90 else 0.0,
                "hard_metrics": merged_hard_metrics,
            }
        )
    return summary


def run_child(
    args: argparse.Namespace,
    passthrough: list[str],
    *,
    prompt_kind: str,
    prompt_text: str,
    batch_size: int,
    scenario: str,
    optimization_profile: str,
    run_index: int,
    output_dir: Path,
) -> dict:
    safe_prompt = safe_filename_part(prompt_kind)
    child_output = output_dir / f"{optimization_profile}_{safe_prompt}_b{batch_size}_{scenario}_r{run_index}.json"
    cmd = [
        sys.executable,
        str(BENCHMARK_ONLINE),
        "--ir-dir",
        str(args.ir_dir),
        "--device",
        str(args.device),
        "--batch-sizes",
        str(batch_size),
        "--scenarios",
        scenario,
        "--runs",
        "1",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--min-new-tokens",
        str(args.min_new_tokens),
        "--max-prompt-tokens",
        str(args.max_prompt_tokens),
        "--text",
        prompt_text,
        "--repeat-prompt",
        "--optimization-profile",
        optimization_profile,
        "--output",
        str(child_output),
        *passthrough,
    ]
    record: dict[str, object] = {
        "optimization_profile": optimization_profile,
        "prompt_kind": prompt_kind,
        "prompt_chars": len(prompt_text),
        "batch_size": int(batch_size),
        "scenario": scenario,
        "run": int(run_index),
        "command": cmd,
    }
    if args.dry_run:
        record["dry_run"] = True
        return record
    if args.resume_existing and child_output.exists():
        payload = json.loads(child_output.read_text(encoding="utf-8"))
        child_runs = payload.get("runs") or []
        if child_runs:
            record.update(child_runs[0])
        record["worker_exit_code"] = 0
        record["resumed_existing"] = True
        record["child_output"] = str(child_output)
        return record
    started = time.time()
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    record.update(
        {
            "worker_exit_code": int(completed.returncode),
            "elapsed_ms": (time.time() - started) * 1000.0,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    )
    if completed.returncode != 0:
        return record
    payload = json.loads(child_output.read_text(encoding="utf-8"))
    child_runs = payload.get("runs") or []
    if child_runs:
        record.update(child_runs[0])
    record["child_output"] = str(child_output)
    return record


def run_group_child(
    args: argparse.Namespace,
    passthrough: list[str],
    *,
    prompt_kind: str,
    prompt_text: str,
    optimization_profile: str,
    batch_sizes: list[int],
    scenarios: list[str],
    output_dir: Path,
) -> list[dict]:
    safe_prompt = safe_filename_part(prompt_kind)
    child_output = output_dir / f"{optimization_profile}_{safe_prompt}_group.json"
    cmd = [
        sys.executable,
        str(BENCHMARK_ONLINE),
        "--ir-dir",
        str(args.ir_dir),
        "--device",
        str(args.device),
        "--batch-sizes",
        ",".join(str(item) for item in batch_sizes),
        "--scenarios",
        ",".join(scenarios),
        "--runs",
        str(args.runs),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--min-new-tokens",
        str(args.min_new_tokens),
        "--max-prompt-tokens",
        str(args.max_prompt_tokens),
        "--text",
        prompt_text,
        "--repeat-prompt",
        "--optimization-profile",
        optimization_profile,
        "--output",
        str(child_output),
        *passthrough,
    ]
    base_record: dict[str, object] = {
        "optimization_profile": optimization_profile,
        "prompt_kind": prompt_kind,
        "prompt_chars": len(prompt_text),
        "batch_sizes": list(batch_sizes),
        "scenarios": list(scenarios),
        "runs_requested": int(args.runs),
        "command": cmd,
    }
    if args.dry_run:
        record = dict(base_record)
        record["dry_run"] = True
        return [record]
    if args.resume_existing and child_output.exists():
        payload = json.loads(child_output.read_text(encoding="utf-8"))
        child_runs = payload.get("runs") or []
        records = []
        for child_run in child_runs:
            if not isinstance(child_run, dict):
                continue
            record = {
                **base_record,
                "worker_exit_code": 0,
                "resumed_existing": True,
                "child_output": str(child_output),
            }
            record.update(child_run)
            records.append(record)
        if records:
            return records
    started = time.time()
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    worker_record = {
        **base_record,
        "worker_exit_code": int(completed.returncode),
        "elapsed_ms": (time.time() - started) * 1000.0,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "child_output": str(child_output),
    }
    if completed.returncode != 0:
        return [worker_record]
    payload = json.loads(child_output.read_text(encoding="utf-8"))
    child_runs = payload.get("runs") or []
    records = []
    for child_run in child_runs:
        if not isinstance(child_run, dict):
            continue
        record = dict(worker_record)
        record.update(child_run)
        record["child_output"] = str(child_output)
        records.append(record)
    return records or [worker_record]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a prompt-length x batch-size matrix around native online continuous batching."
    )
    parser.add_argument("--ir-dir", default="openvino/voice_design")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--prompt-lengths", default="short,medium,long,xlong")
    parser.add_argument("--scenarios", default="offline,online")
    parser.add_argument(
        "--profile-set",
        default="baseline",
        help=(
            "Optimization profile set to sweep. Supported: "
            + ", ".join(sorted(OPTIMIZATION_PROFILE_SETS))
            + ". A comma-separated profile list is also accepted."
        ),
    )
    parser.add_argument(
        "--optimization-profiles",
        default=None,
        help="Explicit comma-separated profile list. Overrides --profile-set.",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=2048,
        help="Forwarded to benchmark_online_continuous_batch.py; xlong prompt preset exceeds the child default 512.",
    )
    parser.add_argument(
        "--child-granularity",
        default="cell",
        choices=["cell", "profile-prompt"],
        help=(
            "cell starts one child per profile/prompt/batch/scenario/run for maximum isolation; "
            "profile-prompt starts one child per profile/prompt and sweeps batch/scenario/runs inside it."
        ),
    )
    parser.add_argument("--quality-summary", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse existing child JSON outputs when present; failed/missing children are rerun.",
    )
    parser.add_argument("--output", default="outputs/online_batch/prompt_batch_matrix.json")
    args, passthrough = parser.parse_known_args(argv)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    child_dir = output.parent / f"{output.stem}_children"
    child_dir.mkdir(parents=True, exist_ok=True)
    optimization_profiles = parse_optimization_profiles(args.profile_set, args.optimization_profiles)
    batch_sizes = parse_csv_ints(args.batch_sizes)
    prompt_kinds = parse_csv_strings(args.prompt_lengths)
    scenarios = parse_csv_strings(args.scenarios)
    runs = []
    for optimization_profile in optimization_profiles:
        for prompt_kind in prompt_kinds:
            prompt_text = prompt_for_kind(prompt_kind)
            if args.child_granularity == "profile-prompt":
                records = run_group_child(
                    args,
                    passthrough,
                    prompt_kind=prompt_kind,
                    prompt_text=prompt_text,
                    optimization_profile=optimization_profile,
                    batch_sizes=batch_sizes,
                    scenarios=scenarios,
                    output_dir=child_dir,
                )
                runs.extend(records)
                status = "dry-run" if records[0].get("dry_run") else f"exit={records[0].get('worker_exit_code')}"
                print(
                    f"{status} profile={optimization_profile} prompt={prompt_kind} "
                    f"batch={','.join(str(item) for item in batch_sizes)} scenario={','.join(scenarios)} runs={args.runs}",
                    flush=True,
                )
                continue
            for batch_size in batch_sizes:
                for scenario in scenarios:
                    for run_index in range(int(args.runs)):
                        record = run_child(
                            args,
                            passthrough,
                            prompt_kind=prompt_kind,
                            prompt_text=prompt_text,
                            batch_size=batch_size,
                            scenario=scenario,
                            optimization_profile=optimization_profile,
                            run_index=run_index,
                            output_dir=child_dir,
                        )
                        runs.append(record)
                        status = "dry-run" if record.get("dry_run") else f"exit={record.get('worker_exit_code')}"
                        print(
                            f"{status} profile={optimization_profile} prompt={prompt_kind} "
                            f"batch={batch_size} scenario={scenario} run={run_index}",
                            flush=True,
                        )
    payload = {
        "benchmark": "prompt_batch_matrix",
        "created_at_unix": time.time(),
        "device": args.device,
        "batch_sizes": batch_sizes,
        "prompt_lengths": prompt_kinds,
        "scenarios": scenarios,
        "child_granularity": args.child_granularity,
        "profile_set": args.profile_set,
        "optimization_profiles": optimization_profiles,
        "runs": runs,
        "summary": summarize_runs(runs),
    }
    if args.quality_summary:
        quality_path = Path(args.quality_summary)
        payload["quality_summary_path"] = str(quality_path)
        payload["quality_summary"] = json.loads(quality_path.read_text(encoding="utf-8"))
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()

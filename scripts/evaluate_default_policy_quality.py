from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else None


def result_passed(item: dict[str, Any]) -> bool:
    return bool(item.get("passed", item.get("quality_passed", False)))


def quality_summary_passed(summary: dict[str, Any] | None, *, require_omni: bool) -> tuple[bool, str]:
    if not summary:
        return False, "missing_quality_summary"
    if result_passed(summary):
        return True, "summary_passed"
    winner = summary.get("winner")
    if isinstance(winner, dict) and result_passed(winner):
        if require_omni and not bool((winner.get("omni") or winner.get("omni_judge") or {}).get("pass", False)):
            return False, "winner_missing_omni_pass"
        return True, "winner_passed"
    results = summary.get("results")
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and result_passed(item):
                if require_omni and not bool((item.get("omni") or item.get("omni_judge") or {}).get("pass", False)):
                    continue
                return True, "result_passed"
    return False, "quality_not_passed"


def online_benchmark_passed(
    summary: dict[str, Any] | None,
    *,
    max_ttft_p90_ms: float,
    min_aggregate_tps: float,
    require_sampled: bool,
) -> tuple[bool, str, dict[str, Any]]:
    if not summary:
        return False, "missing_online_benchmark", {}
    runs = summary.get("runs")
    if not isinstance(runs, list) or not runs:
        return False, "no_online_runs", {}
    eligible = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        if require_sampled and not bool(run.get("do_sample", False)):
            continue
        if any("error" in item for item in run.get("requests", []) if isinstance(item, dict)):
            continue
        eligible.append(run)
    if not eligible:
        return False, "no_eligible_online_runs", {}
    worst_ttft_p90 = max(float(item.get("ttft_ms_p90", 0.0)) for item in eligible)
    best_aggregate_tps = max(float(item.get("aggregate_tps", 0.0)) for item in eligible)
    largest_batch = max(int(item.get("batch_size", 1)) for item in eligible)
    metrics = {
        "runs": len(eligible),
        "largest_batch": largest_batch,
        "worst_ttft_p90_ms": worst_ttft_p90,
        "best_aggregate_tps": best_aggregate_tps,
    }
    if worst_ttft_p90 > max_ttft_p90_ms:
        return False, "ttft_p90_over_threshold", metrics
    if best_aggregate_tps < min_aggregate_tps:
        return False, "aggregate_tps_under_threshold", metrics
    return True, "online_benchmark_passed", metrics


def single_arch_gate_passed(
    summary: dict[str, Any] | None,
    *,
    required_modes: tuple[str, ...],
    max_rtf_p90: float,
    max_ttft_p90_ms: float,
) -> tuple[bool, str, dict[str, Any]]:
    if not summary:
        return False, "missing_single_arch_gate", {}
    modes = summary.get("modes")
    if not isinstance(modes, dict):
        return False, "single_arch_gate_missing_modes", {}
    metrics: dict[str, Any] = {"required_modes": list(required_modes)}
    if not bool(summary.get("passed")):
        metrics["summary_passed"] = False
        return False, "single_arch_gate_not_passed", metrics
    for mode in required_modes:
        item = modes.get(mode)
        if not isinstance(item, dict):
            return False, f"single_arch_gate_missing_mode:{mode}", metrics
        rtf_p90 = float(item.get("computed_rtf_p90", float("inf")))
        ttft_p90 = float(item.get("ttft_ms_p90", float("inf")))
        metrics[mode] = {
            "passed": bool(item.get("passed")),
            "computed_rtf_p90": rtf_p90,
            "ttft_ms_p90": ttft_p90,
        }
        if not bool(item.get("passed")):
            return False, f"single_arch_gate_mode_failed:{mode}", metrics
        if rtf_p90 >= max_rtf_p90:
            return False, f"single_arch_gate_rtf_over_threshold:{mode}", metrics
        if ttft_p90 > max_ttft_p90_ms:
            return False, f"single_arch_gate_ttft_over_threshold:{mode}", metrics
    return True, "single_arch_gate_passed", metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Gate sampled generation + online batching as production defaults."
    )
    parser.add_argument("--quality-summary", default="outputs/long_text_quality/quality_summary.json")
    parser.add_argument("--online-benchmark-json", default="outputs/online_batch/benchmark.json")
    parser.add_argument("--single-arch-gate-json", default="outputs/single_arch_gate/quality_summary.json")
    parser.add_argument("--output-json", default="outputs/default_policy_quality/quality_summary.json")
    parser.add_argument("--max-ttft-p90-ms", type=float, default=1500.0)
    parser.add_argument("--max-rtf-p90", type=float, default=1.0)
    parser.add_argument("--min-aggregate-tps", type=float, default=12.0)
    parser.add_argument("--required-modes", default="voice_design,custom_voice,voice_clone")
    parser.add_argument("--no-require-omni", action="store_true")
    parser.add_argument("--allow-greedy-online-benchmark", action="store_true")
    args = parser.parse_args(argv)

    quality_path = Path(args.quality_summary)
    online_path = Path(args.online_benchmark_json)
    single_arch_path = Path(args.single_arch_gate_json)
    quality = load_json(quality_path)
    online = load_json(online_path)
    single_arch = load_json(single_arch_path)
    quality_ok, quality_reason = quality_summary_passed(quality, require_omni=not args.no_require_omni)
    online_ok, online_reason, online_metrics = online_benchmark_passed(
        online,
        max_ttft_p90_ms=args.max_ttft_p90_ms,
        min_aggregate_tps=args.min_aggregate_tps,
        require_sampled=not args.allow_greedy_online_benchmark,
    )
    required_modes = tuple(item.strip().replace("-", "_") for item in args.required_modes.split(",") if item.strip())
    single_arch_ok, single_arch_reason, single_arch_metrics = single_arch_gate_passed(
        single_arch,
        required_modes=required_modes,
        max_rtf_p90=args.max_rtf_p90,
        max_ttft_p90_ms=args.max_ttft_p90_ms,
    )
    passed = bool(quality_ok and online_ok and single_arch_ok)
    payload = {
        "feature": "sampled_online_default",
        "policy": "sampled-online",
        "passed": passed,
        "quality_passed": passed,
        "created_at_unix": time.time(),
        "quality": {
            "path": str(quality_path),
            "passed": quality_ok,
            "reason": quality_reason,
        },
        "online_benchmark": {
            "path": str(online_path),
            "passed": online_ok,
            "reason": online_reason,
            **online_metrics,
        },
        "single_arch_gate": {
            "path": str(single_arch_path),
            "passed": single_arch_ok,
            "reason": single_arch_reason,
            **single_arch_metrics,
        },
        "generation_defaults": {
            "do_sample": True,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.9,
            "repetition_penalty": 1.05,
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"passed={passed} wrote {output}", flush=True)


if __name__ == "__main__":
    main()

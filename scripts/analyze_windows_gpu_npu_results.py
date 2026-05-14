#!/usr/bin/env python3
"""Analyze Windows GPU+NPU probe and benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPECTED_OFFLOAD = {
    "gpu_only": "off",
    "npu_decoder": "decoder",
    "npu_audio": "audio",
    "npu_all": "all",
}


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def is_npu(value: object) -> bool:
    return str(value or "").strip().upper().startswith("NPU")


def scenario_results(summary: dict) -> dict[str, dict]:
    return {
        str(item.get("name")): item
        for item in summary.get("results", [])
        if isinstance(item, dict) and item.get("name")
    }


def metric(result: dict, key: str) -> object:
    return (result.get("summary") or {}).get(key)


def accelerator_average(result: dict, category: str) -> float | None:
    counters = (result.get("summary") or {}).get("accelerator_counters") or {}
    if not isinstance(counters, dict):
        return None
    payload = counters.get(category) or {}
    value = payload.get("utilization_average") if isinstance(payload, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def compute_comparison(results_by_name: dict[str, dict]) -> dict[str, dict]:
    gpu = results_by_name.get("gpu_only") or {}
    gpu_rtf = metric(gpu, "median_computed_rtf")
    gpu_util = accelerator_average(gpu, "gpu")
    comparison = {}
    for name, result in results_by_name.items():
        if name == "gpu_only":
            continue
        rtf = metric(result, "median_computed_rtf")
        gpu_avg = accelerator_average(result, "gpu")
        npu_avg = accelerator_average(result, "npu")
        comparison[name] = {
            "computed_rtf_delta": None if gpu_rtf is None or rtf is None else float(rtf) - float(gpu_rtf),
            "computed_rtf_speedup": None if gpu_rtf is None or not rtf else float(gpu_rtf) / float(rtf),
            "gpu_utilization_average": gpu_avg,
            "npu_utilization_average": npu_avg,
            "gpu_utilization_reduction": (
                None
                if gpu_util is None or gpu_avg is None or float(gpu_util) == 0.0
                else (float(gpu_util) - float(gpu_avg)) / float(gpu_util)
            ),
        }
    return comparison


def get_comparison(summary: dict, results_by_name: dict[str, dict]) -> dict[str, dict]:
    existing = summary.get("comparison")
    if isinstance(existing, dict) and existing:
        return existing
    return compute_comparison(results_by_name)


def validate_probe(
    probe: dict | None,
    *,
    require_probe_ok: bool,
    require_prompt_compile: bool,
    require_audio_compile: bool,
) -> tuple[list[str], list[str], dict]:
    failures = []
    warnings = []
    details: dict = {}
    if probe is None:
        if require_probe_ok or require_prompt_compile or require_audio_compile:
            failures.append("probe_json_missing")
        return failures, warnings, details

    details["probe_status"] = probe.get("status")
    if require_probe_ok and probe.get("status") != "ok":
        failures.append(f"probe_status={probe.get('status')!r} != 'ok'")

    decoder_compile = probe.get("decoder_compile") or []
    details["decoder_compile_count"] = len(decoder_compile) if isinstance(decoder_compile, list) else 0
    if not decoder_compile:
        failures.append("probe decoder_compile is empty")
    else:
        for item in decoder_compile:
            if isinstance(item, dict) and not is_npu(item.get("device")):
                failures.append(f"probe decoder graph {item.get('label')} compiled on {item.get('device')}, expected NPU")

    prompt_compile = probe.get("prompt_compile") or {}
    details["prompt_compile_status"] = prompt_compile.get("status") if isinstance(prompt_compile, dict) else None
    if require_prompt_compile:
        if not isinstance(prompt_compile, dict) or prompt_compile.get("status") != "ok":
            failures.append(f"prompt_compile status={details['prompt_compile_status']!r}, expected ok")
        for item in (prompt_compile.get("graphs") or []) if isinstance(prompt_compile, dict) else []:
            if isinstance(item, dict) and not is_npu(item.get("device")):
                failures.append(f"prompt graph {item.get('label')} compiled on {item.get('device')}, expected NPU")
    elif isinstance(prompt_compile, dict) and prompt_compile.get("status") not in {None, "ok", "skipped", "not_run"}:
        warnings.append(f"prompt_compile status={prompt_compile.get('status')!r}")

    audio_compile = probe.get("audio_encoder_compile") or {}
    details["audio_encoder_compile_status"] = audio_compile.get("status") if isinstance(audio_compile, dict) else None
    if require_audio_compile:
        if not isinstance(audio_compile, dict) or audio_compile.get("status") != "ok":
            failures.append(f"audio_encoder_compile status={details['audio_encoder_compile_status']!r}, expected ok")
        for item in (audio_compile.get("graphs") or []) if isinstance(audio_compile, dict) else []:
            if isinstance(item, dict) and not is_npu(item.get("device")):
                failures.append(f"audio encoder graph {item.get('label')} compiled on {item.get('device')}, expected NPU")
    return failures, warnings, details


def validate_benchmark(
    summary: dict,
    *,
    required_scenarios: list[str],
    min_speedup: float | None,
    max_rtf_regression: float | None,
    min_gpu_utilization_reduction: float | None,
    require_counters: bool,
) -> tuple[list[str], list[str], dict]:
    failures = []
    warnings = []
    results_by_name = scenario_results(summary)
    comparison = get_comparison(summary, results_by_name)
    details = {"scenario_count": len(results_by_name), "comparison": comparison}

    if summary.get("status") not in {"ok", "failed"}:
        failures.append(f"benchmark status={summary.get('status')!r}, expected ok or failed with detailed results")
    if summary.get("status") == "failed":
        acceptance_failures = ((summary.get("acceptance") or {}).get("failures") or [])
        if acceptance_failures:
            failures.extend(str(item) for item in acceptance_failures)

    for scenario in required_scenarios:
        result = results_by_name.get(scenario)
        if result is None:
            failures.append(f"missing scenario {scenario}")
            continue
        if result.get("error"):
            failures.append(f"{scenario}: {result.get('error')}")
        expected = EXPECTED_OFFLOAD.get(scenario)
        effective = metric(result, "npu_offload_effective")
        if expected and effective != expected:
            failures.append(f"{scenario}: npu_offload_effective={effective!r}, expected {expected!r}")
        if scenario.startswith("npu_"):
            if not is_npu(metric(result, "decoder_device")):
                failures.append(f"{scenario}: decoder_device={metric(result, 'decoder_device')!r}, expected NPU")
        if scenario in {"npu_audio", "npu_all"}:
            if not is_npu(metric(result, "encoder_device")):
                failures.append(f"{scenario}: encoder_device={metric(result, 'encoder_device')!r}, expected NPU")
        if scenario == "npu_all":
            prompt_device = metric(result, "prompt_device") or metric(result, "text_embedding_device")
            if not is_npu(prompt_device):
                failures.append(f"{scenario}: prompt/text_embedding device={prompt_device!r}, expected NPU")
        counters = (result.get("summary") or {}).get("accelerator_counters")
        if require_counters:
            if not isinstance(counters, dict):
                failures.append(f"{scenario}: accelerator counters missing")
            elif counters.get("status") != "ok":
                failures.append(f"{scenario}: accelerator counter status={counters.get('status')!r}, expected ok")
        elif isinstance(counters, dict) and counters.get("status") not in {None, "ok"}:
            warnings.append(f"{scenario}: accelerator counter status={counters.get('status')!r}")

    for scenario, metrics in comparison.items():
        speedup = metrics.get("computed_rtf_speedup")
        delta = metrics.get("computed_rtf_delta")
        gpu_reduction = metrics.get("gpu_utilization_reduction")
        if min_speedup is not None and (speedup is None or float(speedup) < float(min_speedup)):
            failures.append(f"{scenario}: computed_rtf_speedup={speedup} < {min_speedup}")
        if max_rtf_regression is not None and (delta is None or float(delta) > float(max_rtf_regression)):
            failures.append(f"{scenario}: computed_rtf_delta={delta} > {max_rtf_regression}")
        if min_gpu_utilization_reduction is not None and (
            gpu_reduction is None or float(gpu_reduction) < float(min_gpu_utilization_reduction)
        ):
            failures.append(f"{scenario}: gpu_utilization_reduction={gpu_reduction} < {min_gpu_utilization_reduction}")

    return failures, warnings, details


def analyze(args: argparse.Namespace) -> dict:
    benchmark = load_json(args.benchmark_summary)
    probe = load_json(args.probe_json) if args.probe_json else None
    required_scenarios = parse_csv(args.require_scenarios) or ["gpu_only", "npu_decoder", "npu_audio"]

    failures, warnings, details = validate_benchmark(
        benchmark,
        required_scenarios=required_scenarios,
        min_speedup=args.min_speedup,
        max_rtf_regression=args.max_rtf_regression,
        min_gpu_utilization_reduction=args.min_gpu_utilization_reduction,
        require_counters=args.require_counters,
    )
    probe_failures, probe_warnings, probe_details = validate_probe(
        probe,
        require_probe_ok=args.require_probe_ok,
        require_prompt_compile=args.require_prompt_compile,
        require_audio_compile=args.require_audio_compile,
    )
    failures.extend(probe_failures)
    warnings.extend(probe_warnings)
    details["probe"] = probe_details

    return {
        "status": "ok" if not failures else "failed",
        "benchmark_summary": str(Path(args.benchmark_summary).resolve()),
        "probe_json": None if not args.probe_json else str(Path(args.probe_json).resolve()),
        "required_scenarios": required_scenarios,
        "checks": {
            "min_speedup": args.min_speedup,
            "max_rtf_regression": args.max_rtf_regression,
            "min_gpu_utilization_reduction": args.min_gpu_utilization_reduction,
            "require_counters": args.require_counters,
            "require_probe_ok": args.require_probe_ok,
            "require_prompt_compile": args.require_prompt_compile,
            "require_audio_compile": args.require_audio_compile,
        },
        "details": details,
        "warnings": warnings,
        "failures": failures,
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-summary", required=True)
    parser.add_argument("--probe-json", default=None)
    parser.add_argument("--require-scenarios", default="gpu_only,npu_decoder,npu_audio")
    parser.add_argument("--min-speedup", type=float, default=None)
    parser.add_argument("--max-rtf-regression", type=float, default=None)
    parser.add_argument("--min-gpu-utilization-reduction", type=float, default=None)
    parser.add_argument("--require-counters", action="store_true")
    parser.add_argument("--require-probe-ok", action="store_true")
    parser.add_argument("--require-prompt-compile", action="store_true")
    parser.add_argument("--require-audio-compile", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    report = analyze(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

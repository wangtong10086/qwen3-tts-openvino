#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACCELERATOR_PREFIXES = ("cuda", "xpu")
CURRENT_REFERENCE_CACHE_SCHEMA = "qwen3_tts_prefill_reference_v5"


@dataclass(frozen=True)
class QualityRequirement:
    name: str
    path: str
    mode: str
    min_batch_size: int
    required_candidate_modes: tuple[str, ...] = ()
    min_prefill_bucket: int | None = None
    require_omni: bool = True
    require_gpu_reference: bool = True
    require_no_hit_max: bool = True


@dataclass(frozen=True)
class RouteRequirement:
    name: str
    mode: str
    request: dict[str, Any]
    expected_allowed: bool
    expected_reason_prefix: str
    long_output: bool = False
    voice_clone_xvector_env: str | None = None


@dataclass(frozen=True)
class OnlineBenchmarkRequirement:
    name: str
    path: str
    scenario: str
    batch_size: int
    min_aggregate_tps: float
    min_per_user_tps_p50: float = 0.0
    max_ttft_p90_ms: float = 2000.0
    require_no_sampled_fallback: bool = True
    require_sampled_used: bool = False


@dataclass(frozen=True)
class LongHotPathRequirement:
    name: str
    path: str
    mode: str
    candidate_mode: str = "runtime"
    max_hot_stream_rtf: float = 1.0
    require_native_stream_decode: bool = True


DEFAULT_REQUIREMENTS = (
    QualityRequirement(
        name="voice_design_batch3_mixed_language",
        path="outputs/prefill_quality_voice_design_batch3_mixed_lang_gpu_ref_omni_v5/quality_summary.json",
        mode="voice_design",
        min_batch_size=3,
        required_candidate_modes=("serial", "dynamic_ragged"),
        min_prefill_bucket=4,
    ),
    QualityRequirement(
        name="voice_design_batch5_mixed_language",
        path="outputs/prefill_quality_voice_design_batch5_mixed_lang_gpu_ref_omni_v5/quality_summary.json",
        mode="voice_design",
        min_batch_size=5,
        required_candidate_modes=("dynamic_ragged",),
        min_prefill_bucket=8,
    ),
    QualityRequirement(
        name="voice_design_batch16_mixed_language",
        path="outputs/prefill_quality_voice_design_batch16_mixed_lang_gpu_ref_omni_v2/quality_summary.json",
        mode="voice_design",
        min_batch_size=16,
        required_candidate_modes=("dynamic_ragged",),
        min_prefill_bucket=8,
    ),
    QualityRequirement(
        name="voice_design_long_batch2_runtime",
        path="outputs/prefill_quality_voice_design_long_batch2_runtime_gpu_ref_omni_v5/quality_summary.json",
        mode="voice_design",
        min_batch_size=2,
        required_candidate_modes=("runtime",),
        min_prefill_bucket=None,
    ),
    QualityRequirement(
        name="custom_voice_vivian_batch8_online",
        path="outputs/prefill_quality_custom_voice_batch8_vivian_gpu_ref_omni_v5/quality_summary.json",
        mode="custom_voice",
        min_batch_size=8,
        required_candidate_modes=("dynamic_ragged",),
        min_prefill_bucket=8,
    ),
    QualityRequirement(
        name="custom_voice_mixed_runtime",
        path="outputs/prefill_quality_custom_voice_mixed_vivian_ryan_batch2_gpu_ref_omni_v5/quality_summary.json",
        mode="custom_voice",
        min_batch_size=2,
        required_candidate_modes=("runtime",),
        min_prefill_bucket=None,
    ),
    QualityRequirement(
        name="custom_voice_all_speakers_runtime_batch9",
        path="outputs/prefill_quality_custom_voice_all_speakers_runtime_batch9_no_names_gpu_ref_omni_v5/quality_summary.json",
        mode="custom_voice",
        min_batch_size=9,
        required_candidate_modes=("runtime",),
        min_prefill_bucket=None,
    ),
    QualityRequirement(
        name="custom_voice_ono_anna_proper_name_runtime",
        path="outputs/prefill_quality_custom_voice_ono_anna_runtime_auto_pron_gpu_ref_omni_v3/quality_summary.json",
        mode="custom_voice",
        min_batch_size=1,
        required_candidate_modes=("runtime",),
        min_prefill_bucket=None,
    ),
    QualityRequirement(
        name="custom_voice_vivian_ryan_mixed_online_rp12",
        path="outputs/prefill_quality_custom_voice_mixed_vivian_ryan_online_rp12_batch2_gpu_ref_omni_v5/quality_summary.json",
        mode="custom_voice",
        min_batch_size=2,
        required_candidate_modes=("serial", "dynamic_ragged"),
        min_prefill_bucket=2,
    ),
    QualityRequirement(
        name="custom_voice_vivian_ryan_batch16_online_rp12",
        path="outputs/prefill_quality_custom_voice_batch16_vivian_ryan_rp12_gpu_ref_omni_v5/quality_summary.json",
        mode="custom_voice",
        min_batch_size=16,
        required_candidate_modes=("dynamic_ragged",),
        min_prefill_bucket=8,
    ),
    QualityRequirement(
        name="custom_voice_ryan_english_online_rp12",
        path="outputs/prefill_quality_custom_voice_ryan_english_online_rp12_gpu_ref_omni_v5/quality_summary.json",
        mode="custom_voice",
        min_batch_size=1,
        required_candidate_modes=("serial",),
        min_prefill_bucket=1,
    ),
    QualityRequirement(
        name="custom_voice_long_vivian_runtime_rp12",
        path="outputs/prefill_quality_custom_voice_long_runtime_vivian_max960_rp12_gpu_ref_omni_v5/quality_summary.json",
        mode="custom_voice",
        min_batch_size=1,
        required_candidate_modes=("runtime",),
        min_prefill_bucket=None,
    ),
    QualityRequirement(
        name="custom_voice_long_vivian_ryan_batch2_runtime_rp12",
        path="outputs/prefill_quality_custom_voice_long_batch2_vivian_ryan_rp12_gpu_ref_omni_v5_seedfix/quality_summary.json",
        mode="custom_voice",
        min_batch_size=2,
        required_candidate_modes=("runtime",),
        min_prefill_bucket=None,
    ),
    QualityRequirement(
        name="voice_clone_icl_batch2_online",
        path="outputs/prefill_quality_voice_clone_online_icl_batch2_gpu_ref_omni_v5/quality_summary.json",
        mode="voice_clone",
        min_batch_size=2,
        required_candidate_modes=("dynamic_ragged",),
        min_prefill_bucket=None,
    ),
    QualityRequirement(
        name="voice_clone_prompt_dict_runtime_batch2",
        path="outputs/prefill_quality_voice_clone_prompt_dict_runtime_batch2_gpu_ref_omni_v5/quality_summary.json",
        mode="voice_clone",
        min_batch_size=2,
        required_candidate_modes=("runtime",),
        min_prefill_bucket=None,
    ),
    QualityRequirement(
        name="voice_clone_prompt_dict_natural_runtime_batch2",
        path="outputs/prefill_quality_voice_clone_prompt_dict_natural_runtime_batch2_gpu_ref_omni_v5/quality_summary.json",
        mode="voice_clone",
        min_batch_size=2,
        required_candidate_modes=("runtime",),
        min_prefill_bucket=None,
    ),
    QualityRequirement(
        name="voice_clone_prompt_dict_natural_online_batch2",
        path="outputs/prefill_quality_voice_clone_prompt_dict_natural_online_batch2_gpu_ref_omni_v5/quality_summary.json",
        mode="voice_clone",
        min_batch_size=2,
        required_candidate_modes=("dynamic_ragged",),
        min_prefill_bucket=2,
    ),
    QualityRequirement(
        name="voice_clone_prompt_dict_natural_online_batch4",
        path="outputs/prefill_quality_voice_clone_prompt_dict_natural_online_batch4_gpu_ref_omni_v5/quality_summary.json",
        mode="voice_clone",
        min_batch_size=4,
        required_candidate_modes=("dynamic_ragged",),
        min_prefill_bucket=4,
    ),
    QualityRequirement(
        name="voice_clone_prompt_dict_natural_online_batch8",
        path="outputs/prefill_quality_voice_clone_prompt_dict_natural_online_batch8_gpu_ref_omni_v5/quality_summary.json",
        mode="voice_clone",
        min_batch_size=8,
        required_candidate_modes=("dynamic_ragged",),
        min_prefill_bucket=8,
    ),
    QualityRequirement(
        name="voice_clone_icl_long_runtime",
        path="outputs/prefill_quality_voice_clone_icl_long_runtime_max960_gpu_ref_omni_v5/quality_summary.json",
        mode="voice_clone",
        min_batch_size=1,
        required_candidate_modes=("runtime",),
        min_prefill_bucket=None,
    ),
    QualityRequirement(
        name="voice_clone_prompt_dict_natural_long_batch2_runtime",
        path="outputs/prefill_quality_voice_clone_prompt_dict_natural_long_batch2_gpu_ref_omni_v5/quality_summary.json",
        mode="voice_clone",
        min_batch_size=2,
        required_candidate_modes=("runtime",),
        min_prefill_bucket=None,
    ),
    QualityRequirement(
        name="voice_clone_xvector_runtime_batch2",
        path="outputs/prefill_quality_voice_clone_xvector_runtime_batch2_greedy_max260_gpu_ref_omni_v5/quality_summary.json",
        mode="voice_clone",
        min_batch_size=2,
        required_candidate_modes=("runtime",),
        min_prefill_bucket=None,
    ),
)


DEFAULT_ONLINE_BENCHMARK_REQUIREMENTS = (
    OnlineBenchmarkRequirement(
        name="online_batch_b1_single_user_realtime",
        path="outputs/online_batch/benchmark_current_sampled_contsub_batch1_2_4_8.json",
        scenario="online",
        batch_size=1,
        min_aggregate_tps=10.0,
        min_per_user_tps_p50=12.0,
        max_ttft_p90_ms=2000.0,
    ),
    OnlineBenchmarkRequirement(
        name="online_batch_b4_throughput",
        path="outputs/online_batch/benchmark_current_sampled_contsub_batch1_2_4_8.json",
        scenario="online",
        batch_size=4,
        min_aggregate_tps=30.0,
        min_per_user_tps_p50=8.0,
        max_ttft_p90_ms=1600.0,
        require_sampled_used=True,
    ),
    OnlineBenchmarkRequirement(
        name="online_batch_b8_throughput",
        path="outputs/online_batch/benchmark_current_sampled_contsub_batch1_2_4_8.json",
        scenario="online",
        batch_size=8,
        min_aggregate_tps=45.0,
        min_per_user_tps_p50=8.0,
        max_ttft_p90_ms=2200.0,
        require_sampled_used=True,
    ),
    OnlineBenchmarkRequirement(
        name="online_batch_b16_throughput_probe",
        path="outputs/online_batch/benchmark_current_sampled_contsub_batch16_probe.json",
        scenario="online",
        batch_size=16,
        min_aggregate_tps=80.0,
        min_per_user_tps_p50=7.5,
        max_ttft_p90_ms=3000.0,
        require_sampled_used=True,
    ),
)


DEFAULT_LONG_HOT_PATH_REQUIREMENTS = (
    LongHotPathRequirement(
        name="long_voice_design_hot_full_ar",
        path="outputs/prefill_quality_voice_design_long_batch2_runtime_gpu_ref_omni_v5/quality_summary.json",
        mode="voice_design",
    ),
    LongHotPathRequirement(
        name="long_custom_voice_hot_full_ar",
        path="outputs/prefill_quality_custom_voice_long_runtime_vivian_max960_rp12_gpu_ref_omni_v5/quality_summary.json",
        mode="custom_voice",
    ),
    LongHotPathRequirement(
        name="long_custom_voice_batch2_hot_full_ar",
        path="outputs/prefill_quality_custom_voice_long_batch2_vivian_ryan_rp12_gpu_ref_omni_v5_seedfix/quality_summary.json",
        mode="custom_voice",
    ),
    LongHotPathRequirement(
        name="long_voice_clone_hot_full_ar",
        path="outputs/prefill_quality_voice_clone_icl_long_runtime_max960_gpu_ref_omni_v5/quality_summary.json",
        mode="voice_clone",
    ),
    LongHotPathRequirement(
        name="long_voice_clone_batch2_hot_full_ar",
        path="outputs/prefill_quality_voice_clone_prompt_dict_natural_long_batch2_gpu_ref_omni_v5/quality_summary.json",
        mode="voice_clone",
    ),
)


DEFAULT_ROUTE_REQUIREMENTS = (
    RouteRequirement(
        name="route_voice_design_short_online_allowed",
        mode="voice_design",
        request={"text": "short"},
        expected_allowed=True,
        expected_reason_prefix="vllm_like_voice_design_supported",
    ),
    RouteRequirement(
        name="route_voice_design_long_full_ar_online_allowed",
        mode="voice_design",
        request={"text": "long", "full_context_text": True},
        expected_allowed=True,
        expected_reason_prefix="vllm_like_voice_design_supported",
    ),
    RouteRequirement(
        name="route_custom_voice_vivian_chinese_online_allowed",
        mode="custom_voice",
        request={"speaker": "Vivian", "language": "Chinese"},
        expected_allowed=True,
        expected_reason_prefix="vllm_like_custom_voice_supported",
    ),
    RouteRequirement(
        name="route_custom_voice_ryan_english_low_rp_online_allowed",
        mode="custom_voice",
        request={"speaker": "Ryan", "language": "English", "generation": {"repetition_penalty": 1.05}},
        expected_allowed=True,
        expected_reason_prefix="vllm_like_custom_voice_supported",
    ),
    RouteRequirement(
        name="route_custom_voice_ryan_english_rp12_allowed",
        mode="custom_voice",
        request={"speaker": "Ryan", "language": "English", "generation": {"repetition_penalty": 1.2}},
        expected_allowed=True,
        expected_reason_prefix="vllm_like_custom_voice_supported",
    ),
    RouteRequirement(
        name="route_voice_clone_icl_online_allowed",
        mode="voice_clone",
        request={"x_vector_only": False},
        expected_allowed=True,
        expected_reason_prefix="vllm_like_voice_clone_supported",
    ),
    RouteRequirement(
        name="route_voice_clone_prompt_natural_icl_online_allowed",
        mode="voice_clone",
        request={
            "voice_clone_prompt": {
                "ref_spk_embedding": [0.0, 1.0],
                "ref_code": [[1] * 16, [2] * 16],
                "x_vector_only_mode": False,
                "icl_mode": True,
                "ref_text": "Reference text.",
            }
        },
        expected_allowed=True,
        expected_reason_prefix="vllm_like_voice_clone_supported",
    ),
    RouteRequirement(
        name="route_voice_clone_xvector_default_online_allowed",
        mode="voice_clone",
        request={"x_vector_only": True},
        expected_allowed=True,
        expected_reason_prefix="vllm_like_voice_clone_xvector_supported",
    ),
    RouteRequirement(
        name="route_voice_clone_prompt_without_ref_code_online_allowed",
        mode="voice_clone",
        request={"voice_clone_prompt": {"ref_spk_embedding": [[0.0, 1.0]]}},
        expected_allowed=True,
        expected_reason_prefix="vllm_like_voice_clone_xvector_supported",
    ),
    RouteRequirement(
        name="route_voice_clone_prompt_false_xvector_without_ref_code_online_allowed",
        mode="voice_clone",
        request={"voice_clone_prompt": {"ref_spk_embedding": [0.0, 1.0], "x_vector_only_mode": False, "icl_mode": True}},
        expected_allowed=True,
        expected_reason_prefix="vllm_like_voice_clone_xvector_supported",
    ),
    RouteRequirement(
        name="route_voice_clone_xvector_explicit_env_allowed",
        mode="voice_clone",
        request={"x_vector_only": True},
        expected_allowed=True,
        expected_reason_prefix="vllm_like_voice_clone_xvector_supported",
        voice_clone_xvector_env="1",
    ),
    RouteRequirement(
        name="route_prefix_codes_fallback",
        mode="voice_design",
        request={"_prefix_codes": [[1, 2, 3]]},
        expected_allowed=False,
        expected_reason_prefix="prefix_codes_not_supported",
    ),
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reference_cache_payload(summary: dict[str, Any]) -> dict[str, Any]:
    cache_dir = (summary.get("reference") or {}).get("cache_dir")
    if not cache_dir:
        return {}
    metadata_path = Path(str(cache_dir)) / "cache_key.json"
    if not metadata_path.exists():
        return {}
    try:
        metadata = load_json(metadata_path)
    except Exception:
        return {}
    payload = metadata.get("payload")
    return payload if isinstance(payload, dict) else {}


def reference_cache_schema(summary: dict[str, Any]) -> str | None:
    value = reference_cache_payload(summary).get("schema")
    return str(value) if value else None


def reference_uses_accelerator(summary: dict[str, Any]) -> bool:
    runtime = (summary.get("reference") or {}).get("runtime") or {}
    device = str(runtime.get("device") or "").strip().lower()
    return device.startswith(ACCELERATOR_PREFIXES)


def candidate_results(summary: dict[str, Any], modes: tuple[str, ...]) -> list[dict[str, Any]]:
    results = list(summary.get("results") or [])
    if not modes:
        return results
    expected = set(modes)
    return [
        item for item in results
        if str(item.get("prefill_mode") or item.get("mode") or "").replace("-", "_") in expected
    ]


def result_batch_size(result: dict[str, Any]) -> int:
    return len(list(result.get("results") or []))


def result_prefill_buckets(result: dict[str, Any]) -> set[int]:
    scheduler = result.get("scheduler") or {}
    values = scheduler.get("prefill_batch_buckets_seen") or []
    buckets = set()
    for value in values:
        try:
            buckets.add(int(value))
        except (TypeError, ValueError):
            pass
    return buckets


def optional_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def audit_requirement(requirement: QualityRequirement) -> dict[str, Any]:
    path = Path(requirement.path)
    failures: list[str] = []
    evidence: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        failures.append(f"missing summary: {path}")
        return {"name": requirement.name, "passed": False, "failures": failures, "evidence": evidence}

    summary = load_json(path)
    evidence.update(
        {
            "mode": summary.get("mode"),
            "passed": summary.get("passed"),
            "omni_enabled": summary.get("omni_enabled"),
            "reference_device": ((summary.get("reference") or {}).get("runtime") or {}).get("device"),
            "reference_cache_schema": reference_cache_schema(summary),
        }
    )
    if summary.get("mode") != requirement.mode:
        failures.append(f"expected mode={requirement.mode}, got {summary.get('mode')}")
    if not bool(summary.get("passed")):
        failures.append("summary did not pass")
    if requirement.require_omni and not bool(summary.get("omni_enabled")):
        failures.append("Omni judge was not enabled")
    if requirement.require_gpu_reference and not reference_uses_accelerator(summary):
        failures.append("reference did not run on CUDA/XPU")

    candidate_modes = candidate_results(summary, requirement.required_candidate_modes)
    seen_modes = [
        str(item.get("prefill_mode") or item.get("mode") or "").replace("-", "_")
        for item in summary.get("results") or []
    ]
    evidence["candidate_modes_seen"] = seen_modes
    if len(candidate_modes) != len(requirement.required_candidate_modes):
        missing = sorted(set(requirement.required_candidate_modes) - set(seen_modes))
        failures.append(f"missing required candidate mode(s): {missing}")

    max_batch = 0
    buckets: set[int] = set()
    for result in candidate_modes:
        mode = str(result.get("prefill_mode") or result.get("mode") or "").replace("-", "_")
        if not bool(result.get("passed")):
            failures.append(f"candidate mode {mode} did not pass")
        if requirement.require_no_hit_max and int(result.get("hit_max_new_tokens_count") or 0) != 0:
            failures.append(f"candidate mode {mode} hit max_new_tokens")
        max_batch = max(max_batch, result_batch_size(result))
        buckets.update(result_prefill_buckets(result))
        for item in result.get("results") or []:
            index = item.get("index")
            if not bool(item.get("quality_passed")):
                failures.append(f"candidate mode {mode} item {index} failed quality")
            if int(item.get("generated_frames") or 0) <= 0:
                failures.append(f"candidate mode {mode} item {index} generated no frames")
            if requirement.require_omni and not bool((item.get("omni_pair") or {}).get("pass")):
                failures.append(f"candidate mode {mode} item {index} failed Omni")

    evidence["max_batch_size"] = max_batch
    evidence["prefill_batch_buckets_seen"] = sorted(buckets)
    if max_batch < requirement.min_batch_size:
        failures.append(f"expected batch size >= {requirement.min_batch_size}, got {max_batch}")
    if requirement.min_prefill_bucket is not None and not any(
        bucket >= requirement.min_prefill_bucket for bucket in buckets
    ):
        failures.append(f"expected prefill bucket >= {requirement.min_prefill_bucket}, got {sorted(buckets)}")

    return {
        "name": requirement.name,
        "passed": not failures,
        "failures": failures,
        "evidence": evidence,
    }


def benchmark_summary_item(summary: dict[str, Any], scenario: str, batch_size: int) -> dict[str, Any] | None:
    for item in summary.get("summary") or []:
        if str(item.get("scenario")) == scenario and int(item.get("batch_size") or 0) == int(batch_size):
            return item
    return None


def matching_benchmark_runs(summary: dict[str, Any], scenario: str, batch_size: int) -> list[dict[str, Any]]:
    return [
        item for item in summary.get("runs") or []
        if str(item.get("scenario")) == scenario and int(item.get("batch_size") or 0) == int(batch_size)
    ]


def nested_p50(source: dict[str, Any], field: str) -> float:
    value = source.get(field)
    if isinstance(value, dict):
        return optional_float(value.get("p50"))
    return optional_float(value)


def nested_p90(source: dict[str, Any], field: str) -> float:
    value = source.get(field)
    if isinstance(value, dict):
        return optional_float(value.get("p90"))
    return optional_float(value)


def audit_online_benchmark_requirement(requirement: OnlineBenchmarkRequirement) -> dict[str, Any]:
    path = Path(requirement.path)
    failures: list[str] = []
    evidence: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "scenario": requirement.scenario,
        "batch_size": requirement.batch_size,
    }
    if not path.exists():
        failures.append(f"missing benchmark summary: {path}")
        return {"name": requirement.name, "passed": False, "failures": failures, "evidence": evidence}

    summary = load_json(path)
    item = benchmark_summary_item(summary, requirement.scenario, requirement.batch_size)
    if item is None:
        failures.append(f"missing benchmark row scenario={requirement.scenario} batch={requirement.batch_size}")
        return {"name": requirement.name, "passed": False, "failures": failures, "evidence": evidence}
    aggregate_tps = optional_float(item.get("aggregate_tps_mean"))
    per_user_tps_p50 = nested_p50(item, "per_user_tps")
    ttft_p90_ms = nested_p90(item, "ttft_ms")
    evidence.update(
        {
            "aggregate_tps_mean": aggregate_tps,
            "per_user_tps_p50": per_user_tps_p50,
            "ttft_p90_ms": ttft_p90_ms,
        }
    )
    if aggregate_tps < requirement.min_aggregate_tps:
        failures.append(f"aggregate_tps_mean {aggregate_tps:.3f} < {requirement.min_aggregate_tps:.3f}")
    if per_user_tps_p50 < requirement.min_per_user_tps_p50:
        failures.append(f"per_user_tps_p50 {per_user_tps_p50:.3f} < {requirement.min_per_user_tps_p50:.3f}")
    if ttft_p90_ms > requirement.max_ttft_p90_ms:
        failures.append(f"ttft_p90_ms {ttft_p90_ms:.1f} > {requirement.max_ttft_p90_ms:.1f}")

    runs = matching_benchmark_runs(summary, requirement.scenario, requirement.batch_size)
    fallback_count = 0
    mismatch_count = 0
    sampled_used_count = 0
    for run in runs:
        scheduler = run.get("scheduler") or {}
        fallback_count += int(scheduler.get("sampled_batch_subcode_fallback_count") or 0)
        mismatch_count += int(scheduler.get("sampled_batch_subcode_mismatch_count") or 0)
        sampled_used_count += int(scheduler.get("sampled_batch_subcode_used_count") or 0)
    evidence.update(
        {
            "runs": len(runs),
            "sampled_batch_subcode_fallback_count": fallback_count,
            "sampled_batch_subcode_mismatch_count": mismatch_count,
            "sampled_batch_subcode_used_count": sampled_used_count,
        }
    )
    if requirement.require_no_sampled_fallback and fallback_count:
        failures.append(f"sampled batch subcode fallback count is {fallback_count}")
    if requirement.require_no_sampled_fallback and mismatch_count:
        failures.append(f"sampled batch subcode mismatch count is {mismatch_count}")
    if requirement.require_sampled_used and sampled_used_count <= 0:
        failures.append("sampled batch subcode was not used")

    return {"name": requirement.name, "passed": not failures, "failures": failures, "evidence": evidence}


def candidate_hot_results(summary: dict[str, Any], candidate_mode: str) -> list[dict[str, Any]]:
    results = []
    for result in summary.get("results") or []:
        mode = str(result.get("prefill_mode") or result.get("mode") or "").replace("-", "_")
        if mode != candidate_mode:
            continue
        results.extend(item for item in result.get("results") or [] if isinstance(item, dict))
    return results


def hot_stream_rtf(item: dict[str, Any]) -> float:
    timings = item.get("timings") or item.get("last_timings") or {}
    if isinstance(timings, dict) and timings.get("stream_rtf") is not None:
        return optional_float(timings.get("stream_rtf"), default=float("inf"))
    return optional_float(item.get("stream_rtf"), default=float("inf"))


def audit_long_hot_path_requirement(requirement: LongHotPathRequirement) -> dict[str, Any]:
    path = Path(requirement.path)
    failures: list[str] = []
    evidence: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        failures.append(f"missing long quality summary: {path}")
        return {"name": requirement.name, "passed": False, "failures": failures, "evidence": evidence}

    summary = load_json(path)
    evidence.update(
        {
            "mode": summary.get("mode"),
            "passed": summary.get("passed"),
            "omni_enabled": summary.get("omni_enabled"),
            "reference_device": ((summary.get("reference") or {}).get("runtime") or {}).get("device"),
            "reference_cache_schema": reference_cache_schema(summary),
        }
    )
    if summary.get("mode") != requirement.mode:
        failures.append(f"expected mode={requirement.mode}, got {summary.get('mode')}")
    if not bool(summary.get("passed")):
        failures.append("summary did not pass")
    if not bool(summary.get("omni_enabled")):
        failures.append("Omni judge was not enabled")
    if not reference_uses_accelerator(summary):
        failures.append("reference did not run on CUDA/XPU")

    items = candidate_hot_results(summary, requirement.candidate_mode)
    evidence["items"] = len(items)
    if not items:
        failures.append(f"missing candidate mode {requirement.candidate_mode}")
    hot_rtfs = []
    decode_paths = []
    for item in items:
        index = item.get("index")
        rtf = hot_stream_rtf(item)
        hot_rtfs.append(rtf)
        timings = item.get("timings") or item.get("last_timings") or {}
        decode_path = str(timings.get("decode_path") or "")
        decode_paths.append(decode_path)
        if not bool(item.get("quality_passed")):
            failures.append(f"item {index} failed quality")
        if bool(item.get("hit_max_new_tokens")):
            failures.append(f"item {index} hit max_new_tokens")
        if rtf > requirement.max_hot_stream_rtf:
            failures.append(f"item {index} hot stream_rtf {rtf:.3f} > {requirement.max_hot_stream_rtf:.3f}")
        if requirement.require_native_stream_decode and not decode_path.startswith("native:stream:"):
            failures.append(f"item {index} did not use native stream decoder: {decode_path or '<missing>'}")
    evidence["hot_stream_rtf_max"] = max(hot_rtfs) if hot_rtfs else None
    evidence["decode_paths"] = sorted(set(decode_paths))
    return {"name": requirement.name, "passed": not failures, "failures": failures, "evidence": evidence}


def audit_route_requirement(requirement: RouteRequirement) -> dict[str, Any]:
    from qwen3_tts_ov.server import online_batch_prompt_family_supported

    allowed, reason = online_batch_prompt_family_supported(
        requirement.request,
        requirement.mode,
        long_output=requirement.long_output,
        voice_clone_xvector_env=requirement.voice_clone_xvector_env,
    )
    failures: list[str] = []
    if bool(allowed) != bool(requirement.expected_allowed):
        failures.append(f"expected allowed={requirement.expected_allowed}, got {allowed}")
    if not str(reason).startswith(requirement.expected_reason_prefix):
        failures.append(f"expected reason prefix {requirement.expected_reason_prefix!r}, got {reason!r}")
    return {
        "name": requirement.name,
        "passed": not failures,
        "failures": failures,
        "evidence": {
            "mode": requirement.mode,
            "request": requirement.request,
            "allowed": allowed,
            "reason": reason,
            "expected_allowed": requirement.expected_allowed,
            "expected_reason_prefix": requirement.expected_reason_prefix,
        },
    }


def parse_requirement(raw: str) -> QualityRequirement:
    # name=path,mode,min_batch[,candidate_modes][,min_prefill_bucket]
    if "=" not in raw:
        raise argparse.ArgumentTypeError("requirement must start with name=...")
    name, rest = raw.split("=", 1)
    fields = [field.strip() for field in rest.split(",")]
    if len(fields) < 3:
        raise argparse.ArgumentTypeError("requirement format: name=path,mode,min_batch[,candidate_modes][,min_prefill_bucket]")
    candidate_modes = tuple(
        item.strip().replace("-", "_")
        for item in (fields[3].split("+") if len(fields) >= 4 and fields[3] else [])
        if item.strip()
    )
    min_prefill_bucket = int(fields[4]) if len(fields) >= 5 and fields[4] else None
    return QualityRequirement(
        name=name,
        path=fields[0],
        mode=fields[1].replace("-", "_"),
        min_batch_size=int(fields[2]),
        required_candidate_modes=candidate_modes,
        min_prefill_bucket=min_prefill_bucket,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit prefill quality summary coverage against original Python reference gates.")
    parser.add_argument(
        "--require",
        action="append",
        type=parse_requirement,
        default=None,
        help="Requirement: name=path,mode,min_batch[,candidate_mode+candidate_mode][,min_prefill_bucket].",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--require-current-reference-cache-schema",
        action="store_true",
        help=(
            "Fail summaries whose Python reference cache metadata is older than "
            f"{CURRENT_REFERENCE_CACHE_SCHEMA}. Historical evidence remains useful by default; enable this for "
            "release-quality fresh audits."
        ),
    )
    args = parser.parse_args(argv)

    requirements = tuple(args.require or DEFAULT_REQUIREMENTS)
    results = [audit_requirement(requirement) for requirement in requirements]
    online_benchmark_results = [] if args.require else [
        audit_online_benchmark_requirement(requirement)
        for requirement in DEFAULT_ONLINE_BENCHMARK_REQUIREMENTS
    ]
    long_hot_path_results = [] if args.require else [
        audit_long_hot_path_requirement(requirement)
        for requirement in DEFAULT_LONG_HOT_PATH_REQUIREMENTS
    ]
    route_results = [] if args.require else [audit_route_requirement(requirement) for requirement in DEFAULT_ROUTE_REQUIREMENTS]
    stale_reference_schema = [
        item
        for item in [*results, *long_hot_path_results]
        if item.get("evidence", {}).get("reference_cache_schema") != CURRENT_REFERENCE_CACHE_SCHEMA
    ]
    if args.require_current_reference_cache_schema:
        for item in stale_reference_schema:
            schema = item.get("evidence", {}).get("reference_cache_schema")
            item["passed"] = False
            item["failures"].append(
                f"reference cache schema {schema or '<missing>'} != {CURRENT_REFERENCE_CACHE_SCHEMA}; refresh this quality summary"
            )
    summary = {
        "feature": "prefill_quality_coverage_audit",
        "passed": all(bool(item["passed"]) for item in [*results, *online_benchmark_results, *long_hot_path_results, *route_results]),
        "current_reference_cache_schema": CURRENT_REFERENCE_CACHE_SCHEMA,
        "stale_reference_cache_schema_count": len(stale_reference_schema),
        "stale_reference_cache_schema_names": [str(item.get("name")) for item in stale_reference_schema],
        "requirements": results,
        "online_benchmark_requirements": online_benchmark_results,
        "long_hot_path_requirements": long_hot_path_results,
        "route_requirements": route_results,
    }
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        schema = item["evidence"].get("reference_cache_schema") or "<missing>"
        schema_note = "" if schema == CURRENT_REFERENCE_CACHE_SCHEMA else f" reference_schema={schema}"
        print(f"{status} {item['name']} {item['evidence'].get('path')}{schema_note}")
        for failure in item["failures"]:
            print(f"  - {failure}")
    for item in online_benchmark_results:
        status = "PASS" if item["passed"] else "FAIL"
        evidence = item["evidence"]
        print(
            f"{status} {item['name']} {evidence.get('path')} "
            f"agg_tps={optional_float(evidence.get('aggregate_tps_mean')):.2f} "
            f"per_user_tps_p50={optional_float(evidence.get('per_user_tps_p50')):.2f} "
            f"ttft_p90={optional_float(evidence.get('ttft_p90_ms')):.1f}ms"
        )
        for failure in item["failures"]:
            print(f"  - {failure}")
    for item in long_hot_path_results:
        status = "PASS" if item["passed"] else "FAIL"
        evidence = item["evidence"]
        schema = evidence.get("reference_cache_schema") or "<missing>"
        schema_note = "" if schema == CURRENT_REFERENCE_CACHE_SCHEMA else f" reference_schema={schema}"
        print(
            f"{status} {item['name']} {evidence.get('path')} "
            f"hot_stream_rtf_max={evidence.get('hot_stream_rtf_max')}{schema_note}"
        )
        for failure in item["failures"]:
            print(f"  - {failure}")
    for item in route_results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status} {item['name']} {item['evidence'].get('reason')}")
        for failure in item["failures"]:
            print(f"  - {failure}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

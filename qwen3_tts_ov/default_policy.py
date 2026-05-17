from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_POLICY_ENV = "QWEN3_TTS_OV_DEFAULT_POLICY"
DEFAULT_POLICY_SUMMARY_ENV = "QWEN3_TTS_OV_DEFAULT_POLICY_SUMMARY"
DEFAULT_POLICY_SUMMARY_PATH = Path("outputs/default_policy_quality/quality_summary.json")
PACKAGE_POLICY_SUMMARY_PATH = Path(__file__).with_name("default_policy_summary.json")

SAMPLED_ONLINE_POLICY_NAMES = {
    "sampled-online",
    "sampled_online",
    "sampling-online",
    "sampling_online",
    "do-sample-online",
    "do_sample_online",
}


def env_default_policy() -> str | None:
    value = os.environ.get(DEFAULT_POLICY_ENV)
    if value is None:
        return None
    return str(value).strip().lower()


def load_policy_summary(path: str | Path | None = None) -> dict[str, Any] | None:
    summary_path = Path(path or os.environ.get(DEFAULT_POLICY_SUMMARY_ENV) or DEFAULT_POLICY_SUMMARY_PATH)
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def single_arch_gate_passed(summary: dict[str, Any]) -> bool:
    gate = summary.get("single_arch_gate")
    if not isinstance(gate, dict):
        return False
    if not bool(gate.get("passed")):
        return False
    for mode in ("voice_design", "custom_voice", "voice_clone"):
        item = gate.get(mode)
        if isinstance(item, dict) and not bool(item.get("passed", True)):
            return False
    return True


def sampled_online_default_passed(path: str | Path | None = None) -> tuple[bool, str]:
    """Return whether sampled generation + online batching may be enabled by default.

    The default is intentionally conservative. It only passes when explicitly
    forced by env or when a quality summary produced by the project gate says
    the sampled-online policy passed.
    """

    env_value = env_default_policy()
    if env_value in {"0", "false", "off", "no", "greedy", "legacy"}:
        return False, f"env:{DEFAULT_POLICY_ENV}"
    if env_value in {"1", "true", "on", "yes", "force", "sampled", *SAMPLED_ONLINE_POLICY_NAMES}:
        return True, f"env:{DEFAULT_POLICY_ENV}"

    explicit_path = path or os.environ.get(DEFAULT_POLICY_SUMMARY_ENV)
    candidate_paths = [Path(explicit_path)] if explicit_path else [DEFAULT_POLICY_SUMMARY_PATH, PACKAGE_POLICY_SUMMARY_PATH]
    missing_paths: list[str] = []
    summary = None
    summary_path = candidate_paths[0]
    invalid_reason = None
    for candidate in candidate_paths:
        summary_path = candidate
        if not candidate.exists():
            missing_paths.append(str(candidate))
            continue
        try:
            summary = load_policy_summary(candidate)
            break
        except Exception as exc:
            invalid_reason = f"invalid:{candidate}:{exc}"
            summary = None
            break
    if summary is None:
        if invalid_reason:
            return False, invalid_reason
        return False, "missing:" + ",".join(missing_paths)
    if not isinstance(summary, dict):
        return False, f"invalid:{summary_path}:not_object"
    if bool(summary.get("passed", summary.get("quality_passed", False))):
        if single_arch_gate_passed(summary):
            return True, f"summary:{summary_path}"
        return False, f"summary_missing_single_arch_gate:{summary_path}"
    results = summary.get("results")
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            name = str(item.get("feature") or item.get("policy") or item.get("profile") or "").lower()
            if any(policy in name for policy in SAMPLED_ONLINE_POLICY_NAMES) and bool(
                item.get("passed", item.get("quality_passed", False))
            ):
                if single_arch_gate_passed(item):
                    return True, f"summary:{summary_path}"
                return False, f"summary_missing_single_arch_gate:{summary_path}"
    return False, f"summary_not_passed:{summary_path}"


def resolve_generation_defaults(
    *,
    explicit_do_sample: bool | None,
    explicit_repetition_penalty: float | None,
    fallback_do_sample: bool = False,
    fallback_repetition_penalty: float = 1.0,
    sampled_repetition_penalty: float = 1.05,
) -> tuple[bool, float, dict[str, Any]]:
    default_ok, source = sampled_online_default_passed()
    do_sample_defaulted = explicit_do_sample is None
    repetition_penalty_defaulted = explicit_repetition_penalty is None
    if explicit_do_sample is None:
        do_sample = bool(default_ok or fallback_do_sample)
    else:
        do_sample = bool(explicit_do_sample)
    if explicit_repetition_penalty is None:
        repetition_penalty = float(sampled_repetition_penalty if do_sample else fallback_repetition_penalty)
    else:
        repetition_penalty = float(explicit_repetition_penalty)
    metadata = {
        "default_policy": "sampled_online",
        "default_policy_passed": bool(default_ok),
        "default_policy_source": source,
        "do_sample_defaulted": bool(do_sample_defaulted),
        "repetition_penalty_defaulted": bool(repetition_penalty_defaulted),
    }
    return do_sample, repetition_penalty, metadata

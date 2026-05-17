from __future__ import annotations

from collections.abc import Mapping
from typing import Any


TRUE_STRINGS = {"1", "true", "yes", "on", "require", "required", "enabled"}
FALSE_STRINGS = {"0", "false", "no", "off", "disabled", ""}


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in TRUE_STRINGS
    return False


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def _containers(metrics: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    containers: list[Mapping[str, Any]] = [metrics]
    for key in ("timings", "audio_timings", "final_timings"):
        value = metrics.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    for container in list(containers):
        value = container.get("native_timing")
        if isinstance(value, Mapping):
            containers.append(value)
    return containers


def lookup(metrics: Mapping[str, Any], key: str, default: Any = None) -> Any:
    for container in _containers(metrics):
        if key in container and container[key] is not None:
            return container[key]
    return default


def evaluate_fast_path(
    metrics: Mapping[str, Any],
    *,
    require_request_metrics: bool = True,
    require_paged_kv: bool = True,
    require_stream_decoder: bool = True,
    require_zero_host_copy_fallback: bool = True,
) -> dict[str, Any]:
    """Return a stable fast-path verdict for health and benchmark JSON.

    `require_request_metrics=False` is intended for `/health`, where only
    configured runtime state is available. Completed request metrics should use
    the default strict mode so fallback and host-copy regressions are visible.
    """

    failures: list[str] = []
    missing: list[str] = []

    native_audio = lookup(metrics, "native_audio_pipeline")
    native_pipeline = lookup(metrics, "native_pipeline")
    if not (truthy(native_audio) or truthy(native_pipeline)):
        failures.append("native_audio_pipeline=false")

    if require_paged_kv:
        paged_kv = lookup(metrics, "paged_kv")
        native_paged_kv = lookup(metrics, "native_paged_kv")
        if not (truthy(paged_kv) or truthy(native_paged_kv)):
            failures.append("paged_kv=false")
        backend = lookup(metrics, "paged_kv_backend")
        if require_request_metrics and backend not in (None, "native_paged_attention"):
            failures.append(f"paged_kv_backend={backend}")

    if require_stream_decoder:
        decode_path = lookup(metrics, "decode_path")
        if decode_path is None:
            if require_request_metrics:
                missing.append("decode_path")
        elif "stream:" not in str(decode_path):
            failures.append(f"decode_path={decode_path}")

    for key in ("fallback", "unroll_fallback", "codegen_fusion_fallback"):
        value = lookup(metrics, key)
        if truthy(value):
            failures.append(key)

    static_decode_failure = lookup(metrics, "native_paged_static_decode_failure")
    if static_decode_failure:
        failures.append(f"native_paged_static_decode_failure={static_decode_failure}")

    native_timing = lookup(metrics, "native_timing")
    if require_request_metrics and not isinstance(native_timing, Mapping):
        missing.append("native_timing")

    if require_zero_host_copy_fallback:
        for key in (
            "host_copy_fallback_count",
            "subcode_host_copy_fallback_count",
            "split_subcode_hidden_bind_fallback_count",
        ):
            value = lookup(metrics, key)
            parsed = _as_int(value)
            if parsed is None:
                if require_request_metrics:
                    missing.append(key)
                continue
            if parsed != 0:
                failures.append(f"{key}={parsed}")
        for key in (
            "split_subcode_remote_next_embed_fallback_count",
            "decode_step_prebind_fallback_count",
        ):
            value = lookup(metrics, key)
            if value is None:
                continue
            parsed = _as_int(value)
            if parsed is None:
                if require_request_metrics:
                    missing.append(key)
                continue
            if parsed != 0:
                failures.append(f"{key}={parsed}")

    ok = not failures and not missing
    reason_parts = failures + [f"missing:{item}" for item in missing]
    return {
        "fast_path_ok": ok,
        "fast_path_failure_reason": "ok" if ok else ",".join(reason_parts),
        "fast_path_failures": failures,
        "fast_path_missing_metrics": missing,
    }

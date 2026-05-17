"""Runtime profile policy.

The public runtime has been intentionally collapsed to one production path:
``fastest``.  The low-level ``cache``/``no-cache`` mode names remain only
because exported OpenVINO manifests and diagnostic constructors still use them
internally.
"""

FASTEST_PROFILE_NAME = "fastest"
FASTEST_MODE = "no-cache"
FASTEST_GRAPH_VARIANT = "int8_sym_paged_talker_split"
FASTEST_CACHE_KERNEL = "exact"
FASTEST_CACHE_STEP = "fused"
FASTEST_CODEGEN_UNROLL = 1
FASTEST_CODEGEN_SCHEDULE = "current"
FASTEST_CODEGEN_DECODE_UNROLL = "off"
FASTEST_PREFERRED_CACHE_BUCKET = 0
FASTEST_REPETITION_PENALTY = 1.0
FASTEST_CHUNK_STRATEGY = "smooth"
FASTEST_NATIVE_PIPELINE = "require"
FASTEST_NATIVE_BUFFER_REUSE = "off"
FASTEST_NATIVE_PAGED_KV = "require"
FASTEST_NATIVE_PAGED_KV_GQA = "on"
FASTEST_NATIVE_PAGED_KV_PRECISION = "u8"
FASTEST_NATIVE_PAGED_KV_BLOCK_SIZE = 16
FASTEST_NATIVE_PAGED_KV_SPLIT_SUBCODE = "on"
FASTEST_NATIVE_PAGED_KV_SCORE_AGGREGATION = "on"
FASTEST_NATIVE_CODEGEN_FUSION = "split"
FASTEST_NATIVE_CODEGEN_DEVICE = "GPU"
FASTEST_NATIVE_DYNAMIC_QUANTIZATION_GROUP_SIZE = 32

KV_CACHE_PROFILE_CHOICES = ("auto", "fp16", "bf16", "u8", "u8-input", "u8-all")
NPU_OFFLOAD_CHOICES = ("off", "auto", "decoder", "audio", "all", "require")

KV_CACHE_PROFILE_OPTIONS = {
    "fp16": {
        "native_paged_kv_precision": "f16",
        "native_paged_kv_cache_input_precision": "f32",
        "native_paged_kv_block_size": FASTEST_NATIVE_PAGED_KV_BLOCK_SIZE,
    },
    "bf16": {
        "native_paged_kv_precision": "bf16",
        "native_paged_kv_cache_input_precision": "f32",
        "native_paged_kv_block_size": FASTEST_NATIVE_PAGED_KV_BLOCK_SIZE,
    },
    "u8": {
        "native_paged_kv_precision": "u8",
        "native_paged_kv_cache_input_precision": "f32",
        "native_paged_kv_block_size": FASTEST_NATIVE_PAGED_KV_BLOCK_SIZE,
    },
    "u8-input": {
        "native_paged_kv_precision": "f16",
        "native_paged_kv_cache_input_precision": "u8",
        "native_paged_kv_block_size": FASTEST_NATIVE_PAGED_KV_BLOCK_SIZE,
    },
    "u8-all": {
        "native_paged_kv_precision": "u8",
        "native_paged_kv_cache_input_precision": "u8",
        "native_paged_kv_block_size": FASTEST_NATIVE_PAGED_KV_BLOCK_SIZE,
    },
}

RUNTIME_MODE_CHOICES = (FASTEST_PROFILE_NAME, "no-cache", "cache")
REALTIME_PROFILE_CHOICES = (FASTEST_PROFILE_NAME, "auto")
PUBLIC_REALTIME_PROFILE_CHOICES = REALTIME_PROFILE_CHOICES
CODEGEN_UNROLL_CHOICES = ("profile", "1", "4", "6", "8", "12")
CODEGEN_SCHEDULE_CHOICES = ("current",)


def _fastest_profile_options() -> dict:
    return {
        "realtime_profile": FASTEST_PROFILE_NAME,
        "mode": FASTEST_MODE,
        "cache_kernel": FASTEST_CACHE_KERNEL,
        "cache_step": FASTEST_CACHE_STEP,
        "graph_variant": FASTEST_GRAPH_VARIANT,
        "codegen_unroll": str(FASTEST_CODEGEN_UNROLL),
        "codegen_schedule": FASTEST_CODEGEN_SCHEDULE,
        "codegen_decode_unroll": FASTEST_CODEGEN_DECODE_UNROLL,
        "preferred_cache_bucket": str(FASTEST_PREFERRED_CACHE_BUCKET),
        "repetition_penalty": FASTEST_REPETITION_PENALTY,
        "native_pipeline": FASTEST_NATIVE_PIPELINE,
        "native_buffer_reuse": FASTEST_NATIVE_BUFFER_REUSE,
        "native_paged_kv": FASTEST_NATIVE_PAGED_KV,
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": FASTEST_NATIVE_PAGED_KV_PRECISION,
        "native_paged_kv_cache_input_precision": "f32",
        "native_paged_kv_block_size": str(FASTEST_NATIVE_PAGED_KV_BLOCK_SIZE),
        "native_paged_kv_split_subcode": "1",
        "native_paged_kv_score_aggregation": "1",
        "native_codegen_fusion": FASTEST_NATIVE_CODEGEN_FUSION,
        "native_codegen_device": FASTEST_NATIVE_CODEGEN_DEVICE,
        "native_dynamic_quantization_group_size": str(FASTEST_NATIVE_DYNAMIC_QUANTIZATION_GROUP_SIZE),
    }


REALTIME_BENCHMARK_PROFILE_OPTIONS = {
    FASTEST_PROFILE_NAME: _fastest_profile_options(),
}


def normalize_kv_cache_profile(profile: str | None) -> str:
    normalized = str(profile or "auto").strip().lower().replace("_", "-")
    if normalized not in KV_CACHE_PROFILE_CHOICES:
        raise ValueError(f"kv_cache_profile must be one of {', '.join(KV_CACHE_PROFILE_CHOICES)}")
    return normalized


def kv_cache_profile_options(profile: str | None) -> dict:
    normalized = normalize_kv_cache_profile(profile)
    if normalized == "auto":
        return {}
    return dict(KV_CACHE_PROFILE_OPTIONS[normalized])


def kv_cache_profile_from_options(
    precision: str | None,
    cache_input_precision: str | None,
    block_size: int | str | None,
) -> str:
    precision_text = str(precision or FASTEST_NATIVE_PAGED_KV_PRECISION).strip().lower()
    cache_input_text = str(cache_input_precision or "f32").strip().lower()
    try:
        block_size_int = int(block_size or FASTEST_NATIVE_PAGED_KV_BLOCK_SIZE)
    except Exception:
        block_size_int = FASTEST_NATIVE_PAGED_KV_BLOCK_SIZE
    for name, options in KV_CACHE_PROFILE_OPTIONS.items():
        if (
            precision_text == options["native_paged_kv_precision"]
            and cache_input_text == options["native_paged_kv_cache_input_precision"]
            and block_size_int == int(options["native_paged_kv_block_size"])
        ):
            return name
    return "custom"


def kv_cache_precision_bytes(precision: str | None) -> int:
    normalized = str(precision or FASTEST_NATIVE_PAGED_KV_PRECISION).strip().lower()
    if normalized in {"u8", "i8", "uint8", "int8"}:
        return 1
    if normalized in {"f16", "bf16", "float16", "bfloat16"}:
        return 2
    if normalized in {"f32", "float32"}:
        return 4
    if normalized in {"u4", "i4", "uint4", "int4"}:
        return 1
    return 2


def effective_runtime_options(
    mode: str,
    cache_kernel: str,
    cache_step: str,
    graph_variant: str,
) -> tuple[str, str, str, str]:
    if mode == FASTEST_PROFILE_NAME:
        return FASTEST_MODE, FASTEST_CACHE_KERNEL, FASTEST_CACHE_STEP, FASTEST_GRAPH_VARIANT
    return mode, cache_kernel, cache_step, graph_variant


def apply_realtime_profile(
    realtime_profile: str,
    mode: str,
    cache_kernel: str,
    cache_step: str,
    graph_variant: str,
) -> tuple[str, str, str, str]:
    if realtime_profile in {FASTEST_PROFILE_NAME, "auto"}:
        return FASTEST_MODE, FASTEST_CACHE_KERNEL, FASTEST_CACHE_STEP, FASTEST_GRAPH_VARIANT
    return effective_runtime_options(mode, cache_kernel, cache_step, graph_variant)


def effective_codegen_unroll(requested_mode: str, graph_variant: str, codegen_unroll: str | int | None) -> int:
    value = "profile" if codegen_unroll is None else str(codegen_unroll).strip().lower()
    if value == "profile":
        return FASTEST_CODEGEN_UNROLL
    if value not in {"1", "4", "6", "8", "12"}:
        raise ValueError("codegen_unroll must be one of profile, 1, 4, 6, 8, 12")
    return int(value)


def is_fastest_or_norepeat_mode(mode: str | None) -> bool:
    return str(mode or "").strip().lower().replace("_", "-") == FASTEST_PROFILE_NAME


def fastest_runtime_defaults() -> dict:
    defaults = _fastest_profile_options()
    defaults.update(
        {
            "chunk_strategy": FASTEST_CHUNK_STRATEGY,
            "native_paged_kv_gqa": FASTEST_NATIVE_PAGED_KV_GQA,
            "native_paged_kv_block_size": FASTEST_NATIVE_PAGED_KV_BLOCK_SIZE,
            "native_paged_kv_split_subcode": FASTEST_NATIVE_PAGED_KV_SPLIT_SUBCODE,
            "native_paged_kv_score_aggregation": FASTEST_NATIVE_PAGED_KV_SCORE_AGGREGATION,
            "native_dynamic_quantization_group_size": FASTEST_NATIVE_DYNAMIC_QUANTIZATION_GROUP_SIZE,
        }
    )
    return defaults


def normalize_codegen_schedule(codegen_schedule: str | None) -> str:
    value = str(codegen_schedule or "current").strip().lower().replace("_", "-")
    if value not in CODEGEN_SCHEDULE_CHOICES:
        raise ValueError(f"codegen_schedule must be one of {', '.join(CODEGEN_SCHEDULE_CHOICES)}")
    return value


def scheduled_codegen_unrolls(codegen_schedule: str, primary_unroll: int) -> tuple[int, ...]:
    normalize_codegen_schedule(codegen_schedule)
    return (int(primary_unroll),)


def missing_graph_variant_message(graph_variant: str, available: str) -> str:
    message = f"graph variant {graph_variant!r} not found in manifest; available variants: {available}"
    if graph_variant == FASTEST_GRAPH_VARIANT:
        message += (
            ". Generate the production fastest variant after exporting paged-KV seed graphs with: "
            "uv run python scripts/compress_openvino_weights.py --ir-dir auto --preset fastest"
        )
    if graph_variant in {"int8_sym_batch_fused_gqa", "int8_sym_batch_fused_gqa_selective"}:
        message += (
            ". Generate the production online-batching variant with: "
            "uv run python scripts/compress_openvino_weights.py --ir-dir auto --preset minimal-online-gqa"
        )
    return message

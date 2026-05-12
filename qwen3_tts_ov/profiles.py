FASTEST_PROFILE_NAME = "fastest"
FASTEST_MODE = "realtime-int8-sym-fused-cachedsub-norepeat"
FASTEST_GRAPH_VARIANT = "int8_sym_fused_cachedsub"
FASTEST_CACHE_KERNEL = "exact"
FASTEST_CACHE_STEP = "fused"
FASTEST_CODEGEN_UNROLL = 4
FASTEST_CODEGEN_SCHEDULE = "current"
FASTEST_CODEGEN_DECODE_UNROLL = "auto"
FASTEST_PREFERRED_CACHE_BUCKET = 96
FASTEST_REPETITION_PENALTY = 1.0
FASTEST_CHUNK_STRATEGY = "smooth"
FASTEST_NATIVE_PIPELINE = "require"
FASTEST_NATIVE_BUFFER_REUSE = "on"

RUNTIME_MODE_CHOICES = (
    "fastest",
    "no-cache",
    "cache",
    "fast-cache",
    "fused-no-cache",
    "realtime-int8",
    "realtime-int8-sym",
    "realtime-int8-sym-norepeat",
    "realtime-fp16-fused-rms",
    "realtime-int8-sym-fused-rms",
    "realtime-fp16-sdpa-fused-rms",
    "realtime-int8-sym-sdpa-fused-rms",
    "realtime-fp16-fused-cachedsub",
    "realtime-int8-sym-fused-cachedsub",
    "realtime-int8-sym-fused-cachedsub-norepeat",
    "realtime-fp16-sdpa-fused-cachedsub",
    "realtime-int8-sym-sdpa-fused-cachedsub",
    "realtime-int8-sym-sdpa-fused-cachedsub-norepeat",
    "realtime-fp16-fused-cachedsub-rms",
    "realtime-int8-sym-fused-cachedsub-rms",
)
REALTIME_PROFILE_CHOICES = (
    "fastest",
    "fp16",
    "int8",
    "int8-sym",
    "int8-sym-norepeat",
    "fp16-fused-rms",
    "int8-sym-fused-rms",
    "fp16-sdpa-fused-rms",
    "int8-sym-sdpa-fused-rms",
    "fp16-fused-cachedsub",
    "int8-sym-fused-cachedsub",
    "int8-sym-fused-cachedsub-norepeat",
    "fp16-sdpa-fused-cachedsub",
    "int8-sym-sdpa-fused-cachedsub",
    "int8-sym-sdpa-fused-cachedsub-norepeat",
    "fp16-fused-cachedsub-rms",
    "int8-sym-fused-cachedsub-rms",
    "auto",
)
PUBLIC_REALTIME_PROFILE_CHOICES = ("fastest", "auto")
CODEGEN_UNROLL_CHOICES = ("profile", "1", "4", "6", "8", "12")
CODEGEN_SCHEDULE_CHOICES = ("current", "ll-v2", "balanced-v2")
REALTIME_BENCHMARK_PROFILE_OPTIONS = {
    "fastest": {
        "realtime_profile": FASTEST_PROFILE_NAME,
        "codegen_unroll": str(FASTEST_CODEGEN_UNROLL),
        "codegen_schedule": FASTEST_CODEGEN_SCHEDULE,
        "codegen_decode_unroll": FASTEST_CODEGEN_DECODE_UNROLL,
        "preferred_cache_bucket": str(FASTEST_PREFERRED_CACHE_BUCKET),
        "repetition_penalty": FASTEST_REPETITION_PENALTY,
    },
    "fp16_fused": {"realtime_profile": "fp16", "codegen_unroll": "1", "codegen_schedule": "current"},
    "int8_fused": {"realtime_profile": "int8", "codegen_unroll": "1", "codegen_schedule": "current"},
    "int8_sym_unroll4": {"realtime_profile": "int8-sym", "codegen_unroll": "4", "codegen_schedule": "current"},
    "native_int8_sym_norepeat_unroll4_decode_bucket96": {
        "realtime_profile": "int8-sym-norepeat",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
        "repetition_penalty": 1.0,
    },
    "int8_sym_ll_v2": {"realtime_profile": "int8-sym", "codegen_unroll": "4", "codegen_schedule": "ll-v2"},
    "int8_sym_balanced_v2": {"realtime_profile": "int8-sym", "codegen_unroll": "4", "codegen_schedule": "balanced-v2"},
    "fp16_fused_rms": {"realtime_profile": "fp16-fused-rms", "codegen_unroll": "4", "codegen_schedule": "current"},
    "int8_sym_fused_rms": {"realtime_profile": "int8-sym-fused-rms", "codegen_unroll": "4", "codegen_schedule": "current"},
    "fp16_sdpa_fused_rms": {"realtime_profile": "fp16-sdpa-fused-rms", "codegen_unroll": "4", "codegen_schedule": "current"},
    "int8_sym_sdpa_fused_rms": {
        "realtime_profile": "int8-sym-sdpa-fused-rms",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
    },
    "fp16_fused_cachedsub": {"realtime_profile": "fp16-fused-cachedsub", "codegen_unroll": "4", "codegen_schedule": "current"},
    "int8_sym_fused_cachedsub": {
        "realtime_profile": "int8-sym-fused-cachedsub",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
    },
    "native_fp16_fused_cachedsub_norepeat_bucket96": {
        "realtime_profile": "fp16-fused-cachedsub",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "auto",
        "preferred_cache_bucket": "96",
        "repetition_penalty": 1.0,
    },
    "native_int8_sym_fused_cachedsub_norepeat_bucket96": {
        "realtime_profile": "int8-sym-fused-cachedsub-norepeat",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "auto",
        "preferred_cache_bucket": "96",
        "repetition_penalty": 1.0,
    },
    "native_int8_sym_fused_cachedsub_norepeat_bucket80": {
        "realtime_profile": "int8-sym-fused-cachedsub-norepeat",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "auto",
        "preferred_cache_bucket": "80",
        "repetition_penalty": 1.0,
    },
    "native_int8_sym_fused_cachedsub_norepeat_bucket112": {
        "realtime_profile": "int8-sym-fused-cachedsub-norepeat",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "auto",
        "preferred_cache_bucket": "112",
        "repetition_penalty": 1.0,
    },
    "native_int8_sym_fused_cachedsub_norepeat_unroll8_bucket96": {
        "realtime_profile": "int8-sym-fused-cachedsub-norepeat",
        "codegen_unroll": "8",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "auto",
        "preferred_cache_bucket": "96",
        "repetition_penalty": 1.0,
    },
    "native_int8_sym_fused_cachedsub_norepeat_unroll12_bucket96": {
        "realtime_profile": "int8-sym-fused-cachedsub-norepeat",
        "codegen_unroll": "12",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "auto",
        "preferred_cache_bucket": "96",
        "repetition_penalty": 1.0,
    },
    "native_fp16_sdpa_fused_cachedsub_norepeat_bucket96": {
        "realtime_profile": "fp16-sdpa-fused-cachedsub",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "auto",
        "preferred_cache_bucket": "96",
        "repetition_penalty": 1.0,
    },
    "native_int8_sym_sdpa_fused_cachedsub_norepeat_bucket96": {
        "realtime_profile": "int8-sym-sdpa-fused-cachedsub-norepeat",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "auto",
        "preferred_cache_bucket": "96",
        "repetition_penalty": 1.0,
    },
    "native_fp16_fused_cachedsub_rms_norepeat_bucket96": {
        "realtime_profile": "fp16-fused-cachedsub-rms",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "auto",
        "preferred_cache_bucket": "96",
        "repetition_penalty": 1.0,
    },
    "native_int8_sym_fused_cachedsub_rms_norepeat_bucket96": {
        "realtime_profile": "int8-sym-fused-cachedsub-rms",
        "codegen_unroll": "4",
        "codegen_schedule": "current",
        "codegen_decode_unroll": "auto",
        "preferred_cache_bucket": "96",
        "repetition_penalty": 1.0,
    },
}


def effective_runtime_options(
    mode: str,
    cache_kernel: str,
    cache_step: str,
    graph_variant: str,
) -> tuple[str, str, str, str]:
    if mode == FASTEST_PROFILE_NAME:
        return effective_runtime_options(FASTEST_MODE, cache_kernel, cache_step, graph_variant)
    if mode == "fast-cache":
        return "cache", "sdpa", "split", "int8_cachedsub"
    if mode == "realtime-int8":
        return "cache", "exact", "fused", "int8_fused"
    if mode in {"realtime-int8-sym", "realtime-int8-sym-norepeat"}:
        return "cache", "exact", "fused", "int8_sym_fused"
    if mode == "realtime-fp16-fused-rms":
        return "cache", "exact", "fused", "fp16_fused_rms"
    if mode == "realtime-int8-sym-fused-rms":
        return "cache", "exact", "fused", "int8_sym_fused_rms"
    if mode == "realtime-fp16-sdpa-fused-rms":
        return "cache", "sdpa", "fused", "fp16_sdpa_fused_rms"
    if mode == "realtime-int8-sym-sdpa-fused-rms":
        return "cache", "sdpa", "fused", "int8_sym_sdpa_fused_rms"
    if mode == "realtime-fp16-fused-cachedsub":
        return "cache", "exact", "fused", "fp16_fused_cachedsub"
    if mode in {"realtime-int8-sym-fused-cachedsub", "realtime-int8-sym-fused-cachedsub-norepeat"}:
        return "cache", "exact", "fused", "int8_sym_fused_cachedsub"
    if mode == "realtime-fp16-sdpa-fused-cachedsub":
        return "cache", "sdpa", "fused", "fp16_sdpa_fused_cachedsub"
    if mode in {"realtime-int8-sym-sdpa-fused-cachedsub", "realtime-int8-sym-sdpa-fused-cachedsub-norepeat"}:
        return "cache", "sdpa", "fused", "int8_sym_sdpa_fused_cachedsub"
    if mode == "realtime-fp16-fused-cachedsub-rms":
        return "cache", "exact", "fused", "fp16_fused_cachedsub_rms"
    if mode == "realtime-int8-sym-fused-cachedsub-rms":
        return "cache", "exact", "fused", "int8_sym_fused_cachedsub_rms"
    return mode, cache_kernel, cache_step, graph_variant


def apply_realtime_profile(
    realtime_profile: str,
    mode: str,
    cache_kernel: str,
    cache_step: str,
    graph_variant: str,
) -> tuple[str, str, str, str]:
    if realtime_profile in {FASTEST_PROFILE_NAME, "auto"}:
        return effective_runtime_options(FASTEST_MODE, cache_kernel, cache_step, graph_variant)
    if realtime_profile == "int8":
        return effective_runtime_options("realtime-int8", cache_kernel, cache_step, graph_variant)
    if realtime_profile in {"int8-sym", "int8-sym-norepeat", "auto"}:
        return effective_runtime_options("realtime-int8-sym", cache_kernel, cache_step, graph_variant)
    if realtime_profile == "fp16-fused-rms":
        return effective_runtime_options("realtime-fp16-fused-rms", cache_kernel, cache_step, graph_variant)
    if realtime_profile == "int8-sym-fused-rms":
        return effective_runtime_options("realtime-int8-sym-fused-rms", cache_kernel, cache_step, graph_variant)
    if realtime_profile == "fp16-sdpa-fused-rms":
        return effective_runtime_options("realtime-fp16-sdpa-fused-rms", cache_kernel, cache_step, graph_variant)
    if realtime_profile == "int8-sym-sdpa-fused-rms":
        return effective_runtime_options("realtime-int8-sym-sdpa-fused-rms", cache_kernel, cache_step, graph_variant)
    if realtime_profile == "fp16-fused-cachedsub":
        return effective_runtime_options("realtime-fp16-fused-cachedsub", cache_kernel, cache_step, graph_variant)
    if realtime_profile == "int8-sym-fused-cachedsub":
        return effective_runtime_options("realtime-int8-sym-fused-cachedsub", cache_kernel, cache_step, graph_variant)
    if realtime_profile == "int8-sym-fused-cachedsub-norepeat":
        return effective_runtime_options("realtime-int8-sym-fused-cachedsub-norepeat", cache_kernel, cache_step, graph_variant)
    if realtime_profile == "fp16-sdpa-fused-cachedsub":
        return effective_runtime_options("realtime-fp16-sdpa-fused-cachedsub", cache_kernel, cache_step, graph_variant)
    if realtime_profile == "int8-sym-sdpa-fused-cachedsub":
        return effective_runtime_options("realtime-int8-sym-sdpa-fused-cachedsub", cache_kernel, cache_step, graph_variant)
    if realtime_profile == "int8-sym-sdpa-fused-cachedsub-norepeat":
        return effective_runtime_options("realtime-int8-sym-sdpa-fused-cachedsub-norepeat", cache_kernel, cache_step, graph_variant)
    if realtime_profile == "fp16-fused-cachedsub-rms":
        return effective_runtime_options("realtime-fp16-fused-cachedsub-rms", cache_kernel, cache_step, graph_variant)
    if realtime_profile == "int8-sym-fused-cachedsub-rms":
        return effective_runtime_options("realtime-int8-sym-fused-cachedsub-rms", cache_kernel, cache_step, graph_variant)
    return effective_runtime_options(mode, cache_kernel, cache_step, graph_variant)


def effective_codegen_unroll(requested_mode: str, graph_variant: str, codegen_unroll: str | int | None) -> int:
    value = "profile" if codegen_unroll is None else str(codegen_unroll).strip().lower()
    if value == "profile":
        unroll_variants = {
            "int8_sym_fused",
            "fp16_fused_rms",
            "int8_sym_fused_rms",
            "fp16_sdpa_fused_rms",
            "int8_sym_sdpa_fused_rms",
            "fp16_fused_cachedsub",
            "int8_sym_fused_cachedsub",
            "fp16_sdpa_fused_cachedsub",
            "int8_sym_sdpa_fused_cachedsub",
            "fp16_fused_cachedsub_rms",
            "int8_sym_fused_cachedsub_rms",
        }
        unroll_modes = {
            FASTEST_PROFILE_NAME,
            "realtime-int8-sym",
            "realtime-int8-sym-norepeat",
            "realtime-fp16-fused-rms",
            "realtime-int8-sym-fused-rms",
            "realtime-fp16-sdpa-fused-rms",
            "realtime-int8-sym-sdpa-fused-rms",
            "realtime-fp16-fused-cachedsub",
            "realtime-int8-sym-fused-cachedsub",
            "realtime-int8-sym-fused-cachedsub-norepeat",
            "realtime-fp16-sdpa-fused-cachedsub",
            "realtime-int8-sym-sdpa-fused-cachedsub",
            "realtime-int8-sym-sdpa-fused-cachedsub-norepeat",
            "realtime-fp16-fused-cachedsub-rms",
            "realtime-int8-sym-fused-cachedsub-rms",
        }
        if requested_mode in unroll_modes or graph_variant in unroll_variants:
            return 4
        return 1
    if value not in {"1", "4", "6", "8", "12"}:
        raise ValueError("codegen_unroll must be one of profile, 1, 4, 6, 8, 12")
    return int(value)


def is_fastest_or_norepeat_mode(mode: str | None) -> bool:
    normalized = str(mode or "").strip().lower().replace("_", "-")
    return normalized == FASTEST_PROFILE_NAME or normalized.endswith("-norepeat")


def fastest_runtime_defaults() -> dict:
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
        "chunk_strategy": FASTEST_CHUNK_STRATEGY,
        "native_pipeline": FASTEST_NATIVE_PIPELINE,
        "native_buffer_reuse": FASTEST_NATIVE_BUFFER_REUSE,
    }


def normalize_codegen_schedule(codegen_schedule: str | None) -> str:
    value = str(codegen_schedule or "current").strip().lower().replace("_", "-")
    if value not in CODEGEN_SCHEDULE_CHOICES:
        raise ValueError(f"codegen_schedule must be one of {', '.join(CODEGEN_SCHEDULE_CHOICES)}")
    return value


def scheduled_codegen_unrolls(codegen_schedule: str, primary_unroll: int) -> tuple[int, ...]:
    schedule = normalize_codegen_schedule(codegen_schedule)
    if schedule == "ll-v2":
        return (4, 12)
    if schedule == "balanced-v2":
        return (8, 12)
    return (int(primary_unroll),)


def missing_graph_variant_message(graph_variant: str, available: str) -> str:
    message = f"graph variant {graph_variant!r} not found in manifest; available variants: {available}"
    if graph_variant in {"int8_fused", "int8_sym_fused"}:
        mode = "int8_sym" if graph_variant == "int8_sym_fused" else "int8_asym"
        message += (
            ". Generate it with: uv run python scripts/compress_openvino_weights.py "
            f"--ir-dir auto --variant {graph_variant} --mode {mode}"
        )
    if graph_variant in {"fp16_fused_rms", "fp16_sdpa_fused_rms"}:
        message += (
            ". Generate it by re-exporting the model with: uv run python -m qwen3_tts_ov export "
            "--out-dir openvino/voice_design --rms-export-mode canonical"
        )
    if graph_variant in {"int8_sym_fused_rms", "int8_sym_sdpa_fused_rms"}:
        source_variant = "fp16_sdpa_fused_rms" if "sdpa" in graph_variant else "fp16_fused_rms"
        message += (
            ". Generate it after RMS export with: uv run python scripts/compress_openvino_weights.py "
            f"--ir-dir auto --source-variant {source_variant} --variant {graph_variant} --mode int8_sym"
        )
    if graph_variant == "fp16_fused_cachedsub":
        message += (
            ". Generate it by re-exporting fused codegen graphs with: uv run python -m qwen3_tts_ov export "
            "--out-dir openvino/voice_design --fused-subcode-mode cached"
        )
    if graph_variant == "int8_sym_fused_cachedsub":
        message += (
            ". Generate it after cached-subcode export with: uv run python scripts/compress_openvino_weights.py "
            "--ir-dir auto --source-variant fp16_fused_cachedsub --variant int8_sym_fused_cachedsub --mode int8_sym"
        )
    if graph_variant == "fp16_sdpa_fused_cachedsub":
        message += (
            ". Generate it by re-exporting fused codegen graphs with: uv run python -m qwen3_tts_ov export "
            "--out-dir openvino/voice_design --fused-cache-kernels sdpa --fused-subcode-mode cached"
        )
    if graph_variant == "int8_sym_sdpa_fused_cachedsub":
        message += (
            ". Generate it after cached-subcode SDPA export with: uv run python scripts/compress_openvino_weights.py "
            "--ir-dir auto --source-variant fp16_sdpa_fused_cachedsub --variant int8_sym_sdpa_fused_cachedsub --mode int8_sym --fused-cache-kernels sdpa"
        )
    if graph_variant == "fp16_fused_cachedsub_rms":
        message += (
            ". Generate it by re-exporting fused codegen graphs with: uv run python -m qwen3_tts_ov export "
            "--out-dir openvino/voice_design --fused-subcode-mode cached --rms-export-mode canonical"
        )
    if graph_variant == "int8_sym_fused_cachedsub_rms":
        message += (
            ". Generate it after cached-subcode RMS export with: uv run python scripts/compress_openvino_weights.py "
            "--ir-dir auto --source-variant fp16_fused_cachedsub_rms --variant int8_sym_fused_cachedsub_rms --mode int8_sym"
        )
    return message

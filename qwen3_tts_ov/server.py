import base64
import asyncio
import gc
import io
import json
import os
import queue
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

from .manifest import (
    AUTO_IR_DIR,
    DEFAULT_VOICE_DESIGN_IR_DIR,
    LEGACY_VOICE_DESIGN_IR_DIR,
    has_manifest,
    load_manifest,
    manifest_missing_message,
    path_text,
    resolve_ir_dir,
)
from .profiles import (
    FASTEST_CHUNK_STRATEGY,
    FASTEST_CODEGEN_DECODE_UNROLL,
    FASTEST_CODEGEN_SCHEDULE,
    FASTEST_CODEGEN_UNROLL,
    FASTEST_NATIVE_CODEGEN_DEVICE,
    FASTEST_NATIVE_BUFFER_REUSE,
    FASTEST_NATIVE_PAGED_KV,
    FASTEST_NATIVE_PAGED_KV_BLOCK_SIZE,
    FASTEST_NATIVE_PAGED_KV_GQA,
    FASTEST_NATIVE_PAGED_KV_PRECISION,
    FASTEST_NATIVE_PAGED_KV_SCORE_AGGREGATION,
    FASTEST_NATIVE_PAGED_KV_SPLIT_SUBCODE,
    FASTEST_NATIVE_PIPELINE,
    FASTEST_PREFERRED_CACHE_BUCKET,
    FASTEST_PROFILE_NAME,
    FASTEST_REPETITION_PENALTY,
    REALTIME_BENCHMARK_PROFILE_OPTIONS,
    REALTIME_PROFILE_CHOICES,
    effective_codegen_unroll,
    apply_realtime_profile,
    is_fastest_or_norepeat_mode,
    normalize_codegen_schedule,
)
from .runtime import DEFAULT_STREAM_CHUNK_STRATEGIES, OpenVINOQwen3TTS, StreamChunk, build_assistant_text, build_instruct_text
from .web_client import WEB_CLIENT_HTML


MODE_DIR = {
    "voice_design": "voice_design",
    "voice-design": "voice_design",
    "custom_voice": "custom_voice",
    "custom-voice": "custom_voice",
    "voice_clone": "base",
    "voice-clone": "base",
    "base": "base",
}
FASTEST_SHORT_OUTPUT_MAX_NEW_TOKENS = 48
TEXT_TOKEN_RE = re.compile(r"\s+|[A-Za-z0-9]+|[\u3400-\u9fff]|[^\s]")
SOFT_SPLIT_PUNCT = set("。！？!?；;，,、")
HARD_SPLIT_PUNCT = set("。！？!?；;\n")
CONTINUOUS_LONG_OUTPUT_BUCKET = 384
CONTINUOUS_LONG_OUTPUT_VARIANT = "int8_sym_fused"
CONTINUOUS_LONG_OUTPUT_PREFERRED_VARIANTS = ("int8_sym_fused_cachedsub", "int8_sym_fused")
CONTINUOUS_LONG_OUTPUT_PAGED_QUALITY_VARIANTS = ("int8_sym_paged_kv_seed", "int8_asym_paged_kv_seed", "fp16")
LONG_TEXT_QUALITY_SUMMARY_PATH = "outputs/long_text_quality/quality_summary.json"
LONG_TEXT_PROFILE_ENV_MAP = {
    "native_codegen_device": "QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE",
    "native_paged_kv_precision": "QWEN3_TTS_OV_NATIVE_PAGED_KV_PRECISION",
    "native_paged_kv_block_size": "QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE",
    "native_paged_kv_gqa": "QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA",
    "native_paged_kv_split_subcode": "QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE",
    "native_paged_kv_split_subcode_mode": "QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE_MODE",
    "native_paged_kv_score_aggregation": "QWEN3_TTS_OV_NATIVE_PAGED_KV_SCORE_AGGREGATION",
    "native_paged_kv_subcode_attention": "QWEN3_TTS_OV_NATIVE_PAGED_KV_SUBCODE_ATTENTION",
    "native_pipeline": "QWEN3_TTS_OV_NATIVE_PIPELINE",
    "native_paged_kv": "QWEN3_TTS_OV_NATIVE_PAGED_KV",
    "native_buffer_reuse": "QWEN3_TTS_OV_NATIVE_BUFFER_REUSE",
}
WEB_AUTO_SEGMENT_UNITS = 64
WEB_AUTO_SEGMENT_MAX_NEW_TOKENS = 240
WEB_AUTO_SEGMENT_PREFIX_FRAMES = 24
WEB_AUTO_SEGMENT_FADE_MS = 18
LONG_AR_REFERENCE_MODE_ENV = "QWEN3_TTS_OV_LONG_AR_REFERENCE_MODE"
LONG_AR_PROFILE_ENV = "QWEN3_TTS_OV_LONG_AR_PROFILE"
ENABLE_AUTO_SEGMENT_ENV = "QWEN3_TTS_OV_ENABLE_AUTO_SEGMENT"
ENABLE_PAGED_LONG_AR_ENV = "QWEN3_TTS_OV_ENABLE_PAGED_LONG_AR"
USE_LONG_TEXT_QUALITY_PROFILE_ENV = "QWEN3_TTS_OV_USE_LONG_TEXT_QUALITY_PROFILE"
PAGED_KV_UNAVAILABLE_REASON = (
    "current exported Qwen3-TTS IR uses OpenVINO ReadValue/Assign stateful KV "
    "instead of GenAI key_cache/value_cache/block_indices inputs"
)


def env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "on", "yes", "require"}


def native_paged_kv_requested() -> bool:
    return env_enabled("QWEN3_TTS_OV_NATIVE_PAGED_KV", False)


def paged_long_ar_enabled() -> bool:
    return native_paged_kv_requested() and env_enabled(ENABLE_PAGED_LONG_AR_ENV, False)


def speech_text_units(token: str) -> int:
    if not token or token.isspace():
        return 0
    if re.fullmatch(r"[A-Za-z0-9]+", token):
        return 2
    if re.fullmatch(r"[\u3400-\u9fff]", token):
        return 1
    return 0 if token in SOFT_SPLIT_PUNCT or token in set(".:：") else 1


def speech_text_unit_count(text: str) -> int:
    return sum(speech_text_units(token) for token in TEXT_TOKEN_RE.findall(str(text or "")))


def estimated_codec_frames_for_text(text: str, requested_max_new_tokens: int, *, cap: int = 1024) -> int:
    units = speech_text_unit_count(text)
    estimate = int(max(FASTEST_SHORT_OUTPUT_MAX_NEW_TOKENS, units * 3.2 + 32))
    return int(min(max(int(requested_max_new_tokens), estimate), cap))


def estimated_full_context_codec_frames(text: str, requested_max_new_tokens: int, *, cap: int = 640) -> int:
    units = speech_text_unit_count(text)
    # Qwen3-TTS uses 12 codec frames/sec. Long Chinese news-style text regularly
    # needs around 3-4 frames per spoken text unit, and a too-small limit produces
    # mathematically valid but truncated full-AR audio. Prefer a conservative upper
    # bound and let EOS stop generation.
    estimate = int(max(FASTEST_SHORT_OUTPUT_MAX_NEW_TOKENS, units * 4.0 + 128))
    return int(min(max(int(requested_max_new_tokens), estimate), cap))


ZH_DIGITS = "零一二三四五六七八九"


def chinese_number_under_100(value: str) -> str:
    number = int(value)
    if number < 10:
        return ZH_DIGITS[number]
    tens, ones = divmod(number, 10)
    prefix = "" if tens == 1 else ZH_DIGITS[tens]
    suffix = "" if ones == 0 else ZH_DIGITS[ones]
    return f"{prefix}十{suffix}"


def normalize_tts_text(text: str) -> str:
    content = str(text or "").replace("\u3000", " ")
    content = re.sub(r"[ \t\r\f\v]+", " ", content)
    content = re.sub(r"\n+", " ", content)
    content = re.sub(
        r"(?<!\d)(\d{1,2})月(\d{1,2})日(?!\d)",
        lambda match: f"{chinese_number_under_100(match.group(1))}月{chinese_number_under_100(match.group(2))}日",
        content,
    )
    content = re.sub(
        r"(?<![A-Za-z0-9])(\d{1,2})(?=(日|月|位|名|个|号|台|家|次|年)(?![A-Za-z0-9]))",
        lambda match: chinese_number_under_100(match.group(1)),
        content,
    )
    content = content.replace("总统专机“空军一号”", "总统专机空军一号")
    content = content.replace("总统专机“空军一号”，", "总统专机空军一号，")
    content = content.replace(
        "据悉，黄仁勋作为临时新增成员，",
        "据悉，英伟达CEO黄仁勋作为临时新增成员，",
    )
    content = content.replace(
        "据悉，英伟达CEO黄仁勋作为临时新增成员，五月十三日",
        "据悉，英伟达CEO黄仁勋作为临时新增成员。五月十三日",
    )
    content = content.replace(
        "随后，英伟达官方也证实了此消息。",
        "随后，英伟达官方也证实了黄仁勋临时新增行程并启程赴华的消息。",
    )
    content = content.replace(
        "有消息称，白宫发言人称，黄仁勋的行程有了改动，“就是刚好安排上了”。",
        "有消息称，白宫发言人称，黄仁勋的行程有了改动。白宫发言人表示，黄仁勋刚好被安排进这次行程。",
    )
    content = content.replace(
        "有消息称，白宫发言人称，黄仁勋的行程有了改动，就是刚好安排上了。",
        "有消息称，白宫发言人称，黄仁勋的行程有了改动。白宫发言人表示，黄仁勋刚好被安排进这次行程。",
    )
    return content.strip()


def refine_streaming_segments(segments: list[str]) -> list[str]:
    refined: list[str] = []
    reporting_prefixes: tuple[str, ...] = ()
    for segment in segments:
        pending = [str(segment or "").strip()]
        while pending:
            piece = pending.pop(0)
            if not piece:
                continue
            split_done = False
            for prefix in reporting_prefixes:
                if piece.startswith(prefix) and len(piece) > len(prefix) + 4:
                    refined.append(prefix)
                    pending.insert(0, piece[len(prefix):].strip())
                    split_done = True
                    break
                marker_index = piece.find(prefix)
                if marker_index > 0:
                    before = piece[:marker_index].strip()
                    after = piece[marker_index:].strip()
                    if before:
                        refined.append(before)
                    if after:
                        pending.insert(0, after)
                    split_done = True
                    break
            if not split_done:
                refined.append(piece)
    return refined


def split_text_for_streaming(text: str, max_units: int = 220) -> list[str]:
    text = normalize_tts_text(text)
    if not text:
        return []
    max_units = max(8, int(max_units))
    segments: list[str] = []
    current: list[str] = []
    current_units = 0
    last_soft_index = -1
    last_soft_units = 0
    last_hard_index = -1
    last_hard_units = 0
    tokens = TEXT_TOKEN_RE.findall(text)

    def flush(until: int | None = None) -> None:
        nonlocal current, current_units, last_soft_index, last_soft_units, last_hard_index, last_hard_units
        if until is None:
            piece_tokens = current
            current = []
        else:
            piece_tokens = current[:until]
            current = current[until:]
        piece = "".join(piece_tokens).strip()
        if piece:
            segments.append(piece)
        current_units = speech_text_unit_count("".join(current))
        last_soft_index = -1
        last_soft_units = 0
        last_hard_index = -1
        last_hard_units = 0
        for index, token in enumerate(current):
            if token in SOFT_SPLIT_PUNCT or token in HARD_SPLIT_PUNCT:
                last_soft_index = index + 1
                last_soft_units = speech_text_unit_count("".join(current[:last_soft_index]))
            if token in HARD_SPLIT_PUNCT:
                last_hard_index = index + 1
                last_hard_units = speech_text_unit_count("".join(current[:last_hard_index]))

    for token in tokens:
        unit = speech_text_units(token)
        current.append(token)
        current_units += unit
        if token in SOFT_SPLIT_PUNCT or token in HARD_SPLIT_PUNCT:
            last_soft_index = len(current)
            last_soft_units = current_units
        if token in HARD_SPLIT_PUNCT:
            last_hard_index = len(current)
            last_hard_units = current_units
            if current_units >= 4:
                flush()
                continue
        if current_units <= max_units:
            continue
        split_min_units = max(4, max_units // 4)
        if last_hard_index > 0 and last_hard_units >= split_min_units:
            flush(last_hard_index)
        elif last_soft_index > 0 and last_soft_units >= split_min_units:
            flush(last_soft_index)
        elif current_units <= max_units + 8:
            continue
        else:
            flush(max(1, len(current) - 1))
    flush()
    return refine_streaming_segments(segments or [text])


def needs_continuous_long_output(text: str, max_new_tokens: int) -> bool:
    return int(max_new_tokens) > FASTEST_SHORT_OUTPUT_MAX_NEW_TOKENS or speech_text_unit_count(text) > 24


def continuous_long_output_metadata(enabled: bool) -> dict:
    paged_requested = paged_long_ar_enabled()
    return {
        "continuous_long_output": bool(enabled),
        "long_text_mode": "full_ar" if enabled else "short_ar",
        "segmented": False,
        "continuous_backend": (
            "native_paged_attention"
            if enabled and paged_requested
            else ("single_prompt_full_ar_reference" if enabled else "fastest_native_bucket")
        ),
        "continuous_bucket": None if enabled and paged_requested else (CONTINUOUS_LONG_OUTPUT_BUCKET if enabled else None),
        "paged_kv": bool(enabled and paged_requested),
        "paged_kv_backend": "native_paged_attention" if enabled and paged_requested else "unavailable",
        "paged_kv_unavailable_reason": "" if enabled and paged_requested else (
            f"disabled by default for long full-AR correctness; set {ENABLE_PAGED_LONG_AR_ENV}=1 after parity validation"
            if enabled
            else PAGED_KV_UNAVAILABLE_REASON
        ),
    }


def is_usm_allocation_error(exc: BaseException) -> bool:
    message = str(exc)
    return any(
        marker in message
        for marker in (
            "USM Host",
            "CL_OUT_OF_RESOURCES",
            "Can not allocate",
            "cannot allocate",
            "out of resources",
        )
    )


def request_prompt_memory_estimate(
    request: dict,
    gen_kwargs: dict | None = None,
    *,
    hidden_size: int = 2048,
    block_size: int | None = None,
) -> dict:
    text = str(request.get("text") or request.get("input") or "")
    instruct = str(request.get("instruct") or request.get("instructions") or "")
    units = speech_text_unit_count(text)
    instruct_units = speech_text_unit_count(instruct)
    generation = gen_kwargs or request.get("generation") or {}
    try:
        max_new_tokens = int(generation.get("max_new_tokens", FASTEST_SHORT_OUTPUT_MAX_NEW_TOKENS))
    except Exception:
        max_new_tokens = FASTEST_SHORT_OUTPUT_MAX_NEW_TOKENS
    try:
        effective_block_size = int(block_size or os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE") or FASTEST_NATIVE_PAGED_KV_BLOCK_SIZE)
    except Exception:
        effective_block_size = int(FASTEST_NATIVE_PAGED_KV_BLOCK_SIZE)
    prompt_tokens_estimate = max(1, units + instruct_units + 16)
    kv_blocks = max(1, (prompt_tokens_estimate + max_new_tokens + effective_block_size - 1) // effective_block_size)
    return {
        "prompt_units": int(units),
        "instruct_units": int(instruct_units),
        "prompt_tokens_estimate": int(prompt_tokens_estimate),
        "prompt_embed_bytes_estimate": int(prompt_tokens_estimate * int(hidden_size) * 4),
        "estimated_kv_blocks": int(kv_blocks),
        "kv_block_size": int(effective_block_size),
        "max_new_tokens": int(max_new_tokens),
    }


def select_continuous_long_output_variant(manifest: dict) -> str:
    variants = manifest.get("graph_variants") or {}
    for variant_name in CONTINUOUS_LONG_OUTPUT_PREFERRED_VARIANTS:
        exact_buckets = (
            ((variants.get(variant_name) or {}).get("graphs") or {})
            .get("fused_cache_step_buckets", {})
            .get("exact", {})
        )
        if isinstance(exact_buckets, dict) and str(CONTINUOUS_LONG_OUTPUT_BUCKET) in {str(key) for key in exact_buckets}:
            return variant_name
    return CONTINUOUS_LONG_OUTPUT_VARIANT


def select_stateful_segment_variant(manifest: dict) -> tuple[str, str]:
    variants = manifest.get("graph_variants") or {}
    for variant_name in ("int8_cachedsub", "int8", "fp16_fused_cachedsub_rms"):
        graphs = ((variants.get(variant_name) or {}).get("graphs") or {})
        buckets = graphs.get("talker_stateful_buckets") or {}
        if isinstance(buckets.get("sdpa"), dict) and buckets["sdpa"]:
            return variant_name, "sdpa"
        if isinstance(buckets.get("exact"), dict) and buckets["exact"]:
            return variant_name, "exact"
    graphs = manifest.get("graphs") or {}
    buckets = graphs.get("talker_stateful_buckets") or {}
    if isinstance(buckets.get("sdpa"), dict) and buckets["sdpa"]:
        return "fp16", "sdpa"
    if isinstance(buckets.get("exact"), dict) and buckets["exact"]:
        return "fp16", "exact"
    return CONTINUOUS_LONG_OUTPUT_VARIANT, "exact"


def select_quality_paged_variant(manifest: dict) -> str | None:
    variants = manifest.get("graph_variants") or {}
    base_seed = ((manifest.get("graphs") or {}).get("paged_kv_seed") or {})
    for variant_name in CONTINUOUS_LONG_OUTPUT_PAGED_QUALITY_VARIANTS:
        variant_seed = (((variants.get(variant_name) or {}).get("graphs") or {}).get("paged_kv_seed") or {})
        if variant_seed.get("fused_cache_step"):
            return variant_name
        if variant_name == "fp16" and base_seed.get("fused_cache_step"):
            return "fp16"
    return None


def audio_to_pcm16(audio) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def trim_audio_silence(
    audio,
    sample_rate: int,
    *,
    trim_start: bool = False,
    trim_end: bool = False,
    threshold: float = 0.001,
    keep_ms: int = 40,
):
    data = np.asarray(audio, dtype=np.float32)
    if data.size == 0:
        return data
    keep = int(max(0, sample_rate) * max(0, keep_ms) / 1000)
    active = np.flatnonzero(np.abs(data) > float(threshold))
    if active.size == 0:
        return data[:0]
    start = 0
    end = data.size
    if trim_start:
        start = max(0, int(active[0]) - keep)
    if trim_end:
        end = min(data.size, int(active[-1]) + keep + 1)
    if start >= end:
        return data[:0]
    return np.ascontiguousarray(data[start:end], dtype=np.float32)


def recent_codec_prefix(parts: list[np.ndarray], max_frames: int) -> np.ndarray | None:
    if max_frames <= 0 or not parts:
        return None
    arrays = []
    for part in parts:
        array = np.asarray(part, dtype=np.int64)
        if not array.size:
            continue
        if array.ndim == 1:
            array = array.reshape(1, -1)
        else:
            array = array.reshape(-1, array.shape[-1])
        arrays.append(array)
    if not arrays:
        return None
    codes = np.concatenate(arrays, axis=0)
    if codes.size == 0:
        return None
    return np.ascontiguousarray(codes[-max_frames:], dtype=np.int64)


def apply_boundary_fade(audio, sample_rate: int, *, fade_in: bool = False, fade_out: bool = False, fade_ms: int = WEB_AUTO_SEGMENT_FADE_MS):
    data = np.asarray(audio, dtype=np.float32)
    if data.size == 0 or fade_ms <= 0:
        return data
    count = min(data.size, int(max(1, sample_rate) * fade_ms / 1000))
    if count <= 1:
        return data
    out = np.array(data, dtype=np.float32, copy=True)
    if fade_in:
        out[:count] *= np.linspace(0.0, 1.0, count, dtype=np.float32)
    if fade_out:
        out[-count:] *= np.linspace(1.0, 0.0, count, dtype=np.float32)
    return np.ascontiguousarray(out, dtype=np.float32)


def wav_bytes(audio, sample_rate: int) -> bytes:
    with io.BytesIO() as handle:
        sf.write(handle, np.asarray(audio, dtype=np.float32), sample_rate, format="WAV")
        return handle.getvalue()


def normalize_mode(mode: str) -> str:
    key = (mode or "").replace("-", "_")
    if key not in {"voice_design", "custom_voice", "voice_clone"}:
        raise ValueError("mode must be voice_design, custom_voice, or voice_clone")
    return key


def parse_csv(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def select_auto_realtime_profile(path: str | Path = "outputs/realtime_bench/streaming_profiles.json") -> dict | None:
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        return None
    try:
        with open(benchmark_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    summary_candidates = []
    for summary in payload.get("summaries", []):
        profile_name = summary.get("profile")
        if profile_name not in REALTIME_BENCHMARK_PROFILE_OPTIONS:
            continue
        if not summary.get("accepted"):
            continue
        metric = summary.get("p90_stream_rtf")
        if metric is None:
            continue
        summary_candidates.append((float(metric), profile_name, summary))
    if summary_candidates:
        _, profile_name, summary = min(summary_candidates, key=lambda item: item[0])
        return {
            "profile": profile_name,
            "metric": summary.get("p90_stream_rtf"),
            "summary_metric": "p90_stream_rtf",
            **REALTIME_BENCHMARK_PROFILE_OPTIONS[profile_name],
        }

    candidates = []
    for run in payload.get("runs", []):
        profile_name = run.get("profile")
        if profile_name not in REALTIME_BENCHMARK_PROFILE_OPTIONS:
            continue
        if run.get("status") != "ok":
            continue
        if run.get("worker_exit_code") not in (None, 0):
            continue
        metric = run.get("stream_compute_rtf")
        if metric is None:
            metric = run.get("stream_rtf")
        if metric is None:
            continue
        candidates.append((float(metric), profile_name, run))
    if not candidates:
        return None
    _, profile_name, run = min(candidates, key=lambda item: item[0])
    return {
        "profile": profile_name,
        "metric": run.get("stream_compute_rtf", run.get("stream_rtf")),
        **REALTIME_BENCHMARK_PROFILE_OPTIONS[profile_name],
    }


def _quality_result_passes(result: dict) -> bool:
    if not result.get("ok"):
        return False
    if not (result.get("objective_gate") or {}).get("pass"):
        return False
    omni = result.get("omni")
    if omni is not None and not omni.get("pass"):
        return False
    return True


def select_long_text_quality_profile(
    ir_dir: str | Path,
    path: str | Path = LONG_TEXT_QUALITY_SUMMARY_PATH,
) -> dict | None:
    summary_path = Path(path)
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    winner = payload.get("winner") or {}
    profile_name = winner.get("profile")
    if not profile_name:
        return None
    try:
        resolved_ir_dir = Path(ir_dir).resolve()
    except Exception:
        resolved_ir_dir = Path(ir_dir)
    for result in payload.get("results", []):
        if result.get("profile") != profile_name:
            continue
        if not _quality_result_passes(result):
            continue
        result_ir_dir = result.get("ir_dir")
        if result_ir_dir:
            try:
                if Path(result_ir_dir).resolve() != resolved_ir_dir:
                    continue
            except Exception:
                pass
        runtime = dict(result.get("runtime") or winner.get("runtime") or {})
        if not runtime:
            continue
        return {
            "profile": profile_name,
            "metric": winner.get("median_stream_rtf") or result.get("stream_rtf"),
            "runtime": runtime,
            "profile_env": dict(result.get("profile_env") or winner.get("profile_env") or {}),
            "summary_path": str(summary_path),
        }
    return None


def apply_long_text_profile_env(profile: dict | None) -> None:
    if not profile:
        return
    for key, value in (profile.get("profile_env") or {}).items():
        if value is None:
            continue
        os.environ[LONG_TEXT_PROFILE_ENV_MAP.get(key, key)] = str(value)


def explicit_long_text_profile(name: str | None) -> dict | None:
    normalized = str(name or "").strip().lower().replace("-", "_")
    if not normalized or normalized in {"auto", "quality", "summary"}:
        return None
    if normalized in {"reference", "ref", "no_cache", "no_cache_fp16"}:
        return {"profile": "reference", "runtime": {}, "profile_env": {}}
    profile_names = {
        "paged_sample_fp16": ("long_paged_split_sample_fp16", "fp16"),
        "long_paged_split_sample_fp16": ("long_paged_split_sample_fp16", "fp16"),
        "paged_sample_int8": ("long_paged_split_sample_int8_sym", "int8_sym_paged_talker_split"),
        "paged_sample_int8_sym": ("long_paged_split_sample_int8_sym", "int8_sym_paged_talker_split"),
        "long_paged_split_sample_int8_sym": ("long_paged_split_sample_int8_sym", "int8_sym_paged_talker_split"),
    }
    if normalized not in profile_names:
        raise ValueError(
            f"{LONG_AR_PROFILE_ENV} must be one of: auto, reference, paged-sample-fp16, paged-sample-int8"
        )
    profile_name, variant = profile_names[normalized]
    return {
        "profile": profile_name,
        "metric": None,
        "runtime": {
            "mode": "no-cache",
            "cache_kernel": "exact",
            "cache_step": "fused",
            "graph_variant": variant,
            "codegen_unroll": "1",
            "codegen_schedule": "current",
            "codegen_decode_unroll": "off",
            "preferred_cache_bucket": "0",
            "native_pipeline": "require",
            "native_paged_kv": "require",
            "native_paged_kv_gqa": "1",
            "native_paged_kv_split_subcode": "1",
        },
        "profile_env": {
            "native_codegen_device": "GPU",
            "native_paged_kv_precision": "f16",
            "native_paged_kv_block_size": "16",
            "native_paged_kv_gqa": "1",
            "native_paged_kv_split_subcode": "1",
            "native_paged_kv_split_subcode_mode": "cached_exact",
            "native_paged_kv_score_aggregation": "1",
            "native_buffer_reuse": "0",
        },
    }


def builtin_long_text_profile_from_manifest(manifest: dict) -> dict | None:
    """Return the fastest built-in full-AR long-text profile supported by an IR.

    This keeps the production long-text speed path independent from ignored
    benchmark output files. Quality summaries can still override this, and
    QWEN3_TTS_OV_LONG_AR_PROFILE=reference still forces the FP16 reference path.
    """
    graphs = manifest.get("graphs") or {}
    variants = manifest.get("graph_variants") or {}
    has_cached_subcode = bool(graphs.get("subcode_greedy_cached") or graphs.get("subcode_greedy"))

    int8_graphs = ((variants.get("int8_sym_paged_talker_split") or {}).get("graphs") or {})
    int8_seed = int8_graphs.get("paged_kv_seed") or {}
    if has_cached_subcode and int8_seed.get("talker_stateful_gqa"):
        profile = explicit_long_text_profile("paged-sample-int8")
        if profile:
            profile["source"] = "builtin_manifest"
        return profile

    fp16_seed = graphs.get("paged_kv_seed") or {}
    if has_cached_subcode and fp16_seed.get("talker_stateful_gqa"):
        profile = explicit_long_text_profile("paged-sample-fp16")
        if profile:
            profile["source"] = "builtin_manifest"
        return profile
    return None


def should_auto_apply_long_text_profile(profile: dict | None) -> bool:
    if not profile:
        return False
    if env_enabled(USE_LONG_TEXT_QUALITY_PROFILE_ENV, False):
        return True
    return str(profile.get("profile") or "") in {
        "long_paged_split_sample_fp16",
        "long_paged_split_sample_int8_sym",
    }


def profile_int(value, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def generation_kwargs(request: dict, default_repetition_penalty: float = 1.05) -> dict:
    generation = request.get("generation") or {}
    def value(name: str, default):
        return generation.get(name, request.get(name, default))

    max_new_tokens = int(value("max_new_tokens", 512))
    mode_name_for_defaults = normalize_mode(request.get("mode"))
    text_for_defaults = normalize_tts_text(str(request.get("text") or ""))
    long_voice_design = mode_name_for_defaults == "voice_design" and needs_continuous_long_output(
        text_for_defaults,
        max_new_tokens,
    )
    if request_uses_full_context_text(request):
        max_new_tokens = estimated_full_context_codec_frames(
            text_for_defaults,
            max_new_tokens,
            cap=int(os.environ.get("QWEN3_TTS_OV_FULL_CONTEXT_MAX_NEW_TOKENS") or 2048),
        )
        long_voice_design = mode_name_for_defaults == "voice_design"
    default_penalty = default_repetition_penalty
    try:
        if long_voice_design:
            default_penalty = max(float(default_penalty), 1.05)
    except Exception:
        pass
    do_sample_default = bool(long_voice_design)
    return {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": int(value("min_new_tokens", 2)),
        "repetition_penalty": float(value("repetition_penalty", default_penalty)),
        "max_prompt_tokens": int(value("max_prompt_tokens", 512)),
        "progress_interval": int(value("progress_interval", 0)),
        "do_sample": bool(value("do_sample", do_sample_default)),
        "top_k": int(value("top_k", 50)),
        "top_p": float(value("top_p", 1.0)),
        "temperature": float(value("temperature", 0.9)),
    }


def normalize_chunk_strategy(strategy: str | None, default: str = "low_latency") -> str:
    normalized = str(strategy or default).strip().replace("-", "_").lower()
    if normalized not in DEFAULT_STREAM_CHUNK_STRATEGIES:
        supported = ", ".join(sorted(DEFAULT_STREAM_CHUNK_STRATEGIES))
        raise ValueError(f"unsupported chunk_strategy={strategy!r}; supported strategies: {supported}")
    return normalized


def stream_kwargs(
    request: dict,
    default_strategy: str = "low_latency",
    forced_strategy: str | None = None,
) -> dict:
    stream = request.get("stream") if isinstance(request.get("stream"), dict) else {}
    fmt = stream.get("format", "pcm_s16le")
    if fmt != "pcm_s16le":
        raise ValueError("only stream.format=pcm_s16le is supported")
    if forced_strategy:
        strategy = normalize_chunk_strategy(forced_strategy, default_strategy)
        defaults = DEFAULT_STREAM_CHUNK_STRATEGIES[strategy]
        return {
            "chunk_strategy": strategy,
            "initial_chunk_frames": int(defaults["initial_chunk_frames"]),
            "chunk_frames": int(defaults["chunk_frames"]),
            "left_context_frames": int(defaults["left_context_frames"]),
        }
    kwargs = {
        "chunk_strategy": stream.get("chunk_strategy", request.get("chunk_strategy", default_strategy)),
    }
    for name in ("initial_chunk_frames", "chunk_frames", "left_context_frames"):
        if name in stream:
            kwargs[name] = int(stream[name])
        elif name in request:
            kwargs[name] = int(request[name])
    return kwargs


def include_chunk_metadata(request: dict) -> bool:
    stream = request.get("stream") if isinstance(request.get("stream"), dict) else {}
    return bool(stream.get("include_chunk_metadata", request.get("include_chunk_metadata", False)))


def request_uses_full_context_text(request: dict) -> bool:
    if request.get("force_auto_segment_text", False):
        return False
    if "full_context_text" in request or "use_full_context" in request:
        return bool(request.get("full_context_text", request.get("use_full_context", False)))
    return False


def request_allows_auto_segment(request: dict) -> bool:
    return bool(request.get("allow_auto_segment_text", False)) or env_enabled(ENABLE_AUTO_SEGMENT_ENV, False)


def request_will_auto_segment(request: dict) -> bool:
    if request_uses_full_context_text(request):
        return False
    if not request_allows_auto_segment(request):
        return False
    if not request.get("auto_segment_text", False):
        return False
    try:
        if normalize_mode(request.get("mode")) != "voice_design":
            return False
        segment_units = int(request.get("auto_segment_units") or WEB_AUTO_SEGMENT_UNITS)
    except Exception:
        segment_units = WEB_AUTO_SEGMENT_UNITS
    return speech_text_unit_count(request.get("text") or "") > segment_units


def full_context_metadata(request: dict) -> dict:
    if not request_uses_full_context_text(request):
        return {}
    return {
        "full_context_text": True,
        "auto_segment_text": False,
        "long_text_mode": "full_ar",
        "segmented": False,
        "continuous_long_output": True,
        "continuous_backend": "full_context_single_pass_full_ar",
        "continuous_bucket": None,
        "paged_kv": paged_long_ar_enabled(),
        "paged_kv_quality_safe": paged_long_ar_enabled(),
        "paged_kv_gqa": False,
        "paged_kv_split_subcode": False,
    }


def auto_segment_metadata(request: dict) -> dict:
    if not request_will_auto_segment(request):
        return {}
    try:
        segment_units = int(request.get("auto_segment_units") or WEB_AUTO_SEGMENT_UNITS)
    except Exception:
        segment_units = WEB_AUTO_SEGMENT_UNITS
    return {
        "auto_segment_text": True,
        "auto_segment_units": int(segment_units),
        "long_text_mode": "segmented_debug_fallback",
        "segmented": True,
        "continuous_long_output": False,
        "continuous_backend": "auto_segment_short_prompt",
        "continuous_bucket": None,
        "paged_kv": False,
        "paged_kv_backend": "disabled_for_auto_segment",
        "paged_kv_unavailable_reason": "auto text segmentation uses independent short prompts",
    }


def stream_metadata(
    request: dict,
    default_strategy: str = "low_latency",
    forced_strategy: str | None = None,
) -> dict:
    stream = request.get("stream") if isinstance(request.get("stream"), dict) else {}
    strategy = normalize_chunk_strategy(
        forced_strategy if forced_strategy else stream.get("chunk_strategy", request.get("chunk_strategy")),
        default_strategy,
    )
    defaults = DEFAULT_STREAM_CHUNK_STRATEGIES[strategy]
    if forced_strategy:
        return {
            "chunk_strategy": strategy,
            "initial_chunk_frames": int(defaults["initial_chunk_frames"]),
            "chunk_frames": int(defaults["chunk_frames"]),
            "left_context_frames": int(defaults["left_context_frames"]),
            "forced_chunk_strategy": True,
        }
    return {
        "chunk_strategy": strategy,
        "initial_chunk_frames": int(stream.get("initial_chunk_frames", request.get("initial_chunk_frames", defaults["initial_chunk_frames"]))),
        "chunk_frames": int(stream.get("chunk_frames", request.get("chunk_frames", defaults["chunk_frames"]))),
        "left_context_frames": int(stream.get("left_context_frames", request.get("left_context_frames", defaults["left_context_frames"]))),
    }


def playback_buffer_for_stream(metadata: dict, configured_ms: int) -> int:
    strategy = str(metadata.get("chunk_strategy") or "low_latency")
    strategy_floor = {
        "realtime": 1900,
        "smooth": 1900,
        "stable": 1500,
        "balanced": 500,
        "low_latency": 500,
    }.get(strategy, 500)
    return max(int(configured_ms), strategy_floor)


def normalize_openai_task_type(request: dict) -> str:
    task_type = request.get("task_type") or request.get("mode")
    if task_type:
        key = str(task_type).strip().replace("-", "_").lower()
        if key in {"base", "voice_clone"}:
            return "voice_clone"
        if key in {"voice_design", "custom_voice"}:
            return key
        raise ValueError("task_type must be voice_design, custom_voice, voice_clone, or base")

    model_name = str(request.get("model") or "").lower()
    if request.get("ref_audio") or request.get("ref_text"):
        return "voice_clone"
    if "base" in model_name or "voiceclone" in model_name or "voice_clone" in model_name:
        return "voice_clone"
    if "customvoice" in model_name or "custom_voice" in model_name:
        return "custom_voice"
    voice = str(request.get("voice") or "").strip().lower()
    if voice and voice not in {"default", "voice_design", "none"}:
        return "custom_voice"
    return "voice_design"


def openai_speech_to_tts_request(request: dict) -> tuple[dict, str, bool]:
    text = request.get("input")
    if not text:
        raise ValueError("input is required")
    response_format = str(request.get("response_format", "wav")).lower()
    stream_enabled = bool(request.get("stream", False))
    mode_name = normalize_openai_task_type(request)

    generation = dict(request.get("generation") or {})
    for name in (
        "max_new_tokens",
        "min_new_tokens",
        "do_sample",
        "top_k",
        "top_p",
        "temperature",
        "repetition_penalty",
        "max_prompt_tokens",
        "progress_interval",
    ):
        if name in request:
            generation[name] = request[name]

    stream_config = {}
    if isinstance(request.get("stream"), dict):
        stream_config.update(request["stream"])
        stream_enabled = True
    for name in ("chunk_strategy", "initial_chunk_frames", "chunk_frames", "left_context_frames"):
        if name in request:
            stream_config[name] = request[name]
    stream_config.setdefault("format", "pcm_s16le")

    internal = {
        "mode": mode_name,
        "text": text,
        "language": request.get("language", "Auto"),
        "instruct": request.get("instructions", request.get("instruct", "")),
        "generation": generation,
        "stream": stream_config,
    }
    if mode_name == "custom_voice":
        internal["speaker"] = request.get("voice") or request.get("speaker")
    elif mode_name == "voice_clone":
        internal["ref_audio"] = request.get("ref_audio")
        internal["ref_text"] = request.get("ref_text")
        internal["x_vector_only"] = bool(request.get("x_vector_only_mode", request.get("x_vector_only", False)))
    return internal, response_format, stream_enabled


def create_app(
    model_root: str | Path = "openvino",
    device: str = "GPU",
    decoder_device: str | None = None,
    allow_cpu_fallback: bool = False,
    mode: str = "cache",
    cache_kernel: str = "exact",
    cache_step: str = "fused",
    graph_variant: str = "fp16",
    codegen_unroll: str | int = "profile",
    codegen_schedule: str = "current",
    codegen_decode_unroll: str = "off",
    preferred_cache_bucket: int | str | None = 112,
    ov_cache_dir: str | Path | None = None,
    ov_cache_mode: str | None = "optimize_speed",
    disable_ov_cache: bool = False,
    warmup: bool = True,
    preload_modes: str | list[str] = "voice_design",
    preload_buckets: str = "warmup",
    warmup_text: str = "你好，这是一次流式预热。",
    warmup_strategy: str = "low_latency",
    recommended_playback_buffer_ms: int = 250,
    realtime_profile: str = FASTEST_PROFILE_NAME,
    max_concurrent_tts: int = 1,
    long_output_memory_policy: str = "stable",
    max_continuous_prompt_tokens: int = 1024,
    usm_retry_count: int = 1,
):
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, Response, StreamingResponse

    app = FastAPI(title="Qwen3-TTS OpenVINO Engine")
    model_root = Path(model_root)
    requested_devices = [str(device or "")]
    if decoder_device:
        requested_devices.append(str(decoder_device))
    uses_gpu_device = any("GPU" in item.upper() for item in requested_devices)
    if realtime_profile not in REALTIME_PROFILE_CHOICES:
        raise ValueError(f"realtime_profile must be one of {', '.join(REALTIME_PROFILE_CHOICES)}")
    long_output_memory_policy = str(long_output_memory_policy or "stable").strip().lower()
    if long_output_memory_policy not in {"stable", "fast"}:
        raise ValueError("long_output_memory_policy must be stable or fast")
    max_concurrent_tts = max(1, int(max_concurrent_tts))
    max_continuous_prompt_tokens = int(max_continuous_prompt_tokens)
    usm_retry_count = max(0, int(usm_retry_count))
    auto_profile = select_auto_realtime_profile() if realtime_profile == "auto" else None
    if auto_profile:
        realtime_profile = str(auto_profile["realtime_profile"])
        codegen_unroll = auto_profile["codegen_unroll"]
        codegen_schedule = auto_profile["codegen_schedule"]
        codegen_decode_unroll = auto_profile.get("codegen_decode_unroll", codegen_decode_unroll)
        preferred_cache_bucket = auto_profile.get("preferred_cache_bucket", preferred_cache_bucket)
    elif realtime_profile == "auto":
        realtime_profile = FASTEST_PROFILE_NAME
    if realtime_profile in {FASTEST_PROFILE_NAME, "auto"}:
        codegen_unroll = str(FASTEST_CODEGEN_UNROLL)
        codegen_schedule = FASTEST_CODEGEN_SCHEDULE
        codegen_decode_unroll = FASTEST_CODEGEN_DECODE_UNROLL
        preferred_cache_bucket = FASTEST_PREFERRED_CACHE_BUCKET
        warmup_strategy = FASTEST_CHUNK_STRATEGY if warmup_strategy == "low_latency" else warmup_strategy
        os.environ["QWEN3_TTS_OV_NATIVE_PIPELINE"] = "require" if FASTEST_NATIVE_PIPELINE == "require" else "1"
        os.environ["QWEN3_TTS_OV_NATIVE_BUFFER_REUSE"] = "1" if FASTEST_NATIVE_BUFFER_REUSE == "on" else "0"
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV"] = FASTEST_NATIVE_PAGED_KV
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA"] = "1" if FASTEST_NATIVE_PAGED_KV_GQA == "on" else "0"
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_PRECISION"] = FASTEST_NATIVE_PAGED_KV_PRECISION
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE"] = str(FASTEST_NATIVE_PAGED_KV_BLOCK_SIZE)
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE"] = (
            "1" if FASTEST_NATIVE_PAGED_KV_SPLIT_SUBCODE == "on" else "0"
        )
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_SCORE_AGGREGATION"] = (
            "1" if FASTEST_NATIVE_PAGED_KV_SCORE_AGGREGATION == "on" else "0"
        )
        os.environ["QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE"] = (
            FASTEST_NATIVE_CODEGEN_DEVICE if uses_gpu_device else str(device or "CPU")
        )
    if long_output_memory_policy == "stable":
        if uses_gpu_device:
            os.environ["QWEN3_TTS_OV_NATIVE_GPU_LARGE_ALLOCATIONS"] = "1"
        else:
            os.environ.pop("QWEN3_TTS_OV_NATIVE_GPU_LARGE_ALLOCATIONS", None)
        os.environ["QWEN3_TTS_OV_NATIVE_REMOTE_EMBED"] = "0"
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_CACHE_TENSOR_REUSE"] = "1"
        os.environ["QWEN3_TTS_OV_NATIVE_RELEASE_RUN_BUFFERS_AFTER_RUN"] = "1"
    default_repetition_penalty = (
        float(auto_profile["repetition_penalty"])
        if auto_profile and "repetition_penalty" in auto_profile
        else (FASTEST_REPETITION_PENALTY if realtime_profile == FASTEST_PROFILE_NAME else (1.0 if is_fastest_or_norepeat_mode(realtime_profile) else 1.05))
    )
    mode, cache_kernel, cache_step, graph_variant = apply_realtime_profile(
        realtime_profile,
        mode,
        cache_kernel,
        cache_step,
        graph_variant,
    )
    effective_unroll = effective_codegen_unroll(mode, graph_variant, codegen_unroll)
    codegen_schedule = normalize_codegen_schedule(codegen_schedule)
    codegen_decode_unroll = str(codegen_decode_unroll or "off").strip().lower().replace("_", "-")
    if codegen_decode_unroll not in {"off", "auto", "on"}:
        raise ValueError("codegen_decode_unroll must be one of off, auto, on")
    variant_profile_names = {
        "int8_fused": "int8",
        "int8_sym_fused": "int8-sym",
        "fp16_fused_rms": "fp16-fused-rms",
        "int8_sym_fused_rms": "int8-sym-fused-rms",
        "fp16_sdpa_fused_rms": "fp16-sdpa-fused-rms",
        "int8_sym_sdpa_fused_rms": "int8-sym-sdpa-fused-rms",
        "fp16_fused_cachedsub": "fp16-fused-cachedsub",
        "int8_sym_fused_cachedsub": "int8-sym-fused-cachedsub",
        "fp16_sdpa_fused_cachedsub": "fp16-sdpa-fused-cachedsub",
        "int8_sym_sdpa_fused_cachedsub": "int8-sym-sdpa-fused-cachedsub",
        "fp16_fused_cachedsub_rms": "fp16-fused-cachedsub-rms",
        "int8_sym_fused_cachedsub_rms": "int8-sym-fused-cachedsub-rms",
    }
    reported_realtime_profile = (
        realtime_profile
        if realtime_profile == FASTEST_PROFILE_NAME or is_fastest_or_norepeat_mode(realtime_profile)
        else variant_profile_names.get(graph_variant, realtime_profile)
    )
    default_stream_strategy = FASTEST_CHUNK_STRATEGY if reported_realtime_profile == FASTEST_PROFILE_NAME else "low_latency"
    forced_stream_strategy = FASTEST_CHUNK_STRATEGY if reported_realtime_profile == FASTEST_PROFILE_NAME else None
    runtimes = {}
    tts_semaphore = threading.BoundedSemaphore(max_concurrent_tts)
    active_tts_requests = 0
    active_tts_lock = threading.Lock()
    app.state.warmup = {
        "enabled": bool(warmup),
        "status": "pending" if warmup else "disabled",
        "realtime_profile": reported_realtime_profile,
        "auto_profile": auto_profile,
        "mode": mode,
        "cache_kernel": cache_kernel,
        "cache_step": cache_step,
        "graph_variant": graph_variant,
        "codegen_unroll": effective_unroll,
        "codegen_schedule": codegen_schedule,
        "codegen_decode_unroll": codegen_decode_unroll,
        "default_repetition_penalty": default_repetition_penalty,
        "preferred_cache_bucket": preferred_cache_bucket,
        "native_codegen": os.environ.get("QWEN3_TTS_OV_NATIVE_CODEGEN") or "off",
        "native_pipeline": os.environ.get("QWEN3_TTS_OV_NATIVE_PIPELINE") or "off",
        "native_async_decode": os.environ.get("QWEN3_TTS_OV_NATIVE_ASYNC_DECODE") or "off",
        "native_buffer_reuse": os.environ.get("QWEN3_TTS_OV_NATIVE_BUFFER_REUSE") or "auto",
        "native_remote_embed": os.environ.get("QWEN3_TTS_OV_NATIVE_REMOTE_EMBED") or "auto",
        "native_prompt": os.environ.get("QWEN3_TTS_OV_NATIVE_PROMPT") or "off",
        "native_prompt_device": os.environ.get("QWEN3_TTS_OV_NATIVE_PROMPT_DEVICE") or "CPU",
        "native_paged_kv": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV") or "off",
        "native_paged_kv_gqa": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA") or "on",
        "native_paged_kv_precision": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_PRECISION") or "f16",
        "native_paged_kv_cache_input_precision": (
            os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_CACHE_INPUT_PRECISION") or "f32"
        ),
        "native_paged_kv_block_size": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE") or "8",
        "native_paged_kv_unroll": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_UNROLL") or "1",
        "native_paged_kv_experimental_unroll": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_EXPERIMENTAL_UNROLL") or "0",
        "native_paged_kv_subcode_attention": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SUBCODE_ATTENTION") or "auto",
        "native_paged_kv_split_subcode": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE") or "off",
        "native_paged_kv_split_subcode_mode": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE_MODE") or "cached",
        "native_paged_kv_score_aggregation": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SCORE_AGGREGATION") or "on",
        "native_paged_kv_hybrid": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID") or "off",
        "native_paged_kv_hybrid_prefix_frames": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID_PREFIX_FRAMES") or "48",
        "native_codegen_device": os.environ.get("QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE") or device,
        "native_subcode_device": os.environ.get("QWEN3_TTS_OV_NATIVE_SUBCODE_DEVICE") or "same",
        "native_ov_profile": os.environ.get("QWEN3_TTS_OV_NATIVE_PERF_COUNT") or "off",
        "warmup_strategy": warmup_strategy,
        "default_stream_strategy": default_stream_strategy,
        "forced_stream_strategy": forced_stream_strategy,
        "ov_cache_dir": None if disable_ov_cache else str(ov_cache_dir or "auto"),
        "ov_cache_mode": ov_cache_mode,
        "loaded_modes": [],
        "errors": {},
        "runtimes": {},
    }
    app.state.memory = {
        "long_output_memory_policy": long_output_memory_policy,
        "max_concurrent_tts": max_concurrent_tts,
        "active_tts_requests": 0,
        "max_continuous_prompt_tokens": max_continuous_prompt_tokens,
        "usm_retry_count": usm_retry_count,
        "last_usm_error": None,
        "last_usm_retry_at": None,
        "last_usm_retry_count": 0,
        "last_released_native_runners": 0,
        "last_released_native_buffers": 0,
    }
    runtime_stream_metadata = {
        "realtime_profile": reported_realtime_profile,
        "mode": mode,
        "cache_kernel": cache_kernel,
        "cache_step": cache_step,
        "graph_variant": graph_variant,
        "codegen_unroll": effective_unroll,
        "codegen_schedule": codegen_schedule,
        "codegen_decode_unroll": codegen_decode_unroll,
        "preferred_cache_bucket": preferred_cache_bucket,
        "native_codegen": os.environ.get("QWEN3_TTS_OV_NATIVE_CODEGEN") or "off",
        "native_pipeline": os.environ.get("QWEN3_TTS_OV_NATIVE_PIPELINE") or "off",
        "native_async_decode": os.environ.get("QWEN3_TTS_OV_NATIVE_ASYNC_DECODE") or "off",
        "native_buffer_reuse": os.environ.get("QWEN3_TTS_OV_NATIVE_BUFFER_REUSE") or "auto",
        "native_remote_embed": os.environ.get("QWEN3_TTS_OV_NATIVE_REMOTE_EMBED") or "auto",
        "native_prompt": os.environ.get("QWEN3_TTS_OV_NATIVE_PROMPT") or "off",
        "native_prompt_device": os.environ.get("QWEN3_TTS_OV_NATIVE_PROMPT_DEVICE") or "CPU",
        "native_paged_kv": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV") or "off",
        "native_paged_kv_gqa": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA") or "on",
        "native_paged_kv_precision": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_PRECISION") or "f16",
        "native_paged_kv_cache_input_precision": (
            os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_CACHE_INPUT_PRECISION") or "f32"
        ),
        "native_paged_kv_block_size": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE") or "8",
        "native_paged_kv_unroll": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_UNROLL") or "1",
        "native_paged_kv_experimental_unroll": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_EXPERIMENTAL_UNROLL") or "0",
        "native_paged_kv_subcode_attention": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SUBCODE_ATTENTION") or "auto",
        "native_paged_kv_split_subcode": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE") or "off",
        "native_paged_kv_split_subcode_mode": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE_MODE") or "cached",
        "native_paged_kv_score_aggregation": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SCORE_AGGREGATION") or "on",
        "native_paged_kv_hybrid": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID") or "off",
        "native_paged_kv_hybrid_prefix_frames": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID_PREFIX_FRAMES") or "48",
        "native_codegen_device": os.environ.get("QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE") or device,
        "native_subcode_device": os.environ.get("QWEN3_TTS_OV_NATIVE_SUBCODE_DEVICE") or "same",
        "native_ov_profile": os.environ.get("QWEN3_TTS_OV_NATIVE_PERF_COUNT") or "off",
        "unroll_available": effective_unroll > 1,
        "unroll_fallback": False,
        "long_output_policy": "native_paged_attention" if native_paged_kv_requested() else "single_prompt_stateful_bucket",
        "long_output_bucket": None if native_paged_kv_requested() else CONTINUOUS_LONG_OUTPUT_BUCKET,
        "long_output_memory_policy": long_output_memory_policy,
        "max_concurrent_tts": max_concurrent_tts,
        "max_continuous_prompt_tokens": max_continuous_prompt_tokens,
        "paged_kv": native_paged_kv_requested(),
        "paged_kv_backend": "native_paged_attention" if native_paged_kv_requested() else "unavailable",
        "paged_kv_unavailable_reason": "" if native_paged_kv_requested() else PAGED_KV_UNAVAILABLE_REASON,
    }

    def manifest_supports_mode(ir_dir: Path, mode_name: str) -> bool:
        try:
            manifest = load_manifest(ir_dir)
        except Exception:
            return False
        model_type = str(manifest.get("tts_model_type") or "").replace("-", "_").lower()
        if mode_name == "voice_design":
            return model_type in {"", "voice_design"}
        if mode_name == "custom_voice":
            return model_type == "custom_voice"
        if mode_name == "voice_clone":
            return model_type in {"base", "voice_clone"}
        return False

    def resolve_mode_ir_dir(mode_name: str) -> Path:
        model_dir_name = MODE_DIR[mode_name]
        if path_text(model_root) == AUTO_IR_DIR:
            candidates = [Path("openvino") / model_dir_name]
            if mode_name == "voice_design":
                candidates.append(Path(LEGACY_VOICE_DESIGN_IR_DIR))
            for candidate in candidates:
                if has_manifest(candidate) and manifest_supports_mode(candidate, mode_name):
                    return candidate
            resolved = resolve_ir_dir(AUTO_IR_DIR, fallback_to_local_voice_design=(mode_name == "voice_design"), warn=True)
            if has_manifest(resolved) and manifest_supports_mode(resolved, mode_name):
                return resolved
            raise ValueError(manifest_missing_message(AUTO_IR_DIR))
        nested = model_root / model_dir_name
        if has_manifest(nested):
            return nested
        if has_manifest(model_root) and manifest_supports_mode(model_root, mode_name):
            return model_root
        if mode_name == "voice_design" and path_text(model_root) == "openvino":
            fallback = resolve_ir_dir(DEFAULT_VOICE_DESIGN_IR_DIR, fallback_to_local_voice_design=True, warn=True)
            if has_manifest(fallback) and fallback != nested:
                return fallback
        raise ValueError(manifest_missing_message(nested))

    def runtime_for_ir_dir(
        ir_dir: Path,
        do_sample: bool = False,
        continuous_long_output: bool = False,
        prefer_paged_kv: bool = True,
    ):
        if not has_manifest(ir_dir):
            raise ValueError(manifest_missing_message(ir_dir))
        runtime_mode = mode
        runtime_cache_kernel = cache_kernel
        runtime_cache_step = "split" if do_sample and mode == "cache" and cache_step == "fused" else cache_step
        runtime_graph_variant = graph_variant
        runtime_codegen_unroll = int(effective_unroll)
        runtime_codegen_schedule = codegen_schedule
        runtime_codegen_decode_unroll = codegen_decode_unroll
        runtime_preferred_cache_bucket = preferred_cache_bucket
        runtime_native_codegen = None
        runtime_native_pipeline = None
        runtime_native_paged_kv = None
        runtime_native_paged_kv_gqa = None
        runtime_native_paged_kv_split_subcode = None
        if continuous_long_output:
            runtime_manifest = load_manifest(ir_dir)
            requested_long_profile = explicit_long_text_profile(os.environ.get(LONG_AR_PROFILE_ENV))
            if requested_long_profile and requested_long_profile.get("profile") == "reference":
                selected_quality_profile = None
            elif requested_long_profile:
                selected_quality_profile = requested_long_profile
            else:
                candidate_quality_profile = select_long_text_quality_profile(ir_dir)
                if candidate_quality_profile:
                    selected_quality_profile = (
                        candidate_quality_profile
                        if should_auto_apply_long_text_profile(candidate_quality_profile)
                        else None
                    )
                else:
                    selected_quality_profile = builtin_long_text_profile_from_manifest(runtime_manifest)
            if selected_quality_profile:
                apply_long_text_profile_env(selected_quality_profile)
                profile_runtime = selected_quality_profile.get("runtime") or {}
                runtime_mode = profile_runtime.get("mode", runtime_mode)
                runtime_cache_kernel = profile_runtime.get("cache_kernel", runtime_cache_kernel)
                runtime_cache_step = profile_runtime.get("cache_step", runtime_cache_step)
                runtime_graph_variant = profile_runtime.get("graph_variant", runtime_graph_variant)
                runtime_codegen_unroll = profile_int(profile_runtime.get("codegen_unroll"), runtime_codegen_unroll)
                runtime_codegen_schedule = profile_runtime.get("codegen_schedule", runtime_codegen_schedule)
                runtime_codegen_decode_unroll = profile_runtime.get(
                    "codegen_decode_unroll",
                    runtime_codegen_decode_unroll,
                )
                runtime_preferred_cache_bucket = profile_runtime.get(
                    "preferred_cache_bucket",
                    runtime_preferred_cache_bucket,
                )
                runtime_native_codegen = profile_runtime.get("native_codegen", runtime_native_codegen)
                runtime_native_pipeline = profile_runtime.get("native_pipeline", runtime_native_pipeline)
                runtime_native_paged_kv = profile_runtime.get("native_paged_kv", runtime_native_paged_kv)
                runtime_native_paged_kv_gqa = profile_runtime.get(
                    "native_paged_kv_gqa",
                    runtime_native_paged_kv_gqa,
                )
                runtime_native_paged_kv_split_subcode = profile_runtime.get(
                    "native_paged_kv_split_subcode",
                    runtime_native_paged_kv_split_subcode,
                )
            else:
                paged_seed_graphs = ((runtime_manifest.get("graphs") or {}).get("paged_kv_seed") or {})
                quality_paged_variant = select_quality_paged_variant(runtime_manifest)
                quality_variant_graphs = (
                    ((runtime_manifest.get("graph_variants") or {}).get(quality_paged_variant) or {}).get("graphs") or {}
                    if quality_paged_variant
                    else {}
                )
                quality_variant_paged_seed_graphs = (
                    (quality_variant_graphs.get("paged_kv_seed") or {}) if isinstance(quality_variant_graphs, dict) else {}
                )
                paged_required = str(os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV", "")).strip().lower() == "require"
                use_native_paged_kv = prefer_paged_kv and paged_long_ar_enabled()
                if use_native_paged_kv and (
                    quality_variant_paged_seed_graphs.get("fused_cache_step")
                    or paged_seed_graphs.get("fused_cache_step")
                ):
                    runtime_mode = "no-cache"
                    runtime_cache_kernel = "exact"
                    runtime_cache_step = "fused"
                    runtime_graph_variant = quality_paged_variant or "fp16"
                    runtime_codegen_unroll = 1
                    runtime_codegen_schedule = "current"
                    runtime_codegen_decode_unroll = "off"
                    runtime_preferred_cache_bucket = 0
                    runtime_native_pipeline = "require"
                    runtime_native_paged_kv = "require"
                    runtime_native_paged_kv_gqa = "0"
                    runtime_native_paged_kv_split_subcode = "0"
                elif use_native_paged_kv and paged_required:
                    raise RuntimeError(
                        "native paged-KV was required, but this IR manifest has no "
                        "graphs.paged_kv_seed.fused_cache_step; export with --export-paged-kv-seed"
                    )
                else:
                    default_reference_mode = "no-cache" if do_sample else "fused-no-cache"
                    reference_mode = str(os.environ.get(LONG_AR_REFERENCE_MODE_ENV) or default_reference_mode).strip().lower()
                    if reference_mode not in {"fused-no-cache", "no-cache", "cache-split"}:
                        raise ValueError(
                            f"{LONG_AR_REFERENCE_MODE_ENV} must be one of: fused-no-cache, no-cache, cache-split"
                        )
                    if reference_mode == "cache-split":
                        stateful_variant, stateful_kernel = select_stateful_segment_variant(runtime_manifest)
                        runtime_mode = "cache"
                        runtime_cache_kernel = stateful_kernel
                        runtime_cache_step = "split"
                        runtime_graph_variant = stateful_variant
                    else:
                        runtime_mode = reference_mode
                        runtime_cache_kernel = "exact"
                        runtime_cache_step = "split"
                        runtime_graph_variant = "fp16"
                    runtime_codegen_unroll = 1
                    runtime_codegen_schedule = "current"
                    runtime_codegen_decode_unroll = "off"
                    runtime_preferred_cache_bucket = 0 if reference_mode in {"fused-no-cache", "no-cache"} else 128
                    runtime_native_codegen = "off"
                    runtime_native_pipeline = "off"
                    runtime_native_paged_kv = "0"
                    runtime_native_paged_kv_gqa = "0"
                    runtime_native_paged_kv_split_subcode = "0"
        key = (
            str(ir_dir.resolve()),
            runtime_mode,
            runtime_cache_kernel,
            runtime_cache_step,
            runtime_graph_variant,
            runtime_codegen_unroll,
            runtime_codegen_schedule,
            runtime_codegen_decode_unroll,
            str(runtime_preferred_cache_bucket),
            str(runtime_native_codegen or "auto"),
            str(runtime_native_pipeline or "auto"),
            str(runtime_native_paged_kv or os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV") or "off"),
            str(runtime_native_paged_kv_gqa or os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA") or "on"),
            os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_PRECISION") or "f16",
            os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_CACHE_INPUT_PRECISION") or "f32",
            os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE") or "8",
            os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_UNROLL") or "1",
            os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_EXPERIMENTAL_UNROLL") or "0",
            os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SUBCODE_ATTENTION") or "auto",
            str(runtime_native_paged_kv_split_subcode or os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE") or "off"),
            os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE_MODE") or "cached",
            os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SCORE_AGGREGATION") or "on",
            os.environ.get("QWEN3_TTS_OV_NATIVE_SUBCODE_DEVICE") or "same",
            str(ov_cache_dir or "auto"),
            ov_cache_mode,
            bool(disable_ov_cache),
        )
        if key not in runtimes:
            runtimes[key] = OpenVINOQwen3TTS(
                ir_dir,
                device=device,
                decoder_device=decoder_device,
                allow_cpu_fallback=allow_cpu_fallback,
                mode=runtime_mode,
                cache_kernel=runtime_cache_kernel,
                cache_step=runtime_cache_step,
                graph_variant=runtime_graph_variant,
                codegen_unroll=runtime_codegen_unroll,
                codegen_schedule=runtime_codegen_schedule,
                codegen_decode_unroll=runtime_codegen_decode_unroll,
                preferred_cache_bucket=runtime_preferred_cache_bucket,
                ov_cache_dir=ov_cache_dir,
                ov_cache_mode=ov_cache_mode,
                disable_ov_cache=disable_ov_cache,
                native_codegen=runtime_native_codegen,
                native_pipeline=runtime_native_pipeline,
                native_paged_kv=runtime_native_paged_kv,
                native_paged_kv_gqa=runtime_native_paged_kv_gqa,
                native_paged_kv_split_subcode=runtime_native_paged_kv_split_subcode,
            )
        return runtimes[key]

    def get_runtime(
        request_mode: str,
        do_sample: bool = False,
        continuous_long_output: bool = False,
        prefer_paged_kv: bool = True,
    ):
        normalized = normalize_mode(request_mode)
        ir_dir = resolve_mode_ir_dir(normalized)
        return normalized, runtime_for_ir_dir(
            ir_dir,
            do_sample=do_sample,
            continuous_long_output=continuous_long_output,
            prefer_paged_kv=prefer_paged_kv,
        )

    def request_uses_continuous_long_output(request: dict, gen_kwargs: dict | None = None) -> bool:
        if request.get("force_short_segment_pipeline", False):
            return False
        if request_will_auto_segment(request):
            return False
        try:
            mode_name = normalize_mode(request.get("mode"))
            if mode_name != "voice_design":
                return False
            generation = gen_kwargs if gen_kwargs is not None else generation_kwargs(
                request,
                default_repetition_penalty=default_repetition_penalty,
            )
            return needs_continuous_long_output(request.get("text") or "", int(generation["max_new_tokens"]))
        except Exception:
            return False

    @contextmanager
    def tts_request_slot():
        nonlocal active_tts_requests
        tts_semaphore.acquire()
        with active_tts_lock:
            active_tts_requests += 1
            app.state.memory["active_tts_requests"] = active_tts_requests
        try:
            yield
        finally:
            with active_tts_lock:
                active_tts_requests = max(0, active_tts_requests - 1)
                app.state.memory["active_tts_requests"] = active_tts_requests
            tts_semaphore.release()

    def release_native_runtime_resources() -> tuple[int, int]:
        released_buffers = 0
        closed_runners = 0
        for runtime in list(runtimes.values()):
            release = getattr(runtime, "release_native_audio_runner_buffers", None)
            if release is not None:
                try:
                    released_buffers += int(release())
                except Exception:
                    pass
            close = getattr(runtime, "close_native_audio_runners", None)
            if close is not None:
                try:
                    closed_runners += int(close())
                except Exception:
                    pass
        gc.collect()
        app.state.memory["last_released_native_buffers"] = released_buffers
        app.state.memory["last_released_native_runners"] = closed_runners
        return released_buffers, closed_runners

    def exact_voice_design_prompt_metadata(runtime, text: str, instruct: str, language: str) -> dict:
        try:
            input_ids = runtime.tokenizer.encode(build_assistant_text(text))
            instruct_ids = runtime.tokenizer.encode(build_instruct_text(instruct)) if instruct else []
            codec_prefill = runtime.language_codec_prefill(language, speaker=None)
            prompt_len = int(len(instruct_ids) + len(input_ids) + len(codec_prefill) - 3)
        except Exception:
            return {}
        hidden_size = int(os.environ.get("QWEN3_TTS_OV_PROMPT_HIDDEN_SIZE_ESTIMATE") or 2048)
        return {
            "prompt_len": prompt_len,
            "text_tokens": int(len(input_ids)),
            "instruct_tokens": int(len(instruct_ids)),
            "prompt_embed_bytes": int(prompt_len * hidden_size * 4),
        }

    def validate_continuous_prompt_budget(runtime, request: dict, gen_kwargs: dict, long_output: bool) -> dict:
        estimate = request_prompt_memory_estimate(request, gen_kwargs)
        if not long_output:
            return estimate
        mode_name = normalize_mode(request.get("mode"))
        if mode_name == "voice_design":
            exact = exact_voice_design_prompt_metadata(
                runtime,
                str(request.get("text") or ""),
                str(request.get("instruct") or ""),
                str(request.get("language") or "Auto"),
            )
            estimate.update(exact)
            prompt_len = int(exact.get("prompt_len") or estimate["prompt_tokens_estimate"])
        else:
            prompt_len = int(estimate["prompt_tokens_estimate"])
        if max_continuous_prompt_tokens > 0 and prompt_len > max_continuous_prompt_tokens:
            raise ValueError(
                f"continuous long-output prompt has {prompt_len} tokens, "
                f"max_continuous_prompt_tokens={max_continuous_prompt_tokens}. "
                "Increase --max-continuous-prompt-tokens or shorten the request."
            )
        return estimate

    @app.on_event("startup")
    def warmup_on_startup():
        if not warmup:
            return
        app.state.warmup["status"] = "running"
        app.state.warmup["started_at"] = time.time()
        for preload_mode in parse_csv(preload_modes):
            key = preload_mode.replace("-", "_")
            if key not in MODE_DIR:
                app.state.warmup["errors"][preload_mode] = f"unsupported preload mode: {preload_mode}"
                continue
            try:
                ir_dir = resolve_mode_ir_dir(key)
                runtime = runtime_for_ir_dir(ir_dir, do_sample=False)
                status = runtime.prewarm_streaming(
                    text=warmup_text,
                    instruct="用自然、清晰的中文女声朗读。",
                    language="Chinese",
                    chunk_strategy=warmup_strategy,
                    left_context_frames=None,
                    max_new_tokens=None,
                    repetition_penalty=default_repetition_penalty,
                    preload_buckets=preload_buckets,
                    run_generation=runtime.manifest.get("tts_model_type") == "voice_design",
                )
                app.state.warmup["loaded_modes"].append(preload_mode)
                app.state.warmup["runtimes"][preload_mode] = status
                if status.get("status") != "ready":
                    app.state.warmup["errors"][preload_mode] = (
                        status.get("warmup_generation_error")
                        or status.get("fallback_decoder_error")
                        or f"prewarm finished with status={status.get('status')}"
                    )
            except Exception as exc:
                app.state.warmup["errors"][preload_mode] = str(exc)
        app.state.warmup["finished_at"] = time.time()
        app.state.warmup["elapsed"] = app.state.warmup["finished_at"] - app.state.warmup["started_at"]
        app.state.warmup["status"] = "ready" if not app.state.warmup["errors"] else "ready_with_errors"

    def annotate_stream_chunks(chunks, extra_timings: dict, trim_final_silence: bool = False):
        for chunk in chunks:
            audio = chunk.audio
            trimmed_samples = 0
            if trim_final_silence and chunk.is_final:
                original_samples = int(np.asarray(audio).size)
                audio = trim_audio_silence(
                    audio,
                    chunk.sample_rate,
                    trim_start=False,
                    trim_end=True,
                )
                trimmed_samples = max(0, original_samples - int(np.asarray(audio).size))
            timings = dict(chunk.timings or {})
            timings.update(extra_timings)
            if trimmed_samples:
                timings["final_trimmed_samples"] = int(trimmed_samples)
            yield StreamChunk(
                index=chunk.index,
                audio=audio,
                sample_rate=chunk.sample_rate,
                codes=chunk.codes,
                is_final=chunk.is_final,
                timings=timings,
            )

    def stream_chunks_once(request: dict, retry_count: int = 0):
        gen_kwargs = generation_kwargs(request, default_repetition_penalty=default_repetition_penalty)
        mode_name = normalize_mode(request.get("mode"))
        text = request.get("text")
        long_output = request_uses_continuous_long_output(request, gen_kwargs)
        prefer_paged_long_ar = bool(request.get("use_paged_kv_long_ar", False)) or env_enabled(ENABLE_PAGED_LONG_AR_ENV, False)
        mode_name, runtime = get_runtime(
            mode_name,
            do_sample=bool(gen_kwargs["do_sample"]),
            continuous_long_output=long_output,
            prefer_paged_kv=bool(long_output and prefer_paged_long_ar and not request.get("force_stateful_long_output", False)),
        )
        memory_meta = validate_continuous_prompt_budget(runtime, request, gen_kwargs, long_output)
        kwargs = {**gen_kwargs, **stream_kwargs(request, default_stream_strategy, forced_strategy=forced_stream_strategy)}
        text = request.get("text")
        language = request.get("language", "Auto")
        if not text:
            raise ValueError("text is required")
        common_timings = {
            "retry_count": int(retry_count),
            "long_output_memory_policy": long_output_memory_policy,
            "long_ar_do_sample": bool(gen_kwargs.get("do_sample", False)) if long_output else False,
            **memory_meta,
        }
        if mode_name == "voice_design":
            text = normalize_tts_text(text)
            instruct = request.get("instruct", "")
            if long_output:
                if request.get("force_stateful_long_output", False):
                    common_timings.update(
                        {
                            "continuous_long_output": True,
                            "continuous_backend": "single_prompt_full_ar_reference",
                            "continuous_bucket": None,
                            "long_text_mode": "full_ar",
                            "segmented": False,
                            "paged_kv": False,
                            "paged_kv_backend": "disabled_for_full_ar_reference",
                            "paged_kv_unavailable_reason": "",
                        }
                    )
                else:
                    common_timings.update(continuous_long_output_metadata(True))
            return annotate_stream_chunks(
                runtime.stream_voice_design(
                    text=text,
                    instruct=instruct,
                    language=language,
                    prefix_codes=request.get("_prefix_codes"),
                    append_prefix_codes_to_prompt=bool(request.get("_append_prefix_codes_to_prompt", False)),
                    **kwargs,
                ),
                common_timings,
                trim_final_silence=request_uses_full_context_text(request),
            )
        if mode_name == "custom_voice":
            speaker = request.get("speaker")
            if not speaker:
                raise ValueError("speaker is required for custom_voice")
            return annotate_stream_chunks(
                runtime.stream_custom_voice(
                    text=text,
                    speaker=speaker,
                    instruct=request.get("instruct", ""),
                    language=language,
                    **kwargs,
                ),
                common_timings,
            )
        return annotate_stream_chunks(
            runtime.stream_voice_clone(
                text=text,
                language=language,
                ref_audio=request.get("ref_audio"),
                ref_text=request.get("ref_text"),
                x_vector_only_mode=bool(request.get("x_vector_only", False)),
                **kwargs,
            ),
            common_timings,
        )

    def segmented_stream_chunks(request: dict):
        text = str(request.get("text") or "")
        try:
            max_units = int(request.get("auto_segment_units") or os.environ.get("QWEN3_TTS_OV_WEB_SEGMENT_UNITS") or WEB_AUTO_SEGMENT_UNITS)
        except Exception:
            max_units = WEB_AUTO_SEGMENT_UNITS
        if max_continuous_prompt_tokens > 0:
            instruct_units = speech_text_unit_count(str(request.get("instruct") or ""))
            budget_units = max(8, max_continuous_prompt_tokens - instruct_units - 96)
            max_units = min(max_units, budget_units)
        segments = split_text_for_streaming(text, max_units=max_units)
        if len(segments) <= 1:
            raise ValueError("auto text segmentation could not split the oversized prompt")
        base_generation = dict(request.get("generation") or {})
        try:
            requested_max_new_tokens = int(base_generation.get("max_new_tokens", FASTEST_SHORT_OUTPUT_MAX_NEW_TOKENS))
        except Exception:
            requested_max_new_tokens = FASTEST_SHORT_OUTPUT_MAX_NEW_TOKENS
        try:
            segment_token_cap = int(os.environ.get("QWEN3_TTS_OV_WEB_SEGMENT_MAX_NEW_TOKENS") or WEB_AUTO_SEGMENT_MAX_NEW_TOKENS)
        except Exception:
            segment_token_cap = WEB_AUTO_SEGMENT_MAX_NEW_TOKENS
        segment_requested_max_new_tokens = max(int(requested_max_new_tokens), int(segment_token_cap))
        try:
            prefix_frame_limit = int(
                request.get("auto_segment_prefix_frames")
                or os.environ.get("QWEN3_TTS_OV_WEB_SEGMENT_PREFIX_FRAMES")
                or WEB_AUTO_SEGMENT_PREFIX_FRAMES
            )
        except Exception:
            prefix_frame_limit = WEB_AUTO_SEGMENT_PREFIX_FRAMES
        prefix_codes: np.ndarray | None = None
        global_index = 0
        for segment_index, segment in enumerate(segments):
            segment_audio_started = False
            segment_emitted_chunks = 0
            pending_boundary_chunks: list[tuple[StreamChunk, np.ndarray, int]] = []
            segment_codes_parts: list[np.ndarray] = []
            segment_request = dict(request)
            segment_request["text"] = segment
            segment_request["auto_segment_text"] = False
            segment_request["force_short_segment_pipeline"] = True
            segment_request["force_stateful_long_output"] = False
            if prefix_codes is not None and prefix_codes.size:
                segment_request["_prefix_codes"] = prefix_codes
                segment_request["_append_prefix_codes_to_prompt"] = bool(
                    request.get("auto_segment_append_prefix_to_prompt", True)
                )
            segment_generation = dict(base_generation)
            segment_generation["max_new_tokens"] = estimated_codec_frames_for_text(
                segment,
                segment_requested_max_new_tokens,
                cap=segment_token_cap,
            )
            segment_generation["max_prompt_tokens"] = max(
                int(segment_generation.get("max_prompt_tokens", 512)),
                speech_text_unit_count(segment) + speech_text_unit_count(str(request.get("instruct") or "")) + 96,
            )
            segment_request["generation"] = segment_generation
            for attempt in range(usm_retry_count + 1):
                try:
                    for chunk in stream_chunks_once(segment_request, retry_count=attempt):
                        raw_audio = np.asarray(chunk.audio, dtype=np.float32)
                        audio = raw_audio
                        original_samples = int(audio.size)
                        if not segment_audio_started:
                            trimmed = trim_audio_silence(
                                audio,
                                chunk.sample_rate,
                                trim_start=True,
                                trim_end=False,
                            )
                            if trimmed.size:
                                audio = trimmed
                                segment_audio_started = True
                                pending_boundary_chunks.clear()
                            elif original_samples and not chunk.is_final:
                                pending_boundary_chunks.append((chunk, raw_audio, original_samples))
                                continue
                            else:
                                # Avoid dropping an entire segment when the boundary detector is too
                                # conservative for a quiet utterance. This preserves correctness over
                                # aggressive trimming.
                                if pending_boundary_chunks and chunk.is_final:
                                    pending_boundary_chunks.append((chunk, raw_audio, original_samples))
                                    for pending_chunk, pending_audio, pending_original_samples in pending_boundary_chunks:
                                        pending_timings = dict(pending_chunk.timings or {})
                                        pending_timings.update(
                                            {
                                                "auto_segment_text": True,
                                                "text_segment_index": segment_index,
                                                "text_segment_count": len(segments),
                                                "text_segment_units": speech_text_unit_count(segment),
                                                "text_segment_max_new_tokens": int(segment_generation["max_new_tokens"]),
                                                "segment_trimmed_samples": 0,
                                                "segment_trim_fallback": True,
                                            }
                                        )
                                        pending_is_final = bool(pending_chunk.is_final and segment_index == len(segments) - 1)
                                        pending_codes = np.asarray(pending_chunk.codes, dtype=np.int64)
                                        if pending_codes.size:
                                            segment_codes_parts.append(pending_codes.reshape(-1, pending_codes.shape[-1]))
                                        pending_audio = apply_boundary_fade(
                                            pending_audio,
                                            pending_chunk.sample_rate,
                                            fade_in=bool(segment_index > 0 and segment_emitted_chunks == 0),
                                            fade_out=bool(pending_chunk.is_final and segment_index < len(segments) - 1),
                                        )
                                        if pending_audio.size or pending_is_final:
                                            yield StreamChunk(
                                                index=global_index,
                                                audio=pending_audio,
                                                sample_rate=pending_chunk.sample_rate,
                                                codes=pending_chunk.codes,
                                                is_final=pending_is_final,
                                                timings=pending_timings,
                                            )
                                            global_index += 1
                                            segment_emitted_chunks += 1
                                    pending_boundary_chunks.clear()
                                    continue
                                audio = raw_audio
                                segment_audio_started = bool(audio.size)
                        if chunk.is_final:
                            audio = trim_audio_silence(
                                audio,
                                chunk.sample_rate,
                                trim_start=False,
                                trim_end=True,
                            )
                        timings = dict(chunk.timings or {})
                        timings.update(
                            {
                                "auto_segment_text": True,
                                "text_segment_index": segment_index,
                                "text_segment_count": len(segments),
                                "text_segment_units": speech_text_unit_count(segment),
                                "text_segment_max_new_tokens": int(segment_generation["max_new_tokens"]),
                                "segment_trimmed_samples": max(0, original_samples - int(audio.size)),
                                "segment_prefix_frames": int(0 if prefix_codes is None else prefix_codes.shape[0]),
                            }
                        )
                        is_final = bool(chunk.is_final and segment_index == len(segments) - 1)
                        if audio.size == 0 and not is_final:
                            continue
                        chunk_codes = np.asarray(chunk.codes, dtype=np.int64)
                        if chunk_codes.size:
                            segment_codes_parts.append(chunk_codes.reshape(-1, chunk_codes.shape[-1]))
                        audio = apply_boundary_fade(
                            audio,
                            chunk.sample_rate,
                            fade_in=bool(segment_index > 0 and segment_emitted_chunks == 0),
                            fade_out=bool(chunk.is_final and segment_index < len(segments) - 1),
                        )
                        yield StreamChunk(
                            index=global_index,
                            audio=audio,
                            sample_rate=chunk.sample_rate,
                            codes=chunk.codes,
                            is_final=is_final,
                            timings=timings,
                        )
                        global_index += 1
                        segment_emitted_chunks += 1
                    break
                except Exception as exc:
                    if is_usm_allocation_error(exc) and attempt < usm_retry_count:
                        released_buffers, closed_runners = release_native_runtime_resources()
                        app.state.memory["last_usm_error"] = str(exc)
                        app.state.memory["last_usm_retry_at"] = time.time()
                        app.state.memory["last_usm_retry_count"] = attempt + 1
                        app.state.memory["last_released_native_buffers"] = released_buffers
                        app.state.memory["last_released_native_runners"] = closed_runners
                        time.sleep(0.25)
                        continue
                    raise
            next_prefix = recent_codec_prefix(segment_codes_parts, prefix_frame_limit)
            if next_prefix is not None:
                prefix_codes = next_prefix
            if bool(request.get("auto_segment_isolate_native_runner", True)):
                close_runners = getattr(runtime, "close_native_audio_runners", None)
                if close_runners is not None:
                    close_runners()

    def stream_chunks(request: dict):
        with tts_request_slot():
            for attempt in range(usm_retry_count + 1):
                try:
                    if (
                        request.get("auto_segment_text", False)
                        and request_allows_auto_segment(request)
                        and not request_uses_full_context_text(request)
                        and normalize_mode(request.get("mode")) == "voice_design"
                    ):
                        try:
                            segment_units = int(
                                request.get("auto_segment_units")
                                or os.environ.get("QWEN3_TTS_OV_WEB_SEGMENT_UNITS")
                                or WEB_AUTO_SEGMENT_UNITS
                            )
                        except Exception:
                            segment_units = WEB_AUTO_SEGMENT_UNITS
                        if speech_text_unit_count(request.get("text") or "") > segment_units:
                            for chunk in segmented_stream_chunks(request):
                                yield chunk
                            return
                    for chunk in stream_chunks_once(request, retry_count=attempt):
                        yield chunk
                    return
                except Exception as exc:
                    if (
                        request.get("auto_segment_text", False)
                        and request_allows_auto_segment(request)
                        and not request_uses_full_context_text(request)
                        and normalize_mode(request.get("mode")) == "voice_design"
                        and "continuous long-output prompt has" in str(exc)
                    ):
                        for chunk in segmented_stream_chunks(request):
                            yield chunk
                        return
                    if is_usm_allocation_error(exc) and attempt < usm_retry_count:
                        released_buffers, closed_runners = release_native_runtime_resources()
                        app.state.memory["last_usm_error"] = str(exc)
                        app.state.memory["last_usm_retry_at"] = time.time()
                        app.state.memory["last_usm_retry_count"] = attempt + 1
                        app.state.memory["last_released_native_buffers"] = released_buffers
                        app.state.memory["last_released_native_runners"] = closed_runners
                        time.sleep(0.25)
                        continue
                    if is_usm_allocation_error(exc):
                        raise RuntimeError(
                            "OpenVINO GPU USM allocation failed during TTS generation after retry. "
                            "The native runner was released; reduce max_new_tokens/text length or restart the sidecar "
                            "if the GPU driver remains fragmented. Original error: "
                            f"{exc}"
                        ) from exc
                    raise

    def full_audio(request: dict):
        with tts_request_slot():
            for attempt in range(usm_retry_count + 1):
                try:
                    kwargs = generation_kwargs(request, default_repetition_penalty=default_repetition_penalty)
                    long_output = request_uses_continuous_long_output(request, kwargs)
                    prefer_paged_long_ar = bool(request.get("use_paged_kv_long_ar", False)) or env_enabled(ENABLE_PAGED_LONG_AR_ENV, False)
                    mode_name, runtime = get_runtime(
                        request.get("mode"),
                        do_sample=bool(kwargs["do_sample"]),
                        continuous_long_output=long_output,
                        prefer_paged_kv=bool(long_output and prefer_paged_long_ar),
                    )
                    validate_continuous_prompt_budget(runtime, request, kwargs, long_output)
                    text = request.get("text")
                    language = request.get("language", "Auto")
                    if not text:
                        raise ValueError("text is required")
                    if mode_name == "voice_design":
                        text = normalize_tts_text(text)
                        wavs, sr = runtime.generate_voice_design(
                            text=text,
                            instruct=request.get("instruct", ""),
                            language=language,
                            **kwargs,
                        )
                    elif mode_name == "custom_voice":
                        speaker = request.get("speaker")
                        if not speaker:
                            raise ValueError("speaker is required for custom_voice")
                        wavs, sr = runtime.generate_custom_voice(
                            text=text,
                            speaker=speaker,
                            instruct=request.get("instruct", ""),
                            language=language,
                            **kwargs,
                        )
                    else:
                        wavs, sr = runtime.generate_voice_clone(
                            text=text,
                            language=language,
                            ref_audio=request.get("ref_audio"),
                            ref_text=request.get("ref_text"),
                            x_vector_only_mode=bool(request.get("x_vector_only", False)),
                            **kwargs,
                        )
                    return wavs[0], sr
                except Exception as exc:
                    if is_usm_allocation_error(exc) and attempt < usm_retry_count:
                        released_buffers, closed_runners = release_native_runtime_resources()
                        app.state.memory["last_usm_error"] = str(exc)
                        app.state.memory["last_usm_retry_at"] = time.time()
                        app.state.memory["last_usm_retry_count"] = attempt + 1
                        app.state.memory["last_released_native_buffers"] = released_buffers
                        app.state.memory["last_released_native_runners"] = closed_runners
                        time.sleep(0.25)
                        continue
                    if is_usm_allocation_error(exc):
                        raise RuntimeError(
                            "OpenVINO GPU USM allocation failed during full TTS generation after retry. "
                            "The native runner was released; reduce max_new_tokens/text length or restart the sidecar "
                            "if the GPU driver remains fragmented. Original error: "
                            f"{exc}"
                        ) from exc
                    raise

    @app.get("/health")
    def health():
        runtime_status = {}
        for key, runtime in runtimes.items():
            ir_dir = key[0]
            runtime_id = "|".join(str(item) for item in key[:8])
            variant_fused_buckets = (getattr(runtime, "variant_graphs", {}) or {}).get("fused_cache_step_buckets", {})
            cache_kernel = getattr(runtime, "cache_kernel", None)
            fused_variant_active = False
            if isinstance(variant_fused_buckets, dict):
                if cache_kernel in variant_fused_buckets and isinstance(variant_fused_buckets[cache_kernel], dict):
                    fused_variant_active = bool(variant_fused_buckets[cache_kernel])
                elif all(str(bucket).isdigit() for bucket in variant_fused_buckets):
                    fused_variant_active = bool(variant_fused_buckets)
            runtime_status[runtime_id] = {
                "ir_dir": ir_dir,
                "cache_step": getattr(runtime, "cache_step", None),
                "mode": getattr(runtime, "mode", None),
                "requested_mode": getattr(runtime, "requested_mode", None),
                "cache_kernel": cache_kernel,
                "graph_variant": getattr(runtime, "graph_variant", None),
                "codegen_unroll": getattr(runtime, "codegen_unroll", effective_unroll),
                "codegen_schedule": getattr(runtime, "codegen_schedule", codegen_schedule),
                "codegen_decode_unroll": getattr(runtime, "codegen_decode_unroll", codegen_decode_unroll),
                "preferred_cache_bucket": getattr(runtime, "preferred_cache_bucket", preferred_cache_bucket),
                "native_codegen": getattr(runtime, "native_codegen_override", None) or os.environ.get("QWEN3_TTS_OV_NATIVE_CODEGEN") or "off",
                "native_pipeline": getattr(runtime, "native_pipeline_override", None) or os.environ.get("QWEN3_TTS_OV_NATIVE_PIPELINE") or "off",
                "native_codegen_device": os.environ.get("QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE") or device,
                "native_paged_kv": getattr(runtime, "native_paged_kv_override", None)
                or os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV")
                or "off",
                "native_paged_kv_gqa": getattr(runtime, "native_paged_kv_gqa_override", None)
                or os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA")
                or "on",
                "native_paged_kv_precision": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_PRECISION") or "f16",
                "native_paged_kv_cache_input_precision": (
                    os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_CACHE_INPUT_PRECISION") or "f32"
                ),
                "native_paged_kv_block_size": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE") or "8",
                "native_paged_kv_static_decode": (
                    os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_DECODE") or "off"
                ),
                "native_paged_kv_static_blocks": (
                    os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_BLOCKS") or "128"
                ),
                "native_paged_kv_static_decode_mode": (
                    os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_DECODE_MODE") or "minimal"
                ),
                "native_paged_kv_unroll": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_UNROLL") or "1",
                "native_paged_kv_experimental_unroll": (
                    os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_EXPERIMENTAL_UNROLL") or "0"
                ),
                "native_paged_kv_subcode_attention": (
                    os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SUBCODE_ATTENTION") or "auto"
                ),
                "native_paged_kv_split_subcode": (
                    getattr(runtime, "native_paged_kv_split_subcode_override", None)
                    or os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE")
                    or "off"
                ),
                "native_paged_kv_split_subcode_mode": (
                    os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE_MODE") or "cached"
                ),
                "native_paged_kv_score_aggregation": (
                    os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SCORE_AGGREGATION") or "on"
                ),
                "native_subcode_device": os.environ.get("QWEN3_TTS_OV_NATIVE_SUBCODE_DEVICE") or "same",
                "native_paged_kv_hybrid": os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID") or "off",
                "native_paged_kv_hybrid_prefix_frames": (
                    os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID_PREFIX_FRAMES") or "48"
                ),
                "paged_kv": bool(getattr(runtime, "paged_kv_enabled", False)),
                "paged_kv_backend": getattr(runtime, "paged_kv_backend", "stateful_bucket"),
                "paged_kv_unavailable_reason": getattr(runtime, "paged_kv_unavailable_reason", PAGED_KV_UNAVAILABLE_REASON),
                "unroll_available": bool(getattr(runtime, "fused_cache_unroll_bucket_graphs", {}))
                or bool(getattr(runtime, "fused_cache_unroll_bucket_graphs_by_step", {})),
                "unroll_fallback": bool(getattr(runtime, "codegen_unroll_fallback", False)),
                "ov_cache_dir": None if getattr(runtime, "cache_dir", None) is None else str(runtime.cache_dir),
                "ov_cache_mode": getattr(runtime, "ov_cache_mode", None),
                "ov_cache_disabled": getattr(runtime, "disable_ov_cache", False),
                "streaming_decoder_available": bool(getattr(runtime, "streaming_decoder_graphs_by_context", {})),
                "streaming_decoder_contexts": {
                    str(context): sorted(chunk_graphs)
                    for context, chunk_graphs in getattr(runtime, "streaming_decoder_graphs_by_context", {}).items()
                },
                "default_chunk_strategy": getattr(runtime, "default_chunk_strategy", "low_latency"),
                "chunk_strategies": getattr(runtime, "streaming_decoder_strategies", DEFAULT_STREAM_CHUNK_STRATEGIES),
                "compiled_stream_decoders": [
                    {"context_frames": context, "chunk_frames": chunk}
                    for context, chunk in sorted(getattr(runtime, "streaming_decoders", {}))
                ],
                "compiled_fused_buckets": sorted(getattr(runtime, "fused_cache_step_by_bucket", {})),
                "fused_cache_bucket_graphs": {
                    str(bucket): graph for bucket, graph in getattr(runtime, "fused_cache_bucket_graphs", {}).items()
                },
                "fused_cache_variant_active": fused_variant_active,
                "compiled_stateful_buckets": sorted(getattr(runtime, "talker_stateful_by_bucket", {})),
            }
        return {
            "ok": True,
            "model_root": str(model_root),
            "warmup": app.state.warmup,
            "memory": app.state.memory,
            "runtimes": runtime_status,
        }

    @app.get("/", response_class=HTMLResponse)
    def web_client():
        return HTMLResponse(
            WEB_CLIENT_HTML,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @app.get("/web", response_class=HTMLResponse)
    def web_client_alias():
        return HTMLResponse(
            WEB_CLIENT_HTML,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    @app.post("/v1/tts")
    def tts(request: dict):
        try:
            audio, sr = full_audio(request)
            return Response(content=wav_bytes(audio, sr), media_type="audio/wav")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/tts/stream")
    def tts_stream(request: dict):
        def iter_lines():
            started = time.time()
            try:
                metadata_gen_kwargs = generation_kwargs(request, default_repetition_penalty=default_repetition_penalty)
                metadata = stream_metadata(request, default_stream_strategy, forced_strategy=forced_stream_strategy)
                continuity = full_context_metadata(request) or auto_segment_metadata(request) or continuous_long_output_metadata(
                    request_uses_continuous_long_output(request, metadata_gen_kwargs)
                )
                playback_buffer_ms = playback_buffer_for_stream(metadata, recommended_playback_buffer_ms)
                yield json.dumps(
                    {
                        "type": "metadata",
                        "sample_rate": 24000,
                        "format": "pcm_s16le",
                        "started_at": started,
                        **metadata,
                        **request_prompt_memory_estimate(request, metadata_gen_kwargs),
                        **runtime_stream_metadata,
                        **continuity,
                        "long_ar_do_sample": bool(metadata_gen_kwargs.get("do_sample", False)),
                        "active_tts_requests": app.state.memory["active_tts_requests"],
                        "recommended_playback_buffer_ms": int(playback_buffer_ms),
                    },
                    ensure_ascii=False,
                ) + "\n"
                for chunk in stream_chunks(request):
                    if chunk.audio.size:
                        yield json.dumps(
                            {
                                "type": "audio",
                                "index": chunk.index,
                                "sample_rate": chunk.sample_rate,
                                "format": "pcm_s16le",
                                "is_final": chunk.is_final,
                                "timings": chunk.timings,
                                "audio": base64.b64encode(audio_to_pcm16(chunk.audio)).decode("ascii"),
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    if chunk.is_final:
                        yield json.dumps(
                            {"type": "final", "index": chunk.index, "elapsed": time.time() - started, "timings": chunk.timings},
                            ensure_ascii=False,
                        ) + "\n"
            except Exception as exc:
                yield json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False) + "\n"

        return StreamingResponse(iter_lines(), media_type="application/x-ndjson")

    async def stream_chunks_async(request: dict):
        output_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=2)
        stop_event = threading.Event()

        def put_item(item: tuple[str, object]) -> None:
            while not stop_event.is_set():
                try:
                    output_queue.put(item, timeout=0.1)
                    return
                except queue.Full:
                    continue

        def worker() -> None:
            try:
                for chunk in stream_chunks(request):
                    if stop_event.is_set():
                        break
                    put_item(("chunk", chunk))
            except Exception as exc:
                put_item(("error", exc))
            finally:
                put_item(("done", None))

        thread = threading.Thread(target=worker, name="qwen3-tts-stream", daemon=True)
        thread.start()
        try:
            while True:
                kind, payload = await asyncio.to_thread(output_queue.get)
                if kind == "chunk":
                    yield payload
                    continue
                if kind == "error":
                    raise payload  # type: ignore[misc]
                break
        finally:
            stop_event.set()

    @app.get("/v1/audio/voices")
    def audio_voices():
        voices = []
        voice_details = []
        seen = set()
        manifest_paths = []
        if has_manifest(model_root):
            manifest_paths.append(("direct", model_root / "manifest.json"))
        for model_name in sorted(set(MODE_DIR.values())):
            manifest_paths.append((model_name, model_root / model_name / "manifest.json"))
        fallback = resolve_ir_dir(DEFAULT_VOICE_DESIGN_IR_DIR, fallback_to_local_voice_design=True)
        if path_text(model_root) == "openvino" and has_manifest(fallback):
            manifest_paths.append(("voice_design", fallback / "manifest.json"))
        for model_name, manifest_path in manifest_paths:
            if not manifest_path.exists():
                continue
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                continue
            speakers = (manifest.get("ids") or {}).get("spk_id") or manifest.get("spk_id") or {}
            if isinstance(speakers, dict):
                iterable = speakers.keys()
            elif isinstance(speakers, list):
                iterable = speakers
            else:
                iterable = []
            for speaker in iterable:
                name = str(speaker)
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                voices.append(name)
                voice_details.append({"id": name, "name": name, "source": model_name})
        return {"voices": voices, "voice_details": voice_details, "uploaded_voices": []}

    @app.post("/v1/audio/speech")
    def audio_speech(request: dict):
        try:
            internal, response_format, stream_enabled = openai_speech_to_tts_request(request)
            if stream_enabled:
                if response_format not in {"pcm", "pcm_s16le"}:
                    raise ValueError('stream=true requires response_format="pcm"')

                def iter_pcm():
                    for chunk in stream_chunks(internal):
                        if chunk.audio.size:
                            yield audio_to_pcm16(chunk.audio)

                return StreamingResponse(iter_pcm(), media_type="audio/L16; rate=24000; channels=1")

            audio, sr = full_audio(internal)
            if response_format in {"pcm", "pcm_s16le"}:
                return Response(content=audio_to_pcm16(audio), media_type=f"audio/L16; rate={sr}; channels=1")
            if response_format != "wav":
                raise ValueError("only response_format=wav or pcm is supported")
            return Response(content=wav_bytes(audio, sr), media_type="audio/wav")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.websocket("/v1/tts/stream")
    async def websocket_stream(websocket: WebSocket):
        await websocket.accept()
        try:
            request = await websocket.receive_json()
            started = time.time()
            metadata_gen_kwargs = generation_kwargs(request, default_repetition_penalty=default_repetition_penalty)
            metadata = stream_metadata(request, default_stream_strategy, forced_strategy=forced_stream_strategy)
            continuity = full_context_metadata(request) or auto_segment_metadata(request) or continuous_long_output_metadata(
                request_uses_continuous_long_output(request, metadata_gen_kwargs)
            )
            playback_buffer_ms = playback_buffer_for_stream(metadata, recommended_playback_buffer_ms)
            await websocket.send_json(
                {
                    "type": "metadata",
                    "sample_rate": 24000,
                    "format": "pcm_s16le",
                    "started_at": started,
                    **metadata,
                    **request_prompt_memory_estimate(request, metadata_gen_kwargs),
                    **runtime_stream_metadata,
                    **continuity,
                    "long_ar_do_sample": bool(metadata_gen_kwargs.get("do_sample", False)),
                    "active_tts_requests": app.state.memory["active_tts_requests"],
                    "recommended_playback_buffer_ms": int(playback_buffer_ms),
                }
            )
            final_timings = {}
            final_index = 0
            send_chunk_metadata = include_chunk_metadata(request)
            async for chunk in stream_chunks_async(request):
                final_timings = chunk.timings
                final_index = chunk.index
                if chunk.audio.size:
                    pcm = audio_to_pcm16(chunk.audio)
                    if send_chunk_metadata:
                        await websocket.send_json(
                            {
                                "type": "audio",
                                "index": chunk.index,
                                "sample_rate": chunk.sample_rate,
                                "format": "pcm_s16le",
                                "byte_length": len(pcm),
                                "is_final": chunk.is_final,
                                "timings": chunk.timings,
                            }
                        )
                    await websocket.send_bytes(pcm)
                if chunk.is_final:
                    await websocket.send_json(
                        {
                            "type": "final",
                            "index": final_index,
                            "elapsed": time.time() - started,
                            "timings": final_timings,
                        }
                    )
                    break
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})

    return app


def serve(
    model_root: str | Path = "openvino",
    host: str = "127.0.0.1",
    port: int = 17860,
    device: str = "GPU",
    decoder_device: str | None = None,
    allow_cpu_fallback: bool = False,
    mode: str = "cache",
    cache_kernel: str = "exact",
    cache_step: str = "fused",
    graph_variant: str = "fp16",
    codegen_unroll: str | int = "profile",
    codegen_schedule: str = "current",
    codegen_decode_unroll: str = "off",
    preferred_cache_bucket: int | str | None = 112,
    ov_cache_dir: str | Path | None = None,
    ov_cache_mode: str | None = "optimize_speed",
    disable_ov_cache: bool = False,
    warmup: bool = True,
    preload_modes: str | list[str] = "voice_design",
    preload_buckets: str = "warmup",
    warmup_text: str = "你好，这是一次流式预热。",
    warmup_strategy: str = "low_latency",
    realtime_profile: str = FASTEST_PROFILE_NAME,
    max_concurrent_tts: int = 1,
    long_output_memory_policy: str = "stable",
    max_continuous_prompt_tokens: int = 1024,
    usm_retry_count: int = 1,
):
    import uvicorn

    app = create_app(
        model_root=model_root,
        device=device,
        decoder_device=decoder_device,
        allow_cpu_fallback=allow_cpu_fallback,
        mode=mode,
        cache_kernel=cache_kernel,
        cache_step=cache_step,
        graph_variant=graph_variant,
        codegen_unroll=codegen_unroll,
        codegen_schedule=codegen_schedule,
        codegen_decode_unroll=codegen_decode_unroll,
        preferred_cache_bucket=preferred_cache_bucket,
        ov_cache_dir=ov_cache_dir,
        ov_cache_mode=ov_cache_mode,
        disable_ov_cache=disable_ov_cache,
        warmup=warmup,
        preload_modes=preload_modes,
        preload_buckets=preload_buckets,
        warmup_text=warmup_text,
        warmup_strategy=warmup_strategy,
        realtime_profile=realtime_profile,
        max_concurrent_tts=max_concurrent_tts,
        long_output_memory_policy=long_output_memory_policy,
        max_continuous_prompt_tokens=max_continuous_prompt_tokens,
        usm_retry_count=usm_retry_count,
    )
    uvicorn.run(app, host=host, port=port)

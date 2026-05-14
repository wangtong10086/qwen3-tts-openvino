#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


DEFAULT_TEXT = (
    "This is a long-text quality and latency check for the Qwen3-TTS OpenVINO runtime. "
    "Pass --text-file for real validation."
)
DEFAULT_INSTRUCT = "Read with a natural, clear, consistent voice."
DEFAULT_LANGUAGE = "Chinese"
DEFAULT_PROFILES = "quality"
DEFAULT_OMNI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_OMNI_MAX_AUDIO_MB = 9.5
DEFAULT_SEGMENT_SECONDS = 45.0


PROFILE_ENV_MAP = {
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


LONG_TEXT_PROFILES: dict[str, dict[str, Any]] = {
    "long_reference_no_cache_fp16_sample": {
        "description": "Correctness-first full-AR reference: FP16 no-cache, single prompt, sampled first codebook.",
        "default_safe": True,
        "runtime": {
            "mode": "no-cache",
            "cache_kernel": "exact",
            "cache_step": "split",
            "graph_variant": "fp16",
            "codegen_unroll": "1",
            "codegen_schedule": "current",
            "codegen_decode_unroll": "off",
            "preferred_cache_bucket": "0",
            "native_pipeline": "off",
            "native_paged_kv": "0",
            "native_paged_kv_gqa": "0",
            "native_paged_kv_split_subcode": "0",
            "repetition_penalty": 1.05,
            "do_sample": True,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.9,
        },
        "env": {
            "native_pipeline": "0",
            "native_paged_kv": "0",
            "native_paged_kv_gqa": "0",
            "native_paged_kv_split_subcode": "0",
            "native_buffer_reuse": "0",
        },
    },
    "long_quality_paged_fullhead_int8_sym": {
        "description": "Quality-first paged KV, full fused seed, INT8_SYM weights.",
        "default_safe": False,
        "runtime": {
            "mode": "no-cache",
            "cache_kernel": "exact",
            "cache_step": "fused",
            "graph_variant": "int8_sym_paged_kv_seed",
            "codegen_unroll": "1",
            "codegen_schedule": "current",
            "codegen_decode_unroll": "off",
            "preferred_cache_bucket": "0",
            "native_pipeline": "require",
            "native_paged_kv": "require",
            "native_paged_kv_gqa": "0",
            "native_paged_kv_split_subcode": "0",
            "repetition_penalty": 1.05,
        },
        "env": {
            "native_codegen_device": "GPU",
            "native_paged_kv_precision": "u8",
            "native_paged_kv_block_size": "16",
            "native_paged_kv_gqa": "0",
            "native_paged_kv_split_subcode": "0",
            "native_paged_kv_score_aggregation": "0",
            "native_buffer_reuse": "0",
        },
    },
    "long_paged_split_sample_fp16": {
        "description": "Full-AR native paged-KV split-subcode, FP16 talker, sampled first codebook.",
        "default_safe": True,
        "runtime": {
            "mode": "no-cache",
            "cache_kernel": "exact",
            "cache_step": "fused",
            "graph_variant": "fp16",
            "codegen_unroll": "1",
            "codegen_schedule": "current",
            "codegen_decode_unroll": "off",
            "preferred_cache_bucket": "0",
            "native_pipeline": "require",
            "native_paged_kv": "require",
            "native_paged_kv_gqa": "1",
            "native_paged_kv_split_subcode": "1",
            "repetition_penalty": 1.05,
            "do_sample": True,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.9,
        },
        "env": {
            "native_codegen_device": "GPU",
            "native_paged_kv_precision": "u8",
            "native_paged_kv_block_size": "16",
            "native_paged_kv_gqa": "1",
            "native_paged_kv_split_subcode": "1",
            "native_paged_kv_split_subcode_mode": "cached_exact",
            "native_paged_kv_score_aggregation": "1",
            "native_buffer_reuse": "0",
        },
    },
    "long_paged_split_sample_int8_sym": {
        "description": "Full-AR native paged-KV split-subcode, INT8_SYM talker, sampled first codebook.",
        "default_safe": True,
        "runtime": {
            "mode": "no-cache",
            "cache_kernel": "exact",
            "cache_step": "fused",
            "graph_variant": "int8_sym_paged_talker_split",
            "codegen_unroll": "1",
            "codegen_schedule": "current",
            "codegen_decode_unroll": "off",
            "preferred_cache_bucket": "0",
            "native_pipeline": "require",
            "native_paged_kv": "require",
            "native_paged_kv_gqa": "1",
            "native_paged_kv_split_subcode": "1",
            "repetition_penalty": 1.05,
            "do_sample": True,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.9,
        },
        "env": {
            "native_codegen_device": "GPU",
            "native_paged_kv_precision": "u8",
            "native_paged_kv_block_size": "16",
            "native_paged_kv_gqa": "1",
            "native_paged_kv_split_subcode": "1",
            "native_paged_kv_split_subcode_mode": "cached_exact",
            "native_paged_kv_score_aggregation": "1",
            "native_buffer_reuse": "0",
        },
    },
    "long_quality_paged_fullhead_fp16": {
        "description": "Quality-first paged KV, full fused seed, FP16 graph baseline.",
        "default_safe": False,
        "runtime": {
            "mode": "no-cache",
            "cache_kernel": "exact",
            "cache_step": "fused",
            "graph_variant": "fp16",
            "codegen_unroll": "1",
            "codegen_schedule": "current",
            "codegen_decode_unroll": "off",
            "preferred_cache_bucket": "0",
            "native_pipeline": "require",
            "native_paged_kv": "require",
            "native_paged_kv_gqa": "0",
            "native_paged_kv_split_subcode": "0",
            "repetition_penalty": 1.05,
        },
        "env": {
            "native_codegen_device": "GPU",
            "native_paged_kv_precision": "u8",
            "native_paged_kv_block_size": "16",
            "native_paged_kv_gqa": "0",
            "native_paged_kv_split_subcode": "0",
            "native_paged_kv_score_aggregation": "0",
            "native_buffer_reuse": "0",
        },
    },
    "long_quality_paged_gqa_no_split": {
        "description": "Paged KV with GQA seed, but no split-subcode path.",
        "default_safe": False,
        "runtime": {
            "mode": "no-cache",
            "cache_kernel": "exact",
            "cache_step": "fused",
            "graph_variant": "int8_sym_paged_kv_seed",
            "codegen_unroll": "1",
            "codegen_schedule": "current",
            "codegen_decode_unroll": "off",
            "preferred_cache_bucket": "0",
            "native_pipeline": "require",
            "native_paged_kv": "require",
            "native_paged_kv_gqa": "1",
            "native_paged_kv_split_subcode": "0",
            "repetition_penalty": 1.05,
        },
        "env": {
            "native_codegen_device": "GPU",
            "native_paged_kv_precision": "u8",
            "native_paged_kv_block_size": "16",
            "native_paged_kv_gqa": "1",
            "native_paged_kv_split_subcode": "0",
            "native_paged_kv_score_aggregation": "0",
            "native_buffer_reuse": "0",
        },
    },
    "long_stateful_int8_sdpa": {
        "description": "Non-paged stateful cache baseline with SDPA split codegen.",
        "default_safe": False,
        "runtime": {
            "mode": "cache",
            "cache_kernel": "sdpa",
            "cache_step": "split",
            "graph_variant": "int8_cachedsub",
            "codegen_unroll": "1",
            "codegen_schedule": "current",
            "codegen_decode_unroll": "off",
            "preferred_cache_bucket": "384",
            "native_pipeline": "off",
            "native_paged_kv": "0",
            "native_paged_kv_gqa": "0",
            "native_paged_kv_split_subcode": "0",
            "repetition_penalty": 1.05,
        },
        "env": {
            "native_pipeline": "0",
            "native_paged_kv": "0",
            "native_buffer_reuse": "0",
        },
    },
    "long_reference_fused_no_cache_fp16": {
        "description": "FP16 fused no-cache reference; slow, but avoids paged-KV and fixed cache buckets.",
        "default_safe": False,
        "runtime": {
            "mode": "fused-no-cache",
            "cache_kernel": "exact",
            "cache_step": "fused",
            "graph_variant": "fp16",
            "codegen_unroll": "1",
            "codegen_schedule": "current",
            "codegen_decode_unroll": "off",
            "preferred_cache_bucket": "0",
            "native_pipeline": "off",
            "native_paged_kv": "0",
            "native_paged_kv_gqa": "0",
            "native_paged_kv_split_subcode": "0",
            "repetition_penalty": 1.05,
        },
        "env": {
            "native_pipeline": "0",
            "native_paged_kv": "0",
            "native_paged_kv_gqa": "0",
            "native_paged_kv_split_subcode": "0",
            "native_buffer_reuse": "0",
        },
    },
    "long_experimental_split_subcode": {
        "description": "Fastest split-subcode paged path; included only to prove or reject quality.",
        "default_safe": False,
        "runtime": {
            "mode": "no-cache",
            "cache_kernel": "exact",
            "cache_step": "fused",
            "graph_variant": "int8_sym_paged_talker_split",
            "codegen_unroll": "1",
            "codegen_schedule": "current",
            "codegen_decode_unroll": "off",
            "preferred_cache_bucket": "0",
            "native_pipeline": "require",
            "native_paged_kv": "require",
            "native_paged_kv_gqa": "1",
            "native_paged_kv_split_subcode": "1",
            "repetition_penalty": 1.05,
        },
        "env": {
            "native_codegen_device": "GPU",
            "native_paged_kv_precision": "u8",
            "native_paged_kv_block_size": "16",
            "native_paged_kv_gqa": "1",
            "native_paged_kv_split_subcode": "1",
            "native_paged_kv_score_aggregation": "1",
            "native_buffer_reuse": "0",
        },
    },
}


TEXT_TOKEN_RE = re.compile(r"\s+|[A-Za-z0-9]+|[\u3400-\u9fff]|[^\s]")


def speech_text_units(token: str) -> int:
    if not token or token.isspace():
        return 0
    if re.fullmatch(r"[A-Za-z0-9]+", token):
        return 2
    if re.fullmatch(r"[\u3400-\u9fff]", token):
        return 1
    return 0 if token in set("。！？!?；;，,、.:：") else 1


def speech_text_unit_count(text: str) -> int:
    return sum(speech_text_units(token) for token in TEXT_TOKEN_RE.findall(str(text or "")))


def estimate_max_new_tokens(text: str, requested: int | None, *, cap: int = 640) -> int:
    requested = int(requested or 0)
    units = speech_text_unit_count(text)
    estimate = int(max(48, units * 4.0 + 128))
    return int(min(max(requested, estimate), cap))


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def parse_profiles(value: str) -> list[str]:
    normalized = str(value or "").strip()
    if normalized in {"", "default", "quality"}:
        return [name for name, config in LONG_TEXT_PROFILES.items() if config.get("default_safe")]
    if normalized == "all":
        return list(LONG_TEXT_PROFILES)
    profiles = [item.strip() for item in normalized.split(",") if item.strip()]
    unknown = [item for item in profiles if item not in LONG_TEXT_PROFILES]
    if unknown:
        known = ", ".join(LONG_TEXT_PROFILES)
        raise ValueError(f"unknown profiles: {', '.join(unknown)}. Known profiles: {known}")
    return profiles


def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].strip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip().strip("\"'")
    return key, value


def load_env_file(env_file: str | Path = ".env") -> None:
    path = Path(env_file)
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except Exception:
        pass
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def load_aliyun_env(env_file: str | Path = ".env") -> dict[str, str | None]:
    load_env_file(env_file)
    return {
        "api_key": _first_env("ALIYUN_API_KEY", "aliyun_api_key", "DASHSCOPE_API_KEY", "dashscope_api_key"),
        "model": _first_env("ALIYUN_MODEL_NAME", "aliyun_model_name", "OMNI_MODEL_NAME", "omni_model_name"),
        "base_url": _first_env("ALIYUN_BASE_URL", "aliyun_base_url", "OPENAI_BASE_URL") or DEFAULT_OMNI_BASE_URL,
    }


def redacted_env_status(config: dict[str, str | None]) -> dict[str, Any]:
    base_url = config.get("base_url") or ""
    return {
        "api_key": "set" if config.get("api_key") else "missing",
        "model": "set" if config.get("model") else "missing",
        "base_url_host": re.sub(r"^https?://", "", base_url).split("/", 1)[0] if base_url else "missing",
    }


def read_text_arg(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    return str(args.text or DEFAULT_TEXT).strip()


def objective_audio_metrics(wav_path: str | Path) -> dict[str, Any]:
    data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    finite = np.isfinite(audio)
    if not finite.all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    abs_audio = np.abs(audio)
    duration_sec = float(audio.shape[0] / sample_rate) if sample_rate else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    peak = float(np.max(abs_audio)) if audio.size else 0.0
    silence_ratio = float(np.mean(abs_audio < 1.0e-3)) if audio.size else 1.0
    clip_ratio = float(np.mean(abs_audio > 0.98)) if audio.size else 0.0
    if audio.size > 1:
        signs = np.signbit(audio)
        zero_crossing_rate = float(np.mean(signs[1:] != signs[:-1]))
        flatline_ratio = float(np.mean(np.abs(np.diff(audio)) < 1.0e-7))
    else:
        zero_crossing_rate = 0.0
        flatline_ratio = 1.0

    window = max(1, int(sample_rate * 0.05))
    if audio.size >= window:
        trimmed = audio[: (audio.size // window) * window].reshape(-1, window)
        window_rms = np.sqrt(np.mean(np.square(trimmed), axis=1))
        low_rms_window_ratio = float(np.mean(window_rms < 2.0e-3))
        high_rms_window_ratio = float(np.mean(window_rms > 0.25))
    else:
        low_rms_window_ratio = 1.0 if rms < 2.0e-3 else 0.0
        high_rms_window_ratio = 1.0 if rms > 0.25 else 0.0

    return {
        "sample_rate": int(sample_rate),
        "samples": int(audio.shape[0]),
        "duration_sec": duration_sec,
        "finite": bool(finite.all()),
        "rms": rms,
        "peak": peak,
        "silence_ratio": silence_ratio,
        "clip_ratio": clip_ratio,
        "zero_crossing_rate": zero_crossing_rate,
        "flatline_ratio": flatline_ratio,
        "low_rms_window_ratio": low_rms_window_ratio,
        "high_rms_window_ratio": high_rms_window_ratio,
        "dc_offset": float(abs(np.mean(audio))) if audio.size else 0.0,
    }


def code_metrics(codes: np.ndarray | None) -> dict[str, Any]:
    if codes is None or np.asarray(codes).size == 0:
        return {
            "frames": 0,
            "groups": 0,
            "adjacent_equal_frame_ratio": 0.0,
            "dominant_token_ratio_max": 0.0,
            "unique_token_ratio_min": 0.0,
        }
    arr = np.asarray(codes)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    frames, groups = arr.shape
    if frames > 1:
        adjacent_equal_frame_ratio = float(np.mean(np.all(arr[1:] == arr[:-1], axis=1)))
    else:
        adjacent_equal_frame_ratio = 0.0
    unique_ratios = []
    dominant_ratios = []
    for group_index in range(groups):
        values, counts = np.unique(arr[:, group_index], return_counts=True)
        unique_ratios.append(float(len(values) / max(frames, 1)))
        dominant_ratios.append(float(np.max(counts) / max(frames, 1)))
    return {
        "frames": int(frames),
        "groups": int(groups),
        "adjacent_equal_frame_ratio": adjacent_equal_frame_ratio,
        "dominant_token_ratio_max": float(max(dominant_ratios) if dominant_ratios else 0.0),
        "unique_token_ratio_min": float(min(unique_ratios) if unique_ratios else 0.0),
    }


def objective_gate(audio: dict[str, Any], codes: dict[str, Any] | None = None) -> dict[str, Any]:
    codes = codes or {}
    failures: list[str] = []

    def metric_float(source: dict[str, Any], key: str, default: float = 0.0) -> float:
        value = source.get(key, default)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def metric_int(source: dict[str, Any], key: str, default: int = 0) -> int:
        value = source.get(key, default)
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    if metric_int(audio, "sample_rate") != 24000:
        failures.append("sample_rate_not_24000")
    if metric_float(audio, "duration_sec") < 0.5:
        failures.append("too_short")
    if not bool(audio.get("finite", False)):
        failures.append("non_finite_audio")
    if metric_float(audio, "rms") < 0.003:
        failures.append("rms_too_low")
    if metric_float(audio, "peak") < 0.02:
        failures.append("peak_too_low")
    if metric_float(audio, "clip_ratio") > 0.01:
        failures.append("clipping")
    if metric_float(audio, "silence_ratio", 1.0) > 0.85:
        failures.append("mostly_silence")
    if metric_float(audio, "low_rms_window_ratio", 1.0) > 0.85:
        failures.append("mostly_low_energy_windows")
    if metric_float(audio, "zero_crossing_rate") > 0.42:
        failures.append("noise_like_high_zcr")
    if metric_float(audio, "flatline_ratio") > 0.95:
        failures.append("flatline")
    if metric_int(codes, "frames") > 12:
        if metric_float(codes, "adjacent_equal_frame_ratio") > 0.75:
            failures.append("codec_frame_collapse")
        if metric_float(codes, "dominant_token_ratio_max") > 0.92:
            failures.append("codec_token_collapse")
    return {"pass": not failures, "failures": failures}


def wav_bytes_for_segment(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def wav_file_bytes(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def audio_bytes_to_data_url(wav_bytes: bytes) -> str:
    encoded = base64.b64encode(wav_bytes).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def segment_wav_for_omni(
    wav_path: str | Path,
    *,
    max_base64_bytes: int,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
) -> list[tuple[str, bytes]]:
    wav_bytes = wav_file_bytes(wav_path)
    if len(base64.b64encode(wav_bytes)) <= max_base64_bytes:
        return [("full", wav_bytes)]

    data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    segment_samples = max(1, int(float(segment_seconds) * int(sample_rate)))
    if audio.shape[0] <= segment_samples:
        return [("full", wav_bytes_for_segment(audio, int(sample_rate)))]

    starts = {
        "head": 0,
        "middle": max(0, (audio.shape[0] - segment_samples) // 2),
        "tail": max(0, audio.shape[0] - segment_samples),
    }
    segments = []
    seen: set[tuple[int, int]] = set()
    for label, start in starts.items():
        end = min(audio.shape[0], start + segment_samples)
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        segments.append((label, wav_bytes_for_segment(audio[start:end], int(sample_rate))))
    return segments


def build_omni_messages(text: str, audio_data_url: str, *, segment_label: str = "full") -> list[dict[str, Any]]:
    prompt = (
        "You are a strict TTS quality evaluator. Evaluate whether the audio is intelligible speech "
        "that correctly speaks the target text, has stable speaker identity, and has no obvious noise, "
        "stutter, truncation, or long-text drift. Return JSON only with this schema: "
        '{"verdict":"pass|fail","intelligibility":0-5,"naturalness":0-5,'
        '"continuity":0-5,"speaker_consistency":0-5,"noise":0-5,'
        '"transcript_match":0-5,"failure_reason":"...","actionable_notes":["..."]}. '
        "Use 5 as best. Mark verdict fail if the audio is mostly noise or the speech cannot be understood. "
        f"Segment: {segment_label}. Target text: {text}"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": audio_data_url, "format": "wav"}},
            ],
        }
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    content = str(text or "").strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object found in Omni response: {content[:200]}")
    return json.loads(content[start : end + 1])


def normalize_omni_result(raw: dict[str, Any], *, segment_label: str) -> dict[str, Any]:
    result = dict(raw)
    result["segment"] = segment_label
    result["verdict"] = str(result.get("verdict", "fail")).strip().lower()
    score_fields = (
        "intelligibility",
        "naturalness",
        "continuity",
        "speaker_consistency",
        "noise",
        "transcript_match",
    )
    scores = []
    for key in score_fields:
        try:
            value = float(result.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        result[key] = value
        scores.append(value)
    result["score_mean"] = float(sum(scores) / len(scores)) if scores else 0.0
    result["pass"] = bool(
        result["verdict"] == "pass"
        and result["intelligibility"] >= 3.5
        and result["continuity"] >= 3.0
        and result["noise"] >= 3.0
        and result["transcript_match"] >= 3.0
    )
    return result


def judge_wav_with_omni(
    wav_path: str | Path,
    *,
    text: str,
    config: dict[str, str | None],
    max_base64_bytes: int,
    segment_seconds: float,
) -> dict[str, Any]:
    if not config.get("api_key") or not config.get("model"):
        raise RuntimeError(
            "Aliyun Omni judge requires API key and model. Set aliyun_api_key and aliyun_model_name "
            "in .env, or pass --skip-omni."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('Missing dependency "openai". Install with `uv pip install -e ".[quality]"`.') from exc

    client = OpenAI(api_key=config["api_key"], base_url=config.get("base_url") or DEFAULT_OMNI_BASE_URL)
    segment_results = []
    for label, wav_bytes in segment_wav_for_omni(
        wav_path,
        max_base64_bytes=max_base64_bytes,
        segment_seconds=segment_seconds,
    ):
        messages = build_omni_messages(text, audio_bytes_to_data_url(wav_bytes), segment_label=label)
        content_parts: list[str] = []
        stream = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            temperature=0,
            stream=True,
        )
        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            delta = getattr(choice, "delta", None)
            value = getattr(delta, "content", None) if delta is not None else None
            if isinstance(value, str):
                content_parts.append(value)
        raw_text = "".join(content_parts).strip()
        raw_result = extract_json_object(raw_text)
        result = normalize_omni_result(raw_result, segment_label=label)
        result["raw_text"] = raw_text
        segment_results.append(result)

    failures = [item for item in segment_results if not item.get("pass")]
    score_mean = float(min((item.get("score_mean", 0.0) for item in segment_results), default=0.0))
    return {
        "pass": not failures,
        "score_min": score_mean,
        "segments": segment_results,
        "failure_reason": "; ".join(str(item.get("failure_reason", "")) for item in failures if item.get("failure_reason")),
    }


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def profile_runtime_kwargs(profile: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(profile.get("runtime", {}))
    allowed = {
        "mode",
        "cache_kernel",
        "cache_step",
        "graph_variant",
        "codegen_unroll",
        "codegen_schedule",
        "codegen_decode_unroll",
        "preferred_cache_bucket",
        "native_pipeline",
        "native_paged_kv",
        "native_paged_kv_gqa",
        "native_paged_kv_split_subcode",
    }
    return {key: value for key, value in runtime.items() if key in allowed}


def profile_generation_kwargs(profile: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(profile.get("runtime", {}))
    return {
        "do_sample": bool(runtime.get("do_sample", config.get("do_sample", False))),
        "top_k": int(runtime.get("top_k", config.get("top_k", 50))),
        "top_p": float(runtime.get("top_p", config.get("top_p", 1.0))),
        "temperature": float(runtime.get("temperature", config.get("temperature", 0.9))),
    }


def apply_profile_env(profile: dict[str, Any]) -> None:
    for key, value in dict(profile.get("env", {})).items():
        env_name = PROFILE_ENV_MAP.get(key, key)
        if value is None:
            continue
        os.environ[env_name] = str(value)


def recent_codec_prefix(codes_parts: list[np.ndarray], frame_limit: int) -> np.ndarray | None:
    if frame_limit <= 0 or not codes_parts:
        return None
    codes = np.concatenate(codes_parts, axis=0)
    if codes.size == 0:
        return None
    return codes[-int(frame_limit) :].astype(np.int64, copy=True)


def worker_main(worker_json: str | Path) -> int:
    config = load_json(worker_json)
    result_path = Path(config["result_json"])
    try:
        profile_name = config["profile"]
        profile = LONG_TEXT_PROFILES[profile_name]
        apply_profile_env(profile)

        from qwen3_tts_ov.manifest import resolve_ir_dir
        from qwen3_tts_ov.runtime import OpenVINOQwen3TTS

        ir_dir = resolve_ir_dir(config["ir_dir"], fallback_to_local_voice_design=True, warn=True)
        runtime_kwargs = profile_runtime_kwargs(profile)
        runtime = OpenVINOQwen3TTS(
            str(ir_dir),
            device=config["device"],
            decoder_device=config.get("decoder_device") or config["device"],
            allow_cpu_fallback=bool(config.get("allow_cpu_fallback", False)),
            precision_hint=config.get("precision_hint", "f16"),
            ov_cache_mode=config.get("ov_cache_mode", "optimize_speed"),
            **runtime_kwargs,
        )

        text = str(config["text"])
        max_new_tokens = int(config["max_new_tokens"])
        repetition_penalty = float(profile.get("runtime", {}).get("repetition_penalty", config["repetition_penalty"]))
        generation_kwargs = profile_generation_kwargs(profile, config)
        audio_parts: list[np.ndarray] = []
        code_parts: list[np.ndarray] = []
        first_audio_ms: float | None = None
        last_timings: dict[str, Any] = {}
        started = time.perf_counter()
        segment_units = int(config.get("segment_units") or 0)
        segment_summaries: list[dict[str, Any]] = []
        judgement_text = text
        if segment_units > 0:
            from qwen3_tts_ov.server import (
                apply_boundary_fade,
                split_text_for_streaming,
                speech_text_unit_count,
                trim_audio_silence,
            )

            segments = split_text_for_streaming(text, max_units=segment_units)
            judgement_text = " ".join(segments)
            prefix_codes: np.ndarray | None = None
            prefix_frame_limit = int(config.get("segment_prefix_frames") or 0)
            segment_cap = int(config.get("segment_max_new_tokens") or max_new_tokens)
            for segment_index, segment in enumerate(segments):
                segment_audio_parts: list[np.ndarray] = []
                segment_codes_parts: list[np.ndarray] = []
                segment_max_new_tokens = estimate_max_new_tokens(segment, segment_cap, cap=segment_cap)
                segment_max_prompt_tokens = max(
                    int(config["max_prompt_tokens"]),
                    speech_text_unit_count(segment) + speech_text_unit_count(config["instruct"]) + 96,
                )
                for chunk in runtime.stream_voice_design(
                    text=segment,
                    instruct=config["instruct"],
                    language=config["language"],
                    max_new_tokens=segment_max_new_tokens,
                    min_new_tokens=int(config["min_new_tokens"]),
                    repetition_penalty=repetition_penalty,
                    max_prompt_tokens=segment_max_prompt_tokens,
                    chunk_strategy=config["chunk_strategy"],
                    **generation_kwargs,
                    prefix_codes=prefix_codes,
                    append_prefix_codes_to_prompt=bool(config.get("segment_append_prefix_to_prompt", True)),
                ):
                    last_timings = dict(chunk.timings or {})
                    if chunk.audio.size:
                        if first_audio_ms is None:
                            first_audio_ms = (time.perf_counter() - started) * 1000.0
                        segment_audio_parts.append(np.asarray(chunk.audio, dtype=np.float32))
                    if chunk.codes.size:
                        codes_chunk = np.asarray(chunk.codes, dtype=np.int64)
                        code_parts.append(codes_chunk)
                        segment_codes_parts.append(codes_chunk.reshape(-1, codes_chunk.shape[-1]))
                segment_audio = np.zeros((0,), dtype=np.float32)
                if segment_audio_parts:
                    segment_audio = np.concatenate(segment_audio_parts)
                    segment_audio = trim_audio_silence(
                        segment_audio,
                        runtime.sample_rate,
                        trim_start=True,
                        trim_end=True,
                    )
                    segment_audio = apply_boundary_fade(
                        segment_audio,
                        runtime.sample_rate,
                        fade_in=segment_index > 0,
                        fade_out=segment_index < len(segments) - 1,
                    )
                    if segment_audio.size:
                        audio_parts.append(np.asarray(segment_audio, dtype=np.float32))
                segment_summaries.append(
                    {
                        "index": int(segment_index),
                        "text": segment,
                        "units": int(speech_text_unit_count(segment)),
                        "max_new_tokens": int(segment_max_new_tokens),
                        "prefix_frames": int(0 if prefix_codes is None else prefix_codes.shape[0]),
                        "generated_frames": int(sum(part.shape[0] for part in segment_codes_parts)),
                        "audio_samples_after_trim": int(segment_audio.size),
                    }
                )
                next_prefix = recent_codec_prefix(segment_codes_parts, prefix_frame_limit)
                if next_prefix is not None:
                    prefix_codes = next_prefix
                if bool(config.get("segment_isolate_native_runner", True)):
                    close_runners = getattr(runtime, "close_native_audio_runners", None)
                    if close_runners is not None:
                        segment_summaries[-1]["closed_native_runners"] = int(close_runners())
        else:
            for chunk in runtime.stream_voice_design(
                text=text,
                instruct=config["instruct"],
                language=config["language"],
                max_new_tokens=max_new_tokens,
                min_new_tokens=int(config["min_new_tokens"]),
                repetition_penalty=repetition_penalty,
                max_prompt_tokens=int(config["max_prompt_tokens"]),
                chunk_strategy=config["chunk_strategy"],
                **generation_kwargs,
            ):
                last_timings = dict(chunk.timings or {})
                if chunk.audio.size:
                    if first_audio_ms is None:
                        first_audio_ms = (time.perf_counter() - started) * 1000.0
                    audio_parts.append(np.asarray(chunk.audio, dtype=np.float32))
                if chunk.codes.size:
                    code_parts.append(np.asarray(chunk.codes, dtype=np.int64))
        elapsed_sec = time.perf_counter() - started
        audio = np.concatenate(audio_parts) if audio_parts else np.zeros((0,), dtype=np.float32)
        codes = np.concatenate(code_parts, axis=0) if code_parts else np.empty((0, runtime.num_code_groups), dtype=np.int64)
        wav_path = Path(config["wav_path"])
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path, audio, runtime.sample_rate)
        npy_path = Path(config["codes_path"])
        np.save(npy_path, codes)

        audio_metrics = objective_audio_metrics(wav_path)
        codec_metrics = code_metrics(codes)
        gate = objective_gate(audio_metrics, codec_metrics)
        audio_sec = float(audio_metrics.get("duration_sec", 0.0) or 0.0)
        stream_rtf = float(last_timings.get("stream_rtf") or (elapsed_sec / audio_sec if audio_sec > 0 else 0.0))
        stream_compute_rtf = float(last_timings.get("stream_compute_rtf") or 0.0)
        result = {
            "ok": True,
            "profile": profile_name,
            "description": profile.get("description"),
            "wav_path": str(wav_path),
            "codes_path": str(npy_path),
            "ir_dir": str(ir_dir),
            "elapsed_sec": elapsed_sec,
            "audio_sec": audio_sec,
            "first_audio_ms": first_audio_ms,
            "stream_rtf": stream_rtf,
            "stream_compute_rtf": stream_compute_rtf,
            "generated_frames": int(codes.shape[0]),
            "last_timings": last_timings,
            "segmented": bool(segment_units > 0),
            "segments": segment_summaries,
            "judgement_text": judgement_text,
            "objective_audio": audio_metrics,
            "objective_codes": codec_metrics,
            "objective_gate": gate,
            "profile_env": dict(profile.get("env", {})),
            "runtime": {
                **profile.get("runtime", {}),
                "chunk_strategy": config["chunk_strategy"],
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": int(config["min_new_tokens"]),
                "max_prompt_tokens": int(config["max_prompt_tokens"]),
                "segment_units": int(segment_units),
                "segment_prefix_frames": int(config.get("segment_prefix_frames") or 0),
                "segment_max_new_tokens": int(config.get("segment_max_new_tokens") or 0),
                "segment_append_prefix_to_prompt": bool(config.get("segment_append_prefix_to_prompt", True)),
                "segment_isolate_native_runner": bool(config.get("segment_isolate_native_runner", True)),
                **generation_kwargs,
            },
        }
    except Exception as exc:
        result = {
            "ok": False,
            "profile": config.get("profile"),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    write_json(result_path, result)
    return 0


def run_worker_subprocess(worker_config: dict[str, Any], *, script_path: Path) -> dict[str, Any]:
    worker_json = Path(worker_config["worker_json"])
    write_json(worker_json, worker_config)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(script_path), "--worker-json", str(worker_json)],
        cwd=str(Path.cwd()),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    result_path = Path(worker_config["result_json"])
    if result_path.exists():
        result = load_json(result_path)
    else:
        result = {"ok": False, "profile": worker_config.get("profile"), "error": "worker did not write result JSON"}
    result["worker_exit_code"] = int(proc.returncode)
    result["worker_elapsed_sec"] = elapsed
    result["stdout_tail"] = proc.stdout[-4000:]
    result["stderr_tail"] = proc.stderr[-4000:]
    return result


def quality_pass(result: dict[str, Any]) -> bool:
    if not result.get("ok"):
        return False
    if not result.get("objective_gate", {}).get("pass"):
        return False
    omni = result.get("omni")
    if omni is not None and not omni.get("pass"):
        return False
    return True


def median(values: list[float]) -> float:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.median(clean)) if clean else math.inf


def select_winner(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if quality_pass(result):
            by_profile.setdefault(str(result["profile"]), []).append(result)
    if not by_profile:
        return None
    ranked = []
    for profile, items in by_profile.items():
        ranked.append(
            {
                "profile": profile,
                "runs": len(items),
                "median_stream_rtf": median([item.get("stream_rtf", math.inf) for item in items]),
                "median_first_audio_ms": median([item.get("first_audio_ms", math.inf) for item in items]),
                "best_stream_rtf": min(float(item.get("stream_rtf", math.inf)) for item in items),
                "wav_path": items[0].get("wav_path"),
                "runtime": items[0].get("runtime"),
                "profile_env": items[0].get("profile_env"),
            }
        )
    ranked.sort(key=lambda item: (item["median_stream_rtf"], item["median_first_audio_ms"], item["profile"]))
    return ranked[0]


def print_summary(results: list[dict[str, Any]], winner: dict[str, Any] | None) -> None:
    header = f"{'profile':44} {'ok':>3} {'obj':>3} {'omni':>5} {'rtf':>7} {'first_ms':>9} reason"
    print(header)
    print("-" * len(header))
    for item in results:
        obj = item.get("objective_gate", {})
        omni = item.get("omni")
        omni_status = "skip" if omni is None else ("yes" if omni.get("pass") else "no")
        failures = ",".join(obj.get("failures", []) or [])
        if omni and not omni.get("pass") and omni.get("failure_reason"):
            failures = f"{failures}; {omni['failure_reason']}".strip("; ")
        print(
            f"{str(item.get('profile'))[:44]:44} "
            f"{'yes' if item.get('ok') else 'no':>3} "
            f"{'yes' if obj.get('pass') else 'no':>3} "
            f"{omni_status:>5} "
            f"{float(item.get('stream_rtf') or 0.0):7.3f} "
            f"{float(item.get('first_audio_ms') or 0.0):9.1f} "
            f"{failures or item.get('error', '')}"
        )
    if winner:
        print(
            f"\nselected_profile={winner['profile']} "
            f"median_stream_rtf={winner['median_stream_rtf']:.3f} "
            f"median_first_audio_ms={winner['median_first_audio_ms']:.1f}"
        )
    else:
        print("\nselected_profile=none")


def build_worker_config(
    *,
    args: argparse.Namespace,
    profile: str,
    run_index: int,
    text: str,
    max_new_tokens: int,
    out_dir: Path,
) -> dict[str, Any]:
    run_dir = out_dir / safe_slug(profile) / f"run_{run_index:02d}"
    return {
        "worker_json": str(run_dir / "worker_config.json"),
        "result_json": str(run_dir / "worker_result.json"),
        "wav_path": str(run_dir / "audio.wav"),
        "codes_path": str(run_dir / "codes.npy"),
        "profile": profile,
        "ir_dir": args.ir_dir,
        "device": args.device,
        "decoder_device": args.decoder_device or args.device,
        "allow_cpu_fallback": bool(args.allow_cpu_fallback),
        "precision_hint": args.precision_hint,
        "ov_cache_mode": args.ov_cache_mode,
        "text": text,
        "instruct": args.instruct,
        "language": args.language,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "chunk_strategy": args.chunk_strategy,
        "repetition_penalty": args.repetition_penalty,
        "segment_units": int(args.segment_units),
        "segment_prefix_frames": int(args.segment_prefix_frames),
        "segment_max_new_tokens": int(args.segment_max_new_tokens),
        "segment_append_prefix_to_prompt": not bool(args.no_segment_append_prefix_to_prompt),
        "segment_isolate_native_runner": not bool(args.no_segment_isolate_native_runner),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate long-text Qwen3-TTS OpenVINO quality with Omni judge.")
    parser.add_argument("--worker-json", help=argparse.SUPPRESS)
    parser.add_argument("--ir-dir", default="auto")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--decoder-device", default=None)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--precision-hint", default="f16")
    parser.add_argument("--ov-cache-mode", default="optimize_speed")
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--profiles", default=DEFAULT_PROFILES)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--max-new-tokens-cap", type=int, default=2048)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--chunk-strategy", default="smooth")
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--segment-units", type=int, default=0)
    parser.add_argument("--segment-prefix-frames", type=int, default=24)
    parser.add_argument("--segment-max-new-tokens", type=int, default=240)
    parser.add_argument("--no-segment-append-prefix-to-prompt", action="store_true")
    parser.add_argument("--no-segment-isolate-native-runner", action="store_true")
    parser.add_argument("--out-dir", default="outputs/long_text_quality")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--skip-omni", action="store_true")
    parser.add_argument("--objective-only", action="store_true")
    parser.add_argument("--omni-max-audio-mb", type=float, default=DEFAULT_OMNI_MAX_AUDIO_MB)
    parser.add_argument("--segment-seconds", type=float, default=DEFAULT_SEGMENT_SECONDS)
    args = parser.parse_args(argv)

    if args.worker_json:
        return worker_main(args.worker_json)

    text = read_text_arg(args)
    profiles = parse_profiles(args.profiles)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_new_tokens = estimate_max_new_tokens(text, args.max_new_tokens, cap=args.max_new_tokens_cap)
    script_path = Path(__file__).resolve()
    omni_enabled = not bool(args.skip_omni or args.objective_only)
    omni_config = load_aliyun_env(args.env_file) if omni_enabled else {"api_key": None, "model": None, "base_url": None}
    if omni_enabled and (not omni_config.get("api_key") or not omni_config.get("model")):
        status = redacted_env_status(omni_config)
        raise SystemExit(
            "Aliyun Omni judge is enabled but credentials are incomplete. "
            f"status={json.dumps(status, ensure_ascii=False)}. "
            "Set aliyun_api_key and aliyun_model_name in .env, or pass --skip-omni."
        )

    results: list[dict[str, Any]] = []
    for profile in profiles:
        for run_index in range(1, int(args.runs) + 1):
            worker_config = build_worker_config(
                args=args,
                profile=profile,
                run_index=run_index,
                text=text,
                max_new_tokens=max_new_tokens,
                out_dir=out_dir,
            )
            print(f"running profile={profile} run={run_index} max_new_tokens={max_new_tokens}", flush=True)
            result = run_worker_subprocess(worker_config, script_path=script_path)
            if omni_enabled and result.get("ok") and result.get("objective_gate", {}).get("pass"):
                try:
                    result["omni"] = judge_wav_with_omni(
                        result["wav_path"],
                        text=str(result.get("judgement_text") or text),
                        config=omni_config,
                        max_base64_bytes=int(args.omni_max_audio_mb * 1024 * 1024),
                        segment_seconds=float(args.segment_seconds),
                    )
                except Exception as exc:
                    result["omni"] = {
                        "pass": False,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
            elif omni_enabled:
                result["omni"] = {"pass": False, "skipped": "objective_gate_failed_or_worker_failed"}
            results.append(result)
            write_json(Path(worker_config["result_json"]), result)

    winner = select_winner(results)
    summary = {
        "selected_profile": winner.get("profile") if winner else None,
        "text_units": speech_text_unit_count(text),
        "max_new_tokens": max_new_tokens,
        "profiles": profiles,
        "omni_enabled": omni_enabled,
        "omni_env": redacted_env_status(omni_config) if omni_enabled else None,
        "winner": winner,
        "results": results,
    }
    output_json = Path(args.output_json) if args.output_json else out_dir / "quality_summary.json"
    write_json(output_json, summary)
    print_summary(results, winner)
    print(f"summary_json={output_json}")
    return 0 if winner else 2


if __name__ == "__main__":
    raise SystemExit(main())

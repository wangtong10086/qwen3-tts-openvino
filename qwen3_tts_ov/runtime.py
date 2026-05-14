import argparse
from contextlib import contextmanager
import gc
import json
import os
import queue
import shutil
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# os.environ.setdefault("OV_TELEMETRY_DISABLE", "1")
# if "ZE_ENABLE_ALT_DRIVERS" not in os.environ:
#     default_level_zero_driver = "/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.14.37020"
#     if os.path.exists(default_level_zero_driver):
#         os.environ["ZE_ENABLE_ALT_DRIVERS"] = default_level_zero_driver

import numpy as np
import openvino as ov
import regex as re
import soundfile as sf

from .audio import load_audio, speaker_mel_spectrogram
from .cache import build_ov_cache_config, merge_compile_config_with_cache_mode, normalize_ov_cache_mode, resolve_ov_cache_dir
from .manifest import load_manifest, resolve_ir_dir
from .profiles import (
    CODEGEN_SCHEDULE_CHOICES,
    CODEGEN_UNROLL_CHOICES,
    RUNTIME_MODE_CHOICES,
    effective_codegen_unroll,
    effective_runtime_options,
    fastest_runtime_defaults,
    is_fastest_or_norepeat_mode,
    missing_graph_variant_message,
    normalize_codegen_schedule,
    scheduled_codegen_unrolls,
)


PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
NEG_INF = -3.4028234663852886e38
_CACHE_SPACE_WARNED: set[str] = set()
DEFAULT_STREAM_CHUNK_STRATEGIES = {
    "realtime": {
        "initial_chunk_frames": 8,
        "chunk_frames": 12,
        "left_context_frames": 25,
    },
    "low_latency": {
        "initial_chunk_frames": 8,
        "chunk_frames": 12,
        "left_context_frames": 25,
    },
    "smooth": {
        "initial_chunk_frames": 8,
        "chunk_frames": 24,
        "left_context_frames": 25,
    },
    "balanced": {
        "initial_chunk_frames": 12,
        "chunk_frames": 12,
        "left_context_frames": 25,
    },
    "stable": {
        "initial_chunk_frames": 12,
        "chunk_frames": 24,
        "left_context_frames": 25,
    },
}


def normalize_preferred_cache_bucket(value) -> int | None:
    if value is None:
        return 112
    text = str(value).strip().lower()
    if text in {"", "auto"}:
        return 112
    if text in {"none", "off", "false", "0"}:
        return None
    bucket = int(text)
    return bucket if bucket > 0 else None


def env_flag_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "on", "yes"}


def effective_paged_kv_unroll(requested: int | str | None, experimental_enabled: bool = False) -> int:
    value = int(requested or 1)
    if value > 1 and not experimental_enabled:
        return 1
    return max(1, value)


def paged_kv_seed_uses_gqa(seed_key: str | None) -> bool:
    key = str(seed_key or "")
    return key.endswith("_gqa") or "_gqa_" in key


def select_paged_kv_seed_key(
    paged_seed_graphs: dict,
    *,
    prefer_gqa: bool = True,
    split_subcode: bool = False,
    requested_unroll: int = 1,
    subcode_attention: str = "auto",
) -> tuple[str, str, bool]:
    """Select the paged-KV seed graph key, effective subcode mode, and fallback flag."""
    subcode_mode = str(subcode_attention or "auto").strip().lower().replace("-", "_")
    if subcode_mode not in {"auto", "sdpa", "exact"}:
        raise ValueError("QWEN3_TTS_OV_NATIVE_PAGED_KV_SUBCODE_ATTENTION must be one of: auto, sdpa, exact")
    seed_key = (
        ("talker_stateful_gqa" if prefer_gqa else "talker_stateful")
        if split_subcode
        else ("fused_cache_step_gqa" if prefer_gqa else "fused_cache_step")
    )
    if int(requested_unroll or 1) > 1 and not split_subcode:
        unroll_key = (
            f"fused_cache_step_unroll{int(requested_unroll)}_gqa"
            if prefer_gqa
            else f"fused_cache_step_unroll{int(requested_unroll)}"
        )
        if paged_seed_graphs.get(unroll_key):
            seed_key = unroll_key
    fallback = False
    selected_subcode_attention = "split" if split_subcode else "sdpa"
    if subcode_mode == "exact" and not split_subcode:
        exact_key = f"{seed_key}_subcode_exact"
        if paged_seed_graphs.get(exact_key):
            seed_key = exact_key
            selected_subcode_attention = "exact"
        else:
            non_gqa_exact_key = exact_key.replace("_gqa", "")
            if prefer_gqa and paged_seed_graphs.get(non_gqa_exact_key):
                seed_key = non_gqa_exact_key
                selected_subcode_attention = "exact"
            else:
                fallback = True
    elif subcode_mode == "exact" and split_subcode:
        fallback = True
    if not paged_seed_graphs.get(seed_key):
        non_gqa_key = seed_key.replace("_gqa", "")
        if prefer_gqa and paged_seed_graphs.get(non_gqa_key):
            seed_key = non_gqa_key
    return seed_key, selected_subcode_attention, fallback


@contextmanager
def temporary_env(updates: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def usable_ov_cache_dir(cache_dir: str | Path | None) -> Path | None:
    if cache_dir is None:
        return None
    try:
        min_free_gb = float(os.environ.get("QWEN3_TTS_OV_CACHE_MIN_FREE_GB") or "4")
    except ValueError:
        min_free_gb = 4.0
    if min_free_gb <= 0:
        return Path(cache_dir).expanduser()
    path = Path(cache_dir).expanduser()
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free_bytes = shutil.disk_usage(probe).free
    except Exception:
        return path
    min_free_bytes = int(min_free_gb * 1024 * 1024 * 1024)
    if free_bytes >= min_free_bytes:
        return path
    warning_key = str(path)
    if warning_key not in _CACHE_SPACE_WARNED:
        _CACHE_SPACE_WARNED.add(warning_key)
        print(
            f"warning: disabling OpenVINO cache for this compile because {path} has "
            f"{free_bytes / (1024 ** 3):.2f} GiB free, below QWEN3_TTS_OV_CACHE_MIN_FREE_GB={min_free_gb:g}",
            flush=True,
        )
    return None


@dataclass
class VoiceClonePromptItem:
    ref_code: np.ndarray | None
    ref_spk_embedding: np.ndarray
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None = None


@dataclass
class StreamChunk:
    index: int
    audio: np.ndarray
    sample_rate: int
    codes: np.ndarray
    is_final: bool
    timings: dict
    pcm_s16le: bytes | None = None


@lru_cache
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for value in range(2**8):
        if value not in bs:
            bs.append(value)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(value) for value in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Qwen2BPETokenizer:
    def __init__(self, model_dir: str):
        model_path = Path(model_dir)
        with open(model_path / "vocab.json", "r", encoding="utf-8") as f:
            self.encoder = json.load(f)

        merges = []
        with open(model_path / "merges.txt", "r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                line = line.strip()
                if (index == 0 and line.startswith("#version:")) or not line:
                    continue
                merges.append(tuple(line.split()))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
        self.byte_encoder = bytes_to_unicode()
        self.pat = re.compile(PRETOKENIZE_REGEX)

        with open(model_path / "tokenizer_config.json", "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
        special_tokens = []
        self.special_encoder = {}
        for token_id, item in tokenizer_config.get("added_tokens_decoder", {}).items():
            content = item.get("content")
            if content:
                special_tokens.append(content)
                self.special_encoder[content] = int(token_id)
                self.encoder.setdefault(content, int(token_id))
        special_tokens.extend(
            token for token in tokenizer_config.get("additional_special_tokens", []) if token in self.encoder
        )
        self.special_tokens = sorted(set(special_tokens), key=len, reverse=True)
        escaped = [re.escape(token) for token in self.special_tokens]
        self.special_pat = re.compile("(" + "|".join(escaped) + ")") if escaped else None

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                except ValueError:
                    new_word.extend(word[index:])
                    break
                new_word.extend(word[index:next_index])
                index = next_index
                if word[index] == first and index < len(word) - 1 and word[index + 1] == second:
                    new_word.append(first + second)
                    index += 2
                else:
                    new_word.append(word[index])
                    index += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def _tokenize_regular(self, text: str):
        token_ids = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            token_ids.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return token_ids

    def encode(self, text: str):
        if not text:
            return []
        if self.special_pat is None:
            return self._tokenize_regular(text)

        token_ids = []
        for part in self.special_pat.split(text):
            if part == "":
                continue
            if part in self.special_encoder:
                token_ids.append(self.special_encoder[part])
            else:
                token_ids.extend(self._tokenize_regular(part))
        return token_ids


def build_assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def build_instruct_text(instruct: str) -> str:
    return f"<|im_start|>user\n{instruct}<|im_end|>\n"


def build_ref_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n"


def compile_model(
    core: ov.Core,
    model_path: Path,
    device: str,
    cache_dir: Path | None,
    allow_cpu_fallback: bool = False,
    ov_profile: bool = False,
    precision_hint: str = "f16",
    extra_config: dict | None = None,
    ov_cache_mode: str | None = "optimize_speed",
    disable_ov_cache: bool = False,
):
    cache_dir = None if disable_ov_cache else usable_ov_cache_dir(cache_dir)
    config = {
        "INFERENCE_PRECISION_HINT": precision_hint,
    }
    config.update(build_ov_cache_config(cache_dir, ov_cache_mode=ov_cache_mode, disable_ov_cache=False))
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if extra_config:
        config.update({key: value for key, value in extra_config.items() if value is not None})
    if ov_profile:
        config["PERF_COUNT"] = "YES"
    if "GPU" in device:
        config["GPU_ENABLE_LARGE_ALLOCATIONS"] = "YES"
    try:
        return core.compile_model(str(model_path), device, config)
    except Exception as first_error:
        config.pop("GPU_ENABLE_LARGE_ALLOCATIONS", None)
        try:
            return core.compile_model(str(model_path), device, config)
        except Exception:
            if allow_cpu_fallback and "GPU" in device and device != "CPU":
                reason = next((line.strip() for line in str(first_error).splitlines() if line.strip()), repr(first_error))
                print(f"warning: failed to compile {model_path.name} on {device}; falling back to CPU", flush=True)
                print(f"warning: OpenVINO GPU error: {reason}", flush=True)
                fallback_config = build_ov_cache_config(
                    cache_dir,
                    ov_cache_mode=ov_cache_mode,
                    disable_ov_cache=disable_ov_cache,
                )
                if ov_profile:
                    fallback_config["PERF_COUNT"] = "YES"
                return core.compile_model(str(model_path), "CPU", fallback_config)
            raise


def run_compiled(compiled, inputs):
    result = compiled(inputs)
    return [result[compiled.output(index)] for index in range(len(compiled.outputs))]


def run_request(request, compiled, inputs, ov_profiler=None, profile_label: str | None = None):
    result = request.infer(inputs)
    if ov_profiler is not None and profile_label:
        ov_profiler.add_request(profile_label, request)
    return [result[compiled.output(index)] for index in range(len(compiled.outputs))]


class Timings:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.values = {}

    def add(self, name: str, elapsed: float) -> None:
        if self.enabled:
            self.values[name] = self.values.get(name, 0.0) + elapsed

    def print(self, generated_tokens: int) -> None:
        if not self.enabled:
            return
        print("profile:", flush=True)
        for name in sorted(self.values):
            print(f"  {name}: {self.values[name]:.4f}s", flush=True)
        generation = self.values.get("fused_step", 0.0) or (
            self.values.get("talker", 0.0) + self.values.get("subcode", 0.0)
        )
        if generated_tokens and generation:
            print(f"  generation_tokens_per_second: {generated_tokens / generation:.2f}", flush=True)

    def snapshot(self, generated_tokens: int) -> dict:
        values = dict(sorted(self.values.items()))
        generation = values.get("fused_step", 0.0) or (values.get("talker", 0.0) + values.get("subcode", 0.0))
        if generated_tokens and generation:
            values["generation_tokens_per_second"] = generated_tokens / generation
        return values


def seconds_from_duration(value) -> float:
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    return float(value)


class OVProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.ops = {}

    def add_request(self, label: str, request) -> None:
        if not self.enabled:
            return
        for info in request.get_profiling_info():
            key = (label, info.node_name, info.node_type, info.exec_type)
            item = self.ops.setdefault(
                key,
                {
                    "label": label,
                    "node_name": info.node_name,
                    "node_type": info.node_type,
                    "exec_type": info.exec_type,
                    "real_time": 0.0,
                    "cpu_time": 0.0,
                    "count": 0,
                },
            )
            item["real_time"] += seconds_from_duration(info.real_time)
            item["cpu_time"] += seconds_from_duration(info.cpu_time)
            item["count"] += 1

    def top(self, limit: int = 30) -> list[dict]:
        if not self.enabled:
            return []
        return sorted(self.ops.values(), key=lambda item: item["real_time"], reverse=True)[:limit]

    def aggregate(self, field: str) -> list[dict]:
        if not self.enabled:
            return []
        totals = {}
        for item in self.ops.values():
            key = item[field]
            total = totals.setdefault(key, {"name": key, "real_time": 0.0, "cpu_time": 0.0, "count": 0})
            total["real_time"] += item["real_time"]
            total["cpu_time"] += item["cpu_time"]
            total["count"] += item["count"]
        return sorted(totals.values(), key=lambda item: item["real_time"], reverse=True)


def first_code_list(codes: np.ndarray) -> list[int]:
    return codes[:, 0].astype(np.int64).tolist()


def compare_code_tensors(candidate: np.ndarray, reference: np.ndarray) -> dict:
    min_tokens = min(int(candidate.shape[0]), int(reference.shape[0]))
    first_code_divergence = None
    full_code_divergence = None
    for index in range(min_tokens):
        if first_code_divergence is None and int(candidate[index, 0]) != int(reference[index, 0]):
            first_code_divergence = index
        if full_code_divergence is None and not np.array_equal(candidate[index], reference[index]):
            full_code_divergence = index
        if first_code_divergence is not None and full_code_divergence is not None:
            break
    return {
        "candidate_tokens": int(candidate.shape[0]),
        "reference_tokens": int(reference.shape[0]),
        "same_shape": tuple(candidate.shape) == tuple(reference.shape),
        "exact_match": tuple(candidate.shape) == tuple(reference.shape) and np.array_equal(candidate, reference),
        "first_code_divergence": first_code_divergence,
        "full_code_divergence": full_code_divergence,
        "candidate_first_codes": first_code_list(candidate),
        "reference_first_codes": first_code_list(reference),
    }


class OpenVINOQwen3TTS:
    def __init__(
        self,
        ir_dir: str,
        device: str,
        decoder_device: str | None = None,
        allow_cpu_fallback: bool = False,
        mode: str = "cache",
        cache_kernel: str = "exact",
        cache_step: str = "split",
        graph_variant: str = "fp16",
        codegen_unroll: str | int = "profile",
        codegen_schedule: str = "current",
        codegen_decode_unroll: str = "off",
        preferred_cache_bucket: int | str | None = 112,
        precision_hint: str = "f16",
        compile_config: dict | None = None,
        ov_cache_dir: str | Path | None = None,
        ov_cache_mode: str | None = "optimize_speed",
        disable_ov_cache: bool = False,
        native_codegen: str | None = None,
        native_pipeline: str | None = None,
        native_paged_kv: str | None = None,
        native_paged_kv_gqa: str | bool | None = None,
        native_paged_kv_split_subcode: str | bool | None = None,
        calibration_dir: str | None = None,
        calibration_limit: int = 64,
        profile: bool = False,
        ov_profile: bool = False,
        encoder_device: str | None = None,
    ):
        self.ir_dir = resolve_ir_dir(ir_dir, fallback_to_local_voice_design=True, warn=True)
        self.manifest = load_manifest(self.ir_dir)
        model_dir = Path(self.manifest["model_dir"])
        if not model_dir.is_absolute():
            model_dir = self.ir_dir / model_dir
        self.model_dir = str(model_dir)
        self.ids = self.manifest["ids"]
        self.num_code_groups = int(self.manifest["num_code_groups"])
        self.sample_rate = int(self.manifest["sample_rate"])
        self.decode_upsample_rate = int(self.manifest["decode_upsample_rate"])
        self.input_sample_rate = int(self.manifest.get("input_sample_rate", self.sample_rate))
        self.encode_downsample_rate = int(self.manifest.get("encode_downsample_rate", self.decode_upsample_rate))
        self.speaker_encoder_sample_rate = int(self.manifest.get("speaker_encoder_sample_rate", 24000))
        graphs = self.manifest["graphs"]
        stream_config = self.manifest.get("streaming_decoder", {})
        self.streaming_decoder_left_context = int(stream_config.get("left_context_frames", 25))
        self.default_chunk_strategy = self._normalize_chunk_strategy_name(
            stream_config.get("default_strategy") or "low_latency"
        )
        self.streaming_decoder_strategies = self._load_streaming_decoder_strategies(stream_config)
        self.streaming_decoder_graphs_by_context = self._load_streaming_decoder_graphs(stream_config, graphs)
        self.streaming_decoder_graphs = self.streaming_decoder_graphs_by_context.get(self.streaming_decoder_left_context, {})
        self.streaming_decoders = {}
        self.streaming_decoder_requests = {}
        self.last_stream_decode_info = {}
        self.stream_pipeline_decode = True
        self.native_codegen_override = None if native_codegen is None else str(native_codegen).strip().lower()
        self.native_pipeline_override = None if native_pipeline is None else str(native_pipeline).strip().lower()
        self.native_paged_kv_override = None if native_paged_kv is None else str(native_paged_kv).strip().lower()
        self.native_paged_kv_gqa_override = (
            None if native_paged_kv_gqa is None else str(native_paged_kv_gqa).strip().lower()
        )
        self.native_paged_kv_split_subcode_override = (
            None if native_paged_kv_split_subcode is None else str(native_paged_kv_split_subcode).strip().lower()
        )
        self.paged_kv_enabled = False
        self.paged_kv_backend = "stateful_bucket"
        self.paged_kv_unavailable_reason = (
            "current exported Qwen3-TTS IR uses OpenVINO ReadValue/Assign stateful KV "
            "instead of GenAI key_cache/value_cache/block_indices inputs"
        )
        if mode == "fastest":
            fastest = fastest_runtime_defaults()
            mode = fastest["mode"]
            cache_kernel = fastest["cache_kernel"]
            cache_step = fastest["cache_step"]
            graph_variant = fastest["graph_variant"]
            codegen_unroll = fastest["codegen_unroll"]
            codegen_schedule = fastest["codegen_schedule"]
            codegen_decode_unroll = fastest["codegen_decode_unroll"]
            preferred_cache_bucket = fastest["preferred_cache_bucket"]
            os.environ["QWEN3_TTS_OV_NATIVE_PIPELINE"] = "require"
            os.environ["QWEN3_TTS_OV_NATIVE_BUFFER_REUSE"] = "1" if fastest["native_buffer_reuse"] == "on" else "0"
            os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV"] = fastest["native_paged_kv"]
            os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA"] = (
                "1" if fastest["native_paged_kv_gqa"] == "on" else "0"
            )
            os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_PRECISION"] = fastest["native_paged_kv_precision"]
            os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE"] = str(fastest["native_paged_kv_block_size"])
            os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE"] = (
                "1" if fastest["native_paged_kv_split_subcode"] == "on" else "0"
            )
            os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_SCORE_AGGREGATION"] = (
                "1" if fastest["native_paged_kv_score_aggregation"] == "on" else "0"
            )
            os.environ["QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE"] = fastest["native_codegen_device"]
        self.requested_mode = mode
        mode, cache_kernel, cache_step, graph_variant = effective_runtime_options(
            mode,
            cache_kernel,
            cache_step,
            graph_variant,
        )
        self.graph_variant = graph_variant
        self.codegen_unroll = effective_codegen_unroll(self.requested_mode, self.graph_variant, codegen_unroll)
        self.codegen_schedule = normalize_codegen_schedule(codegen_schedule)
        self.codegen_schedule_unrolls = scheduled_codegen_unrolls(self.codegen_schedule, self.codegen_unroll)
        self.codegen_decode_unroll = str(codegen_decode_unroll or "off").strip().lower().replace("_", "-")
        if self.codegen_decode_unroll not in {"off", "auto", "on"}:
            raise ValueError("codegen_decode_unroll must be one of off, auto, on")
        self.codegen_unroll_fallback = False
        self.preferred_cache_bucket = normalize_preferred_cache_bucket(preferred_cache_bucket)
        self.variant_graphs = self._load_graph_variant(graph_variant)
        self.cache_kernel = cache_kernel or self.manifest.get("default_cache_kernel", "exact")
        self.cache_step = cache_step or self.manifest.get("default_cache_step", "fused")
        if self.cache_step not in {"split", "fused"}:
            raise ValueError(f"unsupported cache_step={self.cache_step!r}")
        self.cache_bucket_graphs = self._load_cache_bucket_graphs(graphs, self.cache_kernel, self.variant_graphs)
        self.fused_cache_bucket_graphs = self._load_fused_cache_bucket_graphs(graphs, self.cache_kernel, self.variant_graphs)
        self.fused_cache_unroll_bucket_graphs_by_step = self._load_fused_cache_unroll_bucket_graphs_by_step(
            graphs,
            self.cache_kernel,
            self.variant_graphs,
        )
        self.fused_cache_unroll_norepeat_bucket_graphs_by_step = self._load_fused_cache_decode_unroll_bucket_graphs_by_step(
            graphs,
            self.cache_kernel,
            self.variant_graphs,
            "fused_cache_step_unroll_norepeat_buckets",
        )
        self.fused_cache_unroll_bucket_graphs = self.fused_cache_unroll_bucket_graphs_by_step.get(self.codegen_unroll, {})
        self.fused_cache_decode_unroll_bucket_graphs_by_step = self._load_fused_cache_decode_unroll_bucket_graphs_by_step(
            graphs,
            self.cache_kernel,
            self.variant_graphs,
            "fused_cache_decode_unroll_buckets",
        )
        self.fused_cache_decode_unroll_stateful_mask_bucket_graphs_by_step = self._load_fused_cache_decode_unroll_bucket_graphs_by_step(
            graphs,
            self.cache_kernel,
            self.variant_graphs,
            "fused_cache_decode_unroll_stateful_mask_buckets",
        )
        self.fused_cache_decode_unroll_norepeat_bucket_graphs_by_step = self._load_fused_cache_decode_unroll_bucket_graphs_by_step(
            graphs,
            self.cache_kernel,
            self.variant_graphs,
            "fused_cache_decode_unroll_norepeat_buckets",
        )
        self.fused_cache_decode_unroll_bucket_graphs = self.fused_cache_decode_unroll_bucket_graphs_by_step.get(self.codegen_unroll, {})
        self.fused_cache_decode_unroll_stateful_mask_bucket_graphs = (
            self.fused_cache_decode_unroll_stateful_mask_bucket_graphs_by_step.get(self.codegen_unroll, {})
        )
        self.fused_cache_decode_unroll_norepeat_bucket_graphs = (
            self.fused_cache_decode_unroll_norepeat_bucket_graphs_by_step.get(self.codegen_unroll, {})
        )
        self.max_cache_len = max(self.cache_bucket_graphs) if self.cache_bucket_graphs else 0
        self.mode = mode
        self.device = device
        self.decoder_device = decoder_device or device
        self.encoder_device = encoder_device
        self.speech_encoder_device = encoder_device or self.decoder_device
        self.speaker_encoder_device = encoder_device or self.device
        self.allow_cpu_fallback = allow_cpu_fallback
        self.precision_hint = precision_hint
        self.disable_ov_cache = bool(disable_ov_cache)
        self.ov_cache_mode = normalize_ov_cache_mode(ov_cache_mode)
        self.compile_config = merge_compile_config_with_cache_mode(
            compile_config,
            ov_cache_mode=self.ov_cache_mode,
            disable_ov_cache=self.disable_ov_cache,
        )
        self.cache_dir = resolve_ov_cache_dir(
            self.ir_dir,
            self.manifest,
            device=self.device,
            decoder_device=self.decoder_device,
            mode=self.mode,
            cache_kernel=self.cache_kernel,
            cache_step=self.cache_step,
            graph_variant=self.graph_variant,
            codegen_unroll=self.codegen_unroll,
            codegen_schedule=self.codegen_schedule,
            precision_hint=self.precision_hint,
            compile_config=self.compile_config,
            ov_cache_dir=ov_cache_dir,
            disable_ov_cache=self.disable_ov_cache,
        )
        self.calibration_dir = Path(calibration_dir) if calibration_dir else None
        self.calibration_limit = int(calibration_limit)
        self.calibration_counts = {}
        if self.calibration_dir is not None:
            self.calibration_dir.mkdir(parents=True, exist_ok=True)
        native_pipeline_required = self._native_pipeline_mode() == "require"
        if self.mode == "cache" and not self.cache_bucket_graphs and not native_pipeline_required:
            raise ValueError(f"cache mode requested, but manifest has no {self.cache_kernel!r} stateful talker graph")
        if (
            self.mode == "cache"
            and self.cache_step == "fused"
            and not self.fused_cache_bucket_graphs
            and not native_pipeline_required
        ):
            raise ValueError(f"cache_step=fused requested, but manifest has no {self.cache_kernel!r} fused cache graph")
        if self.mode == "fused-no-cache" and "fused_no_cache_step" not in self.manifest.get("graphs", {}):
            raise ValueError("fused-no-cache mode requested, but manifest has no fused_no_cache_step graph")
        self.timings = Timings(profile)
        self.ov_profiler = OVProfiler(ov_profile)
        self.tokenizer = Qwen2BPETokenizer(self.model_dir)

        self.core = ov.Core()
        print(f"OpenVINO available devices: {self.core.available_devices}", flush=True)
        started = time.time()
        self.text_embedding = compile_model(
            self.core, self.ir_dir / graphs["text_embedding"], device, self.cache_dir, allow_cpu_fallback, ov_profile, self.precision_hint, self.compile_config
        )
        self.text_embedding_request = self.text_embedding.create_infer_request()
        self.codec_embedding = compile_model(
            self.core, self.ir_dir / graphs["codec_embedding"], device, self.cache_dir, allow_cpu_fallback, ov_profile, self.precision_hint, self.compile_config
        )
        self.codec_embedding_request = self.codec_embedding.create_infer_request()
        self.talker = None
        self.talker_request = None
        self.talker_stateful_by_bucket = {}
        self.talker_request_by_bucket = {}
        self.fused_cache_step_by_bucket = {}
        self.fused_cache_request_by_bucket = {}
        self.fused_cache_unroll_step_by_bucket = {}
        self.fused_cache_unroll_request_by_bucket = {}
        self.fused_cache_decode_unroll_step_by_bucket = {}
        self.fused_cache_decode_unroll_request_by_bucket = {}
        self.native_codegen_runners = {}
        self.native_audio_runners = {}
        self.fused_step = None
        self.fused_request = None
        self.last_codegen_info = {}
        compile_python_codegen_core = not native_pipeline_required
        if compile_python_codegen_core and self.mode == "fused-no-cache":
            self.fused_step = compile_model(
                self.core, self.ir_dir / self.graph_name(graphs, "fused_no_cache_step"), device, self.cache_dir, allow_cpu_fallback, ov_profile, self.precision_hint, self.compile_config
            )
            self.fused_request = self.fused_step.create_infer_request()
        elif compile_python_codegen_core and self.mode != "cache":
            self.talker = compile_model(
                self.core,
                self.ir_dir / self.graph_name(graphs, "talker"),
                device,
                self.cache_dir,
                allow_cpu_fallback,
                ov_profile,
                self.precision_hint,
                self.compile_config,
            )
            self.talker_request = self.talker.create_infer_request()
        self.subcode_greedy = None
        self.subcode_request = None
        if (
            compile_python_codegen_core
            and self.mode != "fused-no-cache"
            and not (self.mode == "cache" and self.cache_step == "fused")
        ):
            subcode_graph = self.graph_name(graphs, "subcode_greedy")
            self.subcode_graph_name = subcode_graph
            self.subcode_greedy = compile_model(
                self.core, self.ir_dir / subcode_graph, device, self.cache_dir, allow_cpu_fallback, ov_profile, self.precision_hint, self.compile_config
            )
            self.subcode_request = self.subcode_greedy.create_infer_request()
        self.code_frame_embedding = None
        self.code_frame_embedding_request = None
        if "code_frame_embedding" in graphs:
            self.code_frame_embedding = compile_model(
                self.core,
                self.ir_dir / graphs["code_frame_embedding"],
                device,
                self.cache_dir,
                allow_cpu_fallback,
                ov_profile,
                self.precision_hint,
                self.compile_config,
            )
            self.code_frame_embedding_request = self.code_frame_embedding.create_infer_request()
        self.speech_encoder = None
        self.speech_encoder_request = None
        if "speech_encoder" in graphs:
            self.speech_encoder = compile_model(
                self.core,
                self.ir_dir / graphs["speech_encoder"],
                self.speech_encoder_device,
                self.cache_dir,
                allow_cpu_fallback,
                ov_profile,
                self.precision_hint,
                self.compile_config,
            )
            self.speech_encoder_request = self.speech_encoder.create_infer_request()
        self.speaker_encoder = None
        self.speaker_encoder_request = None
        if "speaker_encoder" in graphs:
            self.speaker_encoder = compile_model(
                self.core,
                self.ir_dir / graphs["speaker_encoder"],
                self.speaker_encoder_device,
                self.cache_dir,
                allow_cpu_fallback,
                ov_profile,
                self.precision_hint,
                self.compile_config,
            )
            self.speaker_encoder_request = self.speaker_encoder.create_infer_request()
        self.decoder_graphs = {int(k): v for k, v in graphs["speech_decoder"].items()}
        self.decoders = {}
        if not allow_cpu_fallback and device == "GPU":
            gpu_models = [
                self.text_embedding,
                self.codec_embedding,
                self.fused_step if self.mode == "fused-no-cache" else self.talker,
            ]
            gpu_models = [compiled for compiled in gpu_models if compiled is not None]
            if self.subcode_greedy is not None:
                gpu_models.append(self.subcode_greedy)
            self.assert_gpu_execution(gpu_models)
        cache_suffix = f" {self.cache_kernel}/{self.cache_step}" if self.mode == "cache" else ""
        variant_suffix = f" variant={self.graph_variant}" if self.graph_variant != "fp16" else ""
        unroll_suffix = f" unroll={self.codegen_unroll}" if self.codegen_unroll > 1 else ""
        schedule_suffix = f" schedule={self.codegen_schedule}" if self.codegen_schedule != "current" else ""
        decode_unroll_suffix = f" decode_unroll={self.codegen_decode_unroll}" if self.codegen_unroll > 1 else ""
        bucket_suffix = f" preferred_bucket={self.preferred_cache_bucket}" if self.preferred_cache_bucket else ""
        config_suffix = f" config={self.compile_config}" if self.compile_config else ""
        print(
            f"compiled {self.requested_mode}{cache_suffix}{variant_suffix}{unroll_suffix}"
            f"{schedule_suffix}{decode_unroll_suffix}{bucket_suffix}{config_suffix} core graphs on {device} in {time.time() - started:.1f}s",
            flush=True,
        )

    def dump_calibration(self, name: str, inputs) -> None:
        if self.calibration_dir is None:
            return
        count = self.calibration_counts.get(name, 0)
        if count >= self.calibration_limit:
            return
        self.calibration_counts[name] = count + 1
        arrays = {f"input_{index}": np.asarray(item) for index, item in enumerate(inputs)}
        np.savez_compressed(self.calibration_dir / f"{name}_{count:04d}.npz", **arrays)

    def _load_graph_variant(self, graph_variant: str) -> dict:
        if graph_variant in {"", "fp16", None}:
            return {}
        variants = self.manifest.get("graph_variants", {})
        if graph_variant not in variants:
            available = ", ".join(sorted(variants)) or "none"
            raise ValueError(missing_graph_variant_message(graph_variant, available))
        return variants[graph_variant].get("graphs", {})

    def graph_name(self, graphs, name: str) -> str:
        return self.variant_graphs.get(name, graphs[name])

    def _load_bucket_graphs(self, bucket_section, kernel: str):
        if not bucket_section:
            return {}
        if kernel in bucket_section and isinstance(bucket_section[kernel], dict):
            return {int(length): graph for length, graph in bucket_section[kernel].items() if graph}
        if all(str(key).isdigit() for key in bucket_section):
            return {int(length): graph for length, graph in bucket_section.items() if graph}
        return {}

    def _load_cache_bucket_graphs(self, graphs, kernel: str, variant_graphs=None):
        bucket_graphs = self._load_bucket_graphs(graphs.get("talker_stateful_buckets", {}), kernel)
        bucket_graphs.update(self._load_bucket_graphs((variant_graphs or {}).get("talker_stateful_buckets", {}), kernel))
        legacy_graph = graphs.get("talker_stateful")
        legacy_len = int(self.manifest.get("max_cache_len", 0) or 0)
        if legacy_graph and legacy_len:
            bucket_graphs.setdefault(legacy_len, legacy_graph)
        return dict(sorted(bucket_graphs.items()))

    def _load_fused_cache_bucket_graphs(self, graphs, kernel: str, variant_graphs=None):
        bucket_graphs = self._load_bucket_graphs(graphs.get("fused_cache_step_buckets", {}), kernel)
        bucket_graphs.update(self._load_bucket_graphs((variant_graphs or {}).get("fused_cache_step_buckets", {}), kernel))
        return dict(sorted(bucket_graphs.items()))

    def _load_unroll_bucket_graphs(self, bucket_section, kernel: str, unroll_steps: int):
        if not bucket_section:
            return {}
        by_kernel = bucket_section.get(kernel, {}) if isinstance(bucket_section, dict) else {}
        if not isinstance(by_kernel, dict):
            return {}
        by_unroll = by_kernel.get(str(unroll_steps), {})
        if isinstance(by_unroll, dict):
            return {int(length): graph for length, graph in by_unroll.items() if graph}
        return {}

    def _load_fused_cache_unroll_bucket_graphs_by_step(self, graphs, kernel: str, variant_graphs=None):
        section = graphs.get("fused_cache_step_unroll_buckets", {})
        variant_section = (variant_graphs or {}).get("fused_cache_step_unroll_buckets", {})
        return self._load_unroll_bucket_graphs_by_step(section, variant_section, kernel)

    def _load_fused_cache_decode_unroll_bucket_graphs_by_step(
        self,
        graphs,
        kernel: str,
        variant_graphs=None,
        section_name: str = "fused_cache_decode_unroll_buckets",
    ):
        section = graphs.get(section_name, {})
        variant_section = (variant_graphs or {}).get(section_name, {})
        return self._load_unroll_bucket_graphs_by_step(section, variant_section, kernel)

    def _load_unroll_bucket_graphs_by_step(self, section, variant_section, kernel: str):
        steps = set()
        for item in (section, variant_section):
            by_kernel = item.get(kernel, {}) if isinstance(item, dict) else {}
            if isinstance(by_kernel, dict):
                steps.update(int(step) for step in by_kernel if str(step).isdigit())
        result = {}
        for step in sorted(steps):
            bucket_graphs = self._load_unroll_bucket_graphs(section, kernel, step)
            bucket_graphs.update(self._load_unroll_bucket_graphs(variant_section, kernel, step))
            if bucket_graphs:
                result[int(step)] = dict(sorted(bucket_graphs.items()))
        return result

    def _load_streaming_decoder_graphs(self, stream_config: dict, graphs: dict) -> dict[int, dict[int, str]]:
        contexts = stream_config.get("contexts")
        if contexts:
            return {
                int(context): {int(chunk): graph for chunk, graph in chunk_graphs.items() if graph}
                for context, chunk_graphs in contexts.items()
                if chunk_graphs
            }

        stream_graphs = stream_config.get("graphs") or graphs.get("streaming_decoder", {})
        if not stream_graphs:
            return {}

        if all(str(key).isdigit() and isinstance(value, dict) for key, value in stream_graphs.items()):
            return {
                int(context): {int(chunk): graph for chunk, graph in chunk_graphs.items() if graph}
                for context, chunk_graphs in stream_graphs.items()
                if chunk_graphs
            }

        left_context = int(stream_config.get("left_context_frames", 25))
        return {left_context: {int(chunk): graph for chunk, graph in stream_graphs.items() if graph}}

    @staticmethod
    def _normalize_chunk_strategy_name(strategy: str | None) -> str:
        return str(strategy or "low_latency").strip().replace("-", "_").lower()

    def _load_streaming_decoder_strategies(self, stream_config: dict) -> dict[str, dict[str, int]]:
        strategies = {name: dict(config) for name, config in DEFAULT_STREAM_CHUNK_STRATEGIES.items()}
        for name, config in (stream_config.get("strategies") or {}).items():
            normalized = self._normalize_chunk_strategy_name(name)
            if not isinstance(config, dict):
                continue
            merged = dict(strategies.get(normalized, {}))
            merged.update(config)
            strategies[normalized] = merged
        for name, config in strategies.items():
            chunk_frames = int(config.get("chunk_frames", 12))
            strategies[name] = {
                "initial_chunk_frames": int(config.get("initial_chunk_frames", chunk_frames)),
                "chunk_frames": chunk_frames,
                "left_context_frames": int(config.get("left_context_frames", self.streaming_decoder_left_context)),
            }
        if self.default_chunk_strategy not in strategies:
            self.default_chunk_strategy = "low_latency"
        return strategies

    def _resolve_stream_chunk_config(
        self,
        chunk_strategy: str | None = None,
        chunk_frames: int | None = None,
        initial_chunk_frames: int | None = None,
        left_context_frames: int | None = None,
    ) -> dict:
        strategy = self._normalize_chunk_strategy_name(chunk_strategy or self.default_chunk_strategy)
        if strategy not in self.streaming_decoder_strategies:
            supported = ", ".join(sorted(self.streaming_decoder_strategies))
            raise ValueError(f"unsupported chunk_strategy={chunk_strategy!r}; supported strategies: {supported}")

        explicit_fixed_chunk = chunk_strategy is None and chunk_frames is not None and initial_chunk_frames is None
        config = dict(self.streaming_decoder_strategies[strategy])
        if chunk_frames is not None:
            config["chunk_frames"] = int(chunk_frames)
        if initial_chunk_frames is not None:
            config["initial_chunk_frames"] = int(initial_chunk_frames)
        elif explicit_fixed_chunk:
            config["initial_chunk_frames"] = int(chunk_frames)
        if left_context_frames is not None:
            config["left_context_frames"] = int(left_context_frames)

        config["strategy"] = strategy
        config["initial_chunk_frames"] = int(config["initial_chunk_frames"])
        config["chunk_frames"] = int(config["chunk_frames"])
        config["left_context_frames"] = int(config["left_context_frames"])
        if config["initial_chunk_frames"] <= 0:
            raise ValueError("initial_chunk_frames must be positive")
        if config["chunk_frames"] <= 0:
            raise ValueError("chunk_frames must be positive")
        if config["left_context_frames"] < 0:
            raise ValueError("left_context_frames must be non-negative")
        return config

    def assert_gpu_execution(self, compiled_models):
        for compiled in compiled_models:
            devices = compiled.get_property("EXECUTION_DEVICES")
            if devices != ["GPU.0"]:
                raise RuntimeError(f"expected strict GPU execution, got {devices}")

    def select_cache_bucket(self, required_len: int) -> int:
        bucket = next((length for length in self.cache_bucket_graphs if length >= required_len), None)
        if bucket is None:
            available = ", ".join(str(length) for length in self.cache_bucket_graphs)
            raise ValueError(
                f"prompt_len + max_new_tokens requires cache length {required_len}, "
                f"but available cache buckets are: {available}"
            )
        return bucket

    @staticmethod
    def select_runtime_bucket(
        available_buckets: dict[int, str],
        required_len: int,
        compiled_buckets=(),
        preferred_min_bucket: int | None = None,
    ) -> int | None:
        compiled_candidates = sorted(
            bucket for bucket in compiled_buckets if bucket in available_buckets and bucket >= required_len
        )
        if compiled_candidates:
            return compiled_candidates[0]
        if preferred_min_bucket is not None and required_len <= preferred_min_bucket and preferred_min_bucket in available_buckets:
            return preferred_min_bucket
        return next((length for length in sorted(available_buckets) if length >= required_len), None)

    def get_talker_stateful(self, required_len: int):
        bucket = self.select_cache_bucket(required_len)
        if bucket not in self.talker_stateful_by_bucket:
            started = time.time()
            graph = self.cache_bucket_graphs[bucket]
            compiled = compile_model(
                self.core,
                self.ir_dir / graph,
                self.device,
                self.cache_dir,
                self.allow_cpu_fallback,
                self.ov_profiler.enabled,
                self.precision_hint,
                self.compile_config,
            )
            if not self.allow_cpu_fallback and self.device == "GPU":
                self.assert_gpu_execution([compiled])
            self.talker_stateful_by_bucket[bucket] = compiled
            self.talker_request_by_bucket[bucket] = compiled.create_infer_request()
            print(f"compiled stateful talker cache bucket {bucket} on {self.device} in {time.time() - started:.1f}s", flush=True)
        return bucket, self.talker_stateful_by_bucket[bucket], self.talker_request_by_bucket[bucket]

    def get_fused_cache_step(self, required_len: int):
        bucket = self.select_runtime_bucket(
            self.fused_cache_bucket_graphs,
            required_len,
            self.fused_cache_step_by_bucket.keys(),
        )
        if bucket is None:
            available = ", ".join(str(length) for length in self.fused_cache_bucket_graphs)
            raise ValueError(
                f"prompt_len + max_new_tokens requires fused cache length {required_len}, "
                f"but available fused cache buckets are: {available}"
            )
        if bucket not in self.fused_cache_step_by_bucket:
            started = time.time()
            graph = self.fused_cache_bucket_graphs[bucket]
            compiled = compile_model(
                self.core,
                self.ir_dir / graph,
                self.device,
                self.cache_dir,
                self.allow_cpu_fallback,
                self.ov_profiler.enabled,
                self.precision_hint,
                self.compile_config,
            )
            if not self.allow_cpu_fallback and self.device == "GPU":
                self.assert_gpu_execution([compiled])
            self.fused_cache_step_by_bucket[bucket] = compiled
            self.fused_cache_request_by_bucket[bucket] = compiled.create_infer_request()
            print(f"compiled fused cache bucket {bucket} on {self.device} in {time.time() - started:.1f}s", flush=True)
        return bucket, self.fused_cache_step_by_bucket[bucket], self.fused_cache_request_by_bucket[bucket]

    def get_fused_cache_unroll_step(
        self,
        required_len: int,
        unroll_steps: int | None = None,
        preferred_min_bucket: int | None = 112,
    ):
        unroll_steps = int(unroll_steps or self.codegen_unroll)
        bucket_graphs = self.fused_cache_unroll_bucket_graphs_by_step.get(unroll_steps, {})
        compiled_buckets = [
            bucket
            for unroll, bucket in self.fused_cache_unroll_step_by_bucket
            if unroll == unroll_steps
        ]
        bucket = self.select_runtime_bucket(
            bucket_graphs,
            required_len,
            compiled_buckets,
            preferred_min_bucket=preferred_min_bucket,
        )
        if bucket is None:
            available = ", ".join(str(length) for length in bucket_graphs)
            raise ValueError(
                f"prompt_len + max_new_tokens requires fused unroll cache length {required_len}, "
                f"but available fused unroll{unroll_steps} cache buckets are: {available}"
            )
        key = (unroll_steps, bucket)
        if key not in self.fused_cache_unroll_step_by_bucket:
            started = time.time()
            graph = bucket_graphs[bucket]
            compiled = compile_model(
                self.core,
                self.ir_dir / graph,
                self.device,
                self.cache_dir,
                self.allow_cpu_fallback,
                self.ov_profiler.enabled,
                self.precision_hint,
                self.compile_config,
            )
            if not self.allow_cpu_fallback and self.device == "GPU":
                self.assert_gpu_execution([compiled])
            self.fused_cache_unroll_step_by_bucket[key] = compiled
            self.fused_cache_unroll_request_by_bucket[key] = compiled.create_infer_request()
            print(
                f"compiled fused cache unroll{unroll_steps} bucket {bucket} on {self.device} "
                f"in {time.time() - started:.1f}s",
                flush=True,
            )
        return bucket, self.fused_cache_unroll_step_by_bucket[key], self.fused_cache_unroll_request_by_bucket[key]

    def get_fused_cache_decode_unroll_step(
        self,
        required_len: int,
        unroll_steps: int | None = None,
        preferred_min_bucket: int | None = 112,
    ):
        unroll_steps = int(unroll_steps or self.codegen_unroll)
        stateful_graphs = self.fused_cache_decode_unroll_stateful_mask_bucket_graphs_by_step.get(unroll_steps, {})
        plain_graphs = self.fused_cache_decode_unroll_bucket_graphs_by_step.get(unroll_steps, {})
        stateful_mask = bool(stateful_graphs)
        bucket_graphs = (
            stateful_graphs
            if stateful_mask
            else plain_graphs
        )
        compiled_buckets = [
            bucket
            for unroll, bucket, compiled_stateful_mask in self.fused_cache_decode_unroll_step_by_bucket
            if unroll == unroll_steps and compiled_stateful_mask == stateful_mask
        ]
        bucket = self.select_runtime_bucket(bucket_graphs, required_len, compiled_buckets, preferred_min_bucket=preferred_min_bucket)
        if bucket is None:
            available = ", ".join(str(length) for length in bucket_graphs)
            raise ValueError(
                f"prompt_len + max_new_tokens requires fused decode unroll cache length {required_len}, "
                f"but available fused decode unroll buckets are: {available}"
            )
        key = (unroll_steps, bucket, stateful_mask)
        if key not in self.fused_cache_decode_unroll_step_by_bucket:
            started = time.time()
            graph = bucket_graphs[bucket]
            compiled = compile_model(
                self.core,
                self.ir_dir / graph,
                self.device,
                self.cache_dir,
                self.allow_cpu_fallback,
                self.ov_profiler.enabled,
                self.precision_hint,
                self.compile_config,
            )
            if not self.allow_cpu_fallback and self.device == "GPU":
                self.assert_gpu_execution([compiled])
            self.fused_cache_decode_unroll_step_by_bucket[key] = compiled
            self.fused_cache_decode_unroll_request_by_bucket[key] = compiled.create_infer_request()
            mask_suffix = " statefulmask" if stateful_mask else ""
            print(
                f"compiled fused cache decode unroll{unroll_steps}{mask_suffix} bucket {bucket} on {self.device} "
                f"in {time.time() - started:.1f}s",
                flush=True,
            )
        return (
            bucket,
            self.fused_cache_decode_unroll_step_by_bucket[key],
            self.fused_cache_decode_unroll_request_by_bucket[key],
            stateful_mask,
        )

    def codegen_unroll_available(self, unroll_steps: int) -> bool:
        return bool(self.fused_cache_unroll_bucket_graphs_by_step.get(int(unroll_steps), {}))

    def select_codegen_unroll_for_step(self, generated_count: int) -> int:
        if self.codegen_schedule == "ll-v2":
            preferred = [4, 8, 6, 12] if generated_count < 8 else [12, 8, 6, 4]
        elif self.codegen_schedule == "balanced-v2":
            preferred = [8, 4, 6, 12] if generated_count < 8 else [12, 8, 6, 4]
        else:
            preferred = [self.codegen_unroll]
        for unroll_steps in preferred:
            if unroll_steps > 1 and self.codegen_unroll_available(unroll_steps):
                return int(unroll_steps)
        if self.codegen_unroll > 1 and self.codegen_unroll_available(self.codegen_unroll):
            return int(self.codegen_unroll)
        return 1

    @staticmethod
    def unroll_required_cache_len(prompt_len: int, max_new_tokens: int, unroll_steps: int) -> int:
        if unroll_steps <= 1:
            return prompt_len + max_new_tokens
        batches = (max_new_tokens + unroll_steps - 1) // unroll_steps
        return prompt_len + batches * unroll_steps - 1

    @staticmethod
    def _copy_matching_states(source_request, target_request) -> None:
        source_states = {state.name: state for state in source_request.query_state()}
        for target_state in target_request.query_state():
            source_state = source_states.get(target_state.name)
            if source_state is not None:
                target_state.state = source_state.state

    @staticmethod
    def _set_request_state(request, name: str, value: np.ndarray) -> bool:
        for state in request.query_state():
            if state.name == name or state.name.startswith(name):
                state.state = ov.Tensor(np.asarray(value))
                return True
        return False

    def _native_codegen_mode(self) -> str:
        if self.native_codegen_override is not None:
            return self.native_codegen_override
        return str(os.environ.get("QWEN3_TTS_OV_NATIVE_CODEGEN", "")).strip().lower()

    def _native_pipeline_mode(self) -> str:
        if self.native_pipeline_override is not None:
            return self.native_pipeline_override
        return str(os.environ.get("QWEN3_TTS_OV_NATIVE_PIPELINE", "")).strip().lower()

    def _native_paged_kv_mode(self) -> str:
        if self.native_paged_kv_override is not None:
            return self.native_paged_kv_override
        return str(os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV", "")).strip().lower()

    def _native_paged_kv_gqa_enabled(self) -> bool:
        value = (
            self.native_paged_kv_gqa_override
            if self.native_paged_kv_gqa_override is not None
            else str(os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA", "1")).strip().lower()
        )
        return value not in {"0", "false", "off", "no"}

    def _native_paged_kv_split_subcode_enabled(self) -> bool:
        if self.native_paged_kv_split_subcode_override is not None:
            return self.native_paged_kv_split_subcode_override in {"1", "true", "on", "yes", "require"}
        return env_flag_enabled(os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE"))

    def _try_generate_codes_native_unroll4_statefulmask(
        self,
        sequence: np.ndarray,
        tts_pad_embed: np.ndarray,
        prompt_len: int,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        initial_unroll: int,
        unroll_required_len: int,
        decode_unroll_enabled: bool,
        decode_unroll_graph_available: bool,
        stream_batches: bool = False,
    ):
        native_mode = self._native_codegen_mode()
        if native_mode not in {"1", "true", "on", "require"}:
            return None
        require = native_mode == "require"
        try:
            if self.mode != "cache" or self.cache_step != "fused":
                raise RuntimeError("native codegen requires cache + fused mode")
            if initial_unroll <= 1 or self.codegen_schedule != "current":
                raise RuntimeError("native codegen currently requires codegen_unroll>1 and codegen_schedule=current")
            if not decode_unroll_enabled or not decode_unroll_graph_available:
                raise RuntimeError("native codegen requires codegen_decode_unroll=auto/on and a decode-unroll graph")
            no_repeat_codegen = abs(float(repetition_penalty) - 1.0) <= 1e-6
            if is_fastest_or_norepeat_mode(self.requested_mode) and not no_repeat_codegen:
                raise RuntimeError(f"{self.requested_mode} requires repetition_penalty=1.0")
            no_repeat_prefill_graphs = self.fused_cache_unroll_norepeat_bucket_graphs_by_step.get(initial_unroll, {})
            no_repeat_decode_graphs = self.fused_cache_decode_unroll_norepeat_bucket_graphs_by_step.get(initial_unroll, {})
            use_no_repeat_graphs = bool(no_repeat_codegen and no_repeat_prefill_graphs and no_repeat_decode_graphs)
            if is_fastest_or_norepeat_mode(self.requested_mode) and not use_no_repeat_graphs:
                raise RuntimeError(
                    f"{self.requested_mode} requires unroll{initial_unroll} no-repeat prefill and decode-unroll graphs"
                )
            prefill_graphs = (
                no_repeat_prefill_graphs
                if use_no_repeat_graphs
                else self.fused_cache_unroll_bucket_graphs_by_step.get(initial_unroll, {})
            )
            decode_graphs = (
                no_repeat_decode_graphs
                if use_no_repeat_graphs
                else self.fused_cache_decode_unroll_stateful_mask_bucket_graphs_by_step.get(initial_unroll, {})
            )
            if not prefill_graphs or not decode_graphs:
                raise RuntimeError(f"native codegen requires unroll{initial_unroll} prefill and decode-unroll graphs")
            bucket = self.select_runtime_bucket(
                prefill_graphs,
                unroll_required_len,
                (),
                preferred_min_bucket=self.preferred_cache_bucket,
            )
            if bucket is None:
                raise RuntimeError("no native prefill bucket can satisfy requested generation length")
            if bucket not in decode_graphs:
                raise RuntimeError(f"decode-unroll graph is missing for bucket {bucket}")

            prefill_graph = self.ir_dir / prefill_graphs[bucket]
            decode_graph = self.ir_dir / decode_graphs[bucket]
            cache_mode = str(self.compile_config.get("CACHE_MODE", "OPTIMIZE_SPEED"))
            cache_dir = None if self.disable_ov_cache else usable_ov_cache_dir(self.cache_dir)
            key = (
                str(prefill_graph),
                str(decode_graph),
                self.device,
                str(cache_dir or ""),
                cache_mode,
            )
            if key not in self.native_codegen_runners:
                from .native_codegen import NativeCodegenRunner

                started = time.time()
                self.native_codegen_runners[key] = NativeCodegenRunner(
                    prefill_graph=prefill_graph,
                    decode_graph=decode_graph,
                    device=self.device,
                    cache_dir=cache_dir,
                    cache_mode=cache_mode,
                )
                print(
                    f"compiled native GenAI-style codegen unroll{initial_unroll} bucket {bucket} on {self.device} "
                    f"in {time.time() - started:.1f}s",
                    flush=True,
                )
            runner = self.native_codegen_runners[key]
            self.codegen_unroll_fallback = False
            self.last_codegen_info = {
                "prompt_len": int(prompt_len),
                "required_cache_len": int(prompt_len + max_new_tokens),
                "unroll_required_cache_len": int(unroll_required_len),
                "selected_bucket": int(bucket),
                "preferred_cache_bucket": self.preferred_cache_bucket,
                "selected_codegen_graph": str(decode_graph.name),
                "selected_prefill_graph": str(prefill_graph.name),
                "codegen_graph_kind": "native_decode_unroll_norepeat" if use_no_repeat_graphs else "native_decode_unroll_statefulmask",
                "codegen_schedule": self.codegen_schedule,
                "scheduled_unrolls": list(self.codegen_schedule_unrolls),
                "active_codegen_unroll": int(initial_unroll),
                "codegen_decode_unroll": self.codegen_decode_unroll,
                "decode_unroll_graph_available": True,
                "decode_unroll_available": True,
                "decode_unroll_stateful_mask": not use_no_repeat_graphs,
                "codegen_no_repeat": bool(use_no_repeat_graphs),
                "native_codegen": True,
                "native_streaming_callbacks": bool(stream_batches),
            }
            if stream_batches:
                def _iter_native_batches():
                    emitted = 0
                    for batch in runner.iter_batches(
                        sequence=sequence,
                        tts_pad_embed=tts_pad_embed,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        repetition_penalty=repetition_penalty,
                        vocab_size=int(self.ids["vocab_size"]),
                        num_code_groups=self.num_code_groups,
                        eos_token_id=int(self.ids["codec_eos_token_id"]),
                    ):
                        batch = np.asarray(batch, dtype=np.int64)
                        for frame in batch:
                            emitted += 1
                            yield frame
                    elapsed_ms = float(getattr(runner, "last_stream_elapsed_ms", 0.0) or 0.0)
                    if elapsed_ms > 0:
                        self.timings.add("fused_step", elapsed_ms / 1000.0)
                    self.last_codegen_info.update(
                        {
                            "native_codegen_ms": elapsed_ms,
                            "native_emitted_frames": int(emitted),
                            "native_remote_embed": bool(getattr(runner, "last_remote_embed", False)),
                        }
                    )

                return _iter_native_batches()
            codes, elapsed_ms = runner.run(
                sequence=sequence,
                tts_pad_embed=tts_pad_embed,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                vocab_size=int(self.ids["vocab_size"]),
                num_code_groups=self.num_code_groups,
                eos_token_id=int(self.ids["codec_eos_token_id"]),
            )
            self.timings.add("fused_step", elapsed_ms / 1000.0)
            self.last_codegen_info.update(
                {
                    "native_codegen_ms": float(elapsed_ms),
                    "native_emitted_frames": int(codes.shape[0]),
                    "native_remote_embed": bool(getattr(runner, "last_remote_embed", False)),
                }
            )
            return np.asarray(codes, dtype=np.int64)
        except Exception as exc:
            if require:
                raise
            print(f"warning: native GenAI-style codegen unavailable; falling back to Python runtime: {exc}", flush=True)
            return None

    def _try_stream_native_audio_pipeline(
        self,
        *,
        text: str,
        instruct: str,
        language: str,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        max_prompt_tokens: int,
        chunk_frames: int | None = None,
        left_context_frames: int | None = None,
        initial_chunk_frames: int | None = None,
        chunk_strategy: str | None = None,
        speaker: str | None = None,
        voice_clone_prompt: VoiceClonePromptItem | None = None,
        ref_text: str | None = None,
        prefix_codes: np.ndarray | None = None,
        append_prefix_codes_to_prompt: bool = False,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ):
        native_mode = self._native_pipeline_mode()
        if native_mode not in {"1", "true", "on", "require"}:
            return None
        require = native_mode == "require"
        try:
            paged_kv_requested = self._native_paged_kv_mode() in {"1", "true", "on", "yes", "require"}
            if not paged_kv_requested and (self.mode != "cache" or self.cache_step != "fused"):
                raise RuntimeError("native audio pipeline requires cache + fused mode")
            initial_unroll = self.select_codegen_unroll_for_step(0)
            if not paged_kv_requested and self.codegen_schedule != "current":
                raise RuntimeError("native audio pipeline currently requires codegen_schedule=current")
            if not paged_kv_requested and initial_unroll <= 1:
                requested_unroll = self.codegen_unroll
                available_unrolls = sorted(
                    set(self.fused_cache_unroll_bucket_graphs_by_step)
                    | set(self.fused_cache_decode_unroll_bucket_graphs_by_step)
                    | set(self.fused_cache_decode_unroll_stateful_mask_bucket_graphs_by_step)
                    | set(self.fused_cache_unroll_norepeat_bucket_graphs_by_step)
                    | set(self.fused_cache_decode_unroll_norepeat_bucket_graphs_by_step)
                )
                available = ", ".join(str(item) for item in available_unrolls) or "none"
                raise RuntimeError(
                    f"native audio pipeline requires an available codegen_unroll>1 graph; "
                    f"requested {requested_unroll}, available unrolls: {available}"
                )
            decode_unroll_enabled = self.codegen_decode_unroll in {"auto", "on"} and self.codegen_schedule == "current"
            no_repeat_codegen = abs(float(repetition_penalty) - 1.0) <= 1e-6
            if is_fastest_or_norepeat_mode(self.requested_mode) and not no_repeat_codegen:
                raise RuntimeError(f"{self.requested_mode} requires repetition_penalty=1.0")
            decode_unroll_graph_available = bool(
                self.fused_cache_decode_unroll_bucket_graphs_by_step.get(initial_unroll, {})
                or self.fused_cache_decode_unroll_stateful_mask_bucket_graphs_by_step.get(initial_unroll, {})
                or self.fused_cache_decode_unroll_norepeat_bucket_graphs_by_step.get(initial_unroll, {})
            )
            if not paged_kv_requested and (not decode_unroll_enabled or not decode_unroll_graph_available):
                raise RuntimeError("native audio pipeline requires codegen_decode_unroll=auto/on and decode-unroll graphs")

            stream_config = self._resolve_stream_chunk_config(
                chunk_strategy=chunk_strategy,
                chunk_frames=chunk_frames,
                initial_chunk_frames=initial_chunk_frames,
                left_context_frames=left_context_frames,
            )
            first_context = 0
            first_chunk = int(stream_config["initial_chunk_frames"])
            steady_context = int(stream_config["left_context_frames"])
            steady_chunk = int(stream_config["chunk_frames"])
            try:
                first_decoder_graph = self.streaming_decoder_graphs_by_context[first_context][first_chunk]
                steady_decoder_graph = self.streaming_decoder_graphs_by_context[steady_context][steady_chunk]
            except KeyError as exc:
                raise RuntimeError(
                    f"native audio pipeline requires exact streaming decoder graphs c{first_context}_t{first_chunk} "
                    f"and c{steady_context}_t{steady_chunk}"
                ) from exc

            native_prompt_pipeline = (
                str(os.environ.get("QWEN3_TTS_OV_NATIVE_PROMPT", "")).strip().lower() in {"1", "true", "on", "yes"}
                and speaker is None
                and voice_clone_prompt is None
                and ref_text is None
                and prefix_codes is None
                and bool((self.manifest.get("tokenizer_ir") or {}).get("tokenizer"))
            )
            codec_prefill = None
            sequence = None
            tts_pad_embed = None
            if native_prompt_pipeline:
                input_ids = self.tokenizer.encode(build_assistant_text(text))
                instruct_ids = self.tokenizer.encode(build_instruct_text(instruct)) if instruct else []
                if len(input_ids) > max_prompt_tokens:
                    raise ValueError(f"text prompt has {len(input_ids)} tokens, max_prompt_tokens={max_prompt_tokens}")
                if len(instruct_ids) > max_prompt_tokens:
                    raise ValueError(f"instruct prompt has {len(instruct_ids)} tokens, max_prompt_tokens={max_prompt_tokens}")
                codec_prefill = self.language_codec_prefill(language, speaker=None)
                prompt_len = int(len(instruct_ids) + len(input_ids) + len(codec_prefill) - 3)
            else:
                sequence, tts_pad_embed = self.build_prompt(
                    text,
                    instruct,
                    language,
                    max_prompt_tokens,
                    speaker=speaker,
                    voice_clone_prompt=voice_clone_prompt,
                    ref_text=ref_text,
                )
                if append_prefix_codes_to_prompt:
                    sequence = self.append_prefix_code_frames_to_prompt(sequence, tts_pad_embed, prefix_codes)
            prompt_len = int(sequence.shape[1])
            unroll_required_len = self.unroll_required_cache_len(prompt_len, max_new_tokens, initial_unroll)
            if paged_kv_requested:
                paged_seed_graphs = dict((self.manifest.get("graphs", {}) or {}).get("paged_kv_seed", {}) or {})
                paged_seed_graphs.update((self.variant_graphs or {}).get("paged_kv_seed", {}) or {})
                paged_meta = self.manifest.get("paged_kv") or {}
                native_async_env = os.environ.get("QWEN3_TTS_OV_NATIVE_ASYNC_DECODE")
                effective_native_async_decode = (
                    env_flag_enabled(native_async_env)
                    if native_async_env is not None
                    else False
                )
                prefer_gqa = self._native_paged_kv_gqa_enabled()
                paged_split_subcode = self._native_paged_kv_split_subcode_enabled()
                if do_sample and not paged_split_subcode:
                    raise RuntimeError("native paged-KV sampling requires split-subcode mode")
                requested_paged_unroll = int(
                    os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_UNROLL")
                    or paged_meta.get("default_unroll")
                    or 1
                )
                paged_unroll_experimental = env_flag_enabled(
                    os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_EXPERIMENTAL_UNROLL")
                )
                requested_paged_unroll = effective_paged_kv_unroll(
                    requested_paged_unroll,
                    experimental_enabled=paged_unroll_experimental,
                )
                if paged_split_subcode:
                    requested_paged_unroll = 1
                subcode_attention = (
                    str(os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SUBCODE_ATTENTION") or "auto")
                    .strip()
                    .lower()
                    .replace("-", "_")
                )
                paged_seed_key, selected_subcode_attention, paged_subcode_attention_fallback = select_paged_kv_seed_key(
                    paged_seed_graphs,
                    prefer_gqa=prefer_gqa,
                    split_subcode=paged_split_subcode,
                    requested_unroll=requested_paged_unroll,
                    subcode_attention=subcode_attention,
                )
                paged_seed_graph = paged_seed_graphs.get(paged_seed_key)
                if not paged_seed_graph:
                    raise RuntimeError(
                        "native paged-KV pipeline requires graphs.paged_kv_seed.fused_cache_step[_gqa] "
                        "or talker_stateful[_gqa] for split-subcode mode; "
                        "export with --export-paged-kv-seed"
                )
                paged_split_subcode_graph = None
                if paged_split_subcode:
                    graphs_root = self.manifest["graphs"]
                    split_subcode_mode = (
                        str(os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE_MODE") or "cached")
                        .strip()
                        .lower()
                    )
                    if split_subcode_mode not in {"cached", "recompute", "cached_exact", "recompute_exact"}:
                        raise RuntimeError(
                            "QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE_MODE must be one of: "
                            "cached, recompute, cached_exact, recompute_exact"
                        )
                    if split_subcode_mode == "recompute_exact":
                        subcode_graph_name = (
                            (self.variant_graphs or {}).get("subcode_greedy_exact")
                            or graphs_root.get("subcode_greedy_exact")
                        )
                    elif split_subcode_mode == "cached_exact":
                        subcode_graph_name = (
                            (self.variant_graphs or {}).get("subcode_greedy_cached_exact")
                            or graphs_root.get("subcode_greedy_cached_exact")
                        )
                    elif split_subcode_mode == "recompute":
                        subcode_graph_name = (
                            (self.variant_graphs or {}).get("subcode_greedy")
                            or graphs_root.get("subcode_greedy")
                            or (self.variant_graphs or {}).get("subcode_greedy_cached")
                            or graphs_root.get("subcode_greedy_cached")
                        )
                    else:
                        subcode_graph_name = (
                            (self.variant_graphs or {}).get("subcode_greedy_cached")
                            or graphs_root.get("subcode_greedy_cached")
                            or (self.variant_graphs or {}).get("subcode_greedy")
                            or graphs_root.get("subcode_greedy")
                        )
                    if not subcode_graph_name and split_subcode_mode.endswith("_exact"):
                        fallback_mode = "cached" if split_subcode_mode.startswith("cached") else "recompute"
                        fallback_graph_name = (
                            (self.variant_graphs or {}).get("subcode_greedy_cached")
                            or graphs_root.get("subcode_greedy_cached")
                            or (self.variant_graphs or {}).get("subcode_greedy")
                            or graphs_root.get("subcode_greedy")
                        ) if fallback_mode == "cached" else (
                            (self.variant_graphs or {}).get("subcode_greedy")
                            or graphs_root.get("subcode_greedy")
                            or (self.variant_graphs or {}).get("subcode_greedy_cached")
                            or graphs_root.get("subcode_greedy_cached")
                        )
                        if fallback_graph_name:
                            split_subcode_mode = fallback_mode
                            subcode_graph_name = fallback_graph_name
                        else:
                            raise RuntimeError(
                                f"native paged-KV split-subcode mode {split_subcode_mode!r} requires "
                                f"graphs.subcode_greedy{'_cached' if split_subcode_mode.startswith('cached') else ''}_exact; "
                                "re-export with `--subcode-attention-kernels exact`"
                            )
                    if not subcode_graph_name:
                        raise RuntimeError(
                            "native paged-KV split-subcode mode requires graphs.subcode_greedy_cached "
                            "or graphs.subcode_greedy"
                    )
                    paged_split_subcode_graph = self.ir_dir / subcode_graph_name
                paged_kv_heads = int(
                    paged_meta.get("kv_cache_gqa_heads" if paged_kv_seed_uses_gqa(paged_seed_key) else "kv_cache_heads")
                    or (8 if paged_kv_seed_uses_gqa(paged_seed_key) else 16)
                )
                bucket = 0
                use_no_repeat_graphs = False
                prefill_graph = self.ir_dir / paged_seed_graph
                decode_graph = prefill_graph
            else:
                no_repeat_prefill_graphs = self.fused_cache_unroll_norepeat_bucket_graphs_by_step.get(initial_unroll, {})
                no_repeat_decode_graphs = self.fused_cache_decode_unroll_norepeat_bucket_graphs_by_step.get(initial_unroll, {})
                use_no_repeat_graphs = bool(no_repeat_codegen and no_repeat_prefill_graphs and no_repeat_decode_graphs)
                if is_fastest_or_norepeat_mode(self.requested_mode) and not use_no_repeat_graphs:
                    raise RuntimeError(
                        f"{self.requested_mode} requires unroll{initial_unroll} no-repeat prefill and decode-unroll graphs"
                    )
                prefill_graphs = (
                    no_repeat_prefill_graphs
                    if use_no_repeat_graphs
                    else self.fused_cache_unroll_bucket_graphs_by_step.get(initial_unroll, {})
                )
                decode_graphs = (
                    no_repeat_decode_graphs
                    if use_no_repeat_graphs
                    else self.fused_cache_decode_unroll_stateful_mask_bucket_graphs_by_step.get(initial_unroll, {})
                )
                if not prefill_graphs or not decode_graphs:
                    raise RuntimeError(
                        f"native audio pipeline requires unroll{initial_unroll} prefill and decode-unroll graphs"
                    )
                bucket = self.select_runtime_bucket(
                    prefill_graphs,
                    unroll_required_len,
                    (),
                    preferred_min_bucket=self.preferred_cache_bucket,
                )
                if bucket is None or bucket not in decode_graphs:
                    raise RuntimeError("no native audio pipeline bucket can satisfy requested generation length")
                prefill_graph = self.ir_dir / prefill_graphs[bucket]
                decode_graph = self.ir_dir / decode_graphs[bucket]
                effective_native_async_decode = env_flag_enabled(os.environ.get("QWEN3_TTS_OV_NATIVE_ASYNC_DECODE"))

            cache_mode = str(self.compile_config.get("CACHE_MODE", "OPTIMIZE_SPEED"))
            cache_dir = None if self.disable_ov_cache else usable_ov_cache_dir(self.cache_dir)
            first_decoder = self.ir_dir / first_decoder_graph
            steady_decoder = self.ir_dir / steady_decoder_graph
            codegen_device = str(os.environ.get("QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE") or self.device)
            effective_paged_kv_precision = str(os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_PRECISION") or "f16")
            effective_paged_kv_cache_input_precision = str(
                os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_CACHE_INPUT_PRECISION") or "f32"
            )
            effective_paged_kv_score_aggregation = str(
                os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_SCORE_AGGREGATION") or "1"
            ).strip().lower() not in {"0", "false", "off", "no"}
            effective_paged_kv_heads = int(
                os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_HEADS")
                or (paged_kv_heads if paged_kv_requested else 0)
            )
            effective_paged_kv_block_size = int(os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE") or 8)
            effective_paged_kv_head_dim = int(os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_HEAD_DIM") or 128)
            effective_paged_kv_static_decode = env_flag_enabled(
                os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_DECODE")
            )
            effective_paged_kv_static_blocks = int(os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_BLOCKS") or 128)
            effective_paged_kv_static_mode = str(
                os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_DECODE_MODE") or "minimal"
            ).strip().lower()
            effective_paged_split_subcode = bool(paged_kv_requested and self._native_paged_kv_split_subcode_enabled())
            key = (
                "paged_kv" if paged_kv_requested else "stateful_bucket",
                str(prefill_graph),
                str(decode_graph),
                str(paged_split_subcode_graph or "") if paged_kv_requested else "",
                str(first_decoder),
                str(steady_decoder),
                codegen_device,
                self.decoder_device,
                str(cache_dir or ""),
                cache_mode,
                first_context,
                first_chunk,
                steady_context,
                steady_chunk,
                effective_paged_kv_precision if paged_kv_requested else "",
                effective_paged_kv_cache_input_precision if paged_kv_requested else "",
                effective_paged_kv_score_aggregation if paged_kv_requested else False,
                effective_paged_kv_heads if paged_kv_requested else 0,
                effective_paged_kv_block_size if paged_kv_requested else 0,
                effective_paged_kv_head_dim if paged_kv_requested else 0,
                effective_paged_kv_static_decode if paged_kv_requested else False,
                effective_paged_kv_static_blocks if paged_kv_requested else 0,
                effective_paged_kv_static_mode if paged_kv_requested else "",
                effective_paged_split_subcode,
            )
            if key not in self.native_audio_runners:
                from .native_codegen import NativeCodegenRunner

                started = time.time()
                runner = NativeCodegenRunner(
                    prefill_graph=prefill_graph,
                    decode_graph=decode_graph,
                    device=codegen_device,
                    cache_dir=cache_dir,
                    cache_mode=cache_mode,
                    paged_kv=paged_kv_requested,
                    kv_cache_precision=effective_paged_kv_precision,
                    kv_cache_heads=effective_paged_kv_heads,
                    kv_cache_block_size=effective_paged_kv_block_size,
                    kv_cache_head_dim=effective_paged_kv_head_dim,
                    paged_kv_subcode_graph=paged_split_subcode_graph if effective_paged_split_subcode else None,
                )
                runner.set_stream_decoders(
                    first_decoder_graph=first_decoder,
                    steady_decoder_graph=steady_decoder,
                    decoder_device=self.decoder_device,
                    cache_dir=cache_dir,
                    cache_mode=cache_mode,
                    first_context_frames=first_context,
                    first_chunk_frames=first_chunk,
                    steady_context_frames=steady_context,
                    steady_chunk_frames=steady_chunk,
                    num_code_groups=self.num_code_groups,
                    decode_upsample_rate=self.decode_upsample_rate,
                )
                self.native_audio_runners[key] = runner
                print(
                    f"compiled native GenAI C++ audio pipeline "
                    f"{'paged-kv' if paged_kv_requested else f'bucket {bucket}'} "
                    f"stream c{first_context}_t{first_chunk}/c{steady_context}_t{steady_chunk} "
                    f"on {codegen_device}/{self.decoder_device} "
                    f"in {time.time() - started:.1f}s",
                    flush=True,
                )
            runner = self.native_audio_runners[key]
            if native_prompt_pipeline and not getattr(runner, "voice_design_prompt_configured", False):
                tokenizer_dir = self.ir_dir
                tokenizer_ir = self.manifest.get("tokenizer_ir") or {}
                if not (tokenizer_dir / tokenizer_ir.get("tokenizer", "openvino_tokenizer.xml")).exists():
                    raise RuntimeError("native prompt pipeline requires openvino_tokenizer.xml in the IR directory")
                runner.configure_voice_design_prompt(
                    tokenizer_dir=tokenizer_dir,
                    text_embedding_graph=self.ir_dir / self.graph_name(self.manifest["graphs"], "text_embedding"),
                    codec_embedding_graph=self.ir_dir / self.graph_name(self.manifest["graphs"], "codec_embedding"),
                    device=str(os.environ.get("QWEN3_TTS_OV_NATIVE_PROMPT_DEVICE") or "CPU"),
                    ids=self.ids,
                    cache_dir=cache_dir,
                    cache_mode=cache_mode,
                )
                runner.voice_design_prompt_configured = True
            self.codegen_unroll_fallback = False
            self.last_codegen_info = {
                "prompt_len": int(prompt_len),
                "required_cache_len": int(prompt_len + max_new_tokens),
                "unroll_required_cache_len": int(unroll_required_len),
                "selected_bucket": int(bucket),
                "preferred_cache_bucket": self.preferred_cache_bucket,
                "selected_codegen_graph": str(decode_graph.name),
                "selected_prefill_graph": str(prefill_graph.name),
                "selected_paged_split_subcode_graph": (
                    str(paged_split_subcode_graph.name) if paged_kv_requested and paged_split_subcode_graph else None
                ),
                "paged_kv_split_subcode_mode": (
                    split_subcode_mode if paged_kv_requested and paged_split_subcode else None
                ),
                "paged_kv_seed_key": paged_seed_key if paged_kv_requested else None,
                "paged_kv_gqa": bool(paged_kv_requested and paged_kv_seed_uses_gqa(paged_seed_key)),
                "paged_kv_split_subcode": bool(effective_paged_split_subcode) if paged_kv_requested else None,
                "paged_kv_subcode_attention": (
                    selected_subcode_attention
                    if paged_kv_requested and not paged_split_subcode
                    else ("split" if paged_kv_requested and paged_split_subcode else None)
                ),
                "paged_kv_subcode_attention_requested": subcode_attention if paged_kv_requested else None,
                "paged_kv_subcode_attention_fallback": bool(
                    paged_kv_requested and paged_subcode_attention_fallback
                ),
                "paged_kv_unroll": int(requested_paged_unroll) if paged_kv_requested else None,
                "paged_kv_unroll_experimental": bool(paged_kv_requested and requested_paged_unroll > 1),
                "paged_kv_heads": int(effective_paged_kv_heads) if paged_kv_requested else None,
                "paged_kv_block_size": int(effective_paged_kv_block_size) if paged_kv_requested else None,
                "paged_kv_precision": effective_paged_kv_precision if paged_kv_requested else None,
                "paged_kv_cache_input_precision": (
                    effective_paged_kv_cache_input_precision if paged_kv_requested else None
                ),
                "paged_kv_score_aggregation": effective_paged_kv_score_aggregation if paged_kv_requested else None,
                "paged_kv_static_decode_requested": bool(effective_paged_kv_static_decode) if paged_kv_requested else None,
                "paged_kv_static_decode": bool(effective_paged_kv_static_decode) if paged_kv_requested else None,
                "paged_kv_static_blocks": int(effective_paged_kv_static_blocks) if paged_kv_requested else None,
                "paged_kv_static_decode_mode": effective_paged_kv_static_mode if paged_kv_requested else None,
                "codegen_graph_kind": (
                    "native_audio_pipeline_paged_kv"
                    if paged_kv_requested
                    else (
                        "native_audio_pipeline_decode_unroll_norepeat"
                        if use_no_repeat_graphs
                        else "native_audio_pipeline_decode_unroll_statefulmask"
                    )
                ),
                "codegen_schedule": self.codegen_schedule,
                "scheduled_unrolls": list(self.codegen_schedule_unrolls),
                "active_codegen_unroll": int(requested_paged_unroll if paged_kv_requested else initial_unroll),
                "codegen_decode_unroll": self.codegen_decode_unroll,
                "decode_unroll_graph_available": True,
                "decode_unroll_available": True,
                "decode_unroll_stateful_mask": not use_no_repeat_graphs,
                "codegen_no_repeat": bool(use_no_repeat_graphs),
                "native_codegen": True,
                "native_audio_pipeline": True,
                "paged_kv": bool(paged_kv_requested),
                "paged_kv_backend": "native_paged_attention" if paged_kv_requested else "stateful_bucket",
                "native_prompt_pipeline": bool(native_prompt_pipeline),
                "native_async_decode": bool(effective_native_async_decode),
                "native_streaming_callbacks": True,
            }
            self.paged_kv_enabled = bool(paged_kv_requested)
            self.paged_kv_backend = "native_paged_attention" if paged_kv_requested else "stateful_bucket"
            self.paged_kv_unavailable_reason = "" if paged_kv_requested else self.paged_kv_unavailable_reason

            def _iter_audio_chunks():
                chunk_index = 0
                stream_started = time.time()
                stream_audio_ms = 0.0
                stream_compute_ms = 0.0
                emitted_frames = 0
                final_seen = False
                if native_prompt_pipeline:
                    source_iter = runner.iter_voice_design_audio_chunks(
                        text=text,
                        instruct=instruct,
                        codec_prefill=codec_prefill,
                        max_prompt_tokens=max_prompt_tokens,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        repetition_penalty=repetition_penalty,
                        vocab_size=int(self.ids["vocab_size"]),
                        num_code_groups=self.num_code_groups,
                        eos_token_id=int(self.ids["codec_eos_token_id"]),
                        do_sample=do_sample,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                    )
                else:
                    source_iter = runner.iter_audio_chunks(
                        sequence=sequence,
                        tts_pad_embed=tts_pad_embed,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        repetition_penalty=repetition_penalty,
                        vocab_size=int(self.ids["vocab_size"]),
                        num_code_groups=self.num_code_groups,
                        eos_token_id=int(self.ids["codec_eos_token_id"]),
                        prefix_codes=prefix_codes,
                        do_sample=do_sample,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                    )
                for item in source_iter:
                    audio = np.asarray(item["audio"], dtype=np.float32)
                    codes = np.asarray(item["codes"], dtype=np.int64).reshape(-1, self.num_code_groups)
                    emitted_frames += int(codes.shape[0])
                    codegen_ms = float(item.get("codegen_ms", 0.0))
                    decode_ms = float(item.get("decode_ms", 0.0))
                    audio_ms = (float(audio.shape[0]) / float(self.sample_rate) * 1000.0) if audio.size else 0.0
                    compute_ms = codegen_ms + decode_ms
                    pcm_convert_ms = float(item.get("pcm_convert_ms", 0.0) or 0.0)
                    if audio_ms > 0:
                        stream_audio_ms += audio_ms
                        stream_compute_ms += compute_ms
                    stream_elapsed_ms = max(0.0, (time.time() - stream_started) * 1000.0)
                    stream_rtf = (stream_elapsed_ms / stream_audio_ms) if stream_audio_ms > 0 else 0.0
                    stream_compute_rtf = (stream_compute_ms / stream_audio_ms) if stream_audio_ms > 0 else 0.0
                    is_final = bool(item.get("is_final", False))
                    native_timing = item.get("native_timing")
                    if paged_kv_requested and isinstance(native_timing, dict):
                        static_mode = str(native_timing.get("paged_static_decode_mode") or "dynamic")
                        actual_static_decode = bool(native_timing.get("paged_static_decode_enabled")) and static_mode != "dynamic"
                        actual_async_decode = bool(native_timing.get("async_decode", False))
                    else:
                        actual_static_decode = None
                        actual_async_decode = None
                    timings = {
                        **self.timings.snapshot(emitted_frames),
                        **dict(getattr(self, "last_codegen_info", {}) or {}),
                        "decode_path": f"native:stream:c{first_context if chunk_index == 0 else steady_context}_t{first_chunk if chunk_index == 0 else steady_chunk}",
                        "decode_context_frames": int(first_context if chunk_index == 0 else steady_context),
                        "decode_chunk_graph_frames": int(first_chunk if chunk_index == 0 else steady_chunk),
                        "decode_ms": decode_ms,
                        "fallback": False,
                        "codegen_ms": codegen_ms,
                        "chunk_compute_ms": compute_ms,
                        "chunk_audio_ms": audio_ms,
                        "rtf": (compute_ms / audio_ms) if audio_ms > 0 else 0.0,
                        "stream_audio_ms": stream_audio_ms,
                        "stream_compute_ms": stream_compute_ms,
                        "stream_elapsed_ms": stream_elapsed_ms,
                        "stream_rtf": stream_rtf,
                        "stream_compute_rtf": stream_compute_rtf,
                        "queue_hint_ms": max(0.0, audio_ms - compute_ms),
                        "queue_wait_ms": 0.0,
                        "producer_lag_ms": max(0.0, codegen_ms - audio_ms),
                        "strategy": stream_config["strategy"],
                        "codegen_unroll": int((getattr(self, "last_codegen_info", {}) or {}).get("active_codegen_unroll", getattr(self, "codegen_unroll", 1))),
                        "unroll_fallback": False,
                        "initial_chunk_frames": int(first_chunk),
                        "configured_chunk_frames": int(steady_chunk),
                        "native_audio_pipeline": True,
                        "native_prompt_pipeline": bool(native_prompt_pipeline),
                        "native_async_decode": actual_async_decode,
                        "native_sampling": bool(do_sample and paged_kv_requested and effective_paged_split_subcode),
                        "do_sample": bool(do_sample),
                        "top_k": int(top_k),
                        "top_p": float(top_p),
                        "temperature": float(temperature),
                        "native_remote_embed": bool(item.get("remote_embed", getattr(runner, "last_remote_embed", False))),
                        "native_ov_profile": item.get("native_ov_profile"),
                        "native_timing": native_timing,
                        "pcm_convert_ms": pcm_convert_ms,
                        "paged_kv_static_decode_actual": actual_static_decode,
                        "paged_kv_static_decode": actual_static_decode if paged_kv_requested else None,
                        "pipeline_decode": True,
                        "chunk_frames": int(codes.shape[0]),
                        "emitted_frames": int(emitted_frames),
                        "prefix_frames": int(0 if prefix_codes is None else np.asarray(prefix_codes).reshape(-1, self.num_code_groups).shape[0]),
                        "is_final": is_final,
                    }
                    chunk = StreamChunk(
                        index=chunk_index,
                        audio=audio,
                        sample_rate=self.sample_rate,
                        codes=codes,
                        is_final=is_final,
                        timings=timings,
                        pcm_s16le=item.get("pcm_s16le") or None,
                    )
                    chunk_index += 1
                    yield chunk
                    if is_final:
                        final_seen = True
                if final_seen:
                    self.last_codegen_info.update(
                        {
                            "native_pipeline_ms": float(getattr(runner, "last_audio_stream_elapsed_ms", 0.0) or 0.0),
                            "native_emitted_frames": int(getattr(runner, "last_audio_stream_count", emitted_frames) or emitted_frames),
                            "native_remote_embed": bool(getattr(runner, "last_remote_embed", False)),
                            "native_ov_profile": getattr(runner, "last_profile_json", None),
                            "native_timing": getattr(runner, "last_timing_json", None),
                        }
                    )

            return _iter_audio_chunks()
        except Exception as exc:
            if require:
                raise
            print(f"warning: native GenAI C++ audio pipeline unavailable; falling back to Python runtime: {exc}", flush=True)
            return None

    def close_native_audio_runners(self, runner_kind: str | None = None) -> int:
        closed = 0
        for key, runner in list(self.native_audio_runners.items()):
            kind = key[0] if isinstance(key, tuple) and key else None
            if runner_kind is not None and kind != runner_kind:
                continue
            self.native_audio_runners.pop(key, None)
            try:
                runner.close()
            finally:
                closed += 1
        if closed:
            gc.collect()
        return closed

    def release_native_audio_runner_buffers(self, runner_kind: str | None = None) -> int:
        released = 0
        for key, runner in list(self.native_audio_runners.items()):
            kind = key[0] if isinstance(key, tuple) and key else None
            if runner_kind is not None and kind != runner_kind:
                continue
            release = getattr(runner, "release_run_buffers", None)
            if release is None:
                continue
            release()
            released += 1
        if released:
            gc.collect()
        return released

    def _try_stream_native_hybrid_paged_audio_pipeline(
        self,
        *,
        text: str,
        instruct: str,
        language: str,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        max_prompt_tokens: int,
        chunk_frames: int | None = None,
        left_context_frames: int | None = None,
        initial_chunk_frames: int | None = None,
        chunk_strategy: str | None = None,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ):
        if not env_flag_enabled(os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID")):
            return None
        if (
            str(os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV", "")).strip().lower()
            in {"1", "true", "on", "yes", "require"}
        ):
            return None
        if do_sample or abs(float(repetition_penalty) - 1.0) > 1e-6:
            return None
        if self.mode != "cache" or self.cache_step != "fused":
            return None
        try:
            prefix_frames = int(os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID_PREFIX_FRAMES") or 48)
        except ValueError:
            prefix_frames = 48
        prefix_frames = max(1, prefix_frames)
        if max_new_tokens <= prefix_frames:
            return None

        with temporary_env({"QWEN3_TTS_OV_NATIVE_PAGED_KV": "0"}):
            prefix_iter = self._try_stream_native_audio_pipeline(
                text=text,
                instruct=instruct,
                language=language,
                max_new_tokens=prefix_frames,
                min_new_tokens=min(min_new_tokens, prefix_frames),
                repetition_penalty=repetition_penalty,
                max_prompt_tokens=max_prompt_tokens,
                chunk_frames=chunk_frames,
                left_context_frames=left_context_frames,
                initial_chunk_frames=initial_chunk_frames,
                chunk_strategy=chunk_strategy,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
        if prefix_iter is None:
            return None

        def _iter_hybrid_chunks():
            started = time.time()
            total_audio_ms = 0.0
            total_compute_ms = 0.0
            emitted_frames = 0
            chunk_index = 0
            prefix_parts: list[np.ndarray] = []
            eos_id = int(self.ids["codec_eos_token_id"])

            def rewrite_chunk(chunk: StreamChunk, phase: str, is_final: bool | None = None) -> StreamChunk:
                nonlocal chunk_index, total_audio_ms, total_compute_ms, emitted_frames
                audio = np.asarray(chunk.audio, dtype=np.float32)
                codes = np.asarray(chunk.codes, dtype=np.int64).reshape(-1, self.num_code_groups)
                timings = dict(chunk.timings or {})
                audio_ms = (
                    float(timings.get("chunk_audio_ms", 0.0) or 0.0)
                    or ((float(audio.shape[0]) / float(self.sample_rate) * 1000.0) if audio.size else 0.0)
                )
                codegen_ms = float(timings.get("codegen_ms", 0.0) or 0.0)
                decode_ms = float(timings.get("decode_ms", 0.0) or 0.0)
                compute_ms = (
                    float(timings.get("chunk_compute_ms", 0.0) or 0.0)
                    or codegen_ms + decode_ms
                )
                if audio_ms > 0:
                    total_audio_ms += audio_ms
                    total_compute_ms += compute_ms
                emitted_frames += int(codes.shape[0])
                stream_elapsed_ms = max(0.0, (time.time() - started) * 1000.0)
                final_value = bool(chunk.is_final if is_final is None else is_final)
                timings.update(
                    {
                        "hybrid_paged_kv": True,
                        "hybrid_phase": phase,
                        "hybrid_prefix_frames": int(prefix_frames),
                        "hybrid_prefix_actual_frames": int(sum(part.shape[0] for part in prefix_parts)),
                        "stream_audio_ms": total_audio_ms,
                        "stream_compute_ms": total_compute_ms,
                        "stream_elapsed_ms": stream_elapsed_ms,
                        "stream_rtf": (stream_elapsed_ms / total_audio_ms) if total_audio_ms > 0 else 0.0,
                        "stream_compute_rtf": (total_compute_ms / total_audio_ms) if total_audio_ms > 0 else 0.0,
                        "chunk_compute_ms": compute_ms,
                        "chunk_audio_ms": audio_ms,
                        "rtf": (compute_ms / audio_ms) if audio_ms > 0 else 0.0,
                        "chunk_frames": int(codes.shape[0]),
                        "emitted_frames": int(emitted_frames),
                        "is_final": final_value,
                    }
                )
                out = StreamChunk(
                    index=chunk_index,
                    audio=audio,
                    sample_rate=chunk.sample_rate,
                    codes=codes,
                    is_final=final_value,
                    timings=timings,
                )
                chunk_index += 1
                return out

            saw_eos = False
            last_prefix_chunk: StreamChunk | None = None
            for chunk in prefix_iter:
                codes = np.asarray(chunk.codes, dtype=np.int64).reshape(-1, self.num_code_groups)
                if codes.size:
                    prefix_parts.append(codes.copy())
                    saw_eos = bool(saw_eos or np.any(codes[:, 0] == eos_id))
                last_prefix_chunk = chunk
                continue_after_prefix = (not saw_eos) and (sum(part.shape[0] for part in prefix_parts) < max_new_tokens)
                yield rewrite_chunk(chunk, "fixed_prefix", is_final=(False if chunk.is_final and continue_after_prefix else chunk.is_final))
                if chunk.is_final:
                    break

            if not prefix_parts:
                if last_prefix_chunk is None:
                    return
                return
            prefix = np.concatenate(prefix_parts, axis=0).astype(np.int64, copy=False)
            if saw_eos or prefix.shape[0] >= max_new_tokens:
                return

            release_prefix_runner = (
                str(os.environ.get("QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID_RELEASE_PREFIX_RUNNER") or "1")
                .strip()
                .lower()
                not in {"0", "false", "off", "no"}
            )
            released_prefix_runners = self.close_native_audio_runners("stateful_bucket") if release_prefix_runner else 0
            remaining_tokens = int(max_new_tokens - prefix.shape[0])
            remaining_min_tokens = max(0, int(min_new_tokens - prefix.shape[0]))
            with temporary_env(
                {
                    "QWEN3_TTS_OV_NATIVE_PAGED_KV": "1",
                    "QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA": os.environ.get(
                        "QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA",
                        "1",
                    ),
                }
            ):
                continuation_iter = self._try_stream_native_audio_pipeline(
                    text=text,
                    instruct=instruct,
                    language=language,
                    max_new_tokens=remaining_tokens,
                    min_new_tokens=remaining_min_tokens,
                    repetition_penalty=repetition_penalty,
                    max_prompt_tokens=max_prompt_tokens,
                    chunk_frames=chunk_frames,
                    left_context_frames=left_context_frames,
                    initial_chunk_frames=initial_chunk_frames,
                    chunk_strategy=chunk_strategy,
                    prefix_codes=prefix,
                    append_prefix_codes_to_prompt=True,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                )
            if continuation_iter is None:
                raise RuntimeError("hybrid paged-KV continuation is unavailable after fixed-bucket prefix generation")
            for chunk in continuation_iter:
                rewritten = rewrite_chunk(chunk, "paged_continuation")
                rewritten.timings["hybrid_released_prefix_runners"] = int(released_prefix_runners)
                yield rewritten
                if chunk.is_final:
                    break

        return _iter_hybrid_chunks()

    def ensure_subcode_greedy(self):
        if self.subcode_greedy is not None:
            return
        graphs = self.manifest["graphs"]
        subcode_graph = self.graph_name(graphs, "subcode_greedy")
        started = time.time()
        self.subcode_graph_name = subcode_graph
        self.subcode_greedy = compile_model(
            self.core,
            self.ir_dir / subcode_graph,
            self.device,
            self.cache_dir,
            self.allow_cpu_fallback,
            self.ov_profiler.enabled,
            self.precision_hint,
            self.compile_config,
        )
        self.subcode_request = self.subcode_greedy.create_infer_request()
        if not self.allow_cpu_fallback and self.device == "GPU":
            self.assert_gpu_execution([self.subcode_greedy])
        print(f"compiled subcode greedy on {self.device} in {time.time() - started:.1f}s", flush=True)

    def embed_text(self, token_ids):
        ids = np.asarray([token_ids], dtype=np.int64)
        started = time.time()
        result = run_request(
            self.text_embedding_request,
            self.text_embedding,
            [ids],
            self.ov_profiler,
            "text_embedding",
        )[0].astype(np.float32, copy=False)
        self.timings.add("embedding", time.time() - started)
        return result

    def embed_codec(self, token_ids):
        ids = np.asarray([token_ids], dtype=np.int64)
        started = time.time()
        result = run_request(
            self.codec_embedding_request,
            self.codec_embedding,
            [ids],
            self.ov_profiler,
            "codec_embedding",
        )[0].astype(np.float32, copy=False)
        self.timings.add("embedding", time.time() - started)
        return result

    def embed_code_frames(self, codes: np.ndarray) -> np.ndarray:
        if self.code_frame_embedding_request is None:
            raise RuntimeError(
                "this IR does not include code_frame_embedding.xml; re-export Base/VoiceClone support before using ICL voice clone"
            )
        ids = np.asarray(codes, dtype=np.int64)
        if ids.ndim == 2:
            ids = ids[None, :, :]
        started = time.time()
        result = run_request(
            self.code_frame_embedding_request,
            self.code_frame_embedding,
            [ids],
            self.ov_profiler,
            "code_frame_embedding",
        )[0].astype(np.float32, copy=False)
        self.timings.add("embedding", time.time() - started)
        return result

    def speaker_token_embed(self, speaker: str | None):
        if speaker is None or speaker == "":
            return None
        spk_id = self.ids.get("spk_id", {}).get(speaker.lower())
        if spk_id is None:
            supported = sorted(self.ids.get("spk_id", {}))
            raise ValueError(f"unsupported speaker {speaker!r}; supported: {supported}")
        return self.embed_codec([int(spk_id)])

    def voice_clone_speaker_embed(self, prompt_item: VoiceClonePromptItem | None):
        if prompt_item is None:
            return None
        if prompt_item.x_vector_only_mode or prompt_item.icl_mode:
            embed = np.asarray(prompt_item.ref_spk_embedding, dtype=np.float32)
            if embed.ndim == 1:
                embed = embed.reshape(1, 1, -1)
            elif embed.ndim == 2:
                embed = embed.reshape(1, 1, embed.shape[-1])
            return embed
        return None

    def language_codec_prefill(self, language: str, speaker: str | None = None):
        language_key = language.lower()
        speaker_key = speaker.lower() if speaker else None
        if (
            language_key in {"chinese", "auto"}
            and speaker_key
            and self.ids.get("spk_is_dialect", {}).get(speaker_key) not in (None, False, "")
        ):
            language_key = self.ids["spk_is_dialect"][speaker_key]

        if language_key == "auto":
            return [
                self.ids["codec_nothink_id"],
                self.ids["codec_think_bos_id"],
                self.ids["codec_think_eos_id"],
            ]

        language_ids = self.ids["codec_language_id"]
        if language_key not in language_ids:
            raise ValueError(f"unsupported language {language!r}; supported: {sorted(language_ids)} plus auto")
        return [
            self.ids["codec_think_id"],
            self.ids["codec_think_bos_id"],
            language_ids[language_key],
            self.ids["codec_think_eos_id"],
        ]

    def build_prompt(
        self,
        text: str,
        instruct: str,
        language: str,
        max_prompt_tokens: int,
        speaker: str | None = None,
        voice_clone_prompt: VoiceClonePromptItem | None = None,
        ref_text: str | None = None,
    ):
        input_ids = self.tokenizer.encode(build_assistant_text(text))
        instruct_ids = self.tokenizer.encode(build_instruct_text(instruct)) if instruct else None
        if len(input_ids) > max_prompt_tokens:
            raise ValueError(f"text prompt has {len(input_ids)} tokens, max_prompt_tokens={max_prompt_tokens}")
        if instruct_ids is not None and len(instruct_ids) > max_prompt_tokens:
            raise ValueError(f"instruct prompt has {len(instruct_ids)} tokens, max_prompt_tokens={max_prompt_tokens}")

        prompt_parts = []
        if instruct_ids is not None:
            prompt_parts.append(self.embed_text(instruct_ids))

        tts_special = self.embed_text(
            [
                self.ids["tts_bos_token_id"],
                self.ids["tts_eos_token_id"],
                self.ids["tts_pad_token_id"],
            ]
        )
        tts_bos_embed = tts_special[:, 0:1, :]
        tts_eos_embed = tts_special[:, 1:2, :]
        tts_pad_embed = tts_special[:, 2:3, :]

        codec_prefill = self.language_codec_prefill(language, speaker=speaker)
        codec_prefix = self.embed_codec(codec_prefill)
        codec_tail = self.embed_codec([self.ids["codec_pad_id"], self.ids["codec_bos_id"]])
        speaker_embed = self.voice_clone_speaker_embed(voice_clone_prompt)
        if speaker_embed is None:
            speaker_embed = self.speaker_token_embed(speaker)
        codec_parts = [codec_prefix]
        if speaker_embed is not None:
            codec_parts.append(speaker_embed)
        codec_parts.append(codec_tail)
        codec_input_embedding = np.concatenate(codec_parts, axis=1)

        input_embed = self.embed_text(input_ids)
        role_embed = input_embed[:, :3, :]
        pad_prefix = np.repeat(tts_pad_embed, codec_input_embedding.shape[1] - 2, axis=1)
        prefill_embed = np.concatenate([pad_prefix, tts_bos_embed], axis=1) + codec_input_embedding[:, :-1, :]
        talker_input_embed = np.concatenate([role_embed, prefill_embed], axis=1)

        if voice_clone_prompt is not None and voice_clone_prompt.ref_code is not None and voice_clone_prompt.icl_mode:
            if not ref_text:
                raise ValueError("ref_text is required for ICL voice clone")
            ref_ids = self.tokenizer.encode(build_ref_text(ref_text))
            ref_body_ids = ref_ids[3:-2]
            text_body_ids = input_ids[3:-5]
            text_embed = self.embed_text(ref_body_ids + text_body_ids)
            text_embed = np.concatenate([text_embed, tts_eos_embed], axis=1)
            text_codec_pad = self.embed_codec([self.ids["codec_pad_id"]] * text_embed.shape[1])

            ref_code = np.asarray(voice_clone_prompt.ref_code, dtype=np.int64)
            codec_embed = self.embed_code_frames(ref_code)
            codec_bos_embed = self.embed_codec([self.ids["codec_bos_id"]])
            codec_embed = np.concatenate([codec_bos_embed, codec_embed], axis=1)

            talker_input_embed = np.concatenate(
                [
                    talker_input_embed,
                    text_embed + text_codec_pad,
                    codec_embed + tts_pad_embed,
                ],
                axis=1,
            )
        else:
            text_body = input_embed[:, 3:-5, :]
            text_body_with_eos = np.concatenate([text_body, tts_eos_embed], axis=1)
            text_codec_pad = self.embed_codec([self.ids["codec_pad_id"]] * text_body_with_eos.shape[1])
            final_bos = tts_pad_embed + self.embed_codec([self.ids["codec_bos_id"]])
            talker_input_embed = np.concatenate(
                [
                    talker_input_embed[:, :-1, :],
                    text_body_with_eos + text_codec_pad,
                    final_bos,
                ],
                axis=1,
            )
        prompt_parts.append(talker_input_embed)
        return np.concatenate(prompt_parts, axis=1), tts_pad_embed

    def append_prefix_code_frames_to_prompt(
        self,
        sequence: np.ndarray,
        tts_pad_embed: np.ndarray,
        prefix_codes: np.ndarray | None,
    ) -> np.ndarray:
        if prefix_codes is None:
            return sequence
        prefix = np.asarray(prefix_codes, dtype=np.int64)
        if prefix.size == 0:
            return sequence
        prefix = prefix.reshape(-1, self.num_code_groups)
        if prefix.shape[1] != self.num_code_groups:
            raise ValueError(f"prefix_codes must have {self.num_code_groups} code groups")
        prefix_embeds = self.embed_code_frames(prefix) + tts_pad_embed
        return np.concatenate(
            [
                np.asarray(sequence, dtype=np.float32),
                np.asarray(prefix_embeds, dtype=np.float32),
            ],
            axis=1,
        )

    def make_attention_mask(self, query_positions: np.ndarray, query_len: int, key_len: int | None = None):
        key_len = int(key_len or (int(query_positions.reshape(-1)[-1]) + 1))
        if key_len <= 0:
            raise RuntimeError("attention masks for cache mode require a positive key length")
        columns = np.arange(key_len, dtype=np.int64)
        allowed = columns[None, :] <= query_positions.reshape(-1, 1)
        mask = np.where(allowed, 0.0, NEG_INF).astype(np.float32)
        return mask.reshape(1, 1, query_len, key_len)

    def select_first_code(
        self,
        logits,
        generated_first_codes,
        step: int,
        min_new_tokens: int,
        repetition_penalty: float,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ):
        logits = self.normalize_logits(logits)
        scores = logits[0].astype(np.float32, copy=True)
        eos_id = int(self.ids["codec_eos_token_id"])
        suppress_from = int(self.ids["suppress_from"])
        scores[suppress_from:] = NEG_INF
        scores[eos_id] = logits[0, eos_id]
        if step < min_new_tokens:
            scores[eos_id] = NEG_INF
        if repetition_penalty and repetition_penalty != 1.0:
            for token_id in set(generated_first_codes):
                value = scores[token_id]
                scores[token_id] = value * repetition_penalty if value < 0 else value / repetition_penalty
        if do_sample:
            return self.sample_token(scores, top_k=top_k, top_p=top_p, temperature=temperature)
        return int(np.argmax(scores))

    @staticmethod
    def normalize_logits(logits: np.ndarray) -> np.ndarray:
        if logits.ndim == 3 and logits.shape[1] == 1:
            return logits[:, 0, :]
        return logits

    @staticmethod
    def sample_token(scores: np.ndarray, top_k: int = 50, top_p: float = 1.0, temperature: float = 0.9) -> int:
        finite = np.isfinite(scores)
        if not finite.any():
            return int(np.argmax(scores))
        scaled = scores.astype(np.float64, copy=True) / max(float(temperature), 1e-6)
        scaled[~finite] = -np.inf
        if top_k and top_k > 0 and top_k < scaled.size:
            cutoff = np.partition(scaled, -top_k)[-top_k]
            scaled[scaled < cutoff] = -np.inf
        shifted = scaled - np.nanmax(scaled)
        probs = np.exp(shifted)
        probs[~np.isfinite(probs)] = 0.0
        total = probs.sum()
        if total <= 0:
            return int(np.argmax(scores))
        probs /= total
        if top_p and 0 < top_p < 1.0:
            order = np.argsort(probs)[::-1]
            cumulative = np.cumsum(probs[order])
            keep = cumulative <= top_p
            keep[0] = True
            mask = np.zeros_like(probs, dtype=bool)
            mask[order[keep]] = True
            probs = np.where(mask, probs, 0.0)
            probs /= probs.sum()
        return int(np.random.choice(np.arange(probs.size), p=probs))

    def generate_codes(
        self,
        text: str,
        instruct: str,
        language: str,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        max_prompt_tokens: int,
        progress_interval: int,
        speaker: str | None = None,
        voice_clone_prompt: VoiceClonePromptItem | None = None,
        ref_text: str | None = None,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        prefix_codes: np.ndarray | None = None,
        append_prefix_codes_to_prompt: bool = False,
    ):
        frames = list(
            self.generate_codes_iter(
                text=text,
                instruct=instruct,
                language=language,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                max_prompt_tokens=max_prompt_tokens,
                progress_interval=progress_interval,
                speaker=speaker,
                voice_clone_prompt=voice_clone_prompt,
                ref_text=ref_text,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                prefix_codes=prefix_codes,
                append_prefix_codes_to_prompt=append_prefix_codes_to_prompt,
            )
        )
        if not frames:
            raise RuntimeError("generation stopped before producing any codec token")
        return np.stack(frames, axis=0)

    def generate_codes_iter(
        self,
        text: str,
        instruct: str,
        language: str,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        max_prompt_tokens: int,
        progress_interval: int,
        speaker: str | None = None,
        voice_clone_prompt: VoiceClonePromptItem | None = None,
        ref_text: str | None = None,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        prefix_codes: np.ndarray | None = None,
        append_prefix_codes_to_prompt: bool = False,
    ):
        if self.mode == "cache":
            cache_generator = (
                self.generate_codes_cache_fused_iter
                if self.cache_step == "fused" and not do_sample
                else self.generate_codes_cache_split_iter
            )
            yield from cache_generator(
                text=text,
                instruct=instruct,
                language=language,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                max_prompt_tokens=max_prompt_tokens,
                progress_interval=progress_interval,
                speaker=speaker,
                voice_clone_prompt=voice_clone_prompt,
                ref_text=ref_text,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                prefix_codes=prefix_codes,
                append_prefix_codes_to_prompt=append_prefix_codes_to_prompt,
            )
            return
        if self.mode == "fused-no-cache":
            yield from self.generate_codes_fused_no_cache_iter(
                text=text,
                instruct=instruct,
                language=language,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                max_prompt_tokens=max_prompt_tokens,
                progress_interval=progress_interval,
                speaker=speaker,
                voice_clone_prompt=voice_clone_prompt,
                ref_text=ref_text,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                prefix_codes=prefix_codes,
                append_prefix_codes_to_prompt=append_prefix_codes_to_prompt,
            )
            return
        yield from self.generate_codes_no_cache_iter(
            text=text,
            instruct=instruct,
            language=language,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            max_prompt_tokens=max_prompt_tokens,
            progress_interval=progress_interval,
            speaker=speaker,
            voice_clone_prompt=voice_clone_prompt,
            ref_text=ref_text,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            prefix_codes=prefix_codes,
            append_prefix_codes_to_prompt=append_prefix_codes_to_prompt,
        )

    def generate_codes_no_cache_iter(
        self,
        text: str,
        instruct: str,
        language: str,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        max_prompt_tokens: int,
        progress_interval: int,
        speaker: str | None = None,
        voice_clone_prompt: VoiceClonePromptItem | None = None,
        ref_text: str | None = None,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        prefix_codes: np.ndarray | None = None,
        append_prefix_codes_to_prompt: bool = False,
    ):
        sequence, tts_pad_embed = self.build_prompt(
            text,
            instruct,
            language,
            max_prompt_tokens,
            speaker=speaker,
            voice_clone_prompt=voice_clone_prompt,
            ref_text=ref_text,
        )
        if append_prefix_codes_to_prompt:
            sequence = self.append_prefix_code_frames_to_prompt(sequence, tts_pad_embed, prefix_codes)
        generated_count = 0
        generated_first_codes = []
        started = time.time()
        for step in range(max_new_tokens):
            talker_started = time.time()
            talker_inputs = [sequence.astype(np.float32, copy=False)]
            self.dump_calibration("talker_no_cache", talker_inputs)
            logits, past_hidden = run_request(
                self.talker_request,
                self.talker,
                talker_inputs,
                self.ov_profiler,
                "talker_no_cache",
            )
            self.timings.add("talker", time.time() - talker_started)
            first_code = self.select_first_code(
                logits,
                generated_first_codes,
                step,
                min_new_tokens,
                repetition_penalty,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            if first_code == int(self.ids["codec_eos_token_id"]):
                break

            code_input = np.asarray([[first_code]], dtype=np.int64)
            subcode_started = time.time()
            subcode_inputs = [past_hidden.astype(np.float32), code_input]
            self.dump_calibration("subcode_greedy", subcode_inputs)
            codes, sum_embed = run_request(
                self.subcode_request,
                self.subcode_greedy,
                subcode_inputs,
                self.ov_profiler,
                "subcode_greedy",
            )
            self.timings.add("subcode", time.time() - subcode_started)
            codes = codes.astype(np.int64, copy=False)
            generated_count += 1
            generated_first_codes.append(first_code)

            frame_embed = sum_embed.astype(np.float32, copy=False) + tts_pad_embed
            sequence = np.concatenate([sequence, frame_embed], axis=1)
            if progress_interval and generated_count % progress_interval == 0:
                elapsed = time.time() - started
                print(f"generated {generated_count}/{max_new_tokens} codec tokens in {elapsed:.1f}s", flush=True)
            yield codes[0]

        if generated_count == 0:
            raise RuntimeError("generation stopped before producing any codec token")

    def generate_codes_fused_no_cache_iter(
        self,
        text: str,
        instruct: str,
        language: str,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        max_prompt_tokens: int,
        progress_interval: int,
        speaker: str | None = None,
        voice_clone_prompt: VoiceClonePromptItem | None = None,
        ref_text: str | None = None,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        prefix_codes: np.ndarray | None = None,
        append_prefix_codes_to_prompt: bool = False,
    ):
        if do_sample:
            raise ValueError("do_sample=True is not supported by fused-no-cache OpenVINO graphs; use --mode no-cache")
        sequence, tts_pad_embed = self.build_prompt(
            text,
            instruct,
            language,
            max_prompt_tokens,
            speaker=speaker,
            voice_clone_prompt=voice_clone_prompt,
            ref_text=ref_text,
        )
        if append_prefix_codes_to_prompt:
            sequence = self.append_prefix_code_frames_to_prompt(sequence, tts_pad_embed, prefix_codes)
        generated_count = 0
        generated_first_codes = []
        started = time.time()
        for step in range(max_new_tokens):
            repeated_mask = np.zeros((1, int(self.ids["vocab_size"])), dtype=np.float32)
            for token_id in set(generated_first_codes):
                repeated_mask[0, token_id] = 1.0
            allow_eos = np.asarray([1.0 if step >= min_new_tokens else 0.0], dtype=np.float32)
            penalty = np.asarray([repetition_penalty], dtype=np.float32)

            step_started = time.time()
            first_code, codes, frame_embed = run_request(
                self.fused_request,
                self.fused_step,
                [
                    sequence.astype(np.float32, copy=False),
                    tts_pad_embed.astype(np.float32, copy=False),
                    repeated_mask,
                    allow_eos,
                    penalty,
                ],
                self.ov_profiler,
                "fused_no_cache_step",
            )
            self.timings.add("fused_step", time.time() - step_started)
            first_code_int = int(first_code.reshape(-1)[0])
            if first_code_int == int(self.ids["codec_eos_token_id"]):
                break

            codes = codes.astype(np.int64, copy=False)
            generated_count += 1
            generated_first_codes.append(first_code_int)
            sequence = np.concatenate([sequence, frame_embed.astype(np.float32, copy=False)], axis=1)
            if progress_interval and generated_count % progress_interval == 0:
                elapsed = time.time() - started
                print(f"generated {generated_count}/{max_new_tokens} codec tokens in {elapsed:.1f}s", flush=True)
            yield codes[0]

        if generated_count == 0:
            raise RuntimeError("generation stopped before producing any codec token")

    def generate_codes_cache_split_iter(
        self,
        text: str,
        instruct: str,
        language: str,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        max_prompt_tokens: int,
        progress_interval: int,
        speaker: str | None = None,
        voice_clone_prompt: VoiceClonePromptItem | None = None,
        ref_text: str | None = None,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        prefix_codes: np.ndarray | None = None,
        append_prefix_codes_to_prompt: bool = False,
    ):
        sequence, tts_pad_embed = self.build_prompt(
            text,
            instruct,
            language,
            max_prompt_tokens,
            speaker=speaker,
            voice_clone_prompt=voice_clone_prompt,
            ref_text=ref_text,
        )
        if append_prefix_codes_to_prompt:
            sequence = self.append_prefix_code_frames_to_prompt(sequence, tts_pad_embed, prefix_codes)
        self.ensure_subcode_greedy()
        prompt_len = int(sequence.shape[1])
        required_len = prompt_len + max_new_tokens
        cache_len, talker_stateful, talker_request = self.get_talker_stateful(required_len)

        talker_request.reset_state()
        prompt_positions = np.arange(prompt_len, dtype=np.int64)
        prompt_mask = self.make_attention_mask(prompt_positions, prompt_len, prompt_len)

        started = time.time()
        talker_started = time.time()
        logits, past_hidden = run_request(
            talker_request,
            talker_stateful,
            {
                "inputs_embeds": sequence.astype(np.float32, copy=False),
                "cache_position": prompt_positions,
                "attention_mask": prompt_mask,
            },
            self.ov_profiler,
            "talker_cache_prefill",
        )
        self.timings.add("talker", time.time() - talker_started)

        generated_count = 0
        generated_first_codes = []
        for step in range(max_new_tokens):
            first_code = self.select_first_code(
                logits,
                generated_first_codes,
                step,
                min_new_tokens,
                repetition_penalty,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            if first_code == int(self.ids["codec_eos_token_id"]):
                break

            code_input = np.asarray([[first_code]], dtype=np.int64)
            subcode_started = time.time()
            codes, sum_embed = run_request(
                self.subcode_request,
                self.subcode_greedy,
                [past_hidden.astype(np.float32, copy=False), code_input],
                self.ov_profiler,
                "subcode_greedy",
            )
            self.timings.add("subcode", time.time() - subcode_started)
            codes = codes.astype(np.int64, copy=False)
            generated_count += 1
            generated_first_codes.append(first_code)

            if progress_interval and generated_count % progress_interval == 0:
                elapsed = time.time() - started
                print(f"generated {generated_count}/{max_new_tokens} codec tokens in {elapsed:.1f}s", flush=True)
            yield codes[0]

            if step + 1 >= max_new_tokens:
                break

            frame_embed = sum_embed.astype(np.float32, copy=False) + tts_pad_embed
            cache_position = np.asarray([prompt_len + step], dtype=np.int64)
            attention_mask = self.make_attention_mask(cache_position, 1, prompt_len + step + 1)
            talker_started = time.time()
            logits, past_hidden = run_request(
                talker_request,
                talker_stateful,
                {
                    "inputs_embeds": frame_embed.astype(np.float32, copy=False),
                    "cache_position": cache_position,
                    "attention_mask": attention_mask,
                },
                self.ov_profiler,
                "talker_cache_decode",
            )
            self.timings.add("talker", time.time() - talker_started)

        if generated_count == 0:
            raise RuntimeError("generation stopped before producing any codec token")

    def generate_codes_cache_fused_iter(
        self,
        text: str,
        instruct: str,
        language: str,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        max_prompt_tokens: int,
        progress_interval: int,
        speaker: str | None = None,
        voice_clone_prompt: VoiceClonePromptItem | None = None,
        ref_text: str | None = None,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        prefix_codes: np.ndarray | None = None,
        append_prefix_codes_to_prompt: bool = False,
    ):
        if do_sample:
            raise ValueError("do_sample=True is not supported by fused cache OpenVINO graphs; use --cache-step split")
        sequence, tts_pad_embed = self.build_prompt(
            text,
            instruct,
            language,
            max_prompt_tokens,
            speaker=speaker,
            voice_clone_prompt=voice_clone_prompt,
            ref_text=ref_text,
        )
        if append_prefix_codes_to_prompt:
            sequence = self.append_prefix_code_frames_to_prompt(sequence, tts_pad_embed, prefix_codes)
        prompt_len = int(sequence.shape[1])
        required_len = prompt_len + max_new_tokens
        initial_unroll = self.select_codegen_unroll_for_step(0)
        unroll_required_len = self.unroll_required_cache_len(prompt_len, max_new_tokens, initial_unroll)
        use_unroll = initial_unroll > 1
        fused_cache_step = None
        fused_cache_request = None
        unroll_cache_len = None
        unroll_step = None
        unroll_request = None
        decode_unroll_cache_len = None
        decode_unroll_step = None
        decode_unroll_request = None
        decode_unroll_stateful_mask = False
        decode_unroll_ready = False
        decode_unroll_graph_available = bool(
            self.fused_cache_decode_unroll_bucket_graphs_by_step.get(initial_unroll, {})
            or self.fused_cache_decode_unroll_stateful_mask_bucket_graphs_by_step.get(initial_unroll, {})
            or self.fused_cache_decode_unroll_norepeat_bucket_graphs_by_step.get(initial_unroll, {})
        )
        decode_unroll_enabled = self.codegen_decode_unroll in {"auto", "on"} and self.codegen_schedule == "current"
        native_codes = self._try_generate_codes_native_unroll4_statefulmask(
            sequence=sequence,
            tts_pad_embed=tts_pad_embed,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            initial_unroll=initial_unroll,
            unroll_required_len=unroll_required_len,
            decode_unroll_enabled=decode_unroll_enabled,
            decode_unroll_graph_available=decode_unroll_graph_available,
            stream_batches=True,
        )
        if native_codes is not None:
            for frame in native_codes:
                yield frame
            return
        if use_unroll:
            try:
                unroll_cache_len, unroll_step, unroll_request = self.get_fused_cache_unroll_step(
                    unroll_required_len,
                    initial_unroll,
                )
                if self.codegen_decode_unroll == "on" and self.codegen_schedule != "current":
                    raise RuntimeError("codegen_decode_unroll=on is only supported with codegen_schedule=current")
                if self.codegen_decode_unroll == "on" and not decode_unroll_graph_available:
                    raise RuntimeError("codegen_decode_unroll=on requested, but no fused cache decode-unroll graph is available")
                if decode_unroll_enabled and decode_unroll_graph_available:
                    (
                        decode_unroll_cache_len,
                        decode_unroll_step,
                        decode_unroll_request,
                        decode_unroll_stateful_mask,
                    ) = self.get_fused_cache_decode_unroll_step(unroll_required_len, initial_unroll)
                self.codegen_unroll_fallback = False
            except Exception as exc:
                self.codegen_unroll_fallback = True
                use_unroll = False
                print(f"warning: fused cache unroll{initial_unroll} failed; falling back to single-step fused cache: {exc}", flush=True)
        else:
            self.codegen_unroll_fallback = self.codegen_unroll > 1
        if not use_unroll:
            cache_len, fused_cache_step, fused_cache_request = self.get_fused_cache_step(required_len)

        if use_unroll:
            unroll_request.reset_state()
            if decode_unroll_request is not None:
                decode_unroll_request.reset_state()
        else:
            fused_cache_request.reset_state()
        self.last_codegen_info = {
            "prompt_len": int(prompt_len),
            "required_cache_len": int(required_len),
            "unroll_required_cache_len": int(unroll_required_len),
            "selected_bucket": int(unroll_cache_len if use_unroll else cache_len),
            "selected_codegen_graph": (
                self.fused_cache_unroll_bucket_graphs_by_step.get(initial_unroll, {}).get(unroll_cache_len)
                if use_unroll
                else self.fused_cache_bucket_graphs.get(cache_len)
            ),
            "codegen_graph_kind": "prefill_unroll" if use_unroll else "single_fused",
            "codegen_schedule": self.codegen_schedule,
            "scheduled_unrolls": list(self.codegen_schedule_unrolls),
            "active_codegen_unroll": int(initial_unroll if use_unroll else 1),
            "codegen_decode_unroll": self.codegen_decode_unroll,
            "decode_unroll_graph_available": bool(decode_unroll_graph_available),
            "decode_unroll_available": bool(decode_unroll_request is not None),
            "decode_unroll_stateful_mask": bool(decode_unroll_stateful_mask),
        }
        generated_count = 0
        generated_first_codes = []
        repeated_mask = np.zeros((1, int(self.ids["vocab_size"])), dtype=np.float32)
        penalty = np.asarray([repetition_penalty], dtype=np.float32)
        started = time.time()

        next_inputs_embeds = sequence.astype(np.float32, copy=False)
        next_cache_position = np.arange(prompt_len, dtype=np.int64)
        next_attention_mask = self.make_attention_mask(
            next_cache_position,
            prompt_len,
            unroll_cache_len if use_unroll else prompt_len,
        )
        active_unroll = initial_unroll
        active_request = unroll_request
        initialized_unroll_requests = {(initial_unroll, unroll_cache_len)} if use_unroll else set()

        while generated_count < max_new_tokens:
            step = generated_count
            if use_unroll:
                scheduled_unroll = initial_unroll if decode_unroll_ready else self.select_codegen_unroll_for_step(generated_count)
                if scheduled_unroll != active_unroll and not decode_unroll_ready:
                    scheduled_required_len = self.unroll_required_cache_len(prompt_len, max_new_tokens, scheduled_unroll)
                    scheduled_cache_len, scheduled_step, scheduled_request = self.get_fused_cache_unroll_step(
                        scheduled_required_len,
                        scheduled_unroll,
                    )
                    request_key = (scheduled_unroll, scheduled_cache_len)
                    if request_key not in initialized_unroll_requests:
                        scheduled_request.reset_state()
                        self._copy_matching_states(active_request, scheduled_request)
                        initialized_unroll_requests.add(request_key)
                    active_unroll = scheduled_unroll
                    active_request = scheduled_request
                    unroll_cache_len = scheduled_cache_len
                    unroll_step = scheduled_step
                    unroll_request = scheduled_request
                    unroll_required_len = scheduled_required_len
                    next_attention_mask = self.make_attention_mask(next_cache_position, next_inputs_embeds.shape[1], unroll_cache_len)
                current_request = decode_unroll_request if decode_unroll_ready else active_request
                current_step = decode_unroll_step if decode_unroll_ready else unroll_step
                current_label = (
                    f"fused_cache_decode_unroll{active_unroll}"
                    if decode_unroll_ready
                    else f"fused_cache_unroll{active_unroll}"
                )
                current_kind = "decode_unroll_statefulmask" if decode_unroll_ready and decode_unroll_stateful_mask else (
                    "decode_unroll" if decode_unroll_ready else "prefill_unroll"
                )
                allow_eos = np.asarray(
                    [1.0 if step + offset >= min_new_tokens else 0.0 for offset in range(active_unroll)],
                    dtype=np.float32,
                )
                inputs = {
                    "inputs_embeds": next_inputs_embeds,
                    "cache_position": next_cache_position,
                    "tts_pad_embed": tts_pad_embed.astype(np.float32, copy=False),
                    "allow_eos_steps": allow_eos,
                    "repetition_penalty": penalty,
                }
                if decode_unroll_ready:
                    if not decode_unroll_stateful_mask:
                        inputs["repeated_mask"] = repeated_mask
                else:
                    inputs["attention_mask"] = next_attention_mask
                    inputs["repeated_mask"] = repeated_mask
                step_started = time.time()
                outputs = run_request(
                    current_request,
                    current_step,
                    inputs,
                    self.ov_profiler,
                    current_label,
                )
                self.timings.add("fused_step", time.time() - step_started)
                first_codes, codes, frame_embed = outputs[:3]
                repeated_mask_out = None if decode_unroll_stateful_mask else outputs[3]
                self.last_codegen_info.update(
                    {
                        "selected_bucket": int(decode_unroll_cache_len if decode_unroll_ready else unroll_cache_len),
                        "selected_codegen_graph": (
                            (self.fused_cache_decode_unroll_stateful_mask_bucket_graphs if decode_unroll_stateful_mask else self.fused_cache_decode_unroll_bucket_graphs).get(decode_unroll_cache_len)
                            if decode_unroll_ready
                            else self.fused_cache_unroll_bucket_graphs_by_step.get(active_unroll, {}).get(unroll_cache_len)
                        ),
                        "codegen_graph_kind": current_kind,
                        "active_codegen_unroll": int(active_unroll),
                        "unroll_required_cache_len": int(unroll_required_len),
                    }
                )

                first_codes = first_codes.reshape(-1).astype(np.int64, copy=False)
                codes = codes.astype(np.int64, copy=False).reshape(-1, self.num_code_groups)
                stop = False
                for offset in range(min(active_unroll, max_new_tokens - generated_count)):
                    first_code_int = int(first_codes[offset])
                    if first_code_int == int(self.ids["codec_eos_token_id"]):
                        stop = True
                        break
                    generated_count += 1
                    generated_first_codes.append(first_code_int)
                    if progress_interval and generated_count % progress_interval == 0:
                        elapsed = time.time() - started
                        print(f"generated {generated_count}/{max_new_tokens} codec tokens in {elapsed:.1f}s", flush=True)
                    yield codes[offset]
                if stop or generated_count >= max_new_tokens:
                    break
                if repeated_mask_out is not None:
                    repeated_mask = repeated_mask_out.astype(np.float32, copy=False)
                next_inputs_embeds = frame_embed.astype(np.float32, copy=False)
                next_cache_position = np.asarray([prompt_len + generated_count - 1], dtype=np.int64)
                if decode_unroll_request is not None and not decode_unroll_ready:
                    self._copy_matching_states(active_request, decode_unroll_request)
                    if decode_unroll_stateful_mask:
                        self._set_request_state(decode_unroll_request, "repeated_mask", repeated_mask)
                    decode_unroll_ready = True
                if not decode_unroll_ready:
                    next_attention_mask = self.make_attention_mask(next_cache_position, 1, unroll_cache_len)
                continue

            allow_eos = np.asarray([1.0 if step >= min_new_tokens else 0.0], dtype=np.float32)
            step_started = time.time()
            first_code, codes, frame_embed = run_request(
                fused_cache_request,
                fused_cache_step,
                {
                    "inputs_embeds": next_inputs_embeds,
                    "cache_position": next_cache_position,
                    "attention_mask": next_attention_mask,
                    "tts_pad_embed": tts_pad_embed.astype(np.float32, copy=False),
                    "repeated_mask": repeated_mask,
                    "allow_eos": allow_eos,
                    "repetition_penalty": penalty,
                },
                self.ov_profiler,
                "fused_cache_step",
            )
            self.timings.add("fused_step", time.time() - step_started)
            self.last_codegen_info.update(
                {
                    "selected_bucket": int(cache_len),
                    "selected_codegen_graph": self.fused_cache_bucket_graphs.get(cache_len),
                    "codegen_graph_kind": "single_fused",
                }
            )

            first_code_int = int(first_code.reshape(-1)[0])
            if first_code_int == int(self.ids["codec_eos_token_id"]):
                break

            codes = codes.astype(np.int64, copy=False)
            generated_count += 1
            generated_first_codes.append(first_code_int)
            repeated_mask[0, first_code_int] = 1.0

            if progress_interval and generated_count % progress_interval == 0:
                elapsed = time.time() - started
                print(f"generated {generated_count}/{max_new_tokens} codec tokens in {elapsed:.1f}s", flush=True)
            yield codes[0]

            if generated_count >= max_new_tokens:
                break

            next_inputs_embeds = frame_embed.astype(np.float32, copy=False)
            next_cache_position = np.asarray([prompt_len + step], dtype=np.int64)
            next_attention_mask = self.make_attention_mask(next_cache_position, 1, prompt_len + step + 1)

        if generated_count == 0:
            raise RuntimeError("generation stopped before producing any codec token")

    def generate_codes_no_cache(
        self,
        text: str,
        instruct: str,
        language: str,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        max_prompt_tokens: int,
        progress_interval: int,
        speaker: str | None = None,
        voice_clone_prompt: VoiceClonePromptItem | None = None,
        ref_text: str | None = None,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ):
        sequence, tts_pad_embed = self.build_prompt(
            text,
            instruct,
            language,
            max_prompt_tokens,
            speaker=speaker,
            voice_clone_prompt=voice_clone_prompt,
            ref_text=ref_text,
        )
        generated = []
        generated_first_codes = []
        started = time.time()
        for step in range(max_new_tokens):
            talker_started = time.time()
            talker_inputs = [sequence.astype(np.float32, copy=False)]
            self.dump_calibration("talker_no_cache", talker_inputs)
            logits, past_hidden = run_request(
                self.talker_request,
                self.talker,
                talker_inputs,
                self.ov_profiler,
                "talker_no_cache",
            )
            self.timings.add("talker", time.time() - talker_started)
            first_code = self.select_first_code(
                logits,
                generated_first_codes,
                step,
                min_new_tokens,
                repetition_penalty,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            if first_code == int(self.ids["codec_eos_token_id"]):
                break

            code_input = np.asarray([[first_code]], dtype=np.int64)
            subcode_started = time.time()
            subcode_inputs = [past_hidden.astype(np.float32), code_input]
            self.dump_calibration("subcode_greedy", subcode_inputs)
            codes, sum_embed = run_request(
                self.subcode_request,
                self.subcode_greedy,
                subcode_inputs,
                self.ov_profiler,
                "subcode_greedy",
            )
            self.timings.add("subcode", time.time() - subcode_started)
            codes = codes.astype(np.int64, copy=False)
            generated.append(codes[0])
            generated_first_codes.append(first_code)

            frame_embed = sum_embed.astype(np.float32, copy=False) + tts_pad_embed
            sequence = np.concatenate([sequence, frame_embed], axis=1)
            if progress_interval and (step + 1) % progress_interval == 0:
                elapsed = time.time() - started
                print(f"generated {step + 1}/{max_new_tokens} codec tokens in {elapsed:.1f}s", flush=True)

        if not generated:
            raise RuntimeError("generation stopped before producing any codec token")
        return np.stack(generated, axis=0)

    def generate_codes_fused_no_cache(
        self,
        text: str,
        instruct: str,
        language: str,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        max_prompt_tokens: int,
        progress_interval: int,
        speaker: str | None = None,
        voice_clone_prompt: VoiceClonePromptItem | None = None,
        ref_text: str | None = None,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ):
        if do_sample:
            raise ValueError("do_sample=True is not supported by fused-no-cache OpenVINO graphs; use --mode no-cache")
        sequence, tts_pad_embed = self.build_prompt(
            text,
            instruct,
            language,
            max_prompt_tokens,
            speaker=speaker,
            voice_clone_prompt=voice_clone_prompt,
            ref_text=ref_text,
        )
        generated = []
        generated_first_codes = []
        started = time.time()
        for step in range(max_new_tokens):
            repeated_mask = np.zeros((1, int(self.ids["vocab_size"])), dtype=np.float32)
            for token_id in set(generated_first_codes):
                repeated_mask[0, token_id] = 1.0
            allow_eos = np.asarray([1.0 if step >= min_new_tokens else 0.0], dtype=np.float32)
            penalty = np.asarray([repetition_penalty], dtype=np.float32)

            step_started = time.time()
            first_code, codes, frame_embed = run_request(
                self.fused_request,
                self.fused_step,
                [
                    sequence.astype(np.float32, copy=False),
                    tts_pad_embed.astype(np.float32, copy=False),
                    repeated_mask,
                    allow_eos,
                    penalty,
                ],
                self.ov_profiler,
                "fused_no_cache_step",
            )
            self.timings.add("fused_step", time.time() - step_started)
            first_code_int = int(first_code.reshape(-1)[0])
            if first_code_int == int(self.ids["codec_eos_token_id"]):
                break

            codes = codes.astype(np.int64, copy=False)
            generated.append(codes[0])
            generated_first_codes.append(first_code_int)
            sequence = np.concatenate([sequence, frame_embed.astype(np.float32, copy=False)], axis=1)
            if progress_interval and (step + 1) % progress_interval == 0:
                elapsed = time.time() - started
                print(f"generated {step + 1}/{max_new_tokens} codec tokens in {elapsed:.1f}s", flush=True)

        if not generated:
            raise RuntimeError("generation stopped before producing any codec token")
        return np.stack(generated, axis=0)

    def generate_codes_cache_split(
        self,
        text: str,
        instruct: str,
        language: str,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        max_prompt_tokens: int,
        progress_interval: int,
        speaker: str | None = None,
        voice_clone_prompt: VoiceClonePromptItem | None = None,
        ref_text: str | None = None,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ):
        sequence, tts_pad_embed = self.build_prompt(
            text,
            instruct,
            language,
            max_prompt_tokens,
            speaker=speaker,
            voice_clone_prompt=voice_clone_prompt,
            ref_text=ref_text,
        )
        prompt_len = int(sequence.shape[1])
        required_len = prompt_len + max_new_tokens
        cache_len, talker_stateful, talker_request = self.get_talker_stateful(required_len)

        talker_request.reset_state()
        prompt_positions = np.arange(prompt_len, dtype=np.int64)
        prompt_mask = self.make_attention_mask(prompt_positions, prompt_len, prompt_len)

        started = time.time()
        talker_started = time.time()
        logits, past_hidden = run_request(
            talker_request,
            talker_stateful,
            {
                "inputs_embeds": sequence.astype(np.float32, copy=False),
                "cache_position": prompt_positions,
                "attention_mask": prompt_mask,
            },
            self.ov_profiler,
            "talker_cache_prefill",
        )
        self.timings.add("talker", time.time() - talker_started)

        generated = []
        generated_first_codes = []
        for step in range(max_new_tokens):
            first_code = self.select_first_code(
                logits,
                generated_first_codes,
                step,
                min_new_tokens,
                repetition_penalty,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            if first_code == int(self.ids["codec_eos_token_id"]):
                break

            code_input = np.asarray([[first_code]], dtype=np.int64)
            subcode_started = time.time()
            codes, sum_embed = run_request(
                self.subcode_request,
                self.subcode_greedy,
                [past_hidden.astype(np.float32, copy=False), code_input],
                self.ov_profiler,
                "subcode_greedy",
            )
            self.timings.add("subcode", time.time() - subcode_started)
            codes = codes.astype(np.int64, copy=False)
            generated.append(codes[0])
            generated_first_codes.append(first_code)

            if progress_interval and (step + 1) % progress_interval == 0:
                elapsed = time.time() - started
                print(f"generated {step + 1}/{max_new_tokens} codec tokens in {elapsed:.1f}s", flush=True)

            if step + 1 >= max_new_tokens:
                break

            frame_embed = sum_embed.astype(np.float32, copy=False) + tts_pad_embed
            cache_position = np.asarray([prompt_len + step], dtype=np.int64)
            attention_mask = self.make_attention_mask(cache_position, 1, prompt_len + step + 1)
            talker_started = time.time()
            logits, past_hidden = run_request(
                talker_request,
                talker_stateful,
                {
                    "inputs_embeds": frame_embed.astype(np.float32, copy=False),
                    "cache_position": cache_position,
                    "attention_mask": attention_mask,
                },
                self.ov_profiler,
                "talker_cache_decode",
            )
            self.timings.add("talker", time.time() - talker_started)

        if not generated:
            raise RuntimeError("generation stopped before producing any codec token")
        return np.stack(generated, axis=0)

    def generate_codes_cache_fused(
        self,
        text: str,
        instruct: str,
        language: str,
        max_new_tokens: int,
        min_new_tokens: int,
        repetition_penalty: float,
        max_prompt_tokens: int,
        progress_interval: int,
        speaker: str | None = None,
        voice_clone_prompt: VoiceClonePromptItem | None = None,
        ref_text: str | None = None,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ):
        if do_sample:
            raise ValueError("do_sample=True is not supported by fused cache OpenVINO graphs; use --cache-step split")
        sequence, tts_pad_embed = self.build_prompt(
            text,
            instruct,
            language,
            max_prompt_tokens,
            speaker=speaker,
            voice_clone_prompt=voice_clone_prompt,
            ref_text=ref_text,
        )
        prompt_len = int(sequence.shape[1])
        required_len = prompt_len + max_new_tokens
        cache_len, fused_cache_step, fused_cache_request = self.get_fused_cache_step(required_len)

        fused_cache_request.reset_state()
        generated = []
        generated_first_codes = []
        repeated_mask = np.zeros((1, int(self.ids["vocab_size"])), dtype=np.float32)
        penalty = np.asarray([repetition_penalty], dtype=np.float32)
        started = time.time()

        next_inputs_embeds = sequence.astype(np.float32, copy=False)
        next_cache_position = np.arange(prompt_len, dtype=np.int64)
        next_attention_mask = self.make_attention_mask(next_cache_position, prompt_len, prompt_len)

        for step in range(max_new_tokens):
            allow_eos = np.asarray([1.0 if step >= min_new_tokens else 0.0], dtype=np.float32)
            step_started = time.time()
            first_code, codes, frame_embed = run_request(
                fused_cache_request,
                fused_cache_step,
                {
                    "inputs_embeds": next_inputs_embeds,
                    "cache_position": next_cache_position,
                    "attention_mask": next_attention_mask,
                    "tts_pad_embed": tts_pad_embed.astype(np.float32, copy=False),
                    "repeated_mask": repeated_mask,
                    "allow_eos": allow_eos,
                    "repetition_penalty": penalty,
                },
                self.ov_profiler,
                "fused_cache_step",
            )
            self.timings.add("fused_step", time.time() - step_started)

            first_code_int = int(first_code.reshape(-1)[0])
            if first_code_int == int(self.ids["codec_eos_token_id"]):
                break

            codes = codes.astype(np.int64, copy=False)
            generated.append(codes[0])
            generated_first_codes.append(first_code_int)
            repeated_mask[0, first_code_int] = 1.0

            if progress_interval and (step + 1) % progress_interval == 0:
                elapsed = time.time() - started
                print(f"generated {step + 1}/{max_new_tokens} codec tokens in {elapsed:.1f}s", flush=True)

            if step + 1 >= max_new_tokens:
                break

            next_inputs_embeds = frame_embed.astype(np.float32, copy=False)
            next_cache_position = np.asarray([prompt_len + step], dtype=np.int64)
            next_attention_mask = self.make_attention_mask(next_cache_position, 1, prompt_len + step + 1)

        if not generated:
            raise RuntimeError("generation stopped before producing any codec token")
        return np.stack(generated, axis=0)

    def encode_audio_codes(self, audio, sr: int | None = None) -> np.ndarray:
        if self.speech_encoder_request is None:
            raise RuntimeError("this IR does not include speech_encoder.xml; export a Base model with clone support")
        wav = load_audio(audio, target_sr=self.input_sample_rate, sr=sr)
        input_values = wav.reshape(1, -1).astype(np.float32)
        padding_mask = np.ones_like(input_values, dtype=np.int64)
        started = time.time()
        outputs = run_request(
            self.speech_encoder_request,
            self.speech_encoder,
            [input_values, padding_mask],
            self.ov_profiler,
            "speech_encoder",
        )
        self.timings.add("encode", time.time() - started)
        codes = outputs[0].astype(np.int64, copy=False)
        if codes.ndim == 3:
            codes = codes[0]
        valid = int(np.ceil(len(wav) / max(self.encode_downsample_rate, 1)))
        return codes[:valid]

    def extract_speaker_embedding(self, audio, sr: int | None = None) -> np.ndarray:
        if self.speaker_encoder_request is None:
            raise RuntimeError("this IR does not include speaker_encoder.xml; export a Base model with clone support")
        wav = load_audio(audio, target_sr=self.speaker_encoder_sample_rate, sr=sr)
        mels = speaker_mel_spectrogram(wav, sr=self.speaker_encoder_sample_rate).astype(np.float32)
        started = time.time()
        outputs = run_request(
            self.speaker_encoder_request,
            self.speaker_encoder,
            [mels.reshape(1, mels.shape[0], mels.shape[1])],
            self.ov_profiler,
            "speaker_encoder",
        )
        self.timings.add("speaker", time.time() - started)
        embed = outputs[0].astype(np.float32, copy=False)
        return embed.reshape(-1)

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode: bool = False):
        if not x_vector_only_mode and (ref_text is None or ref_text == ""):
            raise ValueError("ref_text is required when x_vector_only_mode=False")
        ref_code = None if x_vector_only_mode else self.encode_audio_codes(ref_audio)
        spk_embed = self.extract_speaker_embedding(ref_audio)
        return [
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=spk_embed,
                x_vector_only_mode=bool(x_vector_only_mode),
                icl_mode=not bool(x_vector_only_mode),
                ref_text=ref_text,
            )
        ]

    @classmethod
    def from_ir(cls, ir_dir: str | Path, **kwargs):
        realtime_profile = kwargs.pop("realtime_profile", None)
        if realtime_profile in {None, "fastest"} and "mode" not in kwargs:
            kwargs.update({"mode": "fastest"})
        elif realtime_profile not in {None, "fastest"}:
            raise ValueError("Python API supports realtime_profile='fastest'; use low-level mode kwargs for dev experiments")
        return cls(str(ir_dir), **kwargs)

    @staticmethod
    def _ensure_list(value):
        return value if isinstance(value, list) else [value]

    def generate_voice_design(
        self,
        text,
        instruct,
        language=None,
        max_new_tokens: int = 512,
        min_new_tokens: int = 2,
        repetition_penalty: float = 1.05,
        max_prompt_tokens: int = 512,
        progress_interval: int = 8,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ):
        texts = self._ensure_list(text)
        instructs = self._ensure_list(instruct)
        languages = self._ensure_list(language if language is not None else "Auto")
        if len(instructs) == 1 and len(texts) > 1:
            instructs *= len(texts)
        if len(languages) == 1 and len(texts) > 1:
            languages *= len(texts)
        if not (len(texts) == len(instructs) == len(languages)):
            raise ValueError("text, instruct, and language batch sizes must match")
        wavs = []
        for item_text, item_instruct, item_language in zip(texts, instructs, languages):
            codes = self.generate_codes(
                text=item_text,
                instruct=item_instruct or "",
                language=item_language,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                max_prompt_tokens=max_prompt_tokens,
                progress_interval=progress_interval,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            wavs.append(self.decode(codes))
        return wavs, self.sample_rate

    def generate_custom_voice(
        self,
        text,
        speaker,
        language=None,
        instruct=None,
        max_new_tokens: int = 512,
        min_new_tokens: int = 2,
        repetition_penalty: float = 1.05,
        max_prompt_tokens: int = 512,
        progress_interval: int = 8,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ):
        texts = self._ensure_list(text)
        speakers = self._ensure_list(speaker)
        languages = self._ensure_list(language if language is not None else "Auto")
        instructs = self._ensure_list(instruct if instruct is not None else "")
        if len(speakers) == 1 and len(texts) > 1:
            speakers *= len(texts)
        if len(languages) == 1 and len(texts) > 1:
            languages *= len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs *= len(texts)
        if not (len(texts) == len(speakers) == len(languages) == len(instructs)):
            raise ValueError("text, speaker, language, and instruct batch sizes must match")
        wavs = []
        for item_text, item_speaker, item_language, item_instruct in zip(texts, speakers, languages, instructs):
            codes = self.generate_codes(
                text=item_text,
                instruct=item_instruct or "",
                language=item_language,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                max_prompt_tokens=max_prompt_tokens,
                progress_interval=progress_interval,
                speaker=item_speaker,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            wavs.append(self.decode(codes))
        return wavs, self.sample_rate

    def _normalize_voice_clone_prompt(self, prompt, text_count: int):
        if isinstance(prompt, dict):
            items = []
            for index in range(len(prompt["ref_spk_embedding"])):
                ref_codes = prompt.get("ref_code")
                items.append(
                    VoiceClonePromptItem(
                        ref_code=None if ref_codes is None else ref_codes[index],
                        ref_spk_embedding=prompt["ref_spk_embedding"][index],
                        x_vector_only_mode=bool(prompt.get("x_vector_only_mode", [False])[index]),
                        icl_mode=bool(prompt.get("icl_mode", [True])[index]),
                    )
                )
        else:
            items = prompt
        if not isinstance(items, list):
            items = [items]
        if len(items) == 1 and text_count > 1:
            items = items * text_count
        if len(items) != text_count:
            raise ValueError("voice_clone_prompt and text batch sizes must match")
        return items

    def generate_voice_clone(
        self,
        text,
        language=None,
        ref_audio=None,
        ref_text=None,
        x_vector_only_mode: bool = False,
        voice_clone_prompt=None,
        max_new_tokens: int = 512,
        min_new_tokens: int = 2,
        repetition_penalty: float = 1.05,
        max_prompt_tokens: int = 512,
        progress_interval: int = 8,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ):
        texts = self._ensure_list(text)
        languages = self._ensure_list(language if language is not None else "Auto")
        if len(languages) == 1 and len(texts) > 1:
            languages *= len(texts)
        if len(texts) != len(languages):
            raise ValueError("text and language batch sizes must match")
        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError("either voice_clone_prompt or ref_audio is required")
            prompts = self.create_voice_clone_prompt(ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode)
        else:
            prompts = self._normalize_voice_clone_prompt(voice_clone_prompt, len(texts))
        if len(prompts) == 1 and len(texts) > 1:
            prompts = prompts * len(texts)
        wavs = []
        for item_text, item_language, prompt in zip(texts, languages, prompts):
            item_ref_text = prompt.ref_text if prompt.ref_text is not None else ref_text
            if self._native_pipeline_mode() in {"1", "true", "on", "require"} and self.talker_request is None:
                chunks = list(
                    self.stream_voice_clone(
                        text=item_text,
                        language=item_language,
                        ref_text=item_ref_text,
                        voice_clone_prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        repetition_penalty=repetition_penalty,
                        max_prompt_tokens=max_prompt_tokens,
                        progress_interval=progress_interval,
                        do_sample=do_sample,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                    )
                )
                audio_parts = [np.asarray(chunk.audio, dtype=np.float32) for chunk in chunks if chunk.audio.size]
                if not audio_parts:
                    raise RuntimeError("voice clone generation stopped before producing any audio chunk")
                wavs.append(np.concatenate(audio_parts, axis=0))
                continue
            codes = self.generate_codes(
                text=item_text,
                instruct="",
                language=item_language,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                max_prompt_tokens=max_prompt_tokens,
                progress_interval=progress_interval,
                voice_clone_prompt=prompt,
                ref_text=item_ref_text,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            decode_codes = codes
            ref_code = prompt.ref_code
            if ref_code is not None:
                decode_codes = np.concatenate([np.asarray(ref_code, dtype=np.int64), codes], axis=0)
            wav = self.decode(decode_codes)
            if ref_code is not None:
                ref_len = int(np.asarray(ref_code).shape[0])
                total_len = max(int(decode_codes.shape[0]), 1)
                cut = int(ref_len / total_len * wav.shape[0])
                wav = wav[cut:]
            wavs.append(wav)
        return wavs, self.sample_rate

    @staticmethod
    def _ensure_scalar(value, name: str):
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"streaming {name} only supports a single item")
            return value[0]
        return value

    def _stream_decoder_key(
        self,
        context_frames: int,
        new_frames: int,
        preferred_chunk_frames: int,
        left_context_frames: int,
    ):
        candidates = self._stream_decoder_key_candidates(
            context_frames=context_frames,
            new_frames=new_frames,
            preferred_chunk_frames=preferred_chunk_frames,
            left_context_frames=left_context_frames,
        )
        return candidates[0] if candidates else None

    def _stream_decoder_key_candidates(
        self,
        context_frames: int,
        new_frames: int,
        preferred_chunk_frames: int,
        left_context_frames: int,
    ) -> list[tuple[int, int]]:
        if not self.streaming_decoder_graphs_by_context:
            return []

        context_candidates = []
        if context_frames in self.streaming_decoder_graphs_by_context:
            context_candidates.append(context_frames)
        if context_frames == 0 and 0 in self.streaming_decoder_graphs_by_context:
            context_candidates.append(0)
        if context_frames > 0 and left_context_frames in self.streaming_decoder_graphs_by_context:
            context_candidates.append(left_context_frames)
        if context_frames > 0:
            context_candidates.extend(
                context
                for context in sorted(self.streaming_decoder_graphs_by_context)
                if context >= context_frames
            )
            context_candidates.extend(sorted(self.streaming_decoder_graphs_by_context))

        seen_contexts = set()
        results = []
        for context in context_candidates:
            if context in seen_contexts:
                continue
            seen_contexts.add(context)
            chunk_graphs = self.streaming_decoder_graphs_by_context.get(context, {})
            if not chunk_graphs:
                continue
            chunk_candidates = self._stream_decoder_chunk_key_candidates(chunk_graphs, new_frames, preferred_chunk_frames)
            results.extend((context, chunk) for chunk in chunk_candidates)
        return results

    @staticmethod
    def _stream_decoder_chunk_key(chunk_graphs: dict[int, str], new_frames: int, preferred_chunk_frames: int):
        candidates = OpenVINOQwen3TTS._stream_decoder_chunk_key_candidates(
            chunk_graphs,
            new_frames,
            preferred_chunk_frames,
        )
        return candidates[0] if candidates else None

    @staticmethod
    def _stream_decoder_chunk_key_candidates(
        chunk_graphs: dict[int, str],
        new_frames: int,
        preferred_chunk_frames: int,
    ) -> list[int]:
        candidates = []
        if preferred_chunk_frames in chunk_graphs and preferred_chunk_frames >= new_frames:
            candidates.append(preferred_chunk_frames)
        candidates.extend(key for key in sorted(chunk_graphs) if key >= new_frames and key not in candidates)
        if chunk_graphs:
            largest = max(chunk_graphs)
            if largest not in candidates:
                candidates.append(largest)
        return candidates

    def _get_stream_decoder(self, context_frames: int, chunk_frames: int):
        key = (int(context_frames), int(chunk_frames))
        if key not in self.streaming_decoders:
            started = time.time()
            graph = self.streaming_decoder_graphs_by_context[int(context_frames)][int(chunk_frames)]
            compiled = compile_model(
                self.core,
                self.ir_dir / graph,
                self.decoder_device,
                self.cache_dir,
                self.allow_cpu_fallback,
                self.ov_profiler.enabled,
                self.precision_hint,
                self.compile_config,
            )
            if not self.allow_cpu_fallback and self.decoder_device == "GPU":
                self.assert_gpu_execution([compiled])
            self.streaming_decoders[key] = compiled
            self.streaming_decoder_requests[key] = compiled.create_infer_request()
            print(
                f"compiled streaming decoder context {context_frames} chunk {chunk_frames} "
                f"on {self.decoder_device} in {time.time() - started:.1f}s",
                flush=True,
            )
        return self.streaming_decoders[key], self.streaming_decoder_requests[key]

    def _pad_stream_context(self, window_codes: np.ndarray, context_frames: int, target_context_frames: int):
        if target_context_frames <= context_frames:
            return window_codes, context_frames
        if context_frames <= 0:
            return window_codes, context_frames
        pad_count = target_context_frames - context_frames
        pad_frame = window_codes[:1]
        padding = np.repeat(pad_frame, pad_count, axis=0)
        return np.concatenate([padding, window_codes], axis=0), target_context_frames

    def decode_stream_window(
        self,
        window_codes: np.ndarray,
        context_frames: int,
        new_frames: int,
        chunk_frames: int = 12,
        left_context_frames: int = 25,
    ) -> np.ndarray:
        if new_frames <= 0:
            return np.zeros((0,), dtype=np.float32)

        stream_keys = self._stream_decoder_key_candidates(
            context_frames=context_frames,
            new_frames=new_frames,
            preferred_chunk_frames=chunk_frames,
            left_context_frames=left_context_frames,
        )
        last_stream_error = None
        for stream_key in stream_keys:
            target_context, target_chunk = stream_key
            padded_window, effective_context = self._pad_stream_context(window_codes, context_frames, target_context)
            started = time.time()
            try:
                compiled, request = self._get_stream_decoder(target_context, target_chunk)
                audio = run_request(
                    request,
                    compiled,
                    [np.asarray(padded_window, dtype=np.int64).reshape(1, -1, self.num_code_groups)],
                    self.ov_profiler,
                    f"speech_decoder_stream_c{target_context}_t{target_chunk}",
                )[0][0].astype(np.float32, copy=False)
                elapsed = time.time() - started
                self.timings.add("decode", elapsed)
                self.last_stream_decode_info = {
                    "decode_path": f"stream:c{target_context}_t{target_chunk}",
                    "decode_context_frames": int(effective_context),
                    "decode_chunk_graph_frames": int(target_chunk),
                    "decode_ms": elapsed * 1000.0,
                    "fallback": False,
                }
                return audio[: new_frames * self.decode_upsample_rate]
            except Exception as exc:
                last_stream_error = exc
                failed_key = (int(target_context), int(target_chunk))
                self.streaming_decoders.pop(failed_key, None)
                self.streaming_decoder_requests.pop(failed_key, None)
                print(
                    f"warning: streaming decoder c{target_context}_t{target_chunk} failed; "
                    f"trying fallback path: {exc}",
                    flush=True,
                )

        started = time.time()
        audio = self.decode(window_codes)
        elapsed = time.time() - started
        start = context_frames * self.decode_upsample_rate
        end = start + new_frames * self.decode_upsample_rate
        self.last_stream_decode_info = {
            "decode_path": "fallback:speech_decoder",
            "decode_context_frames": int(context_frames),
            "decode_chunk_graph_frames": None,
            "decode_ms": elapsed * 1000.0,
            "fallback": True,
            "stream_decoder_error": str(last_stream_error) if last_stream_error is not None else None,
        }
        return audio[start:end]

    def stream_decode_codes(
        self,
        code_iter,
        prefix_codes: np.ndarray | None = None,
        chunk_frames: int | None = None,
        left_context_frames: int | None = None,
        initial_chunk_frames: int | None = None,
        chunk_strategy: str | None = None,
    ):
        stream_config = self._resolve_stream_chunk_config(
            chunk_strategy=chunk_strategy,
            chunk_frames=chunk_frames,
            initial_chunk_frames=initial_chunk_frames,
            left_context_frames=left_context_frames,
        )
        strategy = stream_config["strategy"]
        initial_chunk_frames = int(stream_config["initial_chunk_frames"])
        chunk_frames = int(stream_config["chunk_frames"])
        left_context_frames = int(stream_config["left_context_frames"])
        if getattr(self, "stream_pipeline_decode", False):
            yield from self._stream_decode_codes_pipelined(
                code_iter,
                prefix_codes=prefix_codes,
                chunk_frames=chunk_frames,
                left_context_frames=left_context_frames,
                initial_chunk_frames=initial_chunk_frames,
                chunk_strategy=strategy,
            )
            return

        all_codes = []
        if prefix_codes is not None:
            prefix = np.asarray(prefix_codes, dtype=np.int64)
            if prefix.ndim == 1:
                prefix = prefix.reshape(1, -1)
            if prefix.shape[-1] != self.num_code_groups:
                raise ValueError(f"prefix_codes must have {self.num_code_groups} code groups")
            all_codes.extend(prefix.reshape(-1, self.num_code_groups))
        prefix_frames = len(all_codes)
        emitted_frames = prefix_frames
        pending_frames = 0
        chunk_index = 0
        stream_started = time.time()
        codegen_started = stream_started
        stream_audio_ms = 0.0
        stream_compute_ms = 0.0

        def emit(is_final: bool):
            nonlocal emitted_frames, pending_frames, chunk_index, codegen_started, stream_audio_ms, stream_compute_ms
            total_frames = len(all_codes)
            new_frames = total_frames - emitted_frames
            emit_started = time.time()
            codegen_ms = max(0.0, (emit_started - codegen_started) * 1000.0)
            if new_frames > 0:
                context_start = max(0, emitted_frames - left_context_frames)
                context_frames = emitted_frames - context_start
                window = np.stack(all_codes[context_start:total_frames], axis=0).astype(np.int64, copy=False)
                decode_started = time.time()
                audio = self.decode_stream_window(
                    window,
                    context_frames=context_frames,
                    new_frames=new_frames,
                    chunk_frames=chunk_frames,
                    left_context_frames=left_context_frames,
                )
                decode_ms = (time.time() - decode_started) * 1000.0
                decode_info = dict(getattr(self, "last_stream_decode_info", {}) or {})
                decode_ms = float(decode_info.get("decode_ms", decode_ms))
                codes = np.stack(all_codes[emitted_frames:total_frames], axis=0).astype(np.int64, copy=False)
                emitted_frames = total_frames
                pending_frames = 0
            else:
                audio = np.zeros((0,), dtype=np.float32)
                codes = np.empty((0, self.num_code_groups), dtype=np.int64)
                decode_ms = 0.0
                decode_info = {"decode_path": "none", "fallback": False}

            audio_ms = (float(audio.shape[0]) / float(self.sample_rate) * 1000.0) if audio.size else 0.0
            compute_ms = codegen_ms + decode_ms
            rtf = (compute_ms / audio_ms) if audio_ms > 0 else 0.0
            if audio_ms > 0:
                stream_audio_ms += audio_ms
                stream_compute_ms += compute_ms
            stream_elapsed_ms = max(0.0, (time.time() - stream_started) * 1000.0)
            stream_rtf = (stream_elapsed_ms / stream_audio_ms) if stream_audio_ms > 0 else 0.0
            stream_compute_rtf = (stream_compute_ms / stream_audio_ms) if stream_audio_ms > 0 else 0.0
            queue_hint_ms = max(0.0, audio_ms - compute_ms)
            producer_lag_ms = max(0.0, codegen_ms - audio_ms)

            chunk = StreamChunk(
                index=chunk_index,
                audio=audio,
                sample_rate=self.sample_rate,
                codes=codes,
                is_final=is_final,
                timings={
                    **self.timings.snapshot(max(emitted_frames - prefix_frames, 0)),
                    **decode_info,
                    **dict(getattr(self, "last_codegen_info", {}) or {}),
                    "codegen_ms": codegen_ms,
                    "decode_ms": decode_ms,
                    "chunk_compute_ms": compute_ms,
                    "chunk_audio_ms": audio_ms,
                    "rtf": rtf,
                    "stream_audio_ms": stream_audio_ms,
                    "stream_compute_ms": stream_compute_ms,
                    "stream_elapsed_ms": stream_elapsed_ms,
                    "stream_rtf": stream_rtf,
                    "stream_compute_rtf": stream_compute_rtf,
                    "queue_hint_ms": queue_hint_ms,
                    "queue_wait_ms": 0.0,
                    "producer_lag_ms": producer_lag_ms,
                    "strategy": strategy,
                    "codegen_unroll": int(getattr(self, "codegen_unroll", 1)),
                    "unroll_fallback": bool(getattr(self, "codegen_unroll_fallback", False)),
                    "initial_chunk_frames": int(initial_chunk_frames),
                    "configured_chunk_frames": int(chunk_frames),
                    "chunk_frames": int(codes.shape[0]),
                    "emitted_frames": int(max(emitted_frames - prefix_frames, 0)),
                    "prefix_frames": int(prefix_frames),
                    "is_final": bool(is_final),
                },
            )
            chunk_index += 1
            codegen_started = time.time()
            return chunk

        def current_target_frames() -> int:
            return initial_chunk_frames if chunk_index == 0 else chunk_frames

        for code in code_iter:
            frame = np.asarray(code, dtype=np.int64).reshape(-1)
            if frame.shape[0] != self.num_code_groups:
                raise ValueError(f"stream code frame must have {self.num_code_groups} code groups")
            all_codes.append(frame)
            pending_frames += 1
            if pending_frames >= current_target_frames():
                yield emit(False)

        if pending_frames > 0:
            yield emit(True)
        else:
            yield emit(True)

    def _stream_decode_codes_pipelined(
        self,
        code_iter,
        prefix_codes: np.ndarray | None = None,
        chunk_frames: int | None = None,
        left_context_frames: int | None = None,
        initial_chunk_frames: int | None = None,
        chunk_strategy: str | None = None,
    ):
        stream_config = self._resolve_stream_chunk_config(
            chunk_strategy=chunk_strategy,
            chunk_frames=chunk_frames,
            initial_chunk_frames=initial_chunk_frames,
            left_context_frames=left_context_frames,
        )
        strategy = stream_config["strategy"]
        initial_chunk_frames = int(stream_config["initial_chunk_frames"])
        chunk_frames = int(stream_config["chunk_frames"])
        left_context_frames = int(stream_config["left_context_frames"])

        out_queue = queue.Queue(maxsize=4)
        sentinel = object()

        def producer():
            all_codes = []
            if prefix_codes is not None:
                prefix = np.asarray(prefix_codes, dtype=np.int64)
                if prefix.ndim == 1:
                    prefix = prefix.reshape(1, -1)
                if prefix.shape[-1] != self.num_code_groups:
                    raise ValueError(f"prefix_codes must have {self.num_code_groups} code groups")
                all_codes.extend(prefix.reshape(-1, self.num_code_groups))

            prefix_frames = len(all_codes)
            emitted_frames = prefix_frames
            pending_frames = 0
            chunk_index = 0
            codegen_started = time.time()
            stream_started = codegen_started
            stream_audio_ms = 0.0
            stream_compute_ms = 0.0
            pending = deque()

            def build_decode_job(is_final: bool):
                nonlocal emitted_frames, pending_frames, chunk_index, codegen_started
                total_frames = len(all_codes)
                new_frames = total_frames - emitted_frames
                submit_time = time.time()
                codegen_ms = max(0.0, (submit_time - codegen_started) * 1000.0)
                current_index = chunk_index
                preferred_chunk_frames = initial_chunk_frames if current_index == 0 else chunk_frames
                chunk_index += 1

                if new_frames > 0:
                    context_start = max(0, emitted_frames - left_context_frames)
                    context_frames = emitted_frames - context_start
                    window = np.stack(all_codes[context_start:total_frames], axis=0).astype(np.int64, copy=True)
                    codes = np.stack(all_codes[emitted_frames:total_frames], axis=0).astype(np.int64, copy=True)
                    emitted_frames = total_frames
                    pending_frames = 0
                else:
                    context_frames = 0
                    window = np.empty((0, self.num_code_groups), dtype=np.int64)
                    codes = np.empty((0, self.num_code_groups), dtype=np.int64)
                emitted_relative = int(max(emitted_frames - prefix_frames, 0))

                codegen_started = time.time()

                def decode_job():
                    nonlocal stream_audio_ms, stream_compute_ms
                    decode_job_started = time.time()
                    queue_wait_ms = max(0.0, (decode_job_started - submit_time) * 1000.0)
                    if new_frames > 0:
                        decode_started = time.time()
                        audio = self.decode_stream_window(
                            window,
                            context_frames=context_frames,
                            new_frames=new_frames,
                            chunk_frames=preferred_chunk_frames,
                            left_context_frames=left_context_frames,
                        )
                        decode_ms = (time.time() - decode_started) * 1000.0
                        decode_info = dict(getattr(self, "last_stream_decode_info", {}) or {})
                        decode_ms = float(decode_info.get("decode_ms", decode_ms))
                    else:
                        audio = np.zeros((0,), dtype=np.float32)
                        decode_ms = 0.0
                        decode_info = {"decode_path": "none", "fallback": False}

                    audio_ms = (float(audio.shape[0]) / float(self.sample_rate) * 1000.0) if audio.size else 0.0
                    effective_compute_ms = max(codegen_ms, decode_ms)
                    rtf = (effective_compute_ms / audio_ms) if audio_ms > 0 else 0.0
                    if audio_ms > 0:
                        stream_audio_ms += audio_ms
                        stream_compute_ms += effective_compute_ms
                    stream_elapsed_ms = max(0.0, (time.time() - stream_started) * 1000.0)
                    stream_rtf = (stream_elapsed_ms / stream_audio_ms) if stream_audio_ms > 0 else 0.0
                    stream_compute_rtf = (stream_compute_ms / stream_audio_ms) if stream_audio_ms > 0 else 0.0
                    queue_hint_ms = max(0.0, audio_ms - effective_compute_ms)
                    producer_lag_ms = max(0.0, codegen_ms - audio_ms)
                    return StreamChunk(
                        index=current_index,
                        audio=audio,
                        sample_rate=self.sample_rate,
                        codes=codes,
                        is_final=is_final,
                        timings={
                            **self.timings.snapshot(emitted_relative),
                            **decode_info,
                            **dict(getattr(self, "last_codegen_info", {}) or {}),
                            "codegen_ms": codegen_ms,
                            "decode_ms": decode_ms,
                            "chunk_compute_ms": effective_compute_ms,
                            "chunk_audio_ms": audio_ms,
                            "rtf": rtf,
                            "stream_audio_ms": stream_audio_ms,
                            "stream_compute_ms": stream_compute_ms,
                            "stream_elapsed_ms": stream_elapsed_ms,
                            "stream_rtf": stream_rtf,
                            "stream_compute_rtf": stream_compute_rtf,
                            "queue_hint_ms": queue_hint_ms,
                            "queue_wait_ms": queue_wait_ms,
                            "producer_lag_ms": producer_lag_ms,
                            "strategy": strategy,
                            "codegen_unroll": int(getattr(self, "codegen_unroll", 1)),
                            "unroll_fallback": bool(getattr(self, "codegen_unroll_fallback", False)),
                            "initial_chunk_frames": int(initial_chunk_frames),
                            "configured_chunk_frames": int(chunk_frames),
                            "pipeline_decode": True,
                            "chunk_frames": int(codes.shape[0]),
                            "emitted_frames": emitted_relative,
                            "prefix_frames": int(prefix_frames),
                            "is_final": bool(is_final),
                        },
                    )

                return decode_job

            def drain_ready(wait: bool = False):
                while pending and (wait or pending[0].done()):
                    out_queue.put(pending.popleft().result())

            def current_target_frames() -> int:
                return initial_chunk_frames if chunk_index == 0 else chunk_frames

            try:
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen3_tts_decode") as executor:
                    for code in code_iter:
                        frame = np.asarray(code, dtype=np.int64).reshape(-1)
                        if frame.shape[0] != self.num_code_groups:
                            raise ValueError(f"stream code frame must have {self.num_code_groups} code groups")
                        all_codes.append(frame)
                        pending_frames += 1
                        drain_ready(False)

                        if pending_frames >= current_target_frames():
                            future = executor.submit(build_decode_job(False))
                            pending.append(future)
                            if chunk_index == 1:
                                out_queue.put(future.result())
                                pending.pop()
                            elif len(pending) >= 2:
                                drain_ready(True)

                    if pending_frames > 0:
                        pending.append(executor.submit(build_decode_job(True)))
                    else:
                        pending.append(executor.submit(build_decode_job(True)))
                    drain_ready(True)
            except Exception as exc:
                out_queue.put(exc)
            finally:
                out_queue.put(sentinel)

        thread = threading.Thread(target=producer, name="qwen3_tts_stream_producer", daemon=True)
        thread.start()
        while True:
            item = out_queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                thread.join(timeout=0.1)
                raise item
            yield item
        thread.join(timeout=0.1)

    def stream_voice_design(
        self,
        text,
        instruct,
        language=None,
        chunk_frames: int | None = None,
        left_context_frames: int | None = None,
        initial_chunk_frames: int | None = None,
        chunk_strategy: str | None = None,
        max_new_tokens: int = 512,
        min_new_tokens: int = 2,
        repetition_penalty: float = 1.05,
        max_prompt_tokens: int = 512,
        progress_interval: int = 8,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        prefix_codes: np.ndarray | None = None,
        append_prefix_codes_to_prompt: bool = False,
    ):
        scalar_text = self._ensure_scalar(text, "text")
        scalar_instruct = self._ensure_scalar(instruct, "instruct") or ""
        scalar_language = self._ensure_scalar(language if language is not None else "Auto", "language")
        if prefix_codes is None:
            hybrid_chunks = self._try_stream_native_hybrid_paged_audio_pipeline(
                text=scalar_text,
                instruct=scalar_instruct,
                language=scalar_language,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                max_prompt_tokens=max_prompt_tokens,
                chunk_frames=chunk_frames,
                left_context_frames=left_context_frames,
                initial_chunk_frames=initial_chunk_frames,
                chunk_strategy=chunk_strategy,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            if hybrid_chunks is not None:
                yield from hybrid_chunks
                return
        native_chunks = self._try_stream_native_audio_pipeline(
            text=scalar_text,
            instruct=scalar_instruct,
            language=scalar_language,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            max_prompt_tokens=max_prompt_tokens,
            chunk_frames=chunk_frames,
            left_context_frames=left_context_frames,
            initial_chunk_frames=initial_chunk_frames,
            chunk_strategy=chunk_strategy,
            prefix_codes=prefix_codes,
            append_prefix_codes_to_prompt=append_prefix_codes_to_prompt,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        if native_chunks is not None:
            yield from native_chunks
            return
        codes = self.generate_codes_iter(
            text=scalar_text,
            instruct=scalar_instruct,
            language=scalar_language,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            max_prompt_tokens=max_prompt_tokens,
            progress_interval=progress_interval,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            prefix_codes=prefix_codes,
            append_prefix_codes_to_prompt=append_prefix_codes_to_prompt,
        )
        yield from self.stream_decode_codes(
            codes,
            prefix_codes=prefix_codes,
            chunk_frames=chunk_frames,
            left_context_frames=left_context_frames,
            initial_chunk_frames=initial_chunk_frames,
            chunk_strategy=chunk_strategy,
        )

    def stream_custom_voice(
        self,
        text,
        speaker,
        language=None,
        instruct=None,
        chunk_frames: int | None = None,
        left_context_frames: int | None = None,
        initial_chunk_frames: int | None = None,
        chunk_strategy: str | None = None,
        max_new_tokens: int = 512,
        min_new_tokens: int = 2,
        repetition_penalty: float = 1.05,
        max_prompt_tokens: int = 512,
        progress_interval: int = 8,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ):
        native_chunks = self._try_stream_native_audio_pipeline(
            text=self._ensure_scalar(text, "text"),
            instruct=self._ensure_scalar(instruct if instruct is not None else "", "instruct") or "",
            language=self._ensure_scalar(language if language is not None else "Auto", "language"),
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            max_prompt_tokens=max_prompt_tokens,
            chunk_frames=chunk_frames,
            left_context_frames=left_context_frames,
            initial_chunk_frames=initial_chunk_frames,
            chunk_strategy=chunk_strategy,
            speaker=self._ensure_scalar(speaker, "speaker"),
            do_sample=do_sample,
        )
        if native_chunks is not None:
            yield from native_chunks
            return
        codes = self.generate_codes_iter(
            text=self._ensure_scalar(text, "text"),
            instruct=self._ensure_scalar(instruct if instruct is not None else "", "instruct") or "",
            language=self._ensure_scalar(language if language is not None else "Auto", "language"),
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            max_prompt_tokens=max_prompt_tokens,
            progress_interval=progress_interval,
            speaker=self._ensure_scalar(speaker, "speaker"),
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        yield from self.stream_decode_codes(
            codes,
            chunk_frames=chunk_frames,
            left_context_frames=left_context_frames,
            initial_chunk_frames=initial_chunk_frames,
            chunk_strategy=chunk_strategy,
        )

    def stream_voice_clone(
        self,
        text,
        language=None,
        ref_audio=None,
        ref_text=None,
        x_vector_only_mode: bool = False,
        voice_clone_prompt=None,
        chunk_frames: int | None = None,
        left_context_frames: int | None = None,
        initial_chunk_frames: int | None = None,
        chunk_strategy: str | None = None,
        max_new_tokens: int = 512,
        min_new_tokens: int = 2,
        repetition_penalty: float = 1.05,
        max_prompt_tokens: int = 512,
        progress_interval: int = 8,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ):
        text = self._ensure_scalar(text, "text")
        language = self._ensure_scalar(language if language is not None else "Auto", "language")
        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError("either voice_clone_prompt or ref_audio is required")
            prompt = self.create_voice_clone_prompt(ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode)[0]
        else:
            prompt = self._normalize_voice_clone_prompt(voice_clone_prompt, 1)[0]
        item_ref_text = prompt.ref_text if prompt.ref_text is not None else ref_text
        prefix = np.asarray(prompt.ref_code, dtype=np.int64) if prompt.ref_code is not None else None
        native_chunks = self._try_stream_native_audio_pipeline(
            text=text,
            instruct="",
            language=language,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            max_prompt_tokens=max_prompt_tokens,
            chunk_frames=chunk_frames,
            left_context_frames=left_context_frames,
            initial_chunk_frames=initial_chunk_frames,
            chunk_strategy=chunk_strategy,
            voice_clone_prompt=prompt,
            ref_text=item_ref_text,
            prefix_codes=prefix,
            do_sample=do_sample,
        )
        if native_chunks is not None:
            yield from native_chunks
            return
        codes = self.generate_codes_iter(
            text=text,
            instruct="",
            language=language,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            max_prompt_tokens=max_prompt_tokens,
            progress_interval=progress_interval,
            voice_clone_prompt=prompt,
            ref_text=item_ref_text,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        yield from self.stream_decode_codes(
            codes,
            prefix_codes=prefix,
            chunk_frames=chunk_frames,
            left_context_frames=left_context_frames,
            initial_chunk_frames=initial_chunk_frames,
            chunk_strategy=chunk_strategy,
        )

    def prewarm_streaming(
        self,
        text: str = "你好，这是一次流式预热。",
        instruct: str = "用自然、清晰的中文女声朗读。",
        language: str = "Chinese",
        chunk_frames: int | None = None,
        initial_chunk_frames: int | None = None,
        chunk_strategy: str | None = None,
        left_context_frames: int | None = None,
        max_new_tokens: int | None = None,
        repetition_penalty: float = 1.05,
        preload_buckets: str = "warmup",
        run_generation: bool = True,
        compile_fallback_decoder: bool = False,
    ) -> dict:
        started = time.time()
        stream_config = self._resolve_stream_chunk_config(
            chunk_strategy=chunk_strategy,
            chunk_frames=chunk_frames,
            initial_chunk_frames=initial_chunk_frames,
            left_context_frames=left_context_frames,
        )
        strategy = stream_config["strategy"]
        initial_chunk_frames = int(stream_config["initial_chunk_frames"])
        chunk_frames = int(stream_config["chunk_frames"])
        left_context_frames = int(stream_config["left_context_frames"])
        max_new_tokens = int(max_new_tokens or (initial_chunk_frames + chunk_frames))
        native_pipeline_active = self._native_pipeline_mode() in {"1", "true", "on", "require"}
        status = {
            "enabled": True,
            "status": "running",
            "mode": self.mode,
            "cache_step": self.cache_step,
            "codegen_unroll": int(self.codegen_unroll),
            "codegen_schedule": self.codegen_schedule,
            "scheduled_unrolls": list(self.codegen_schedule_unrolls),
            "codegen_decode_unroll": self.codegen_decode_unroll,
            "preferred_cache_bucket": self.preferred_cache_bucket,
            "unroll_available": any(self.codegen_unroll_available(item) for item in self.codegen_schedule_unrolls),
            "unroll_fallback": bool(self.codegen_unroll_fallback),
            "chunk_strategy": strategy,
            "initial_chunk_frames": initial_chunk_frames,
            "chunk_frames": chunk_frames,
            "left_context_frames": left_context_frames,
            "preload_buckets": preload_buckets,
            "repetition_penalty": float(repetition_penalty),
            "compile_fallback_decoder": bool(compile_fallback_decoder),
            "compiled_buckets": [],
            "compiled_unroll_buckets": [],
            "compiled_decode_unroll_buckets": [],
            "bucket_errors": {},
            "compiled_stream_decoders": [],
            "stream_decoder_errors": {},
            "streaming_decoder_available": bool(self.streaming_decoder_graphs_by_context),
            "native_pipeline_prewarm": bool(native_pipeline_active),
        }

        def compile_stream_candidates(label: str, candidates: list[tuple[int, int]]):
            for context, chunk in candidates:
                try:
                    self._get_stream_decoder(context, chunk)
                    item = {"context_frames": int(context), "chunk_frames": int(chunk), "label": label}
                    if item not in status["compiled_stream_decoders"]:
                        status["compiled_stream_decoders"].append(item)
                    return True
                except Exception as exc:
                    status["stream_decoder_errors"][f"{label}:c{context}_t{chunk}"] = str(exc)
            return False

        if native_pipeline_active:
            status["skipped_python_graph_prewarm"] = True
        else:
            compile_stream_candidates(
                "initial",
                self._stream_decoder_key_candidates(
                    context_frames=0,
                    new_frames=initial_chunk_frames,
                    preferred_chunk_frames=initial_chunk_frames,
                    left_context_frames=left_context_frames,
                ),
            )
            compile_stream_candidates(
                "steady",
                self._stream_decoder_key_candidates(
                    context_frames=min(initial_chunk_frames, left_context_frames),
                    new_frames=chunk_frames,
                    preferred_chunk_frames=chunk_frames,
                    left_context_frames=left_context_frames,
                ),
            )

        if self.mode == "cache" and not native_pipeline_active:
            scheduled_unrolls = [item for item in self.codegen_schedule_unrolls if item > 1 and self.codegen_unroll_available(item)]
            if self.cache_step == "fused" and scheduled_unrolls:
                available_buckets = {}
                for unroll_steps in scheduled_unrolls:
                    available_buckets.update(self.fused_cache_unroll_bucket_graphs_by_step.get(unroll_steps, {}))
                available_buckets = dict(sorted(available_buckets.items()))
            elif self.cache_step == "fused":
                available_buckets = self.fused_cache_bucket_graphs
            else:
                available_buckets = self.cache_bucket_graphs
            bucket_mode = str(preload_buckets or "warmup").strip().lower()
            if bucket_mode == "all":
                buckets = list(available_buckets)
            elif bucket_mode in {"", "none", "off", "false", "0"}:
                buckets = []
            elif bucket_mode in {"warmup", "auto", "required", "first"}:
                preferred = self.preferred_cache_bucket or min(available_buckets, default=0)
                buckets = [next((item for item in sorted(available_buckets) if item >= preferred), max(available_buckets))] if available_buckets else []
            else:
                requested = [int(item.strip()) for item in bucket_mode.split(",") if item.strip()]
                buckets = [bucket for bucket in requested if bucket in available_buckets]
            for bucket in buckets:
                try:
                    if self.cache_step == "fused" and scheduled_unrolls:
                        for unroll_steps in scheduled_unrolls:
                            if bucket not in self.fused_cache_unroll_bucket_graphs_by_step.get(unroll_steps, {}):
                                continue
                            self.get_fused_cache_unroll_step(bucket, unroll_steps, preferred_min_bucket=None)
                            status["compiled_unroll_buckets"].append(
                                {"unroll": int(unroll_steps), "bucket": int(bucket)}
                            )
                            if self.codegen_decode_unroll in {"auto", "on"} and self.codegen_schedule == "current":
                                decode_graphs = self.fused_cache_decode_unroll_bucket_graphs_by_step.get(unroll_steps, {})
                                decode_stateful_graphs = self.fused_cache_decode_unroll_stateful_mask_bucket_graphs_by_step.get(
                                    unroll_steps,
                                    {},
                                )
                                if bucket in decode_graphs or bucket in decode_stateful_graphs:
                                    self.get_fused_cache_decode_unroll_step(bucket, unroll_steps, preferred_min_bucket=None)
                                    status["compiled_decode_unroll_buckets"].append(
                                        {"unroll": int(unroll_steps), "bucket": int(bucket)}
                                    )
                    elif self.cache_step == "fused":
                        self.get_fused_cache_step(bucket)
                    else:
                        self.get_talker_stateful(bucket)
                    status["compiled_buckets"].append(int(bucket))
                except Exception as exc:
                    status["bucket_errors"][str(bucket)] = str(exc)
                    break

        if compile_fallback_decoder and self.decoder_graphs:
            try:
                self.decode(np.zeros((1, self.num_code_groups), dtype=np.int64))
                status["fallback_decoder_compiled"] = True
            except Exception as exc:
                status["fallback_decoder_error"] = str(exc)

        if run_generation:
            try:
                chunks = list(
                    self.stream_voice_design(
                        text=text,
                        instruct=instruct,
                        language=language,
                        chunk_frames=chunk_frames,
                        initial_chunk_frames=initial_chunk_frames,
                        chunk_strategy=strategy,
                        left_context_frames=left_context_frames,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=1,
                        repetition_penalty=repetition_penalty,
                        progress_interval=0,
                    )
                )
                status["warmup_chunks"] = len(chunks)
            except Exception as exc:
                status["warmup_generation_error"] = str(exc)
                status["warmup_chunks"] = 0
        else:
            status["warmup_chunks"] = 0
        has_errors = any(
            [
                status["bucket_errors"],
                status["stream_decoder_errors"],
                status.get("fallback_decoder_error"),
                status.get("warmup_generation_error"),
            ]
        )
        status["status"] = "ready_with_errors" if has_errors else "ready"
        status["elapsed"] = time.time() - started
        self.prewarm_status = status
        return status

    def decode(self, codes: np.ndarray):
        token_count = int(codes.shape[0])
        available = sorted(self.decoder_graphs)
        if not available:
            if self.streaming_decoder_graphs_by_context and not getattr(self, "_decode_stream_fallback_active", False):
                self._decode_stream_fallback_active = True
                try:
                    audio_parts = [
                        chunk.audio
                        for chunk in self.stream_decode_codes(
                            (np.asarray(frame, dtype=np.int64) for frame in np.asarray(codes, dtype=np.int64)),
                            chunk_strategy="stable",
                        )
                        if chunk.audio.size
                    ]
                finally:
                    self._decode_stream_fallback_active = False
                if audio_parts:
                    return np.concatenate(audio_parts).astype(np.float32, copy=False)
            raise RuntimeError(
                "manifest has no full speech_decoder graph and streaming decoder fallback did not produce audio"
            )
        bucket = next((item for item in available if item >= token_count), None)
        if bucket is None:
            bucket = available[-1]
            print(f"warning: truncating {token_count} codec tokens to decoder bucket {bucket}", flush=True)
            codes = codes[:bucket]
            token_count = bucket

        if bucket not in self.decoders:
            self.decoders[bucket] = compile_model(
                self.core,
                self.ir_dir / self.decoder_graphs[bucket],
                self.decoder_device,
                self.cache_dir,
                self.allow_cpu_fallback,
                self.ov_profiler.enabled,
                self.precision_hint,
                self.compile_config,
            )
            if not self.allow_cpu_fallback and self.decoder_device == "GPU":
                self.assert_gpu_execution([self.decoders[bucket]])
        padded = np.full((1, bucket, self.num_code_groups), -1, dtype=np.int64)
        padded[0, :token_count, :] = codes
        started = time.time()
        decoder_request = self.decoders[bucket].create_infer_request()
        audio = run_request(
            decoder_request,
            self.decoders[bucket],
            [padded],
            self.ov_profiler,
            f"speech_decoder_t{bucket}",
        )[0][0].astype(np.float32, copy=False)
        self.timings.add("decode", time.time() - started)
        return audio[: token_count * self.decode_upsample_rate]


def build_compile_config(args) -> dict:
    config = {}
    if args.gpu_performance_profile == "latency-high":
        config.update(
            {
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "1",
                "MODEL_PRIORITY": "HIGH",
                "GPU_QUEUE_PRIORITY": "HIGH",
                "GPU_HOST_TASK_PRIORITY": "HIGH",
                "GPU_QUEUE_THROTTLE": "LOW",
            }
        )
    elif args.gpu_performance_profile == "throughput":
        config.update(
            {
                "PERFORMANCE_HINT": "THROUGHPUT",
                "GPU_QUEUE_THROTTLE": "LOW",
            }
        )

    if args.performance_hint != "default":
        config["PERFORMANCE_HINT"] = args.performance_hint.upper()
    if args.num_streams is not None:
        config["NUM_STREAMS"] = str(args.num_streams)
    if args.performance_requests is not None:
        config["PERFORMANCE_HINT_NUM_REQUESTS"] = int(args.performance_requests)
    if args.model_priority:
        config["MODEL_PRIORITY"] = args.model_priority
    if args.gpu_queue_priority:
        config["GPU_QUEUE_PRIORITY"] = args.gpu_queue_priority
    if args.gpu_host_task_priority:
        config["GPU_HOST_TASK_PRIORITY"] = args.gpu_host_task_priority
    if args.gpu_queue_throttle:
        config["GPU_QUEUE_THROTTLE"] = args.gpu_queue_throttle
    if args.gpu_loop_unrolling != "default":
        config["GPU_ENABLE_LOOP_UNROLLING"] = args.gpu_loop_unrolling == "true"
    if args.gpu_sdpa_optimization != "default":
        config["GPU_ENABLE_SDPA_OPTIMIZATION"] = args.gpu_sdpa_optimization == "true"
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-dir", default="openvino/voice_design")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--decoder-device", default=None)
    parser.add_argument("--mode", default="no-cache", choices=RUNTIME_MODE_CHOICES)
    parser.add_argument("--cache-kernel", default="exact", choices=["exact", "sdpa"])
    parser.add_argument("--cache-step", default="split", choices=["split", "fused"])
    parser.add_argument("--graph-variant", default="fp16")
    parser.add_argument("--codegen-unroll", default="profile", choices=CODEGEN_UNROLL_CHOICES)
    parser.add_argument("--codegen-schedule", default="current", choices=CODEGEN_SCHEDULE_CHOICES)
    parser.add_argument("--codegen-decode-unroll", default="off", choices=["off", "auto", "on"])
    parser.add_argument("--preferred-cache-bucket", default="112")
    parser.add_argument("--precision-hint", default="f16", choices=["f16", "f32"])
    parser.add_argument("--ov-cache-dir", default=None)
    parser.add_argument("--ov-cache-mode", default="optimize_speed", choices=["optimize_speed", "optimize_size"])
    parser.add_argument("--disable-ov-cache", action="store_true")
    parser.add_argument("--gpu-performance-profile", default="default", choices=["default", "latency-high", "throughput"])
    parser.add_argument("--performance-hint", default="default", choices=["default", "latency", "throughput"])
    parser.add_argument("--num-streams", type=int, default=None)
    parser.add_argument("--performance-requests", type=int, default=None)
    parser.add_argument("--model-priority", default=None, choices=["LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--gpu-queue-priority", default=None, choices=["LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--gpu-host-task-priority", default=None, choices=["LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--gpu-queue-throttle", default=None, choices=["LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--gpu-loop-unrolling", default="default", choices=["default", "true", "false"])
    parser.add_argument("--gpu-sdpa-optimization", default="default", choices=["default", "true", "false"])
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--no-cpu-fallback", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--ov-profile", action="store_true")
    parser.add_argument("--benchmark-json", default=None)
    parser.add_argument("--compare-no-cache", action="store_true")
    parser.add_argument("--compare-graph-variant", default="fp16")
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument("--calibration-limit", type=int, default=64)
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--batch-jsonl", default=None)
    parser.add_argument("--output-dir", default="outputs/openvino_batch")
    parser.add_argument("--text", default="你好，这是一次完全使用 OpenVINO 的 Qwen 三语音合成测试。")
    parser.add_argument("--instruct", default="A calm young female voice, natural Mandarin pronunciation.")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--progress-interval", type=int, default=8)
    parser.add_argument("--output", default="outputs/openvino.wav")
    parser.add_argument("--skip-torch-assert", action="store_true")
    args = parser.parse_args()

    if not args.skip_torch_assert and "torch" in sys.modules:
        raise RuntimeError("torch was imported before runtime start; full OpenVINO runtime must not import PyTorch")

    compile_config = build_compile_config(args)
    started = time.time()
    runtime = OpenVINOQwen3TTS(
        args.ir_dir,
        args.device,
        args.decoder_device,
        allow_cpu_fallback=args.allow_cpu_fallback and not args.no_cpu_fallback,
        mode=args.mode,
        cache_kernel=args.cache_kernel,
        cache_step=args.cache_step,
        graph_variant=args.graph_variant,
        codegen_unroll=args.codegen_unroll,
        codegen_schedule=args.codegen_schedule,
        codegen_decode_unroll=args.codegen_decode_unroll,
        preferred_cache_bucket=args.preferred_cache_bucket,
        precision_hint=args.precision_hint,
        compile_config=compile_config,
        ov_cache_dir=args.ov_cache_dir,
        ov_cache_mode=args.ov_cache_mode,
        disable_ov_cache=args.disable_ov_cache,
        calibration_dir=args.calibration_dir,
        calibration_limit=args.calibration_limit,
        profile=args.profile,
        ov_profile=args.ov_profile,
    )
    if not args.skip_torch_assert and "torch" in sys.modules:
        raise RuntimeError("torch was imported while initializing runtime")

    if args.batch_jsonl:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        batch_started = time.time()
        with open(args.batch_jsonl, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                item_output = Path(item.get("output") or output_dir / f"item_{line_number:04d}.wav")
                codes = runtime.generate_codes(
                    text=item["text"],
                    instruct=item.get("instruct", args.instruct),
                    language=item.get("language", args.language),
                    max_new_tokens=int(item.get("max_new_tokens", args.max_new_tokens)),
                    min_new_tokens=args.min_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    max_prompt_tokens=args.max_prompt_tokens,
                    progress_interval=args.progress_interval,
                )
                audio = runtime.decode(codes)
                item_output.parent.mkdir(parents=True, exist_ok=True)
                sf.write(item_output, audio, runtime.sample_rate)
                print(f"wrote {item_output} with {codes.shape[0]} codec tokens", flush=True)
                count += 1
        print(f"processed {count} items in {time.time() - batch_started:.1f}s", flush=True)
        runtime.timings.print(0)
        return

    generation_started = time.time()
    codes = runtime.generate_codes(
        text=args.text,
        instruct=args.instruct,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        repetition_penalty=args.repetition_penalty,
        max_prompt_tokens=args.max_prompt_tokens,
        progress_interval=args.progress_interval,
    )
    generation_elapsed = time.time() - generation_started
    print(f"generated code tensor {codes.shape}", flush=True)

    comparison = None
    if args.compare_no_cache and args.mode != "no-cache":
        reference_runtime = OpenVINOQwen3TTS(
            args.ir_dir,
            args.device,
            args.decoder_device,
            allow_cpu_fallback=args.allow_cpu_fallback and not args.no_cpu_fallback,
            mode="no-cache",
            cache_kernel=args.cache_kernel,
            cache_step=args.cache_step,
            graph_variant=args.compare_graph_variant,
            precision_hint=args.precision_hint,
            compile_config=compile_config,
            ov_cache_dir=args.ov_cache_dir,
            ov_cache_mode=args.ov_cache_mode,
            disable_ov_cache=args.disable_ov_cache,
            calibration_dir=None,
            profile=False,
            ov_profile=False,
        )
        reference_codes = reference_runtime.generate_codes(
            text=args.text,
            instruct=args.instruct,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            repetition_penalty=args.repetition_penalty,
            max_prompt_tokens=args.max_prompt_tokens,
            progress_interval=0,
        )
        comparison = compare_code_tensors(codes, reference_codes)
        print(f"compare-no-cache exact_match={comparison['exact_match']} first_code_divergence={comparison['first_code_divergence']}", flush=True)

    output = Path(args.output)
    if not args.skip_decode:
        audio = runtime.decode(codes)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, audio, runtime.sample_rate)
        print(f"wrote {output} at {runtime.sample_rate} Hz in {time.time() - started:.1f}s", flush=True)
    else:
        print(f"skipped decode; generation completed in {generation_elapsed:.1f}s", flush=True)

    runtime.timings.print(int(codes.shape[0]))

    if args.benchmark_json:
        benchmark = {
            "mode": runtime.requested_mode,
            "runtime_mode": runtime.mode,
            "cache_kernel": runtime.cache_kernel,
            "cache_step": runtime.cache_step,
            "graph_variant": runtime.graph_variant,
            "precision_hint": args.precision_hint,
            "compile_config": compile_config,
            "device": args.device,
            "decoder_device": args.decoder_device or args.device,
            "generated_tokens": int(codes.shape[0]),
            "code_shape": list(codes.shape),
            "generation_elapsed": generation_elapsed,
            "total_elapsed": time.time() - started,
            "timings": runtime.timings.snapshot(int(codes.shape[0])),
            "comparison": comparison,
            "ov_profile_top": runtime.ov_profiler.top(100),
            "ov_profile_by_type": runtime.ov_profiler.aggregate("node_type"),
            "ov_profile_by_label": runtime.ov_profiler.aggregate("label"),
            "output": None if args.skip_decode else str(output),
        }
        benchmark_path = Path(args.benchmark_json)
        benchmark_path.parent.mkdir(parents=True, exist_ok=True)
        with open(benchmark_path, "w", encoding="utf-8") as f:
            json.dump(benchmark, f, ensure_ascii=False, indent=2)
        print(f"wrote benchmark {benchmark_path}", flush=True)


if __name__ == "__main__":
    main()

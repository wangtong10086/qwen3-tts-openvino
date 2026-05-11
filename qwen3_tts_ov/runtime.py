import argparse
import json
import os
import queue
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


PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
NEG_INF = -3.4028234663852886e38
DEFAULT_STREAM_CHUNK_STRATEGIES = {
    "low_latency": {
        "initial_chunk_frames": 8,
        "chunk_frames": 12,
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
    config = {
        "INFERENCE_PRECISION_HINT": precision_hint,
    }
    config.update(build_ov_cache_config(cache_dir, ov_cache_mode=ov_cache_mode, disable_ov_cache=disable_ov_cache))
    if cache_dir is not None and not disable_ov_cache:
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
        precision_hint: str = "f16",
        compile_config: dict | None = None,
        ov_cache_dir: str | Path | None = None,
        ov_cache_mode: str | None = "optimize_speed",
        disable_ov_cache: bool = False,
        calibration_dir: str | None = None,
        calibration_limit: int = 64,
        profile: bool = False,
        ov_profile: bool = False,
    ):
        self.ir_dir = resolve_ir_dir(ir_dir, fallback_to_local_voice_design=True, warn=True)
        self.manifest = load_manifest(self.ir_dir)
        self.model_dir = self.manifest["model_dir"]
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
        self.requested_mode = mode
        if mode == "fast-cache":
            mode = "cache"
            cache_kernel = "sdpa"
            cache_step = "split"
            graph_variant = "int8_cachedsub"
        self.graph_variant = graph_variant
        self.variant_graphs = self._load_graph_variant(graph_variant)
        self.cache_kernel = cache_kernel or self.manifest.get("default_cache_kernel", "exact")
        self.cache_step = cache_step or self.manifest.get("default_cache_step", "fused")
        if self.cache_step not in {"split", "fused"}:
            raise ValueError(f"unsupported cache_step={self.cache_step!r}")
        self.cache_bucket_graphs = self._load_cache_bucket_graphs(graphs, self.cache_kernel, self.variant_graphs)
        self.fused_cache_bucket_graphs = self._load_fused_cache_bucket_graphs(graphs, self.cache_kernel, self.variant_graphs)
        self.max_cache_len = max(self.cache_bucket_graphs) if self.cache_bucket_graphs else 0
        self.mode = mode
        self.device = device
        self.decoder_device = decoder_device or device
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
        if self.mode == "cache" and not self.cache_bucket_graphs:
            raise ValueError(f"cache mode requested, but manifest has no {self.cache_kernel!r} stateful talker graph")
        if self.mode == "cache" and self.cache_step == "fused" and not self.fused_cache_bucket_graphs:
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
        self.fused_step = None
        self.fused_request = None
        if self.mode == "fused-no-cache":
            self.fused_step = compile_model(
                self.core, self.ir_dir / self.graph_name(graphs, "fused_no_cache_step"), device, self.cache_dir, allow_cpu_fallback, ov_profile, self.precision_hint, self.compile_config
            )
            self.fused_request = self.fused_step.create_infer_request()
        elif self.mode != "cache":
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
        if self.mode != "fused-no-cache" and not (self.mode == "cache" and self.cache_step == "fused"):
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
                self.decoder_device,
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
                device,
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
        config_suffix = f" config={self.compile_config}" if self.compile_config else ""
        print(f"compiled {self.requested_mode}{cache_suffix}{variant_suffix}{config_suffix} core graphs on {device} in {time.time() - started:.1f}s", flush=True)

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
            raise ValueError(f"graph variant {graph_variant!r} not found in manifest; available variants: {available}")
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
        bucket = next((length for length in self.fused_cache_bucket_graphs if length >= required_len), None)
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
        generated_count = 0
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
            generated_count += 1
            generated_first_codes.append(first_code_int)
            repeated_mask[0, first_code_int] = 1.0

            if progress_interval and generated_count % progress_interval == 0:
                elapsed = time.time() - started
                print(f"generated {generated_count}/{max_new_tokens} codec tokens in {elapsed:.1f}s", flush=True)
            yield codes[0]

            if step + 1 >= max_new_tokens:
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

        def emit(is_final: bool):
            nonlocal emitted_frames, pending_frames, chunk_index, codegen_started
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
                    "codegen_ms": codegen_ms,
                    "decode_ms": decode_ms,
                    "chunk_compute_ms": compute_ms,
                    "chunk_audio_ms": audio_ms,
                    "rtf": rtf,
                    "queue_hint_ms": queue_hint_ms,
                    "queue_wait_ms": 0.0,
                    "producer_lag_ms": producer_lag_ms,
                    "strategy": strategy,
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
                            "codegen_ms": codegen_ms,
                            "decode_ms": decode_ms,
                            "chunk_compute_ms": effective_compute_ms,
                            "chunk_audio_ms": audio_ms,
                            "rtf": rtf,
                            "queue_hint_ms": queue_hint_ms,
                            "queue_wait_ms": queue_wait_ms,
                            "producer_lag_ms": producer_lag_ms,
                            "strategy": strategy,
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
    ):
        codes = self.generate_codes_iter(
            text=self._ensure_scalar(text, "text"),
            instruct=self._ensure_scalar(instruct, "instruct") or "",
            language=self._ensure_scalar(language if language is not None else "Auto", "language"),
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
        yield from self.stream_decode_codes(
            codes,
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
        prefix = np.asarray(prompt.ref_code, dtype=np.int64) if prompt.ref_code is not None else None
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
        preload_buckets: str = "warmup",
        run_generation: bool = True,
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
        status = {
            "enabled": True,
            "status": "running",
            "mode": self.mode,
            "cache_step": self.cache_step,
            "chunk_strategy": strategy,
            "initial_chunk_frames": initial_chunk_frames,
            "chunk_frames": chunk_frames,
            "left_context_frames": left_context_frames,
            "preload_buckets": preload_buckets,
            "compiled_buckets": [],
            "bucket_errors": {},
            "compiled_stream_decoders": [],
            "stream_decoder_errors": {},
            "streaming_decoder_available": bool(self.streaming_decoder_graphs_by_context),
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

        if self.mode == "cache":
            available_buckets = self.fused_cache_bucket_graphs if self.cache_step == "fused" else self.cache_bucket_graphs
            bucket_mode = str(preload_buckets or "warmup").strip().lower()
            if bucket_mode == "all":
                buckets = list(available_buckets)
            elif bucket_mode in {"", "none", "off", "false", "0"}:
                buckets = []
            elif bucket_mode in {"warmup", "auto", "required", "first"}:
                buckets = [min(available_buckets)] if available_buckets else []
            else:
                requested = [int(item.strip()) for item in bucket_mode.split(",") if item.strip()]
                buckets = [bucket for bucket in requested if bucket in available_buckets]
            for bucket in buckets:
                try:
                    if self.cache_step == "fused":
                        self.get_fused_cache_step(bucket)
                    else:
                        self.get_talker_stateful(bucket)
                    status["compiled_buckets"].append(int(bucket))
                except Exception as exc:
                    status["bucket_errors"][str(bucket)] = str(exc)
                    break

        if self.decoder_graphs:
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
    parser.add_argument("--mode", default="no-cache", choices=["no-cache", "cache", "fast-cache", "fused-no-cache"])
    parser.add_argument("--cache-kernel", default="exact", choices=["exact", "sdpa"])
    parser.add_argument("--cache-step", default="split", choices=["split", "fused"])
    parser.add_argument("--graph-variant", default="fp16")
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

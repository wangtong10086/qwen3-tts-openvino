#!/usr/bin/env python3
"""Compare paged-KV split-subcode and graph-fused codegen outputs.

This is a correctness gate for experimental codegen fusion work. It uses the
streaming audio path because the native paged-KV pipeline is implemented there,
but it compares generated codec frames, not decoded waveform samples.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


DEFAULT_TEXT = "你好，这是一次用于验证自回归 codegen 融合路径正确性的测试。"
DEFAULT_INSTRUCT = "用自然、清晰、稳定的中文女声朗读。"


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


def run_profile(args: argparse.Namespace, fusion: str, graph_variant: str) -> dict:
    from qwen3_tts_ov.runtime import OpenVINOQwen3TTS

    env = {
        "QWEN3_TTS_OV_NATIVE_PIPELINE": "require",
        "QWEN3_TTS_OV_NATIVE_PAGED_KV": "require",
        "QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA": "1",
        "QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE": "1",
        "QWEN3_TTS_OV_NATIVE_CODEGEN_FUSION": fusion,
        "QWEN3_TTS_OV_NATIVE_PAGED_KV_PRECISION": args.kv_precision,
        "QWEN3_TTS_OV_NATIVE_PAGED_KV_CACHE_INPUT_PRECISION": args.kv_cache_input_precision,
        "QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE": str(args.kv_block_size),
        "QWEN3_TTS_OV_NATIVE_PAGED_KV_SCORE_AGGREGATION": "1",
        "QWEN3_TTS_OV_NATIVE_PAGED_KV_SUBCODE_ATTENTION": args.subcode_attention,
        "QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE_MODE": args.split_subcode_mode,
        "QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE": args.device,
        "QWEN3_TTS_OV_NATIVE_REMOTE_EMBED": args.native_remote_embed,
        "QWEN3_TTS_OV_NATIVE_TRACE_CODEGEN_FRAMES": str(args.trace_frames),
    }
    started = time.time()
    chunks = 0
    frames = []
    last_timings = {}
    with temporary_env(env):
        runtime = OpenVINOQwen3TTS(
            args.ir_dir,
            args.device,
            args.decoder_device,
            mode="no-cache",
            cache_kernel="exact",
            cache_step="fused",
            graph_variant=graph_variant,
            codegen_unroll="1",
            codegen_schedule="current",
            codegen_decode_unroll="off",
            preferred_cache_bucket="0",
            ov_cache_dir=args.ov_cache_dir,
            disable_ov_cache=args.disable_ov_cache,
            allow_cpu_fallback=args.allow_cpu_fallback,
        )
        for chunk in runtime.stream_voice_design(
            args.text,
            args.instruct,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            repetition_penalty=1.0,
            max_prompt_tokens=args.max_prompt_tokens,
            progress_interval=0,
            do_sample=False,
            chunk_strategy=args.chunk_strategy,
        ):
            chunks += 1
            if chunk.codes.size:
                frames.append(np.asarray(chunk.codes, dtype=np.int64).reshape(-1, runtime.num_code_groups))
            last_timings = dict(chunk.timings or {})
    codes = np.concatenate(frames, axis=0) if frames else np.zeros((0, 0), dtype=np.int64)
    return {
        "fusion": fusion,
        "graph_variant": graph_variant,
        "elapsed_ms": (time.time() - started) * 1000.0,
        "chunks": chunks,
        "codes": codes,
        "shape": list(codes.shape),
        "timings": last_timings,
    }


def first_mismatch(left: np.ndarray, right: np.ndarray) -> dict | None:
    if left.shape != right.shape:
        return {"reason": "shape_mismatch", "left_shape": list(left.shape), "right_shape": list(right.shape)}
    mismatch = np.argwhere(left != right)
    if mismatch.size == 0:
        return None
    frame, group = mismatch[0].tolist()
    return {
        "reason": "value_mismatch",
        "frame": int(frame),
        "group": int(group),
        "left_first_code": int(left[frame, 0]),
        "right_first_code": int(right[frame, 0]),
        "left": int(left[frame, group]),
        "right": int(right[frame, group]),
        "first_code_mismatch": bool(left[frame, 0] != right[frame, 0]),
    }


def first_trace_mismatch(left: list[dict], right: list[dict]) -> dict | None:
    if len(left) != len(right):
        return {"reason": "trace_len_mismatch", "left_len": len(left), "right_len": len(right)}
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        left_codes = lhs.get("codes")
        right_codes = rhs.get("codes")
        left_first = lhs.get("first_code")
        right_first = rhs.get("first_code")
        if left_first != right_first or left_codes != right_codes:
            return {
                "reason": "trace_value_mismatch",
                "index": index,
                "frame": lhs.get("frame", index),
                "left_label": lhs.get("label"),
                "right_label": rhs.get("label"),
                "left_first_code": left_first,
                "right_first_code": right_first,
                "left_codes": left_codes,
                "right_codes": right_codes,
                "left_hidden_l2": lhs.get("hidden_l2"),
                "right_hidden_l2": rhs.get("hidden_l2"),
                "left_embed_l2": lhs.get("embed_l2"),
                "right_embed_l2": rhs.get("embed_l2"),
            }
    return None


def dump_prefix(codes: np.ndarray, frames: int) -> list[list[int]]:
    if frames <= 0 or codes.size == 0:
        return []
    return codes[: min(int(frames), codes.shape[0])].astype(np.int64).tolist()


def compare_runs(left: dict, right: dict) -> dict:
    mismatch = first_mismatch(left["codes"], right["codes"])
    left_trace = ((left.get("timings") or {}).get("native_timing") or {}).get("codegen_trace") or []
    right_trace = ((right.get("timings") or {}).get("native_timing") or {}).get("codegen_trace") or []
    trace_mismatch = first_trace_mismatch(left_trace, right_trace)
    status = "ok" if mismatch is None and trace_mismatch is None else "mismatch"
    return {
        "status": status,
        "mismatch": mismatch,
        "trace_mismatch": trace_mismatch,
        "split": {k: v for k, v in left.items() if k != "codes"},
        "graph": {k: v for k, v in right.items() if k != "codes"},
    }


def add_prefix_dump(result: dict, split_codes: np.ndarray, graph_codes: np.ndarray, frames: int) -> None:
    if frames <= 0:
        return
    result["prefix_frames"] = {
        "frames": int(frames),
        "split": dump_prefix(split_codes, frames),
        "graph": dump_prefix(graph_codes, frames),
    }


def classify_result(target: dict, baseline: dict | None, split_variant: str, graph_variant: str) -> tuple[str, str]:
    if baseline and baseline.get("status") != "ok":
        return (
            "structural_mismatch",
            "FP16 split-vs-graph baseline mismatch; graph fusion is not structurally equivalent.",
        )
    if target.get("status") == "ok":
        return ("passed", "target split-vs-graph comparison matched exactly.")
    if split_variant == "fp16" and graph_variant == "fp16":
        return (
            "structural_mismatch",
            "FP16 graph-fused path diverged from FP16 split path.",
        )
    return (
        "quantization_mismatch",
        "FP16 graph-fused baseline passed, but target variant diverged; this points to graph variant precision/quantization drift.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-dir", default="auto")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--decoder-device", default="GPU")
    parser.add_argument("--graph-variant-split", default="int8_sym_paged_talker_split")
    parser.add_argument("--graph-variant-graph", default="fp16")
    parser.add_argument(
        "--baseline-graph-variant-split",
        default="fp16",
        help="Split-path graph variant used for the structural baseline.",
    )
    parser.add_argument(
        "--baseline-graph-variant-graph",
        default="fp16",
        help="Graph-fused graph variant used for the structural baseline.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the FP structural baseline and only compare the requested target pair.",
    )
    parser.add_argument("--kv-precision", default="u8", choices=["f16", "bf16", "u8"])
    parser.add_argument("--kv-cache-input-precision", default="f32", choices=["f32", "f16", "bf16", "u8"])
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--subcode-attention", default="auto", choices=["auto", "sdpa", "exact"])
    parser.add_argument("--split-subcode-mode", default="cached", choices=["cached", "recompute", "cached_exact", "recompute_exact"])
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--chunk-strategy", default="smooth")
    parser.add_argument("--native-remote-embed", default="0", choices=["0", "1", "off", "on"])
    parser.add_argument("--dump-prefix-frames", type=int, default=8)
    parser.add_argument("--trace-frames", type=int, default=16)
    parser.add_argument("--ov-cache-dir", default=None)
    parser.add_argument("--disable-ov-cache", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--allow-mismatch", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        baseline_result = None
        baseline_split = None
        baseline_graph = None
        target_reuses_baseline = (
            not args.skip_baseline
            and args.graph_variant_split == args.baseline_graph_variant_split
            and args.graph_variant_graph == args.baseline_graph_variant_graph
        )
        if not args.skip_baseline:
            baseline_split = run_profile(args, "split", args.baseline_graph_variant_split)
            baseline_graph = run_profile(args, "graph", args.baseline_graph_variant_graph)
            baseline_result = compare_runs(baseline_split, baseline_graph)
            add_prefix_dump(
                baseline_result,
                baseline_split["codes"],
                baseline_graph["codes"],
                args.dump_prefix_frames,
            )

        if target_reuses_baseline and baseline_split is not None and baseline_graph is not None:
            split = baseline_split
            graph = baseline_graph
            target_result = compare_runs(split, graph)
        else:
            split = run_profile(args, "split", args.graph_variant_split)
            graph = run_profile(args, "graph", args.graph_variant_graph)
            target_result = compare_runs(split, graph)
        add_prefix_dump(target_result, split["codes"], graph["codes"], args.dump_prefix_frames)

        classification, classification_reason = classify_result(
            target_result,
            baseline_result,
            args.graph_variant_split,
            args.graph_variant_graph,
        )
        result = {
            **target_result,
            "classification": classification,
            "classification_reason": classification_reason,
            "target": target_result,
            "baseline": baseline_result,
        }
    except Exception as exc:
        result = {"status": "error", "classification": "error", "error": str(exc)}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if result.get("classification") == "passed" or args.allow_mismatch:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

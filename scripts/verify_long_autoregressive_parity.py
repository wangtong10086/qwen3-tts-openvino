#!/usr/bin/env python3
"""Compare long-text autoregressive codec generation with the upstream PyTorch path.

This is an offline debugging tool. It may import PyTorch and the upstream
Qwen3-TTS repo, but the OpenVINO runtime package must remain torch-free.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def read_text_arg(text: str | None, text_file: str | None) -> str:
    if text_file:
        return Path(text_file).read_text(encoding="utf-8").strip()
    if text:
        return text
    raise ValueError("provide --text or --text-file")


def summarize_codes(codes: np.ndarray, first_n: int = 32) -> dict[str, Any]:
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 2:
        raise ValueError(f"expected codec tensor [frames, codebooks], got {codes.shape}")
    first_codebook = codes[:, 0] if codes.size else np.asarray([], dtype=np.int64)
    unique_first = np.unique(first_codebook) if first_codebook.size else np.asarray([], dtype=np.int64)
    return {
        "shape": list(codes.shape),
        "first_frames": codes[:first_n].tolist(),
        "first_codebook_unique": int(unique_first.size),
        "first_codebook_head": first_codebook[:first_n].tolist(),
        "codec_min": int(codes.min()) if codes.size else None,
        "codec_max": int(codes.max()) if codes.size else None,
    }


def first_mismatch(a: np.ndarray, b: np.ndarray) -> dict[str, Any] | None:
    limit = min(a.shape[0], b.shape[0])
    for index in range(limit):
        if not np.array_equal(a[index], b[index]):
            return {
                "frame": int(index),
                "ov": a[index].tolist(),
                "torch": b[index].tolist(),
            }
    if a.shape != b.shape:
        return {"frame": int(limit), "ov_shape": list(a.shape), "torch_shape": list(b.shape)}
    return None


def generate_openvino_codes(args: argparse.Namespace, text: str) -> tuple[np.ndarray, dict[str, Any]]:
    from qwen3_tts_ov.runtime import OpenVINOQwen3TTS

    runtime = OpenVINOQwen3TTS(
        args.ov_ir_dir,
        device=args.ov_device,
        decoder_device=args.ov_decoder_device,
        allow_cpu_fallback=args.allow_cpu_fallback,
        mode=args.ov_mode,
        cache_kernel=args.ov_cache_kernel,
        cache_step=args.ov_cache_step,
        graph_variant=args.ov_graph_variant,
        codegen_unroll=1,
        codegen_schedule="current",
        codegen_decode_unroll="off",
        preferred_cache_bucket=args.ov_preferred_cache_bucket,
        native_codegen="off",
        native_pipeline="off",
        native_paged_kv="0",
        native_paged_kv_gqa="0",
        native_paged_kv_split_subcode="0",
    )
    started = time.time()
    codes = runtime.generate_codes(
        text=text,
        instruct=args.instruct,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        repetition_penalty=args.repetition_penalty,
        max_prompt_tokens=args.max_prompt_tokens,
        progress_interval=args.progress_interval,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
    )
    elapsed = time.time() - started
    sequence, _ = runtime.build_prompt(text, args.instruct, args.language, args.max_prompt_tokens)
    return np.asarray(codes, dtype=np.int64), {
        "elapsed_sec": elapsed,
        "prompt_len": int(sequence.shape[1]),
        "mode": args.ov_mode,
        "cache_kernel": args.ov_cache_kernel,
        "cache_step": args.ov_cache_step,
        "graph_variant": args.ov_graph_variant,
    }


def generate_torch_codes(args: argparse.Namespace, text: str) -> tuple[np.ndarray, dict[str, Any]]:
    qwen_repo = Path(args.qwen3_tts_repo).resolve()
    if str(qwen_repo) not in sys.path:
        sys.path.insert(0, str(qwen_repo))

    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = getattr(torch, args.torch_dtype)
    model = Qwen3TTSModel.from_pretrained(
        args.torch_model_dir,
        device_map=args.torch_device,
        dtype=dtype,
        attn_implementation=args.torch_attn_implementation,
    )
    input_ids = model._tokenize_texts([model._build_assistant_text(text)])
    instruct_ids = [model._tokenize_texts([model._build_instruct_text(args.instruct)])[0]] if args.instruct else [None]
    gen_kwargs = model._merge_generate_kwargs(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )
    started = time.time()
    codes_list, _ = model.model.generate(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        languages=[args.language],
        non_streaming_mode=True,
        **gen_kwargs,
    )
    if args.torch_device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.time() - started
    codes = codes_list[0].detach().cpu().numpy().astype(np.int64)
    return codes, {
        "elapsed_sec": elapsed,
        "prompt_tokens": int(input_ids[0].shape[-1]),
        "instruct_tokens": int(instruct_ids[0].shape[-1]) if instruct_ids[0] is not None else 0,
        "model_dir": args.torch_model_dir,
        "non_streaming_mode": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ov-ir-dir", default="auto")
    parser.add_argument("--ov-device", default="GPU")
    parser.add_argument("--ov-decoder-device", default=None)
    parser.add_argument("--ov-mode", default="no-cache", choices=["no-cache", "fused-no-cache", "cache"])
    parser.add_argument("--ov-cache-kernel", default="exact", choices=["exact", "sdpa"])
    parser.add_argument("--ov-cache-step", default="split", choices=["split", "fused"])
    parser.add_argument("--ov-graph-variant", default="fp16")
    parser.add_argument("--ov-preferred-cache-bucket", default="0")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--qwen3-tts-repo", default="/home/wt/Qwen3-TTS")
    parser.add_argument("--torch-model-dir", default="models/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--torch-device", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--torch-attn-implementation", default="flash_attention_2")
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default="examples/long_text_zh.example.txt")
    parser.add_argument("--instruct", default="用自然、清晰的中文女声朗读。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--first-n", type=int, default=32)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    text = read_text_arg(args.text, args.text_file)
    ov_codes, ov_meta = generate_openvino_codes(args, text)
    result: dict[str, Any] = {
        "ok": True,
        "text_chars": len(text),
        "language": args.language,
        "instruct": args.instruct,
        "max_new_tokens": args.max_new_tokens,
        "openvino": {**ov_meta, **summarize_codes(ov_codes, first_n=args.first_n)},
    }
    if not args.skip_torch:
        torch_codes, torch_meta = generate_torch_codes(args, text)
        result["torch"] = {**torch_meta, **summarize_codes(torch_codes, first_n=args.first_n)}
        mismatch = first_mismatch(ov_codes, torch_codes)
        result["parity"] = {
            "exact": mismatch is None,
            "first_mismatch": mismatch,
            "matched_frames": (
                min(ov_codes.shape[0], torch_codes.shape[0])
                if mismatch is None
                else int(mismatch.get("frame", 0))
            ),
        }

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

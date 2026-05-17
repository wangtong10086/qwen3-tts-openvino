#!/usr/bin/env python3
"""Compare long-text autoregressive codec generation with the upstream PyTorch path.

This is an offline debugging tool. It may import PyTorch and the upstream
Qwen3-TTS repo, but the OpenVINO runtime package must remain torch-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


TORCH_ACCELERATOR_POLICIES = {"gpu", "accelerator", "cuda", "cuda:0", "xpu", "xpu:0"}
TORCH_ACCELERATOR_PREFIXES = ("cuda", "xpu")


def read_text_arg(text: str | None, text_file: str | None) -> str:
    if text_file:
        return Path(text_file).read_text(encoding="utf-8").strip()
    if text:
        return text
    raise ValueError("provide --text or --text-file")


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reference_requires_accelerator(device_policy: str) -> bool:
    return str(device_policy).strip().lower() in TORCH_ACCELERATOR_POLICIES


def reference_uses_accelerator(meta: dict[str, Any]) -> bool:
    device = str(meta.get("device") or "").strip().lower()
    return device.startswith(TORCH_ACCELERATOR_PREFIXES)


def validate_torch_reference_device(device_policy: str, meta: dict[str, Any]) -> None:
    if not reference_requires_accelerator(device_policy):
        return
    if reference_uses_accelerator(meta):
        return
    requested = str(device_policy).strip().lower()
    actual = str(meta.get("device") or "<unknown>")
    raise RuntimeError(
        "Torch reference was requested on GPU/CUDA/XPU, but the generated reference did not run on an "
        f"accelerator (requested={requested}, actual={actual}). Pass --torch-device cpu only when CPU "
        "reference is intentional."
    )


def resolve_torch_device(torch_module, requested: str) -> str:
    policy = str(requested or "gpu").strip().lower()
    if policy in {"gpu", "accelerator"}:
        if torch_module.cuda.is_available():
            return "cuda:0"
        if hasattr(torch_module, "xpu") and torch_module.xpu.is_available():
            return "xpu:0"
        raise RuntimeError(
            "Torch reference requested GPU, but torch sees no CUDA/XPU device. "
            "Pass --torch-device cpu only when CPU reference is intentional."
        )
    if policy == "cuda":
        return "cuda:0"
    if policy == "xpu":
        return "xpu:0"
    if policy == "auto":
        if torch_module.cuda.is_available():
            return "cuda:0"
        if hasattr(torch_module, "xpu") and torch_module.xpu.is_available():
            return "xpu:0"
        return "cpu"
    return policy


def first_model_device(tts_model) -> str:
    try:
        for param in tts_model.model.parameters():
            return str(param.device)
    except Exception:
        pass
    try:
        return str(tts_model.device)
    except Exception:
        return "<unknown>"


def torch_reference_cache_payload(args: argparse.Namespace, text: str) -> dict[str, Any]:
    model_dir = Path(args.torch_model_dir)
    repo = Path(args.qwen3_tts_repo)
    return {
        "schema": "qwen3_tts_long_ar_torch_reference_v1",
        "qwen3_tts_repo": str(repo.resolve()) if repo.exists() else str(repo),
        "torch_model_dir": str(model_dir.resolve()) if model_dir.exists() else str(model_dir),
        "torch_device": str(args.torch_device).strip().lower(),
        "torch_dtype": args.torch_dtype,
        "torch_attn_implementation": args.torch_attn_implementation,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
        "instruct": args.instruct,
        "language": args.language,
        "max_new_tokens": int(args.max_new_tokens),
        "min_new_tokens": int(args.min_new_tokens),
        "repetition_penalty": float(args.repetition_penalty),
        "do_sample": bool(args.do_sample),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "temperature": float(args.temperature),
    }


def torch_reference_cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def cached_torch_reference_ready(cache_dir: Path, payload: dict[str, Any]) -> bool:
    result_json = cache_dir / "result.json"
    metadata_json = cache_dir / "cache_key.json"
    codes_path = cache_dir / "torch_codes.npy"
    if not result_json.exists() or not metadata_json.exists() or not codes_path.exists():
        return False
    try:
        result = load_json(result_json)
        metadata = load_json(metadata_json)
        if metadata.get("payload") != payload:
            return False
        if not bool(result.get("ok", False)):
            return False
        validate_torch_reference_device(str(payload.get("torch_device") or "gpu"), result)
        codes = np.load(codes_path)
        if codes.ndim != 2 or codes.shape[1] <= 0:
            return False
        return True
    except Exception:
        return False


def load_cached_torch_reference(cache_dir: Path, device_policy: str) -> tuple[np.ndarray, dict[str, Any]]:
    result = load_json(cache_dir / "result.json")
    validate_torch_reference_device(device_policy, result)
    codes = np.load(cache_dir / "torch_codes.npy").astype(np.int64, copy=False)
    return codes, dict(result)


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
    resolved_device = resolve_torch_device(torch, args.torch_device)
    attn = str(args.torch_attn_implementation)
    if attn == "auto":
        attn = "flash_attention_2" if resolved_device.startswith("cuda") else "sdpa"
    model = Qwen3TTSModel.from_pretrained(
        args.torch_model_dir,
        device_map=resolved_device,
        dtype=dtype,
        attn_implementation=attn,
    )
    model_device = first_model_device(model)
    if reference_requires_accelerator(args.torch_device) and not model_device.startswith(TORCH_ACCELERATOR_PREFIXES):
        raise RuntimeError(
            "Torch reference requested GPU/CUDA/XPU, but loaded model parameters are not on an accelerator "
            f"(resolved_device={resolved_device}, model_device={model_device})."
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
    if resolved_device.startswith("cuda"):
        torch.cuda.synchronize()
    if resolved_device.startswith("xpu") and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    elapsed = time.time() - started
    codes = codes_list[0].detach().cpu().numpy().astype(np.int64)
    return codes, {
        "ok": True,
        "elapsed_sec": elapsed,
        "device": model_device,
        "resolved_device": resolved_device,
        "requested_device": args.torch_device,
        "dtype": str(dtype),
        "attn": attn,
        "prompt_tokens": int(input_ids[0].shape[-1]),
        "instruct_tokens": int(instruct_ids[0].shape[-1]) if instruct_ids[0] is not None else 0,
        "model_dir": args.torch_model_dir,
        "non_streaming_mode": True,
    }


def get_torch_reference_codes(args: argparse.Namespace, text: str) -> tuple[np.ndarray, dict[str, Any]]:
    payload = torch_reference_cache_payload(args, text)
    key = torch_reference_cache_key(payload)
    cache_dir = Path(args.torch_reference_cache_dir) / key
    if not bool(args.no_torch_reference_cache) and not bool(args.refresh_torch_reference_cache):
        if cached_torch_reference_ready(cache_dir, payload):
            codes, meta = load_cached_torch_reference(cache_dir, str(args.torch_device))
            meta["cache_hit"] = True
            meta["cache_key"] = key
            meta["cache_dir"] = str(cache_dir)
            return codes, meta
    codes, meta = generate_torch_codes(args, text)
    validate_torch_reference_device(str(args.torch_device), meta)
    if not bool(args.no_torch_reference_cache):
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "torch_codes.npy", codes)
        write_json(cache_dir / "result.json", {**meta, "cache_hit": False, "cache_key": key, "cache_dir": str(cache_dir)})
        write_json(
            cache_dir / "cache_key.json",
            {"key": key, "created_at_unix": time.time(), "payload": payload, "result_json": str(cache_dir / "result.json")},
        )
    meta["cache_hit"] = False
    meta["cache_key"] = key
    meta["cache_dir"] = str(cache_dir)
    return codes, meta


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
    parser.add_argument(
        "--torch-device",
        default="gpu",
        help="Device policy for upstream PyTorch reference. Default `gpu` requires CUDA or Intel XPU; use `cpu` only intentionally.",
    )
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--torch-attn-implementation", default="auto")
    parser.add_argument("--torch-reference-cache-dir", default="outputs/long_ar_torch_reference_cache")
    parser.add_argument("--refresh-torch-reference-cache", action="store_true")
    parser.add_argument("--no-torch-reference-cache", action="store_true")
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
        torch_codes, torch_meta = get_torch_reference_codes(args, text)
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

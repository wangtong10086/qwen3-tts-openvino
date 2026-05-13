from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_TEXT = "你好，这是一次用于验证 paged KV 正确性的 OpenVINO 语音生成。"
DEFAULT_INSTRUCT = "用自然、清晰的中文女声朗读。"


PROFILES = {
    "reference_no_cache": {
        "mode": "no-cache",
        "native_pipeline": "off",
        "native_paged_kv": "off",
    },
    "reference_fused_no_cache": {
        "mode": "fused-no-cache",
        "native_pipeline": "off",
        "native_paged_kv": "off",
    },
    "paged_kv_expanded": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "0",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
    },
    "paged_kv_gqa": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
    },
    "paged_kv_gqa_cpu": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "CPU",
    },
    "paged_kv_gqa_split_subcode": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_split_subcode": "1",
        "native_codegen_device": "GPU",
    },
    "paged_kv_gqa_split_subcode_recompute": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_split_subcode": "1",
        "native_paged_kv_split_subcode_mode": "recompute",
        "native_codegen_device": "GPU",
    },
    "paged_kv_gqa_split_subcode_cached_exact": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_split_subcode": "1",
        "native_paged_kv_split_subcode_mode": "cached_exact",
        "native_codegen_device": "GPU",
    },
    "paged_kv_gqa_split_subcode_recompute_exact": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_split_subcode": "1",
        "native_paged_kv_split_subcode_mode": "recompute_exact",
        "native_codegen_device": "GPU",
    },
    "paged_kv_gqa_split_subcode_cpu": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_split_subcode": "1",
        "native_codegen_device": "CPU",
    },
    "paged_kv_gqa_split_subcode_int8": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_split_subcode": "1",
        "native_codegen_device": "GPU",
        "graph_variant": "int8",
    },
    "paged_kv_gqa_split_talker_int8_subcode_int8": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_split_subcode": "1",
        "native_codegen_device": "GPU",
        "graph_variant": "int8_sym_paged_talker_split",
    },
    "paged_kv_gqa_subcode_exact": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_subcode_attention": "exact",
        "native_codegen_device": "GPU",
    },
    "paged_kv_gqa_sdpa_subcode": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_subcode_attention": "sdpa",
        "native_codegen_device": "GPU",
    },
    "paged_kv_gqa_u8": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "u8",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
    },
    "paged_kv_gqa_bf16": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "bf16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
    },
    "paged_kv_gqa_f32": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f32",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
    },
    "paged_kv_gqa_cacheinput_f16": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_cache_input_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
    },
    "paged_kv_gqa_cacheinput_u8": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_cache_input_precision": "u8",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
    },
    "paged_kv_gqa_score_aggregation": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_score_aggregation": "1",
        "native_codegen_device": "GPU",
    },
    "paged_kv_gqa_no_score_aggregation": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_score_aggregation": "0",
        "native_codegen_device": "GPU",
    },
    "paged_kv_gqa_dq32": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_dynamic_quantization_group_size": "32",
    },
    "paged_kv_gqa_dq32_no_score_aggregation": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "native_dynamic_quantization_group_size": "32",
        "native_paged_kv_score_aggregation": "0",
    },
    "paged_kv_gqa_int8_subcode": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "graph_variant": "int8_sym_paged_subcode",
    },
    "paged_kv_gqa_int8_sym": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "graph_variant": "int8_sym_paged_kv_seed",
    },
    "paged_kv_gqa_int8_asym": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_codegen_device": "GPU",
        "graph_variant": "int8_asym_paged_kv_seed",
    },
    "paged_kv_gqa_unroll4_experimental": {
        "mode": "no-cache",
        "native_pipeline": "require",
        "native_paged_kv": "1",
        "native_paged_kv_gqa": "1",
        "native_paged_kv_precision": "f16",
        "native_paged_kv_block_size": "8",
        "native_paged_kv_unroll": "4",
        "native_paged_kv_experimental_unroll": "1",
        "native_codegen_device": "GPU",
    },
}


WORKER = r"""
import json
import os
import numpy as np

from qwen3_tts_ov.runtime import OpenVINOQwen3TTS

profile = json.loads(os.environ["QWEN3_TTS_OV_VERIFY_PROFILE_JSON"])
runtime = OpenVINOQwen3TTS.from_ir(
    os.environ["QWEN3_TTS_OV_VERIFY_IR_DIR"],
    device=os.environ["QWEN3_TTS_OV_VERIFY_DEVICE"],
    decoder_device=os.environ["QWEN3_TTS_OV_VERIFY_DECODER_DEVICE"],
    mode=profile["mode"],
    cache_kernel="exact",
    cache_step="fused",
    graph_variant=profile.get("graph_variant", os.environ.get("QWEN3_TTS_OV_VERIFY_GRAPH_VARIANT", "fp16")),
    codegen_unroll="1",
    codegen_schedule="current",
    codegen_decode_unroll="off",
    native_pipeline=profile.get("native_pipeline"),
    ov_cache_dir=os.environ.get("QWEN3_TTS_OV_VERIFY_OV_CACHE_DIR") or None,
)

if profile.get("native_pipeline") == "require":
    chunks = list(runtime.stream_voice_design(
        text=os.environ["QWEN3_TTS_OV_VERIFY_TEXT"],
        instruct=os.environ["QWEN3_TTS_OV_VERIFY_INSTRUCT"],
        language=os.environ["QWEN3_TTS_OV_VERIFY_LANGUAGE"],
        max_new_tokens=int(os.environ["QWEN3_TTS_OV_VERIFY_MAX_NEW_TOKENS"]),
        min_new_tokens=int(os.environ["QWEN3_TTS_OV_VERIFY_MIN_NEW_TOKENS"]),
        repetition_penalty=1.0,
        max_prompt_tokens=int(os.environ["QWEN3_TTS_OV_VERIFY_MAX_PROMPT_TOKENS"]),
        chunk_strategy="low_latency",
    ))
    code_chunks = [np.asarray(chunk.codes, dtype=np.int64).reshape(-1, runtime.num_code_groups) for chunk in chunks]
    code_chunks = [codes for codes in code_chunks if codes.size]
    codes = np.concatenate(code_chunks, axis=0) if code_chunks else np.zeros((0, runtime.num_code_groups), dtype=np.int64)
    timings = chunks[-1].timings if chunks else {}
else:
    codes = runtime.generate_codes(
        text=os.environ["QWEN3_TTS_OV_VERIFY_TEXT"],
        instruct=os.environ["QWEN3_TTS_OV_VERIFY_INSTRUCT"],
        language=os.environ["QWEN3_TTS_OV_VERIFY_LANGUAGE"],
        max_new_tokens=int(os.environ["QWEN3_TTS_OV_VERIFY_MAX_NEW_TOKENS"]),
        min_new_tokens=int(os.environ["QWEN3_TTS_OV_VERIFY_MIN_NEW_TOKENS"]),
        repetition_penalty=1.0,
        max_prompt_tokens=int(os.environ["QWEN3_TTS_OV_VERIFY_MAX_PROMPT_TOKENS"]),
        progress_interval=0,
    )
    timings = getattr(runtime, "last_codegen_info", {}) or {}

print("RESULT:" + json.dumps({
    "shape": list(codes.shape),
    "codes": codes.tolist(),
    "timings": timings,
}, ensure_ascii=False))
"""


def run_profile(name: str, args: argparse.Namespace) -> dict:
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; available: {', '.join(sorted(PROFILES))}")
    profile = PROFILES[name]
    env = os.environ.copy()
    env.update(
        {
            "QWEN3_TTS_OV_VERIFY_PROFILE_JSON": json.dumps(profile),
            "QWEN3_TTS_OV_VERIFY_IR_DIR": str(args.ir_dir),
            "QWEN3_TTS_OV_VERIFY_DEVICE": args.device,
            "QWEN3_TTS_OV_VERIFY_DECODER_DEVICE": args.decoder_device,
            "QWEN3_TTS_OV_VERIFY_TEXT": args.text,
            "QWEN3_TTS_OV_VERIFY_INSTRUCT": args.instruct,
            "QWEN3_TTS_OV_VERIFY_LANGUAGE": args.language,
            "QWEN3_TTS_OV_VERIFY_MAX_NEW_TOKENS": str(args.max_new_tokens),
            "QWEN3_TTS_OV_VERIFY_MIN_NEW_TOKENS": str(args.min_new_tokens),
            "QWEN3_TTS_OV_VERIFY_MAX_PROMPT_TOKENS": str(args.max_prompt_tokens),
            "QWEN3_TTS_OV_VERIFY_OV_CACHE_DIR": str(args.ov_cache_dir or ""),
            "QWEN3_TTS_OV_VERIFY_GRAPH_VARIANT": args.graph_variant,
        }
    )
    for key, value in profile.items():
        if key == "native_pipeline":
            env["QWEN3_TTS_OV_NATIVE_PIPELINE"] = str(value)
        elif key == "native_paged_kv":
            env["QWEN3_TTS_OV_NATIVE_PAGED_KV"] = str(value)
        elif key == "native_paged_kv_gqa":
            env["QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA"] = str(value)
        elif key == "native_paged_kv_precision":
            env["QWEN3_TTS_OV_NATIVE_PAGED_KV_PRECISION"] = str(value)
        elif key == "native_paged_kv_cache_input_precision":
            env["QWEN3_TTS_OV_NATIVE_PAGED_KV_CACHE_INPUT_PRECISION"] = str(value)
        elif key == "native_paged_kv_block_size":
            env["QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE"] = str(value)
        elif key == "native_paged_kv_unroll":
            env["QWEN3_TTS_OV_NATIVE_PAGED_KV_UNROLL"] = str(value)
        elif key == "native_paged_kv_experimental_unroll":
            env["QWEN3_TTS_OV_NATIVE_PAGED_KV_EXPERIMENTAL_UNROLL"] = str(value)
        elif key == "native_paged_kv_subcode_attention":
            env["QWEN3_TTS_OV_NATIVE_PAGED_KV_SUBCODE_ATTENTION"] = str(value)
        elif key == "native_paged_kv_split_subcode":
            env["QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE"] = str(value)
        elif key == "native_paged_kv_split_subcode_mode":
            env["QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE_MODE"] = str(value)
        elif key == "native_paged_kv_score_aggregation":
            env["QWEN3_TTS_OV_NATIVE_PAGED_KV_SCORE_AGGREGATION"] = str(value)
        elif key == "native_codegen_device":
            env["QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE"] = str(value)
        elif key == "native_dynamic_quantization_group_size":
            env["QWEN3_TTS_OV_NATIVE_DYNAMIC_QUANTIZATION_GROUP_SIZE"] = str(value)
    proc = subprocess.run(
        [sys.executable, "-c", WORKER],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout_sec,
        check=False,
    )
    result = {
        "profile": name,
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-4000:],
    }
    if proc.returncode != 0:
        result["error"] = proc.stderr[-4000:] or proc.stdout[-4000:]
        return result
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT:"):
            result.update(json.loads(line[len("RESULT:") :]))
            return result
    result["error"] = f"missing RESULT line; stdout tail: {proc.stdout[-4000:]}"
    return result


def compare(reference: dict, candidate: dict) -> dict:
    import numpy as np

    if reference.get("returncode") != 0 or candidate.get("returncode") != 0:
        return {"ok": False, "reason": "worker_error"}
    ref = np.asarray(reference["codes"], dtype=np.int64)
    cand = np.asarray(candidate["codes"], dtype=np.int64)
    limit = min(ref.shape[0], cand.shape[0])
    same_shape = ref.shape == cand.shape
    same_prefix = bool(np.array_equal(ref[:limit], cand[:limit]))
    out = {
        "ok": same_shape and same_prefix,
        "same_shape": same_shape,
        "same_prefix": same_prefix,
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
        "compared_frames": int(limit),
    }
    if not same_prefix:
        diff = np.argwhere(ref[:limit] != cand[:limit])
        if diff.size:
            row, col = diff[0].tolist()
            diff_rows = sorted({int(item[0]) for item in diff[:128]})
            out.update(
                {
                    "first_diff": [int(row), int(col)],
                    "reference_value": int(ref[row, col]),
                    "candidate_value": int(cand[row, col]),
                    "diff_count": int(diff.shape[0]),
                    "diff_frames_first_128": diff_rows,
                    "reference_first_diff_frame": ref[row].tolist(),
                    "candidate_first_diff_frame": cand[row].tolist(),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify paged-KV code generation against a reference profile.")
    parser.add_argument("--ir-dir", type=Path, default=Path("openvino_full"))
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--decoder-device", default="GPU")
    parser.add_argument(
        "--reference",
        default="reference_fused_no_cache",
        choices=sorted(PROFILES),
        help="Reference profile. The default matches the fused paged-KV seed graph semantics.",
    )
    parser.add_argument(
        "--candidate",
        default="paged_kv_gqa_cpu",
        choices=sorted(PROFILES),
        help=(
            "Candidate profile. The default runs paged-KV on CPU for token-exact diagnostics; "
            "use paged_kv_gqa or fastest benchmarks for GPU performance checks."
        ),
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--min-new-tokens", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--graph-variant", default="fp16")
    parser.add_argument("--ov-cache-dir", type=Path, default=Path(".cache/qwen3-tts-ov/verify-paged-kv"))
    parser.add_argument("--timeout-sec", type=int, default=360)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--include-codes", action="store_true", help="Include full generated code matrices in JSON output.")
    args = parser.parse_args()

    reference = run_profile(args.reference, args)
    candidate = run_profile(args.candidate, args)
    omitted = set() if args.include_codes else {"codes"}
    result = {
        "reference": {k: v for k, v in reference.items() if k not in omitted},
        "candidate": {k: v for k, v in candidate.items() if k not in omitted},
        "comparison": compare(reference, candidate),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not result["comparison"].get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

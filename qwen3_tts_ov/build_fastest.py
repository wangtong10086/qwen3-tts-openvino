from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_BY_TYPE = {
    "voice_design": "models/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom_voice": "models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "base": "models/Qwen3-TTS-12Hz-1.7B-Base",
}

DEFAULT_OUT_DIR_BY_TYPE = {
    "voice_design": "openvino/voice_design",
    "custom_voice": "openvino/custom_voice",
    "base": "openvino/base",
}

FASTEST_EXPORT_ARGS_PRODUCTION = (
    "--skip-fixed-cache-graphs",
    "--cache-buckets",
    "96",
    "--cache-kernels",
    "exact",
    "--fused-cache-kernels",
    "exact",
    "--fused-subcode-mode",
    "cached",
    "--fused-cache-unroll-steps",
    "",
    "--fused-cache-decode-unroll-steps",
    "",
    "--fused-cache-stateful-mask-steps",
    "",
    "--fused-cache-norepeat-steps",
    "",
    "--export-paged-kv-seed",
    "--paged-kv-unroll-steps",
    "",
    "--paged-kv-subcode-attention-kernels",
    "sdpa",
    "--decoder-tokens",
    "256",
    "--stream-decoder-chunks",
    "12,24",
    "--stream-decoder-first-chunks",
    "8,12",
    "--stream-decoder-left-context",
    "25",
)

FASTEST_EXPORT_ARGS_COMPAT = (
    "--cache-buckets",
    "96,128,192,256,320,384",
    "--cache-kernels",
    "exact",
    "--fused-cache-kernels",
    "exact",
    "--fused-subcode-mode",
    "cached",
    "--fused-cache-unroll-steps",
    "4,6,8,12",
    "--fused-cache-decode-unroll-steps",
    "4,8,12",
    "--fused-cache-stateful-mask-steps",
    "4,8,12",
    "--fused-cache-norepeat-steps",
    "4",
    "--export-paged-kv-seed",
    "--paged-kv-unroll-steps",
    "4",
    "--paged-kv-subcode-attention-kernels",
    "sdpa",
    "--decoder-tokens",
    "64,128,256",
    "--stream-decoder-chunks",
    "8,12,24",
    "--stream-decoder-first-chunks",
    "6,8,12",
    "--stream-decoder-left-context",
    "25",
)


@dataclass(frozen=True)
class BuildStep:
    name: str
    command: list[str]
    skip_reason: str | None = None


def build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    return env


def _read_manifest(ir_dir: Path) -> dict | None:
    manifest_path = ir_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def manifest_has_fastest_variant(manifest: dict | None) -> bool:
    if not manifest:
        return False
    graphs = manifest.get("graphs") or {}
    variants = manifest.get("graph_variants") or {}
    variant_graphs = ((variants.get("int8_sym_paged_talker_split") or {}).get("graphs") or {})
    paged_seed = variant_graphs.get("paged_kv_seed") or {}
    stream_contexts = ((manifest.get("streaming_decoder") or {}).get("contexts") or {})
    has_first_stream = bool((stream_contexts.get("0") or {}).get("8") or (stream_contexts.get("0") or {}).get("12"))
    has_steady_stream = bool((stream_contexts.get("25") or {}).get("24") or (stream_contexts.get("25") or {}).get("12"))
    return bool(
        paged_seed.get("talker_stateful_gqa")
        and (graphs.get("subcode_greedy_cached") or graphs.get("subcode_greedy"))
        and has_first_stream
        and has_steady_stream
    )


def build_fastest_steps(args: argparse.Namespace) -> list[BuildStep]:
    model_type = args.model_type
    model = args.model or DEFAULT_MODEL_BY_TYPE[model_type]
    out_dir = Path(args.out_dir or DEFAULT_OUT_DIR_BY_TYPE[model_type])
    manifest = None if args.clean else _read_manifest(out_dir)
    python = sys.executable
    steps: list[BuildStep] = []

    if not args.skip_submodule:
        if (REPO_ROOT / ".gitmodules").exists():
            steps.append(
                BuildStep(
                    name="submodule",
                    command=["git", "submodule", "update", "--init", "--recursive"],
                )
            )
        else:
            steps.append(BuildStep(name="submodule", command=[], skip_reason=".gitmodules not found"))

    if not args.skip_native:
        native_library = REPO_ROOT / "native" / "build" / "libqwen3_tts_ov_genai.so"
        if native_library.exists() and not args.force_native and not args.clean_native:
            steps.append(BuildStep(name="native", command=[], skip_reason=f"{native_library} already exists"))
        else:
            steps.append(
                BuildStep(
                    name="native",
                    command=[python, "scripts/build_native_codegen.py"],
                )
            )

    if args.skip_export:
        steps.append(BuildStep(name="export", command=[], skip_reason="disabled by --skip-export"))
    elif manifest and not args.force_export:
        steps.append(BuildStep(name="export", command=[], skip_reason=f"{out_dir / 'manifest.json'} already exists"))
    else:
        export_args = FASTEST_EXPORT_ARGS_COMPAT if args.graph_set == "compat" else FASTEST_EXPORT_ARGS_PRODUCTION
        export_cmd = [
            python,
            "-m",
            "qwen3_tts_ov",
            "export",
            "--model",
            str(model),
            "--model-type",
            model_type,
            "--out-dir",
            str(out_dir),
            *export_args,
        ]
        if model_type == "base":
            export_cmd.append("--export-clone-graphs")
        if args.force_export:
            export_cmd.extend(["--force-cache-graphs", "--force-paged-kv-seed"])
        steps.append(BuildStep(name="export", command=export_cmd))

    manifest_after_export = manifest if manifest and not args.force_export else None
    if args.skip_compress:
        steps.append(BuildStep(name="compress", command=[], skip_reason="disabled by --skip-compress"))
    elif manifest_has_fastest_variant(manifest_after_export) and not args.force_compress:
        steps.append(BuildStep(name="compress", command=[], skip_reason="fastest graph variant already exists"))
    else:
        steps.append(
            BuildStep(
                name="compress",
                command=[
                    python,
                    "scripts/compress_openvino_weights.py",
                    "--ir-dir",
                    str(out_dir),
                    "--preset",
                    "fastest",
                    *(["--force"] if args.force_compress else []),
                ],
            )
        )

    if args.skip_warmup:
        steps.append(BuildStep(name="warmup", command=[], skip_reason="disabled by --skip-warmup"))
    else:
        warmup_cmd = [
            python,
            "-m",
            "qwen3_tts_ov",
            "cache-warmup",
            "--ir-dir",
            str(out_dir),
            "--device",
            args.device,
            "--realtime-profile",
            "fastest",
            "--graphs",
            args.warmup_graphs,
            "--preload-buckets",
            args.preload_buckets,
            "--warmup-strategy",
            args.warmup_strategy,
        ]
        if args.decoder_device:
            warmup_cmd.extend(["--decoder-device", args.decoder_device])
        if getattr(args, "encoder_device", None):
            warmup_cmd.extend(["--encoder-device", args.encoder_device])
        if getattr(args, "npu_offload", "off") != "off":
            warmup_cmd.extend(["--npu-offload", args.npu_offload])
        if args.ov_cache_dir:
            warmup_cmd.extend(["--ov-cache-dir", str(args.ov_cache_dir)])
        if args.disable_ov_cache:
            warmup_cmd.append("--disable-ov-cache")
        steps.append(BuildStep(name="warmup", command=warmup_cmd))

    return steps


def print_step(step: BuildStep) -> None:
    if step.skip_reason:
        print(f"[skip] {step.name}: {step.skip_reason}", flush=True)
        return
    print(f"[run] {step.name}: {shlex.join(step.command)}", flush=True)


def run_build_fastest(args: argparse.Namespace) -> dict:
    model_type = args.model_type
    out_dir = Path(args.out_dir or DEFAULT_OUT_DIR_BY_TYPE[model_type])
    clean_actions = []
    if args.clean:
        print(f"[run] clean: remove {out_dir}", flush=True)
        clean_actions.append({"name": "clean", "path": str(out_dir)})
        if not args.dry_run:
            shutil.rmtree(out_dir, ignore_errors=True)
    if args.clean_native:
        native_build = REPO_ROOT / "native" / "build"
        print(f"[run] clean-native: remove {native_build}", flush=True)
        clean_actions.append({"name": "clean-native", "path": str(native_build)})
        if not args.dry_run:
            shutil.rmtree(native_build, ignore_errors=True)
    steps = build_fastest_steps(args)
    executed = []
    skipped = []
    for step in steps:
        print_step(step)
        if step.skip_reason:
            skipped.append({"name": step.name, "reason": step.skip_reason})
            continue
        executed.append({"name": step.name, "command": step.command})
        if not args.dry_run:
            subprocess.run(step.command, cwd=REPO_ROOT, check=True, env=build_subprocess_env())
    summary = {
        "dry_run": bool(args.dry_run),
        "graph_set": args.graph_set,
        "clean": clean_actions,
        "executed": executed,
        "skipped": skipped,
    }
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .model_download import (
    DEFAULT_RELEASE_MODEL_REPO,
    DEFAULT_RELEASE_MODEL_REVISION,
    DEFAULT_RELEASE_MODEL_SUBDIR,
    ensure_release_model_root,
)
from .native_codegen import native_library_candidates
from .profiles import FASTEST_CHUNK_STRATEGY, FASTEST_PROFILE_NAME, KV_CACHE_PROFILE_CHOICES, NPU_OFFLOAD_CHOICES
from .server import serve


def bundle_roots() -> list[Path]:
    roots: list[Path] = []
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        roots.append(Path(frozen_root))
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    roots.append(Path.cwd())
    roots.append(Path(__file__).resolve().parents[1])
    deduped: list[Path] = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def default_model_root() -> Path:
    for root in bundle_roots():
        candidate = root / "openvino"
        if candidate.exists():
            return candidate
    return Path.cwd() / "openvino"


def configure_native_library_env() -> Path | None:
    if os.environ.get("QWEN3_TTS_OV_NATIVE_CODEGEN_LIB"):
        return Path(os.environ["QWEN3_TTS_OV_NATIVE_CODEGEN_LIB"])
    for candidate in native_library_candidates():
        if candidate.exists():
            os.environ["QWEN3_TTS_OV_NATIVE_CODEGEN_LIB"] = str(candidate)
            return candidate
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen3-tts-ov-server",
        description="Run the Qwen3-TTS OpenVINO release sidecar server.",
    )
    parser.add_argument(
        "--model-root",
        default=None,
        help=(
            "OpenVINO IR root. Defaults to ./openvino next to the executable; "
            "if missing, the release server downloads the default Hugging Face IR to the user cache."
        ),
    )
    parser.add_argument("--no-auto-download-model", action="store_true", help="Do not download the default OpenVINO IR when --model-root is missing.")
    parser.add_argument("--model-repo", default=DEFAULT_RELEASE_MODEL_REPO, help="Hugging Face model repo used for automatic IR download.")
    parser.add_argument("--model-revision", default=DEFAULT_RELEASE_MODEL_REVISION, help="Hugging Face revision used for automatic IR download.")
    parser.add_argument("--model-subdir", default=DEFAULT_RELEASE_MODEL_SUBDIR, help="Subdirectory inside the Hugging Face repo used as --model-root after download.")
    parser.add_argument("--model-cache-dir", default=None, help="Directory for downloaded OpenVINO IR. Defaults to the user cache directory.")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--decoder-device", default=None)
    parser.add_argument("--encoder-device", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--prompt-device", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--npu-offload",
        default="off",
        choices=NPU_OFFLOAD_CHOICES,
        help="Windows heterogeneous mode: off, auto, decoder, audio, all, or require.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17860)
    parser.add_argument("--realtime-profile", default=FASTEST_PROFILE_NAME, choices=[FASTEST_PROFILE_NAME, "auto"])
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--preload-modes", default="voice_design")
    parser.add_argument("--preload-buckets", default="warmup")
    parser.add_argument("--warmup-text", default="你好，这是一次流式预热。")
    parser.add_argument(
        "--warmup-strategy",
        default=FASTEST_CHUNK_STRATEGY,
        choices=["realtime", "low_latency", "smooth", "balanced", "stable"],
    )
    parser.add_argument("--ov-cache-dir", default=None)
    parser.add_argument("--disable-ov-cache", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument(
        "--kv-cache-profile",
        default="auto",
        choices=KV_CACHE_PROFILE_CHOICES,
        help="Paged-KV cache memory profile. Default auto uses the fastest default, currently u8.",
    )
    parser.add_argument("--max-concurrent-tts", type=int, default=1)
    parser.add_argument("--long-output-memory-policy", default="stable", choices=["stable", "fast"])
    parser.add_argument(
        "--max-continuous-prompt-tokens",
        default="auto",
        help="Long full-AR prompt budget: auto, 0 to disable, or a positive token limit.",
    )
    parser.add_argument(
        "--max-vram-ratio",
        default=None,
        help="Memory budget ratio used when prompt tokens are auto: 0.8 or 80 means 80%%.",
    )
    parser.add_argument(
        "--kv-cache-preallocation",
        default="auto",
        choices=["auto", "off", "static"],
        help="KV cache planner allocation mode. static also enables native static decode blocks.",
    )
    parser.add_argument(
        "--kv-cache-reserve-mb",
        default="auto",
        help="Non-KV GPU memory reserve for the planner, in MiB, or auto.",
    )
    parser.add_argument(
        "--kv-cache-max-blocks",
        default="auto",
        help="Optional maximum KV cache blocks for the planner.",
    )
    parser.add_argument("--usm-retry-count", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_native_library_env()
    model_root = Path(args.model_root).expanduser() if args.model_root else default_model_root()
    model_download = ensure_release_model_root(
        model_root,
        auto_download=not args.no_auto_download_model,
        repo_id=args.model_repo,
        revision=args.model_revision,
        subdir=args.model_subdir,
        cache_dir=args.model_cache_dir,
    )
    if model_download.status in {"cached", "downloaded"}:
        print(model_download.message, file=sys.stderr, flush=True)
    model_root = model_download.model_root
    serve(
        model_root=model_root,
        host=args.host,
        port=args.port,
        device=args.device,
        decoder_device=args.decoder_device,
        encoder_device=args.encoder_device,
        prompt_device=args.prompt_device,
        npu_offload=args.npu_offload,
        allow_cpu_fallback=args.allow_cpu_fallback,
        realtime_profile=args.realtime_profile,
        ov_cache_dir=args.ov_cache_dir,
        disable_ov_cache=args.disable_ov_cache,
        warmup=not args.no_warmup,
        preload_modes=args.preload_modes,
        preload_buckets=args.preload_buckets,
        warmup_text=args.warmup_text,
        warmup_strategy=args.warmup_strategy,
        max_concurrent_tts=args.max_concurrent_tts,
        long_output_memory_policy=args.long_output_memory_policy,
        max_continuous_prompt_tokens=args.max_continuous_prompt_tokens,
        max_vram_ratio=args.max_vram_ratio,
        kv_cache_profile=args.kv_cache_profile,
        kv_cache_preallocation=args.kv_cache_preallocation,
        kv_cache_reserve_mb=args.kv_cache_reserve_mb,
        kv_cache_max_blocks=args.kv_cache_max_blocks,
        usm_retry_count=args.usm_retry_count,
        model_download_repo=args.model_repo,
        model_download_revision=args.model_revision,
        model_download_subdir=args.model_subdir,
        model_download_cache_dir=args.model_cache_dir,
    )


if __name__ == "__main__":
    main()

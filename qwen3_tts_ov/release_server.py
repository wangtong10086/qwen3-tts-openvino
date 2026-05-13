from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .native_codegen import native_library_candidates
from .profiles import FASTEST_CHUNK_STRATEGY, FASTEST_PROFILE_NAME
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
    parser.add_argument("--model-root", default=None, help="OpenVINO IR root. Defaults to ./openvino next to the executable.")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--decoder-device", default=None)
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
    parser.add_argument("--max-concurrent-tts", type=int, default=1)
    parser.add_argument("--long-output-memory-policy", default="stable", choices=["stable", "fast"])
    parser.add_argument("--max-continuous-prompt-tokens", type=int, default=1024)
    parser.add_argument("--usm-retry-count", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_native_library_env()
    model_root = Path(args.model_root).expanduser() if args.model_root else default_model_root()
    serve(
        model_root=model_root,
        host=args.host,
        port=args.port,
        device=args.device,
        decoder_device=args.decoder_device,
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
        usm_retry_count=args.usm_retry_count,
    )


if __name__ == "__main__":
    main()

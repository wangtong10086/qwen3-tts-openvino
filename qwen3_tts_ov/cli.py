import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .profiles import (
    CODEGEN_SCHEDULE_CHOICES,
    CODEGEN_UNROLL_CHOICES,
    FASTEST_CHUNK_STRATEGY,
    FASTEST_CODEGEN_DECODE_UNROLL,
    FASTEST_CODEGEN_SCHEDULE,
    FASTEST_CODEGEN_UNROLL,
    FASTEST_CACHE_KERNEL,
    FASTEST_CACHE_STEP,
    FASTEST_GRAPH_VARIANT,
    FASTEST_NATIVE_CODEGEN_DEVICE,
    FASTEST_MODE,
    FASTEST_NATIVE_BUFFER_REUSE,
    FASTEST_NATIVE_PAGED_KV,
    FASTEST_NATIVE_PAGED_KV_BLOCK_SIZE,
    FASTEST_NATIVE_PAGED_KV_GQA,
    FASTEST_NATIVE_PAGED_KV_PRECISION,
    FASTEST_NATIVE_PAGED_KV_SCORE_AGGREGATION,
    FASTEST_NATIVE_PAGED_KV_SPLIT_SUBCODE,
    FASTEST_NATIVE_PIPELINE,
    FASTEST_PREFERRED_CACHE_BUCKET,
    FASTEST_PROFILE_NAME,
    PUBLIC_REALTIME_PROFILE_CHOICES,
    REALTIME_PROFILE_CHOICES,
    RUNTIME_MODE_CHOICES,
    is_fastest_or_norepeat_mode,
)
from .runtime import OpenVINOQwen3TTS


def add_runtime_args(
    parser,
    default_ir_dir: str | None = "openvino/voice_design",
    include_generation: bool = True,
    include_output: bool = True,
):
    parser.add_argument(
        "--ir-dir",
        default=default_ir_dir,
        help="OpenVINO IR directory. VoiceDesign commands accept auto.",
    )
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--decoder-device", default=None)
    parser.add_argument(
        "--realtime-profile",
        default=FASTEST_PROFILE_NAME,
        metavar="{" + ",".join(PUBLIC_REALTIME_PROFILE_CHOICES) + "}",
        help="Production runtime profile. Default fastest requires the native C++ pipeline and optimized INT8_SYM IR.",
    )
    advanced = argparse.SUPPRESS
    parser.add_argument("--mode", default="cache", choices=RUNTIME_MODE_CHOICES, help=advanced)
    parser.add_argument("--cache-kernel", default="exact", choices=["exact", "sdpa"], help=advanced)
    parser.add_argument("--cache-step", default="fused", choices=["split", "fused"], help=advanced)
    parser.add_argument("--graph-variant", default="fp16", help=advanced)
    parser.add_argument("--codegen-unroll", default="profile", choices=CODEGEN_UNROLL_CHOICES, help=advanced)
    parser.add_argument("--codegen-schedule", default="current", choices=CODEGEN_SCHEDULE_CHOICES, help=advanced)
    parser.add_argument("--codegen-decode-unroll", default="off", choices=["off", "auto", "on"], help=advanced)
    parser.add_argument("--preferred-cache-bucket", default="112", help=advanced)
    parser.add_argument("--ov-cache-dir", default=None)
    parser.add_argument("--ov-cache-mode", default="optimize_speed", choices=["optimize_speed", "optimize_size"])
    parser.add_argument("--disable-ov-cache", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--native-codegen", default=None, choices=["off", "on", "require"], help=advanced)
    parser.add_argument("--native-pipeline", default=None, choices=["off", "on", "require"], help=advanced)
    parser.add_argument("--native-async-decode", default=None, choices=["auto", "off", "on"], help=advanced)
    parser.add_argument("--native-remote-embed", default=None, choices=["auto", "off", "on"], help=advanced)
    parser.add_argument("--native-buffer-reuse", default=None, choices=["auto", "off", "on"], help=advanced)
    parser.add_argument("--native-prompt", default=None, choices=["off", "on"], help=advanced)
    parser.add_argument("--native-prompt-device", default=None, help=advanced)
    parser.add_argument("--native-paged-kv", default=None, choices=["auto", "off", "on", "require"], help=advanced)
    parser.add_argument("--native-paged-kv-gqa", default=None, choices=["auto", "off", "on"], help=advanced)
    parser.add_argument("--native-paged-kv-precision", default=None, choices=["f16", "bf16", "u8"], help=advanced)
    parser.add_argument("--native-paged-kv-cache-input-precision", default=None, choices=["f32", "f16", "bf16", "u8"], help=advanced)
    parser.add_argument("--native-paged-kv-block-size", default=None, type=int, help=advanced)
    parser.add_argument("--native-paged-kv-static-decode", default=None, choices=["off", "on"], help=advanced)
    parser.add_argument("--native-paged-kv-static-blocks", default=None, type=int, help=advanced)
    parser.add_argument("--native-paged-kv-static-decode-mode", default=None, choices=["minimal", "full"], help=advanced)
    parser.add_argument("--native-paged-kv-unroll", default=None, type=int, help=advanced)
    parser.add_argument("--native-paged-kv-experimental-unroll", default=None, choices=["off", "on"], help=advanced)
    parser.add_argument("--native-paged-kv-subcode-attention", default=None, choices=["auto", "sdpa", "exact"], help=advanced)
    parser.add_argument("--native-paged-kv-split-subcode", default=None, choices=["off", "on"], help=advanced)
    parser.add_argument(
        "--native-paged-kv-split-subcode-mode",
        default=None,
        choices=["cached", "recompute", "cached_exact", "recompute_exact"],
        help=advanced,
    )
    parser.add_argument("--native-paged-kv-score-aggregation", default=None, choices=["off", "on"], help=advanced)
    parser.add_argument("--native-paged-kv-hybrid", default=None, choices=["off", "on"], help=advanced)
    parser.add_argument("--native-paged-kv-hybrid-prefix-frames", default=None, type=int, help=advanced)
    parser.add_argument("--native-codegen-device", default=None, help=advanced)
    parser.add_argument("--native-ov-profile", action="store_true", help=advanced)
    if not include_generation:
        return
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--progress-interval", type=int, default=8)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    if not include_output:
        return
    parser.add_argument("--output", default="outputs/openvino.wav")
    parser.add_argument("--skip-decode", action="store_true")


def build_runtime(args):
    apply_profile_defaults(args)
    apply_native_env(args)
    cache_step = args.cache_step
    if getattr(args, "do_sample", False) and args.mode == "cache" and cache_step == "fused":
        cache_step = "split"
    return OpenVINOQwen3TTS(
        args.ir_dir,
        args.device,
        args.decoder_device,
        allow_cpu_fallback=args.allow_cpu_fallback,
        mode=args.mode,
        cache_kernel=args.cache_kernel,
        cache_step=cache_step,
        graph_variant=args.graph_variant,
        codegen_unroll=args.codegen_unroll,
        codegen_schedule=args.codegen_schedule,
        codegen_decode_unroll=args.codegen_decode_unroll,
        preferred_cache_bucket=args.preferred_cache_bucket,
        ov_cache_dir=args.ov_cache_dir,
        ov_cache_mode=args.ov_cache_mode,
        disable_ov_cache=args.disable_ov_cache,
        profile=args.profile,
    )


def apply_profile_defaults(args):
    if getattr(args, "native_paged_kv", None) in {"on", "require"}:
        args.realtime_profile = "fp16"
        args.mode = "no-cache"
        args.cache_kernel = "exact"
        args.cache_step = "fused"
        args.graph_variant = "fp16"
        args.codegen_unroll = "1"
        args.codegen_schedule = "current"
        args.codegen_decode_unroll = "off"
        args.preferred_cache_bucket = "0"
        if getattr(args, "native_pipeline", None) is None:
            args.native_pipeline = "require" if args.native_paged_kv == "require" else "on"
        return
    realtime_profile = getattr(args, "realtime_profile", None)
    if realtime_profile not in (None, "") and realtime_profile not in REALTIME_PROFILE_CHOICES:
        raise ValueError(f"realtime_profile must be one of {', '.join(REALTIME_PROFILE_CHOICES)}")
    if realtime_profile in {FASTEST_PROFILE_NAME, "auto"}:
        args.mode = FASTEST_MODE
        args.cache_kernel, args.cache_step, args.graph_variant = (
            FASTEST_CACHE_KERNEL,
            FASTEST_CACHE_STEP,
            FASTEST_GRAPH_VARIANT,
        )
        args.codegen_unroll = str(FASTEST_CODEGEN_UNROLL)
        args.codegen_schedule = FASTEST_CODEGEN_SCHEDULE
        args.codegen_decode_unroll = FASTEST_CODEGEN_DECODE_UNROLL
        args.preferred_cache_bucket = str(FASTEST_PREFERRED_CACHE_BUCKET)
        args.native_pipeline = FASTEST_NATIVE_PIPELINE
        args.native_buffer_reuse = FASTEST_NATIVE_BUFFER_REUSE
        args.native_paged_kv = FASTEST_NATIVE_PAGED_KV
        args.native_paged_kv_gqa = FASTEST_NATIVE_PAGED_KV_GQA
        args.native_paged_kv_precision = FASTEST_NATIVE_PAGED_KV_PRECISION
        args.native_paged_kv_block_size = FASTEST_NATIVE_PAGED_KV_BLOCK_SIZE
        args.native_paged_kv_split_subcode = FASTEST_NATIVE_PAGED_KV_SPLIT_SUBCODE
        args.native_paged_kv_score_aggregation = FASTEST_NATIVE_PAGED_KV_SCORE_AGGREGATION
        args.native_codegen_device = FASTEST_NATIVE_CODEGEN_DEVICE
        return


def apply_native_env(args):
    for arg_name, env_name in (
        ("native_codegen", "QWEN3_TTS_OV_NATIVE_CODEGEN"),
        ("native_pipeline", "QWEN3_TTS_OV_NATIVE_PIPELINE"),
    ):
        value = getattr(args, arg_name, None)
        if value is None:
            continue
        if value == "off":
            os.environ.pop(env_name, None)
        elif value == "on":
            os.environ[env_name] = "1"
        else:
            os.environ[env_name] = "require"
    async_decode = getattr(args, "native_async_decode", None)
    if async_decode in (None, "auto", "off"):
        os.environ.pop("QWEN3_TTS_OV_NATIVE_ASYNC_DECODE", None)
    elif async_decode == "on":
        os.environ["QWEN3_TTS_OV_NATIVE_ASYNC_DECODE"] = "1"
    buffer_reuse = getattr(args, "native_buffer_reuse", None)
    if buffer_reuse in (None, "auto"):
        os.environ.pop("QWEN3_TTS_OV_NATIVE_BUFFER_REUSE", None)
    elif buffer_reuse == "off":
        os.environ["QWEN3_TTS_OV_NATIVE_BUFFER_REUSE"] = "0"
    elif buffer_reuse == "on":
        os.environ["QWEN3_TTS_OV_NATIVE_BUFFER_REUSE"] = "1"
    remote_embed = getattr(args, "native_remote_embed", None)
    if remote_embed in (None, "auto"):
        os.environ.pop("QWEN3_TTS_OV_NATIVE_REMOTE_EMBED", None)
    elif remote_embed == "off":
        os.environ["QWEN3_TTS_OV_NATIVE_REMOTE_EMBED"] = "0"
    elif remote_embed == "on":
        os.environ["QWEN3_TTS_OV_NATIVE_REMOTE_EMBED"] = "1"
    native_prompt = getattr(args, "native_prompt", None)
    if native_prompt == "off":
        os.environ.pop("QWEN3_TTS_OV_NATIVE_PROMPT", None)
    elif native_prompt == "on":
        os.environ["QWEN3_TTS_OV_NATIVE_PROMPT"] = "1"
    prompt_device = getattr(args, "native_prompt_device", None)
    if prompt_device:
        os.environ["QWEN3_TTS_OV_NATIVE_PROMPT_DEVICE"] = str(prompt_device)
    paged_kv = getattr(args, "native_paged_kv", None)
    if paged_kv in (None, "auto"):
        os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV", None)
    elif paged_kv == "off":
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV"] = "0"
    elif paged_kv == "on":
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV"] = "1"
    elif paged_kv == "require":
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV"] = "require"
        os.environ["QWEN3_TTS_OV_NATIVE_PIPELINE"] = "require"
    paged_kv_gqa = getattr(args, "native_paged_kv_gqa", None)
    if paged_kv_gqa in (None, "auto"):
        os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA", None)
    elif paged_kv_gqa == "off":
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA"] = "0"
    elif paged_kv_gqa == "on":
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_GQA"] = "1"
    paged_kv_precision = getattr(args, "native_paged_kv_precision", None)
    if paged_kv_precision:
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_PRECISION"] = str(paged_kv_precision)
    paged_kv_cache_input_precision = getattr(args, "native_paged_kv_cache_input_precision", None)
    if paged_kv_cache_input_precision:
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_CACHE_INPUT_PRECISION"] = str(paged_kv_cache_input_precision)
    paged_kv_block_size = getattr(args, "native_paged_kv_block_size", None)
    if paged_kv_block_size:
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_BLOCK_SIZE"] = str(int(paged_kv_block_size))
    paged_kv_static_decode = getattr(args, "native_paged_kv_static_decode", None)
    if paged_kv_static_decode == "off":
        os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_DECODE", None)
    elif paged_kv_static_decode == "on":
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_DECODE"] = "1"
    paged_kv_static_blocks = getattr(args, "native_paged_kv_static_blocks", None)
    if paged_kv_static_blocks:
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_BLOCKS"] = str(int(paged_kv_static_blocks))
    paged_kv_static_decode_mode = getattr(args, "native_paged_kv_static_decode_mode", None)
    if paged_kv_static_decode_mode:
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_STATIC_DECODE_MODE"] = str(paged_kv_static_decode_mode)
    paged_kv_unroll = getattr(args, "native_paged_kv_unroll", None)
    if paged_kv_unroll:
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_UNROLL"] = str(int(paged_kv_unroll))
    paged_kv_experimental_unroll = getattr(args, "native_paged_kv_experimental_unroll", None)
    if paged_kv_experimental_unroll == "off":
        os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV_EXPERIMENTAL_UNROLL", None)
    elif paged_kv_experimental_unroll == "on":
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_EXPERIMENTAL_UNROLL"] = "1"
    paged_kv_subcode_attention = getattr(args, "native_paged_kv_subcode_attention", None)
    if paged_kv_subcode_attention:
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_SUBCODE_ATTENTION"] = str(paged_kv_subcode_attention)
    paged_kv_split_subcode = getattr(args, "native_paged_kv_split_subcode", None)
    if paged_kv_split_subcode == "off":
        os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE", None)
    elif paged_kv_split_subcode == "on":
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE"] = "1"
    paged_kv_split_subcode_mode = getattr(args, "native_paged_kv_split_subcode_mode", None)
    if paged_kv_split_subcode_mode:
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_SPLIT_SUBCODE_MODE"] = str(paged_kv_split_subcode_mode)
    paged_kv_score_aggregation = getattr(args, "native_paged_kv_score_aggregation", None)
    if paged_kv_score_aggregation == "off":
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_SCORE_AGGREGATION"] = "0"
    elif paged_kv_score_aggregation == "on":
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_SCORE_AGGREGATION"] = "1"
    paged_kv_hybrid = getattr(args, "native_paged_kv_hybrid", None)
    if paged_kv_hybrid == "off":
        os.environ.pop("QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID", None)
    elif paged_kv_hybrid == "on":
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID"] = "1"
    paged_kv_hybrid_prefix = getattr(args, "native_paged_kv_hybrid_prefix_frames", None)
    if paged_kv_hybrid_prefix:
        os.environ["QWEN3_TTS_OV_NATIVE_PAGED_KV_HYBRID_PREFIX_FRAMES"] = str(int(paged_kv_hybrid_prefix))
    native_codegen_device = getattr(args, "native_codegen_device", None)
    if native_codegen_device:
        os.environ["QWEN3_TTS_OV_NATIVE_CODEGEN_DEVICE"] = str(native_codegen_device)
    if getattr(args, "native_ov_profile", False):
        os.environ["QWEN3_TTS_OV_NATIVE_PERF_COUNT"] = "1"


def generation_kwargs(args):
    repetition_penalty = args.repetition_penalty
    if (
        getattr(args, "realtime_profile", None) == FASTEST_PROFILE_NAME
        or is_fastest_or_norepeat_mode(getattr(args, "mode", None))
    ) and repetition_penalty == 1.05:
        repetition_penalty = 1.0
    return {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "repetition_penalty": repetition_penalty,
        "max_prompt_tokens": args.max_prompt_tokens,
        "progress_interval": args.progress_interval,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
    }


def write_wavs(wavs, sample_rate: int, output: str):
    output_path = Path(output)
    if len(wavs) == 1:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, wavs[0], sample_rate)
        print(f"wrote {output_path} at {sample_rate} Hz", flush=True)
        return
    output_path.mkdir(parents=True, exist_ok=True)
    for index, wav in enumerate(wavs):
        item_path = output_path / f"item_{index + 1:04d}.wav"
        sf.write(item_path, wav, sample_rate)
        print(f"wrote {item_path} at {sample_rate} Hz", flush=True)


def add_stream_args(parser):
    parser.add_argument("--chunk-dir", default="outputs/stream")
    parser.add_argument("--chunk-strategy", default=None, choices=["realtime", "low_latency", "smooth", "balanced", "stable"])
    parser.add_argument("--initial-chunk-frames", type=int, default=None)
    parser.add_argument("--chunk-frames", type=int, default=None)
    parser.add_argument("--left-context-frames", type=int, default=None)
    parser.add_argument("--stream-format", default="pcm_s16le", choices=["pcm_s16le"])


def stream_kwargs(args):
    chunk_strategy = args.chunk_strategy
    if chunk_strategy is None and getattr(args, "realtime_profile", None) == FASTEST_PROFILE_NAME:
        chunk_strategy = FASTEST_CHUNK_STRATEGY
    return {
        "chunk_strategy": chunk_strategy,
        "initial_chunk_frames": args.initial_chunk_frames,
        "chunk_frames": args.chunk_frames,
        "left_context_frames": args.left_context_frames,
    }


def audio_to_pcm16(audio) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def write_stream_chunks(chunks, chunk_dir: str, output: str):
    chunk_path = Path(chunk_dir)
    chunk_path.mkdir(parents=True, exist_ok=True)
    audio_parts = []
    sample_rate = None
    first_audio_time = None
    started = time.time()
    for chunk in chunks:
        sample_rate = chunk.sample_rate
        if chunk.audio.size:
            if first_audio_time is None:
                first_audio_time = time.time() - started
            item_path = chunk_path / f"chunk_{chunk.index:04d}.pcm"
            item_path.write_bytes(audio_to_pcm16(chunk.audio))
            audio_parts.append(chunk.audio)
            rtf = chunk.timings.get("rtf")
            rtf_text = f"{rtf:.2f}" if isinstance(rtf, (int, float)) else "n/a"
            stream_rtf = chunk.timings.get("stream_rtf")
            stream_rtf_text = f"{stream_rtf:.2f}" if isinstance(stream_rtf, (int, float)) else "n/a"
            decode_path = chunk.timings.get("decode_path", "unknown")
            codegen_ms = float(chunk.timings.get("codegen_ms", 0.0))
            decode_ms = float(chunk.timings.get("decode_ms", 0.0))
            print(
                f"wrote {item_path} frames={chunk.codes.shape[0]} samples={chunk.audio.shape[0]} "
                f"strategy={chunk.timings.get('strategy', 'n/a')} path={decode_path} "
                f"codegen_ms={codegen_ms:.1f} decode_ms={decode_ms:.1f} "
                f"rtf={rtf_text} stream_rtf={stream_rtf_text} final={chunk.is_final}",
                flush=True,
            )
        elif chunk.is_final:
            print(f"stream final index={chunk.index}", flush=True)
    if audio_parts and output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, np.concatenate(audio_parts), sample_rate)
        print(f"wrote {output_path} at {sample_rate} Hz", flush=True)
    if first_audio_time is not None:
        print(f"first audio chunk in {first_audio_time:.3f}s", flush=True)


def run_voice_design(args):
    runtime = build_runtime(args)
    started = time.time()
    if args.skip_decode:
        codes = runtime.generate_codes(
            text=args.text,
            instruct=args.instruct,
            language=args.language,
            **generation_kwargs(args),
        )
        print(f"generated code tensor {codes.shape}", flush=True)
        runtime.timings.print(int(codes.shape[0]))
        return
    wavs, sample_rate = runtime.generate_voice_design(
        text=args.text,
        instruct=args.instruct,
        language=args.language,
        **generation_kwargs(args),
    )
    write_wavs(wavs, sample_rate, args.output)
    runtime.timings.print(0)
    print(f"done in {time.time() - started:.1f}s", flush=True)


def run_custom_voice(args):
    runtime = build_runtime(args)
    started = time.time()
    if args.skip_decode:
        codes = runtime.generate_codes(
            text=args.text,
            instruct=args.instruct or "",
            language=args.language,
            speaker=args.speaker,
            **generation_kwargs(args),
        )
        print(f"generated code tensor {codes.shape}", flush=True)
        runtime.timings.print(int(codes.shape[0]))
        return
    wavs, sample_rate = runtime.generate_custom_voice(
        text=args.text,
        speaker=args.speaker,
        instruct=args.instruct,
        language=args.language,
        **generation_kwargs(args),
    )
    write_wavs(wavs, sample_rate, args.output)
    runtime.timings.print(0)
    print(f"done in {time.time() - started:.1f}s", flush=True)


def run_voice_clone(args):
    runtime = build_runtime(args)
    started = time.time()
    if args.skip_decode:
        prompt = runtime.create_voice_clone_prompt(args.ref_audio, ref_text=args.ref_text, x_vector_only_mode=args.x_vector_only)[0]
        codes = runtime.generate_codes(
            text=args.text,
            instruct="",
            language=args.language,
            voice_clone_prompt=prompt,
            ref_text=args.ref_text,
            **generation_kwargs(args),
        )
        print(f"generated code tensor {codes.shape}", flush=True)
        runtime.timings.print(int(codes.shape[0]))
        return
    wavs, sample_rate = runtime.generate_voice_clone(
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only_mode=args.x_vector_only,
        **generation_kwargs(args),
    )
    write_wavs(wavs, sample_rate, args.output)
    runtime.timings.print(0)
    print(f"done in {time.time() - started:.1f}s", flush=True)


def run_batch(args):
    runtime = build_runtime(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(args.batch_jsonl, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            mode = item["mode"].replace("-", "_")
            out = item.get("output") or str(output_dir / f"item_{line_number:04d}.wav")
            kwargs = generation_kwargs(args)
            kwargs.update({k: item[k] for k in kwargs if k in item})
            if mode == "voice_design":
                wavs, sr = runtime.generate_voice_design(item["text"], item.get("instruct", ""), item.get("language", "Auto"), **kwargs)
            elif mode == "custom_voice":
                wavs, sr = runtime.generate_custom_voice(item["text"], item["speaker"], item.get("language", "Auto"), item.get("instruct", ""), **kwargs)
            elif mode == "voice_clone":
                wavs, sr = runtime.generate_voice_clone(
                    item["text"],
                    item.get("language", "Auto"),
                    ref_audio=item.get("ref_audio"),
                    ref_text=item.get("ref_text"),
                    x_vector_only_mode=bool(item.get("x_vector_only", False)),
                    **kwargs,
                )
            else:
                raise ValueError(f"unsupported batch mode {item['mode']!r}")
            write_wavs(wavs, sr, out)
            count += len(wavs)
    print(f"processed {count} items", flush=True)


def run_stream_voice_design(args):
    runtime = build_runtime(args)
    chunks = runtime.stream_voice_design(
        text=args.text,
        instruct=args.instruct,
        language=args.language,
        **generation_kwargs(args),
        **stream_kwargs(args),
    )
    write_stream_chunks(chunks, args.chunk_dir, args.output)


def run_stream_custom_voice(args):
    runtime = build_runtime(args)
    chunks = runtime.stream_custom_voice(
        text=args.text,
        speaker=args.speaker,
        instruct=args.instruct,
        language=args.language,
        **generation_kwargs(args),
        **stream_kwargs(args),
    )
    write_stream_chunks(chunks, args.chunk_dir, args.output)


def run_stream_voice_clone(args):
    runtime = build_runtime(args)
    chunks = runtime.stream_voice_clone(
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only_mode=args.x_vector_only,
        **generation_kwargs(args),
        **stream_kwargs(args),
    )
    write_stream_chunks(chunks, args.chunk_dir, args.output)


def run_serve(args):
    from .server import serve

    apply_profile_defaults(args)
    apply_native_env(args)
    serve(
        model_root=args.ir_dir or args.model_root,
        host=args.host,
        port=args.port,
        device=args.device,
        decoder_device=args.decoder_device,
        allow_cpu_fallback=args.allow_cpu_fallback,
        mode=args.mode,
        cache_kernel=args.cache_kernel,
        cache_step=args.cache_step,
        graph_variant=args.graph_variant,
        codegen_unroll=args.codegen_unroll,
        codegen_schedule=args.codegen_schedule,
        codegen_decode_unroll=args.codegen_decode_unroll,
        preferred_cache_bucket=args.preferred_cache_bucket,
        realtime_profile=args.realtime_profile,
        ov_cache_dir=args.ov_cache_dir,
        ov_cache_mode=args.ov_cache_mode,
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


def run_cache_warmup_command(args):
    from .cache_warmup import run_cache_warmup, run_single_task

    apply_profile_defaults(args)
    apply_native_env(args)
    if args.single_task_json:
        result = run_single_task(args)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return

    compile_config = json.loads(args.compile_config_json or "{}")
    try:
        summary = run_cache_warmup(args, compile_config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"wrote {output}", flush=True)
    elif args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    else:
        print(json.dumps({key: summary[key] for key in ("ok", "cache_dir", "task_count", "elapsed") if key in summary}, ensure_ascii=False), flush=True)
    if not args.dry_run and not summary.get("ok", True):
        raise SystemExit(1)


def add_cache_warmup_args(parser):
    parser.add_argument(
        "--ir-dir",
        default="auto",
        help="OpenVINO IR directory, or auto to use openvino/voice_design then openvino_full.",
    )
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--decoder-device", default=None)
    parser.add_argument(
        "--realtime-profile",
        default=FASTEST_PROFILE_NAME,
        metavar="{" + ",".join(PUBLIC_REALTIME_PROFILE_CHOICES) + "}",
        help="Production cache warmup profile. Default fastest warms the native realtime graph set.",
    )
    advanced = argparse.SUPPRESS
    parser.add_argument("--mode", default="cache", choices=RUNTIME_MODE_CHOICES, help=advanced)
    parser.add_argument("--cache-kernel", default="exact", choices=["exact", "sdpa"], help=advanced)
    parser.add_argument("--cache-step", default="fused", choices=["split", "fused"], help=advanced)
    parser.add_argument("--graph-variant", default="fp16", help=advanced)
    parser.add_argument("--codegen-unroll", default="profile", choices=CODEGEN_UNROLL_CHOICES, help=advanced)
    parser.add_argument("--codegen-schedule", default="current", choices=CODEGEN_SCHEDULE_CHOICES, help=advanced)
    parser.add_argument("--codegen-decode-unroll", default="off", choices=["off", "auto", "on"], help=advanced)
    parser.add_argument("--preferred-cache-bucket", default="112", help=advanced)
    parser.add_argument("--precision-hint", default="f16", choices=["f16", "f32"])
    parser.add_argument("--ov-cache-dir", default=None)
    parser.add_argument("--ov-cache-mode", default="optimize_speed", choices=["optimize_speed", "optimize_size"])
    parser.add_argument("--disable-ov-cache", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--graphs", default="core,stream,buckets")
    parser.add_argument("--preload-buckets", default="warmup")
    parser.add_argument("--stream-decoders", default="strategy", choices=["strategy", "all"])
    parser.add_argument("--warmup-strategy", default=FASTEST_CHUNK_STRATEGY, choices=["realtime", "low_latency", "smooth", "balanced", "stable"])
    parser.add_argument("--subprocess", dest="subprocess", action="store_true", default=True)
    parser.add_argument("--no-subprocess", dest="subprocess", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--compile-config-json", default="{}")
    parser.add_argument("--single-task-json", default=None, help=argparse.SUPPRESS)


def add_build_fastest_args(parser):
    parser.add_argument(
        "--model-type",
        default="voice_design",
        choices=["voice_design", "custom_voice", "base"],
        help="Model family to prepare. Defaults to the validated VoiceDesign fastest path.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Local PyTorch model directory. Defaults to models/Qwen3-TTS-12Hz-1.7B-<model type>.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="OpenVINO IR output directory. Defaults to openvino/<model type>.",
    )
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--decoder-device", default=None)
    parser.add_argument("--ov-cache-dir", default=None)
    parser.add_argument("--disable-ov-cache", action="store_true")
    parser.add_argument("--preload-buckets", default="warmup")
    parser.add_argument("--warmup-graphs", default="core,stream,buckets")
    parser.add_argument("--warmup-strategy", default=FASTEST_CHUNK_STRATEGY, choices=["realtime", "low_latency", "smooth", "balanced", "stable"])
    parser.add_argument(
        "--graph-set",
        default="production",
        choices=["production", "compat"],
        help="production exports only the fastest runtime graphs; compat also exports legacy fixed-bucket/unroll diagnostic graphs.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove the output IR directory before building.")
    parser.add_argument("--clean-native", action="store_true", help="Remove native/build before compiling the native pipeline.")
    parser.add_argument("--skip-submodule", action="store_true")
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-compress", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--force-native", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-compress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", default=None)


def run_build_fastest_command(args):
    from .build_fastest import run_build_fastest

    run_build_fastest(args)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "export":
        from . import exporter

        sys.argv = [sys.argv[0]] + argv[1:]
        exporter.main()
        return

    parser = argparse.ArgumentParser(prog="qwen3_tts_ov")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export", help="export PyTorch model weights to OpenVINO IR")

    cache_warmup = sub.add_parser("cache-warmup")
    add_cache_warmup_args(cache_warmup)
    cache_warmup.set_defaults(func=run_cache_warmup_command)

    build_fastest = sub.add_parser("build-fastest", help="build native code, export/compress fastest IR, and warm OpenVINO cache")
    add_build_fastest_args(build_fastest)
    build_fastest.set_defaults(func=run_build_fastest_command)

    vd = sub.add_parser("voice-design")
    add_runtime_args(vd, default_ir_dir="auto")
    vd.add_argument("--text", default="你好，这是一次完全使用 OpenVINO 的 Qwen 三语音合成测试。")
    vd.add_argument("--instruct", default="A calm young female voice, natural Mandarin pronunciation.")
    vd.add_argument("--language", default="Auto")
    vd.set_defaults(func=run_voice_design)

    cv = sub.add_parser("custom-voice")
    add_runtime_args(cv, default_ir_dir="openvino/custom_voice")
    cv.add_argument("--text", required=True)
    cv.add_argument("--speaker", required=True)
    cv.add_argument("--instruct", default="")
    cv.add_argument("--language", default="Auto")
    cv.set_defaults(func=run_custom_voice)

    clone = sub.add_parser("voice-clone")
    add_runtime_args(clone, default_ir_dir="openvino/base")
    clone.add_argument("--text", required=True)
    clone.add_argument("--language", default="Auto")
    clone.add_argument("--ref-audio", required=True)
    clone.add_argument("--ref-text", default=None)
    clone.add_argument("--x-vector-only", action="store_true")
    clone.set_defaults(func=run_voice_clone)

    batch = sub.add_parser("batch")
    add_runtime_args(batch, default_ir_dir="auto")
    batch.add_argument("--batch-jsonl", required=True)
    batch.add_argument("--output-dir", default="outputs/openvino_batch")
    batch.set_defaults(func=run_batch)

    stream = sub.add_parser("stream")
    stream_sub = stream.add_subparsers(dest="stream_command", required=True)

    svd = stream_sub.add_parser("voice-design")
    add_runtime_args(svd, default_ir_dir="auto")
    add_stream_args(svd)
    svd.add_argument("--text", default="你好，这是一次流式 OpenVINO 合成测试。")
    svd.add_argument("--instruct", default="A calm young female voice, natural Mandarin pronunciation.")
    svd.add_argument("--language", default="Auto")
    svd.set_defaults(func=run_stream_voice_design)

    scv = stream_sub.add_parser("custom-voice")
    add_runtime_args(scv, default_ir_dir="openvino/custom_voice")
    add_stream_args(scv)
    scv.add_argument("--text", required=True)
    scv.add_argument("--speaker", required=True)
    scv.add_argument("--instruct", default="")
    scv.add_argument("--language", default="Auto")
    scv.set_defaults(func=run_stream_custom_voice)

    sclone = stream_sub.add_parser("voice-clone")
    add_runtime_args(sclone, default_ir_dir="openvino/base")
    add_stream_args(sclone)
    sclone.add_argument("--text", required=True)
    sclone.add_argument("--language", default="Auto")
    sclone.add_argument("--ref-audio", required=True)
    sclone.add_argument("--ref-text", default=None)
    sclone.add_argument("--x-vector-only", action="store_true")
    sclone.set_defaults(func=run_stream_voice_clone)

    serve_parser = sub.add_parser("serve")
    add_runtime_args(serve_parser, default_ir_dir=None, include_generation=False, include_output=False)
    serve_parser.add_argument("--model-root", default="openvino")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=17860)
    serve_parser.add_argument("--no-warmup", action="store_true")
    serve_parser.add_argument("--preload-modes", default="voice_design")
    serve_parser.add_argument("--preload-buckets", default="warmup")
    serve_parser.add_argument("--warmup-text", default="你好，这是一次流式预热。")
    serve_parser.add_argument("--warmup-strategy", default=FASTEST_CHUNK_STRATEGY, choices=["realtime", "low_latency", "smooth", "balanced", "stable"])
    serve_parser.add_argument("--max-concurrent-tts", type=int, default=1)
    serve_parser.add_argument("--long-output-memory-policy", default="stable", choices=["stable", "fast"])
    serve_parser.add_argument(
        "--max-continuous-prompt-tokens",
        default="auto",
        help="Long full-AR prompt budget: auto, 0 to disable, or a positive token limit.",
    )
    serve_parser.add_argument("--usm-retry-count", type=int, default=1)
    serve_parser.set_defaults(func=run_serve)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        message = str(exc)
        if "OpenVINO IR manifest not found" in message:
            print(f"error: {message}", file=sys.stderr)
            raise SystemExit(2) from exc
        raise


if __name__ == "__main__":
    main()

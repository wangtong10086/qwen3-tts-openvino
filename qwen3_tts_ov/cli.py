import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .runtime import OpenVINOQwen3TTS


def add_runtime_args(
    parser,
    default_ir_dir: str | None = "openvino/voice_design",
    include_generation: bool = True,
    include_output: bool = True,
):
    parser.add_argument("--ir-dir", default=default_ir_dir)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--decoder-device", default=None)
    parser.add_argument("--mode", default="cache", choices=["no-cache", "cache", "fast-cache", "fused-no-cache"])
    parser.add_argument("--cache-kernel", default="exact", choices=["exact", "sdpa"])
    parser.add_argument("--cache-step", default="fused", choices=["split", "fused"])
    parser.add_argument("--graph-variant", default="fp16")
    parser.add_argument("--ov-cache-dir", default=None)
    parser.add_argument("--ov-cache-mode", default="optimize_speed", choices=["optimize_speed", "optimize_size"])
    parser.add_argument("--disable-ov-cache", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--profile", action="store_true")
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
        ov_cache_dir=args.ov_cache_dir,
        ov_cache_mode=args.ov_cache_mode,
        disable_ov_cache=args.disable_ov_cache,
        profile=args.profile,
    )


def generation_kwargs(args):
    return {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "repetition_penalty": args.repetition_penalty,
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
    parser.add_argument("--chunk-strategy", default=None, choices=["low_latency", "balanced", "stable"])
    parser.add_argument("--initial-chunk-frames", type=int, default=None)
    parser.add_argument("--chunk-frames", type=int, default=None)
    parser.add_argument("--left-context-frames", type=int, default=None)
    parser.add_argument("--stream-format", default="pcm_s16le", choices=["pcm_s16le"])


def stream_kwargs(args):
    return {
        "chunk_strategy": args.chunk_strategy,
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
            decode_path = chunk.timings.get("decode_path", "unknown")
            codegen_ms = float(chunk.timings.get("codegen_ms", 0.0))
            decode_ms = float(chunk.timings.get("decode_ms", 0.0))
            print(
                f"wrote {item_path} frames={chunk.codes.shape[0]} samples={chunk.audio.shape[0]} "
                f"strategy={chunk.timings.get('strategy', 'n/a')} path={decode_path} "
                f"codegen_ms={codegen_ms:.1f} decode_ms={decode_ms:.1f} "
                f"rtf={rtf_text} final={chunk.is_final}",
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
        ov_cache_dir=args.ov_cache_dir,
        ov_cache_mode=args.ov_cache_mode,
        disable_ov_cache=args.disable_ov_cache,
        warmup=not args.no_warmup,
        preload_modes=args.preload_modes,
        preload_buckets=args.preload_buckets,
        warmup_text=args.warmup_text,
        warmup_strategy=args.warmup_strategy,
    )


def run_cache_warmup_command(args):
    from .cache_warmup import run_cache_warmup, run_single_task

    if args.single_task_json:
        result = run_single_task(args)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return

    compile_config = json.loads(args.compile_config_json or "{}")
    summary = run_cache_warmup(args, compile_config)
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
    parser.add_argument("--ir-dir", default="openvino/voice_design")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--decoder-device", default=None)
    parser.add_argument("--mode", default="cache", choices=["no-cache", "cache", "fast-cache", "fused-no-cache"])
    parser.add_argument("--cache-kernel", default="exact", choices=["exact", "sdpa"])
    parser.add_argument("--cache-step", default="fused", choices=["split", "fused"])
    parser.add_argument("--graph-variant", default="fp16")
    parser.add_argument("--precision-hint", default="f16", choices=["f16", "f32"])
    parser.add_argument("--ov-cache-dir", default=None)
    parser.add_argument("--ov-cache-mode", default="optimize_speed", choices=["optimize_speed", "optimize_size"])
    parser.add_argument("--disable-ov-cache", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--graphs", default="core,stream,buckets")
    parser.add_argument("--preload-buckets", default="warmup")
    parser.add_argument("--stream-decoders", default="strategy", choices=["strategy", "all"])
    parser.add_argument("--warmup-strategy", default="low_latency", choices=["low_latency", "balanced", "stable"])
    parser.add_argument("--subprocess", dest="subprocess", action="store_true", default=True)
    parser.add_argument("--no-subprocess", dest="subprocess", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--compile-config-json", default="{}")
    parser.add_argument("--single-task-json", default=None, help=argparse.SUPPRESS)


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

    vd = sub.add_parser("voice-design")
    add_runtime_args(vd)
    vd.add_argument("--text", default="你好，这是一次完全使用 OpenVINO 的 Qwen 三语音合成测试。")
    vd.add_argument("--instruct", default="A calm young female voice, natural Mandarin pronunciation.")
    vd.add_argument("--language", default="Auto")
    vd.set_defaults(func=run_voice_design)

    cv = sub.add_parser("custom-voice")
    add_runtime_args(cv)
    cv.add_argument("--text", required=True)
    cv.add_argument("--speaker", required=True)
    cv.add_argument("--instruct", default="")
    cv.add_argument("--language", default="Auto")
    cv.set_defaults(func=run_custom_voice)

    clone = sub.add_parser("voice-clone")
    add_runtime_args(clone)
    clone.add_argument("--text", required=True)
    clone.add_argument("--language", default="Auto")
    clone.add_argument("--ref-audio", required=True)
    clone.add_argument("--ref-text", default=None)
    clone.add_argument("--x-vector-only", action="store_true")
    clone.set_defaults(func=run_voice_clone)

    batch = sub.add_parser("batch")
    add_runtime_args(batch)
    batch.add_argument("--batch-jsonl", required=True)
    batch.add_argument("--output-dir", default="outputs/openvino_batch")
    batch.set_defaults(func=run_batch)

    stream = sub.add_parser("stream")
    stream_sub = stream.add_subparsers(dest="stream_command", required=True)

    svd = stream_sub.add_parser("voice-design")
    add_runtime_args(svd)
    add_stream_args(svd)
    svd.add_argument("--text", default="你好，这是一次流式 OpenVINO 合成测试。")
    svd.add_argument("--instruct", default="A calm young female voice, natural Mandarin pronunciation.")
    svd.add_argument("--language", default="Auto")
    svd.set_defaults(func=run_stream_voice_design)

    scv = stream_sub.add_parser("custom-voice")
    add_runtime_args(scv)
    add_stream_args(scv)
    scv.add_argument("--text", required=True)
    scv.add_argument("--speaker", required=True)
    scv.add_argument("--instruct", default="")
    scv.add_argument("--language", default="Auto")
    scv.set_defaults(func=run_stream_custom_voice)

    sclone = stream_sub.add_parser("voice-clone")
    add_runtime_args(sclone)
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
    serve_parser.add_argument("--warmup-strategy", default="low_latency", choices=["low_latency", "balanced", "stable"])
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

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf

from .runtime import OpenVINOQwen3TTS


def add_runtime_args(parser):
    parser.add_argument("--ir-dir", default="openvino/voice_design")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--decoder-device", default=None)
    parser.add_argument("--mode", default="no-cache", choices=["no-cache", "cache", "fast-cache", "fused-no-cache"])
    parser.add_argument("--cache-kernel", default="exact", choices=["exact", "sdpa"])
    parser.add_argument("--cache-step", default="split", choices=["split", "fused"])
    parser.add_argument("--graph-variant", default="fp16")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--progress-interval", type=int, default=8)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--output", default="outputs/openvino.wav")
    parser.add_argument("--skip-decode", action="store_true")


def build_runtime(args):
    return OpenVINOQwen3TTS(
        args.ir_dir,
        args.device,
        args.decoder_device,
        allow_cpu_fallback=args.allow_cpu_fallback,
        mode=args.mode,
        cache_kernel=args.cache_kernel,
        cache_step=args.cache_step,
        graph_variant=args.graph_variant,
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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

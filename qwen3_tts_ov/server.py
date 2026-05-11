import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .manifest import (
    DEFAULT_VOICE_DESIGN_IR_DIR,
    has_manifest,
    load_manifest,
    manifest_missing_message,
    path_text,
    resolve_ir_dir,
)
from .runtime import DEFAULT_STREAM_CHUNK_STRATEGIES, OpenVINOQwen3TTS
from .web_client import WEB_CLIENT_HTML


MODE_DIR = {
    "voice_design": "voice_design",
    "voice-design": "voice_design",
    "custom_voice": "custom_voice",
    "custom-voice": "custom_voice",
    "voice_clone": "base",
    "voice-clone": "base",
    "base": "base",
}


def audio_to_pcm16(audio) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def wav_bytes(audio, sample_rate: int) -> bytes:
    with io.BytesIO() as handle:
        sf.write(handle, np.asarray(audio, dtype=np.float32), sample_rate, format="WAV")
        return handle.getvalue()


def normalize_mode(mode: str) -> str:
    key = (mode or "").replace("-", "_")
    if key not in {"voice_design", "custom_voice", "voice_clone"}:
        raise ValueError("mode must be voice_design, custom_voice, or voice_clone")
    return key


def parse_csv(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def generation_kwargs(request: dict) -> dict:
    generation = request.get("generation") or {}
    def value(name: str, default):
        return generation.get(name, request.get(name, default))

    return {
        "max_new_tokens": int(value("max_new_tokens", 512)),
        "min_new_tokens": int(value("min_new_tokens", 2)),
        "repetition_penalty": float(value("repetition_penalty", 1.05)),
        "max_prompt_tokens": int(value("max_prompt_tokens", 512)),
        "progress_interval": int(value("progress_interval", 0)),
        "do_sample": bool(value("do_sample", False)),
        "top_k": int(value("top_k", 50)),
        "top_p": float(value("top_p", 1.0)),
        "temperature": float(value("temperature", 0.9)),
    }


def normalize_chunk_strategy(strategy: str | None) -> str:
    normalized = str(strategy or "low_latency").strip().replace("-", "_").lower()
    if normalized not in DEFAULT_STREAM_CHUNK_STRATEGIES:
        supported = ", ".join(sorted(DEFAULT_STREAM_CHUNK_STRATEGIES))
        raise ValueError(f"unsupported chunk_strategy={strategy!r}; supported strategies: {supported}")
    return normalized


def stream_kwargs(request: dict) -> dict:
    stream = request.get("stream") if isinstance(request.get("stream"), dict) else {}
    fmt = stream.get("format", "pcm_s16le")
    if fmt != "pcm_s16le":
        raise ValueError("only stream.format=pcm_s16le is supported")
    kwargs = {
        "chunk_strategy": stream.get("chunk_strategy", request.get("chunk_strategy")),
    }
    for name in ("initial_chunk_frames", "chunk_frames", "left_context_frames"):
        if name in stream:
            kwargs[name] = int(stream[name])
        elif name in request:
            kwargs[name] = int(request[name])
    return kwargs


def include_chunk_metadata(request: dict) -> bool:
    stream = request.get("stream") if isinstance(request.get("stream"), dict) else {}
    return bool(stream.get("include_chunk_metadata", request.get("include_chunk_metadata", False)))


def stream_metadata(request: dict) -> dict:
    stream = request.get("stream") if isinstance(request.get("stream"), dict) else {}
    strategy = normalize_chunk_strategy(stream.get("chunk_strategy", request.get("chunk_strategy")))
    defaults = DEFAULT_STREAM_CHUNK_STRATEGIES[strategy]
    return {
        "chunk_strategy": strategy,
        "initial_chunk_frames": int(stream.get("initial_chunk_frames", request.get("initial_chunk_frames", defaults["initial_chunk_frames"]))),
        "chunk_frames": int(stream.get("chunk_frames", request.get("chunk_frames", defaults["chunk_frames"]))),
        "left_context_frames": int(stream.get("left_context_frames", request.get("left_context_frames", defaults["left_context_frames"]))),
    }


def normalize_openai_task_type(request: dict) -> str:
    task_type = request.get("task_type") or request.get("mode")
    if task_type:
        key = str(task_type).strip().replace("-", "_").lower()
        if key in {"base", "voice_clone"}:
            return "voice_clone"
        if key in {"voice_design", "custom_voice"}:
            return key
        raise ValueError("task_type must be voice_design, custom_voice, voice_clone, or base")

    model_name = str(request.get("model") or "").lower()
    if request.get("ref_audio") or request.get("ref_text"):
        return "voice_clone"
    if "base" in model_name or "voiceclone" in model_name or "voice_clone" in model_name:
        return "voice_clone"
    if "customvoice" in model_name or "custom_voice" in model_name:
        return "custom_voice"
    voice = str(request.get("voice") or "").strip().lower()
    if voice and voice not in {"default", "voice_design", "none"}:
        return "custom_voice"
    return "voice_design"


def openai_speech_to_tts_request(request: dict) -> tuple[dict, str, bool]:
    text = request.get("input")
    if not text:
        raise ValueError("input is required")
    response_format = str(request.get("response_format", "wav")).lower()
    stream_enabled = bool(request.get("stream", False))
    mode_name = normalize_openai_task_type(request)

    generation = dict(request.get("generation") or {})
    for name in (
        "max_new_tokens",
        "min_new_tokens",
        "do_sample",
        "top_k",
        "top_p",
        "temperature",
        "repetition_penalty",
        "max_prompt_tokens",
        "progress_interval",
    ):
        if name in request:
            generation[name] = request[name]

    stream_config = {}
    if isinstance(request.get("stream"), dict):
        stream_config.update(request["stream"])
        stream_enabled = True
    for name in ("chunk_strategy", "initial_chunk_frames", "chunk_frames", "left_context_frames"):
        if name in request:
            stream_config[name] = request[name]
    stream_config.setdefault("format", "pcm_s16le")

    internal = {
        "mode": mode_name,
        "text": text,
        "language": request.get("language", "Auto"),
        "instruct": request.get("instructions", request.get("instruct", "")),
        "generation": generation,
        "stream": stream_config,
    }
    if mode_name == "custom_voice":
        internal["speaker"] = request.get("voice") or request.get("speaker")
    elif mode_name == "voice_clone":
        internal["ref_audio"] = request.get("ref_audio")
        internal["ref_text"] = request.get("ref_text")
        internal["x_vector_only"] = bool(request.get("x_vector_only_mode", request.get("x_vector_only", False)))
    return internal, response_format, stream_enabled


def create_app(
    model_root: str | Path = "openvino",
    device: str = "GPU",
    decoder_device: str | None = None,
    allow_cpu_fallback: bool = False,
    mode: str = "cache",
    cache_kernel: str = "exact",
    cache_step: str = "fused",
    graph_variant: str = "fp16",
    ov_cache_dir: str | Path | None = None,
    ov_cache_mode: str | None = "optimize_speed",
    disable_ov_cache: bool = False,
    warmup: bool = True,
    preload_modes: str | list[str] = "voice_design",
    preload_buckets: str = "warmup",
    warmup_text: str = "你好，这是一次流式预热。",
    warmup_strategy: str = "low_latency",
    recommended_playback_buffer_ms: int = 250,
):
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, Response, StreamingResponse

    app = FastAPI(title="Qwen3-TTS OpenVINO Engine")
    model_root = Path(model_root)
    runtimes = {}
    app.state.warmup = {
        "enabled": bool(warmup),
        "status": "pending" if warmup else "disabled",
        "warmup_strategy": warmup_strategy,
        "ov_cache_dir": None if disable_ov_cache else str(ov_cache_dir or "auto"),
        "ov_cache_mode": ov_cache_mode,
        "loaded_modes": [],
        "errors": {},
        "runtimes": {},
    }

    def manifest_supports_mode(ir_dir: Path, mode_name: str) -> bool:
        try:
            manifest = load_manifest(ir_dir)
        except Exception:
            return False
        model_type = str(manifest.get("tts_model_type") or "").replace("-", "_").lower()
        if mode_name == "voice_design":
            return model_type in {"", "voice_design"}
        if mode_name == "custom_voice":
            return model_type == "custom_voice"
        if mode_name == "voice_clone":
            return model_type in {"base", "voice_clone"}
        return False

    def resolve_mode_ir_dir(mode_name: str) -> Path:
        model_dir_name = MODE_DIR[mode_name]
        nested = model_root / model_dir_name
        if has_manifest(nested):
            return nested
        if has_manifest(model_root) and manifest_supports_mode(model_root, mode_name):
            return model_root
        if mode_name == "voice_design" and path_text(model_root) == "openvino":
            fallback = resolve_ir_dir(DEFAULT_VOICE_DESIGN_IR_DIR, fallback_to_local_voice_design=True, warn=True)
            if has_manifest(fallback) and fallback != nested:
                return fallback
        raise ValueError(manifest_missing_message(nested))

    def runtime_for_ir_dir(ir_dir: Path, do_sample: bool = False):
        if not has_manifest(ir_dir):
            raise ValueError(manifest_missing_message(ir_dir))
        effective_cache_step = "split" if do_sample and mode == "cache" and cache_step == "fused" else cache_step
        key = (str(ir_dir.resolve()), effective_cache_step, str(ov_cache_dir or "auto"), ov_cache_mode, bool(disable_ov_cache))
        if key not in runtimes:
            runtimes[key] = OpenVINOQwen3TTS(
                ir_dir,
                device=device,
                decoder_device=decoder_device,
                allow_cpu_fallback=allow_cpu_fallback,
                mode=mode,
                cache_kernel=cache_kernel,
                cache_step=effective_cache_step,
                graph_variant=graph_variant,
                ov_cache_dir=ov_cache_dir,
                ov_cache_mode=ov_cache_mode,
                disable_ov_cache=disable_ov_cache,
            )
        return runtimes[key]

    def get_runtime(request_mode: str, do_sample: bool = False):
        normalized = normalize_mode(request_mode)
        ir_dir = resolve_mode_ir_dir(normalized)
        return normalized, runtime_for_ir_dir(ir_dir, do_sample=do_sample)

    @app.on_event("startup")
    def warmup_on_startup():
        if not warmup:
            return
        app.state.warmup["status"] = "running"
        app.state.warmup["started_at"] = time.time()
        for preload_mode in parse_csv(preload_modes):
            key = preload_mode.replace("-", "_")
            if key not in MODE_DIR:
                app.state.warmup["errors"][preload_mode] = f"unsupported preload mode: {preload_mode}"
                continue
            try:
                ir_dir = resolve_mode_ir_dir(key)
                runtime = runtime_for_ir_dir(ir_dir, do_sample=False)
                status = runtime.prewarm_streaming(
                    text=warmup_text,
                    instruct="用自然、清晰的中文女声朗读。",
                    language="Chinese",
                    chunk_strategy=warmup_strategy,
                    left_context_frames=None,
                    max_new_tokens=None,
                    preload_buckets=preload_buckets,
                    run_generation=runtime.manifest.get("tts_model_type") == "voice_design",
                )
                app.state.warmup["loaded_modes"].append(preload_mode)
                app.state.warmup["runtimes"][preload_mode] = status
            except Exception as exc:
                app.state.warmup["errors"][preload_mode] = str(exc)
        app.state.warmup["finished_at"] = time.time()
        app.state.warmup["elapsed"] = app.state.warmup["finished_at"] - app.state.warmup["started_at"]
        app.state.warmup["status"] = "ready" if not app.state.warmup["errors"] else "ready_with_errors"

    def stream_chunks(request: dict):
        gen_kwargs = generation_kwargs(request)
        mode_name, runtime = get_runtime(request.get("mode"), do_sample=bool(gen_kwargs["do_sample"]))
        kwargs = {**gen_kwargs, **stream_kwargs(request)}
        text = request.get("text")
        language = request.get("language", "Auto")
        if not text:
            raise ValueError("text is required")
        if mode_name == "voice_design":
            instruct = request.get("instruct", "")
            return runtime.stream_voice_design(text=text, instruct=instruct, language=language, **kwargs)
        if mode_name == "custom_voice":
            speaker = request.get("speaker")
            if not speaker:
                raise ValueError("speaker is required for custom_voice")
            return runtime.stream_custom_voice(
                text=text,
                speaker=speaker,
                instruct=request.get("instruct", ""),
                language=language,
                **kwargs,
            )
        return runtime.stream_voice_clone(
            text=text,
            language=language,
            ref_audio=request.get("ref_audio"),
            ref_text=request.get("ref_text"),
            x_vector_only_mode=bool(request.get("x_vector_only", False)),
            **kwargs,
        )

    def full_audio(request: dict):
        kwargs = generation_kwargs(request)
        mode_name, runtime = get_runtime(request.get("mode"), do_sample=bool(kwargs["do_sample"]))
        text = request.get("text")
        language = request.get("language", "Auto")
        if not text:
            raise ValueError("text is required")
        if mode_name == "voice_design":
            wavs, sr = runtime.generate_voice_design(text=text, instruct=request.get("instruct", ""), language=language, **kwargs)
        elif mode_name == "custom_voice":
            speaker = request.get("speaker")
            if not speaker:
                raise ValueError("speaker is required for custom_voice")
            wavs, sr = runtime.generate_custom_voice(
                text=text,
                speaker=speaker,
                instruct=request.get("instruct", ""),
                language=language,
                **kwargs,
            )
        else:
            wavs, sr = runtime.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=request.get("ref_audio"),
                ref_text=request.get("ref_text"),
                x_vector_only_mode=bool(request.get("x_vector_only", False)),
                **kwargs,
            )
        return wavs[0], sr

    @app.get("/health")
    def health():
        runtime_status = {}
        for key, runtime in runtimes.items():
            ir_dir, effective_cache_step = key[0], key[1]
            runtime_status[ir_dir] = {
                "cache_step": effective_cache_step,
                "ov_cache_dir": None if getattr(runtime, "cache_dir", None) is None else str(runtime.cache_dir),
                "ov_cache_mode": getattr(runtime, "ov_cache_mode", None),
                "ov_cache_disabled": getattr(runtime, "disable_ov_cache", False),
                "streaming_decoder_available": bool(getattr(runtime, "streaming_decoder_graphs_by_context", {})),
                "streaming_decoder_contexts": {
                    str(context): sorted(chunk_graphs)
                    for context, chunk_graphs in getattr(runtime, "streaming_decoder_graphs_by_context", {}).items()
                },
                "default_chunk_strategy": getattr(runtime, "default_chunk_strategy", "low_latency"),
                "chunk_strategies": getattr(runtime, "streaming_decoder_strategies", DEFAULT_STREAM_CHUNK_STRATEGIES),
                "compiled_stream_decoders": [
                    {"context_frames": context, "chunk_frames": chunk}
                    for context, chunk in sorted(getattr(runtime, "streaming_decoders", {}))
                ],
                "compiled_fused_buckets": sorted(getattr(runtime, "fused_cache_step_by_bucket", {})),
                "compiled_stateful_buckets": sorted(getattr(runtime, "talker_stateful_by_bucket", {})),
            }
        return {
            "ok": True,
            "model_root": str(model_root),
            "warmup": app.state.warmup,
            "runtimes": runtime_status,
        }

    @app.get("/", response_class=HTMLResponse)
    def web_client():
        return WEB_CLIENT_HTML

    @app.get("/web", response_class=HTMLResponse)
    def web_client_alias():
        return WEB_CLIENT_HTML

    @app.post("/v1/tts")
    def tts(request: dict):
        try:
            audio, sr = full_audio(request)
            return Response(content=wav_bytes(audio, sr), media_type="audio/wav")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/tts/stream")
    def tts_stream(request: dict):
        def iter_lines():
            started = time.time()
            try:
                metadata = stream_metadata(request)
                yield json.dumps(
                    {
                        "type": "metadata",
                        "sample_rate": 24000,
                        "format": "pcm_s16le",
                        "started_at": started,
                        **metadata,
                        "recommended_playback_buffer_ms": int(recommended_playback_buffer_ms),
                    },
                    ensure_ascii=False,
                ) + "\n"
                for chunk in stream_chunks(request):
                    if chunk.audio.size:
                        yield json.dumps(
                            {
                                "type": "audio",
                                "index": chunk.index,
                                "sample_rate": chunk.sample_rate,
                                "format": "pcm_s16le",
                                "is_final": chunk.is_final,
                                "timings": chunk.timings,
                                "audio": base64.b64encode(audio_to_pcm16(chunk.audio)).decode("ascii"),
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    if chunk.is_final:
                        yield json.dumps(
                            {"type": "final", "index": chunk.index, "elapsed": time.time() - started, "timings": chunk.timings},
                            ensure_ascii=False,
                        ) + "\n"
            except Exception as exc:
                yield json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False) + "\n"

        return StreamingResponse(iter_lines(), media_type="application/x-ndjson")

    @app.get("/v1/audio/voices")
    def audio_voices():
        voices = []
        voice_details = []
        seen = set()
        manifest_paths = []
        if has_manifest(model_root):
            manifest_paths.append(("direct", model_root / "manifest.json"))
        for model_name in sorted(set(MODE_DIR.values())):
            manifest_paths.append((model_name, model_root / model_name / "manifest.json"))
        fallback = resolve_ir_dir(DEFAULT_VOICE_DESIGN_IR_DIR, fallback_to_local_voice_design=True)
        if path_text(model_root) == "openvino" and has_manifest(fallback):
            manifest_paths.append(("voice_design", fallback / "manifest.json"))
        for model_name, manifest_path in manifest_paths:
            if not manifest_path.exists():
                continue
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                continue
            speakers = (manifest.get("ids") or {}).get("spk_id") or manifest.get("spk_id") or {}
            if isinstance(speakers, dict):
                iterable = speakers.keys()
            elif isinstance(speakers, list):
                iterable = speakers
            else:
                iterable = []
            for speaker in iterable:
                name = str(speaker)
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                voices.append(name)
                voice_details.append({"id": name, "name": name, "source": model_name})
        return {"voices": voices, "voice_details": voice_details, "uploaded_voices": []}

    @app.post("/v1/audio/speech")
    def audio_speech(request: dict):
        try:
            internal, response_format, stream_enabled = openai_speech_to_tts_request(request)
            if stream_enabled:
                if response_format not in {"pcm", "pcm_s16le"}:
                    raise ValueError('stream=true requires response_format="pcm"')

                def iter_pcm():
                    for chunk in stream_chunks(internal):
                        if chunk.audio.size:
                            yield audio_to_pcm16(chunk.audio)

                return StreamingResponse(iter_pcm(), media_type="audio/L16; rate=24000; channels=1")

            audio, sr = full_audio(internal)
            if response_format in {"pcm", "pcm_s16le"}:
                return Response(content=audio_to_pcm16(audio), media_type=f"audio/L16; rate={sr}; channels=1")
            if response_format != "wav":
                raise ValueError("only response_format=wav or pcm is supported")
            return Response(content=wav_bytes(audio, sr), media_type="audio/wav")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.websocket("/v1/tts/stream")
    async def websocket_stream(websocket: WebSocket):
        await websocket.accept()
        try:
            request = await websocket.receive_json()
            started = time.time()
            metadata = stream_metadata(request)
            await websocket.send_json(
                {
                    "type": "metadata",
                    "sample_rate": 24000,
                    "format": "pcm_s16le",
                    "started_at": started,
                    **metadata,
                    "recommended_playback_buffer_ms": int(recommended_playback_buffer_ms),
                }
            )
            final_timings = {}
            final_index = 0
            send_chunk_metadata = include_chunk_metadata(request)
            for chunk in stream_chunks(request):
                final_timings = chunk.timings
                final_index = chunk.index
                if chunk.audio.size:
                    pcm = audio_to_pcm16(chunk.audio)
                    if send_chunk_metadata:
                        await websocket.send_json(
                            {
                                "type": "audio",
                                "index": chunk.index,
                                "sample_rate": chunk.sample_rate,
                                "format": "pcm_s16le",
                                "byte_length": len(pcm),
                                "is_final": chunk.is_final,
                                "timings": chunk.timings,
                            }
                        )
                    await websocket.send_bytes(pcm)
                if chunk.is_final:
                    await websocket.send_json(
                        {
                            "type": "final",
                            "index": final_index,
                            "elapsed": time.time() - started,
                            "timings": final_timings,
                        }
                    )
                    break
        except WebSocketDisconnect:
            return
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})

    return app


def serve(
    model_root: str | Path = "openvino",
    host: str = "127.0.0.1",
    port: int = 17860,
    device: str = "GPU",
    decoder_device: str | None = None,
    allow_cpu_fallback: bool = False,
    mode: str = "cache",
    cache_kernel: str = "exact",
    cache_step: str = "fused",
    graph_variant: str = "fp16",
    ov_cache_dir: str | Path | None = None,
    ov_cache_mode: str | None = "optimize_speed",
    disable_ov_cache: bool = False,
    warmup: bool = True,
    preload_modes: str | list[str] = "voice_design",
    preload_buckets: str = "warmup",
    warmup_text: str = "你好，这是一次流式预热。",
    warmup_strategy: str = "low_latency",
):
    import uvicorn

    app = create_app(
        model_root=model_root,
        device=device,
        decoder_device=decoder_device,
        allow_cpu_fallback=allow_cpu_fallback,
        mode=mode,
        cache_kernel=cache_kernel,
        cache_step=cache_step,
        graph_variant=graph_variant,
        ov_cache_dir=ov_cache_dir,
        ov_cache_mode=ov_cache_mode,
        disable_ov_cache=disable_ov_cache,
        warmup=warmup,
        preload_modes=preload_modes,
        preload_buckets=preload_buckets,
        warmup_text=warmup_text,
        warmup_strategy=warmup_strategy,
    )
    uvicorn.run(app, host=host, port=port)

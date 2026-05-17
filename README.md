# Qwen3-TTS OpenVINO

OpenVINO-only runtime, exporter, local sidecar server, and native vLLM-like
inference backend for Qwen3-TTS 12 Hz models.

[中文文档](README.zh-CN.md) | [Docs](docs/README.md) | [Examples](examples/README.md)

This is a source repository. It does not store model weights, exported
OpenVINO IR, generated audio, compile caches, or native build outputs.

## Quick Start

For normal use, download the prebuilt runtime from GitHub Releases. The server
auto-downloads the public OpenVINO IR from Hugging Face on first start.

Linux:

```bash
tar --zstd -xf qwen3-tts-ov-server-linux-x64-0.1.3-runtime-minimal.tar.zst
cd qwen3-tts-ov-server-linux-x64-0.1.3-runtime-minimal
./qwen3-tts-ov-server --device GPU
```

Windows:

```powershell
Expand-Archive qwen3-tts-ov-server-windows-x64-0.1.3-runtime-minimal.zip -DestinationPath .
cd qwen3-tts-ov-server-windows-x64-0.1.3-runtime-minimal
.\qwen3-tts-ov-server.exe --device GPU
```

Open `http://127.0.0.1:17860/`.

The Web Demo covers VoiceDesign, CustomVoice, VoiceClone, full-AR long text,
reference-audio upload, one-click model download, request/curl copy,
custom request JSON, and simultaneous multi-request smoke testing.

CLI clients can call the same local sidecar:

```bash
python examples/python/http_tts_wav.py --output outputs/example_http.wav
uv run --with websockets python examples/python/websocket_stream_pcm.py --output outputs/example_ws.wav
```

## Production Architecture

The repository has been consolidated around one production path:

- `fastest` runtime profile.
- Native C++ codec generation pipeline.
- OpenVINO paged-KV attention with U8 KV cache by default.
- vLLM-like online batching scheduler for request admission and decode steps.
- Full-context autoregressive long-text generation; text is not segmented.
- Lazy per-mode residency by default, so VoiceDesign, CustomVoice, and
  VoiceClone do not all stay resident at the same time.

VoiceDesign, CustomVoice, and VoiceClone are all served through the same
sidecar API. VoiceClone uses the Base IR and defaults to ICL cloning with
`ref_audio + ref_text`; `x_vector_only` is opt-in.

## Developer Build

Use this path only when rebuilding IR from local PyTorch weights.

```bash
uv sync --extra native --extra server --extra export

uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU

uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest
```

## Repository Layout

```text
qwen3_tts_ov/  runtime, exporter, CLI, sidecar, web demo
native/        native OpenVINO C++ codec generation backend
scripts/       production build, release, benchmark, and quality gates
docs/          Chinese documentation
examples/      request examples
tests/         unit tests
```

Ignored local artifacts include `models/`, `openvino/`, `openvino_full/`,
`outputs/`, `.venv/`, `.uv-cache/`, native build outputs, and generated audio.

## Documentation

- [Documentation index](docs/README.md)
- [Release usage](docs/release_zh.md)
- [Source quick start](docs/quick_start_zh.md)
- [Runtime APIs](docs/runtime_zh.md)
- [Export and build](docs/export_zh.md)
- [Artifacts policy](docs/artifacts_zh.md)
- [Security](docs/security_zh.md)

The detailed docs are Chinese-first; English quick entry points are provided in
this README, [docs/README.md](docs/README.md), and [examples/README.md](examples/README.md).

## License

Apache-2.0. See [LICENSE](LICENSE).

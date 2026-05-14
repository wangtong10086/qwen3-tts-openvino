# Qwen3-TTS OpenVINO

OpenVINO-only runtime, exporter, local sidecar server, and native acceleration
path for Qwen3-TTS 12 Hz models.

[中文文档](README.zh-CN.md) | [Documentation Index](docs/README.zh-CN.md) | [Examples](examples/README.zh-CN.md)

This repository contains source code. It does not store model weights, exported
OpenVINO IR, generated audio, OpenVINO compile caches, or native build outputs.

## What To Use

| Need | Use |
| --- | --- |
| Run TTS without a Python development environment | [GitHub Release runtime](https://github.com/wangtong10086/qwen3-tts-openvino/releases/tag/v0.1.3). The server auto-downloads the public [Hugging Face OpenVINO IR](https://huggingface.co/waston10086/qwen3-tts-openvino-voice-design) on first start. |
| Rebuild or tune IR from PyTorch weights | `uv run python -m qwen3_tts_ov build-fastest ...` |
| Publish Linux/Windows runtime packages | Push a `v*` tag to run `release-runtime` |

## Quick Start: Prebuilt Runtime

1. Download one runtime package from GitHub Releases:

```text
qwen3-tts-ov-server-linux-x64-0.1.3-runtime-minimal.tar.zst
qwen3-tts-ov-server-windows-x64-0.1.3-runtime-minimal.zip
```

2. Start the sidecar. If no local OpenVINO IR is found, it downloads the default
   public IR to the user cache and continues startup.

Linux:

```bash
tar --zstd -xf qwen3-tts-ov-server-linux-x64-0.1.3-runtime-minimal.tar.zst
cd qwen3-tts-ov-server-linux-x64-0.1.3-runtime-minimal
./qwen3-tts-ov-server \
  --device GPU
```

Windows:

```powershell
Expand-Archive qwen3-tts-ov-server-windows-x64-0.1.3-runtime-minimal.zip -DestinationPath .
cd qwen3-tts-ov-server-windows-x64-0.1.3-runtime-minimal
.\qwen3-tts-ov-server.exe `
  --device GPU
```

On Windows systems with Intel GPU+NPU support, you can offload the streaming
decoder to NPU:

```powershell
.\qwen3-tts-ov-server.exe `
  --device GPU `
  --npu-offload decoder
```

Open `http://127.0.0.1:17860/`.

For offline deployment, download the IR manually and pass
`--model-root qwen3-tts-openvino-ir/openvino_realtime`:

```bash
uv run --with huggingface_hub huggingface-cli download \
  waston10086/qwen3-tts-openvino-voice-design \
  --include "openvino_realtime/**" \
  --local-dir qwen3-tts-openvino-ir
```

## Developer Quick Start

Use this path when you want to export or tune the OpenVINO graphs yourself.

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

See [docs/quick_start_zh.md](docs/quick_start_zh.md) for the full source-build
guide.

## Highlights

- VoiceDesign, CustomVoice, and VoiceClone runtime interfaces. The public
  Hugging Face IR contains `voice_design`, `custom_voice`, and `base`
  directories under `openvino_realtime/`; VoiceClone uses the `base` IR.
- VoiceClone defaults to ICL cloning with `ref_audio` + `ref_text`, so the
  reference audio codec prompt is used. `x_vector_only` is opt-in for
  speaker-embedding-only experiments.
- The public Hugging Face IR is runtime-minimal: it keeps only the validated
  `fastest` production graphs instead of legacy decoder chunks, talker variants,
  or diagnostic graphs.
- Native C++ codec generation pipeline with OpenVINO paged-KV attention.
- Paged-KV cache defaults to U8 storage, roughly half the storage of FP16 KV
  cache; use `--kv-cache-profile fp16` for a conservative baseline.
- Streaming sidecar with WebSocket, HTTP NDJSON, and OpenAI-compatible Speech API.
- Long VoiceDesign requests are generated as one continuous autoregressive
  sequence; audio is chunked only for playback.
- Long prompt budget defaults to `auto`: GPU paths use a KV-cache planner based
  on model context, KV/cache-input precision, and `--max-vram-ratio`; CPU-only
  paths keep a conservative fixed budget. Use `--max-continuous-prompt-tokens`
  to override.
- Production runtime profile: `fastest`, `pcm_s16le`, mono 24 kHz output.

## Documentation

- [Release Usage](docs/release_zh.md): prebuilt runtime packages plus Hugging Face IR.
- [Quick Start](docs/quick_start_zh.md): source checkout, model download, fastest IR build, web demo.
- [Development Guide](docs/development_zh.md): export, native build, release workflow, diagnostics.
- [Runtime Usage](docs/runtime_zh.md): CLI, Python API, sidecar, WebSocket, OpenAI-compatible API.
- [Export Guide](docs/export_zh.md): manual OpenVINO export and compression.
- [Streaming and Long Text](docs/streaming_zh.md): streaming protocol, full-AR long text, quality gate.
- [Artifacts Policy](docs/artifacts_zh.md): model, IR, output, and build artifact handling.
- [Security](docs/security_zh.md): credentials and commit checks.

## Repository Layout

```text
qwen3_tts_ov/  runtime, exporter, CLI, sidecar server, and web client
native/        native OpenVINO C++ codec generation pipeline
scripts/       build, compression, benchmark, and quality helper scripts
devtools/      experimental benchmark, profiling, and diagnostics
docs/          Chinese documentation
examples/      small JSON/JSONL/text examples
tests/         unit tests
```

## License

Apache-2.0. See [LICENSE](LICENSE).

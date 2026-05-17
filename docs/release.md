# Release Usage

This page is the English quick path for end users. The detailed Chinese release
guide is [release_zh.md](release_zh.md).

Before starting, review [Prerequisites](prerequisites.md). For common startup
failures, see [Troubleshooting](troubleshooting.md).

Current release version: `v0.1.4`.

## Download

Runtime packages are published on GitHub Releases:

- Linux: `qwen3-tts-ov-server-linux-x64-0.1.4-runtime-minimal.tar.zst`
- Windows: `qwen3-tts-ov-server-windows-x64-0.1.4-runtime-minimal.zip`
- Release page: <https://github.com/wangtong10086/qwen3-tts-openvino/releases/tag/v0.1.4>

The runtime package does not include model weights or OpenVINO IR. On first
start, the server downloads public IR from Hugging Face unless `--model-root`
already points to a local IR directory.

## Linux

```bash
tar --zstd -xf qwen3-tts-ov-server-linux-x64-0.1.4-runtime-minimal.tar.zst
cd qwen3-tts-ov-server-linux-x64-0.1.4-runtime-minimal
./qwen3-tts-ov-server --device GPU
```

Open:

```text
http://127.0.0.1:17860/
```

## Windows

```powershell
Expand-Archive qwen3-tts-ov-server-windows-x64-0.1.4-runtime-minimal.zip -DestinationPath .
cd qwen3-tts-ov-server-windows-x64-0.1.4-runtime-minimal
.\qwen3-tts-ov-server.exe --device GPU
```

Optional GPU+NPU fallback mode:

```powershell
.\qwen3-tts-ov-server.exe --device GPU --npu-offload auto
```

Use `--npu-offload decoder` only for strict NPU validation; it fails if NPU is
not visible or the decoder cannot compile on NPU.

## Local IR

Pre-download public IR:

```bash
uv run --with huggingface_hub huggingface-cli download \
  waston10086/qwen3-tts-openvino-voice-design \
  --include "openvino_realtime/**" \
  --local-dir qwen3-tts-openvino-ir
```

Start with the downloaded root:

```bash
./qwen3-tts-ov-server --device GPU --model-root qwen3-tts-openvino-ir/openvino_realtime
```

Expected layout:

```text
openvino_realtime/
  voice_design/
  custom_voice/
  base/
```

## API Entry Points

- Web Demo: `http://127.0.0.1:17860/`
- Health: `GET /health`
- Model status: `GET /v1/models`
- Complete WAV: `POST /v1/tts`
- Streaming: `POST` or WebSocket `/v1/tts/stream`
- OpenAI-compatible Speech: `POST /v1/audio/speech`

Detailed fields are documented in [API Reference](api_reference.md).


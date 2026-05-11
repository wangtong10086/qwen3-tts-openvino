# Qwen3-TTS OpenVINO

OpenVINO runtime and exporter for Qwen3-TTS 12 Hz models.

This repository is source-only. It does not include model weights, exported
OpenVINO IR, generated audio, or local virtual environments. See the Chinese
documentation for complete setup and export instructions:

- [中文文档](README.zh-CN.md)
- [Export guide](docs/export_zh.md)
- [Artifacts and large files](docs/artifacts_zh.md)
- [Streaming guide](docs/streaming_zh.md)
- [OpenVINO cache guide](docs/cache_zh.md)
- [Security notes](docs/security_zh.md)

## Features

- OpenVINO runtime entrypoint: `python -m qwen3_tts_ov`
- VoiceDesign inference: `text + instruct + language`
- CustomVoice inference: `text + speaker + optional instruct + language`
- VoiceClone API shape: `text + ref_audio/ref_text or reusable prompt + language`
- Streaming synthesis through Python iterators, CLI chunks, and a local HTTP/WebSocket sidecar
- Exporter for Qwen3-TTS OpenVINO IR
- Runtime import path avoids importing PyTorch; PyTorch is used only for export

## Quick Start

```bash
uv run python -m qwen3_tts_ov --help
```

Export a VoiceDesign model:

```bash
uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --model-type voice_design \
  --out-dir openvino/voice_design \
  --cache-buckets 128,192,256,320,384 \
  --cache-kernels exact,sdpa \
  --fused-cache-kernels exact \
  --decoder-tokens 64,128,256 \
  --stream-decoder-chunks 8,12,24 \
  --stream-decoder-first-chunks 6,8,12 \
  --stream-decoder-left-context 25
```

Run inference:

```bash
uv run python -m qwen3_tts_ov voice-design \
  --ir-dir openvino/voice_design \
  --device GPU \
  --text "你好，这是一次 OpenVINO 推理测试。" \
  --instruct "A calm young female voice, natural Mandarin pronunciation." \
  --language Chinese \
  --output outputs/voice_design.wav
```

Run streaming synthesis:

```bash
uv run python -m qwen3_tts_ov stream voice-design \
  --ir-dir openvino/voice_design \
  --text "Streaming OpenVINO synthesis test." \
  --instruct "A bright and clear narrator voice." \
  --language English \
  --chunk-strategy low_latency \
  --chunk-dir outputs/stream \
  --output outputs/stream.wav
```

## Repository Policy

Large generated assets are intentionally ignored:

- `models/`
- `openvino_full/`
- `openvino/`
- `outputs/`
- `.venv/`

Regenerate them locally by following [docs/export_zh.md](docs/export_zh.md).

## Layout

```text
qwen3_tts_ov/  package, runtime, exporter, sidecar, and CLI
docs/          Chinese guides for export, streaming, cache, artifacts, security
examples/      small JSON/JSONL request examples
scripts/       development helpers for compression, quantization, benchmark
tests/         unit tests for runtime streaming, server mapping, cache config
```

## License

Apache-2.0. See [LICENSE](LICENSE).

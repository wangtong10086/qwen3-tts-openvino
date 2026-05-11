# Qwen3-TTS OpenVINO

OpenVINO runtime and exporter for Qwen3-TTS 12 Hz models.

This repository is source-only. It does not include model weights, exported
OpenVINO IR, generated audio, or local virtual environments. See the Chinese
documentation for complete setup and export instructions:

- [中文文档](README.zh-CN.md)
- [Export guide](docs/export_zh.md)
- [Artifacts and large files](docs/artifacts_zh.md)
- [Security notes](docs/security_zh.md)

## Features

- OpenVINO runtime entrypoint: `python -m qwen3_tts_ov`
- VoiceDesign inference: `text + instruct + language`
- CustomVoice inference: `text + speaker + optional instruct + language`
- VoiceClone API shape: `text + ref_audio/ref_text or reusable prompt + language`
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
  --decoder-tokens 64,128,256
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

## Repository Policy

Large generated assets are intentionally ignored:

- `models/`
- `openvino_full/`
- `openvino/`
- `outputs/`
- `.venv/`

Regenerate them locally by following [docs/export_zh.md](docs/export_zh.md).

## License

Apache-2.0. See [LICENSE](LICENSE).

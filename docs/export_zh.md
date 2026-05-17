# 导出与构建

推荐使用 `build-fastest`，不要手动拼接旧 profile 或诊断图。

## 一键构建

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU
```

该命令会完成 native build、OpenVINO export、INT8_SYM production variant、minimal online-batching variant 和 cache warmup。

CustomVoice：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model-type custom_voice \
  --model models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --out-dir openvino/custom_voice \
  --device GPU
```

Base/VoiceClone：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model-type base \
  --model models/Qwen3-TTS-12Hz-1.7B-Base \
  --out-dir openvino/base \
  --device GPU
```

## 从零重建

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU \
  --clean \
  --clean-native
```

## 手动导出

只有在调试 exporter 时才直接使用：

```bash
uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --model-type voice_design \
  --out-dir openvino/voice_design \
  --skip-fixed-cache-graphs \
  --cache-buckets 96 \
  --cache-kernels exact \
  --fused-cache-kernels exact \
  --fused-subcode-mode cached \
  --export-paged-kv-seed \
  --paged-kv-subcode-attention-kernels sdpa \
  --decoder-tokens 256 \
  --stream-decoder-chunks 12,24 \
  --stream-decoder-first-chunks 8,12 \
  --stream-decoder-left-context 25 \
  --stream-decoder-input-shape static
```

压缩生产 variant：

```bash
uv run python scripts/compress_openvino_weights.py \
  --ir-dir openvino/voice_design \
  --preset fastest

uv run python scripts/compress_openvino_weights.py \
  --ir-dir openvino/voice_design \
  --preset minimal-online-gqa
```

## 产物策略

`openvino/` 和 `models/` 不进入 git。公开 release 使用 Hugging Face 上的 runtime-minimal IR，源码仓库只保留导出代码和文档。

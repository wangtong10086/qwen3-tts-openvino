# OpenVINO 导出指南

本文说明如何从本地 Qwen3-TTS 权重导出 OpenVINO IR。导出阶段会使用 PyTorch；导出的 runtime 推理路径不导入 PyTorch。

## 1. 准备模型

VoiceDesign：

```bash
uv run modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --local_dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

CustomVoice：

```bash
uv run modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local_dir ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

Base/VoiceClone：

```bash
uv run modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local_dir ./models/Qwen3-TTS-12Hz-1.7B-Base
```

## 2. 导出 VoiceDesign

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

## 3. 导出 CustomVoice

```bash
uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --model-type custom_voice \
  --out-dir openvino/custom_voice \
  --cache-buckets 128,192,256,320,384 \
  --cache-kernels exact,sdpa \
  --fused-cache-kernels exact \
  --decoder-tokens 64,128,256
```

## 4. 导出 Base/VoiceClone

VoiceClone 需要额外导出参考音频编码和 speaker embedding 图：

```bash
uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-Base \
  --model-type base \
  --out-dir openvino/base \
  --cache-buckets 128,192,256,320,384 \
  --cache-kernels exact,sdpa \
  --fused-cache-kernels exact \
  --decoder-tokens 64,128,256 \
  --export-clone-graphs
```

## 5. 可选 INT8 权重压缩

```bash
uv run python compress_openvino_weights.py \
  --ir-dir openvino/voice_design \
  --variant int8 \
  --mode int8_asym \
  --include-cached-subcode
```

压缩后可用 `--graph-variant int8_cachedsub` 选择快速路径。

## 6. 校验

```bash
uv run python -m qwen3_tts_ov voice-design \
  --ir-dir openvino/voice_design \
  --device GPU \
  --max-new-tokens 4 \
  --skip-decode \
  --profile
```

如果 OpenVINO 只能看到 CPU，请先确认 Intel GPU runtime、WSL/DXG 或宿主驱动状态。

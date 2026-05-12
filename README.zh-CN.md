# Qwen3-TTS OpenVINO

这是一个面向 Qwen3-TTS 12Hz 模型的 OpenVINO-only 推理仓库。仓库只保存源码、文档和小型示例，不保存模型权重、OpenVINO IR、生成音频、本地虚拟环境或 native 编译产物。

生产主线只推荐 `fastest` profile：

- 必须构建 native C++ pipeline。
- 必须导出 cached-subcode fused graph。
- 必须生成 `int8_sym_fused_cachedsub` 权重压缩 variant。
- 必须包含 no-repeat `unroll4` 和 decode-unroll graph。
- 默认 `chunk_strategy=smooth`，用于降低块间卡顿风险。

## 目录

```text
qwen3_tts_ov/    生产 runtime、exporter、sidecar、web demo 和 CLI
native/          必需的 C++ OpenVINO GenAI-style pipeline 源码
scripts/         生产辅助脚本：native 构建、权重压缩、realtime benchmark
devtools/        实验 benchmark、profiling、旧入口和历史对照脚本
docs/            中文补充文档
examples/        JSON/JSONL 请求示例
tests/           单元测试
```

以下目录由本地生成，默认不进入 git：

```text
models/
openvino/
openvino_full/
outputs/
.venv/
native/build/
```

## 安装

```bash
git submodule update --init --recursive
uv sync --extra native --extra server --extra export
uv run python scripts/build_native_codegen.py
uv run python -m qwen3_tts_ov --help
```

`third_party/openvino.genai` 是 native 构建所需的 pinned submodule。Runtime 推理不导入 PyTorch；导出模型时才需要 export 依赖。

## 下载模型

VoiceDesign 示例：

```bash
uv run modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --local_dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

CustomVoice 和 Base/VoiceClone 需要分别下载对应模型目录。

## 导出最快路径 IR

VoiceDesign：

```bash
uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --model-type voice_design \
  --out-dir openvino/voice_design \
  --cache-buckets 96,128,192,256,320,384 \
  --cache-kernels exact \
  --fused-cache-kernels exact \
  --fused-subcode-mode cached \
  --fused-cache-unroll-steps 4 \
  --fused-cache-norepeat-steps 4 \
  --decoder-tokens 64,128,256 \
  --stream-decoder-chunks 8,12,24 \
  --stream-decoder-first-chunks 8 \
  --stream-decoder-left-context 25
```

CustomVoice 使用相同参数，替换模型和输出目录：

```bash
uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --model-type custom_voice \
  --out-dir openvino/custom_voice \
  --cache-buckets 96,128,192,256,320,384 \
  --cache-kernels exact \
  --fused-cache-kernels exact \
  --fused-subcode-mode cached \
  --fused-cache-unroll-steps 4 \
  --fused-cache-norepeat-steps 4 \
  --decoder-tokens 64,128,256 \
  --stream-decoder-chunks 8,12,24 \
  --stream-decoder-first-chunks 8 \
  --stream-decoder-left-context 25
```

Base/VoiceClone 额外加 `--export-clone-graphs`：

```bash
uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-Base \
  --model-type base \
  --out-dir openvino/base \
  --cache-buckets 96,128,192,256,320,384 \
  --cache-kernels exact \
  --fused-cache-kernels exact \
  --fused-subcode-mode cached \
  --fused-cache-unroll-steps 4 \
  --fused-cache-norepeat-steps 4 \
  --decoder-tokens 64,128,256 \
  --stream-decoder-chunks 8,12,24 \
  --stream-decoder-first-chunks 8 \
  --stream-decoder-left-context 25 \
  --export-clone-graphs
```

压缩最快路径权重：

```bash
uv run python scripts/compress_openvino_weights.py \
  --ir-dir openvino/voice_design \
  --source-variant fp16_fused_cachedsub \
  --variant int8_sym_fused_cachedsub \
  --mode int8_sym \
  --fused-cache-unroll-steps 4
```

对 CustomVoice/Base 重复同样压缩命令，替换 `--ir-dir`。

## 预热与运行

```bash
uv run python -m qwen3_tts_ov cache-warmup \
  --ir-dir openvino/voice_design \
  --device GPU \
  --realtime-profile fastest \
  --graphs core,stream,buckets \
  --preload-buckets warmup
```

启动 sidecar：

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest \
  --preload-modes voice_design \
  --preload-buckets warmup \
  --host 127.0.0.1 \
  --port 17860
```

浏览器测试页：

```text
http://127.0.0.1:17860/
```

## CLI 示例

VoiceDesign 流式：

```bash
uv run python -m qwen3_tts_ov stream voice-design \
  --ir-dir openvino/voice_design \
  --device GPU \
  --realtime-profile fastest \
  --text "你好，这是一次流式 OpenVINO 合成测试。" \
  --instruct "A calm young female voice." \
  --language Chinese \
  --chunk-dir outputs/stream \
  --output outputs/stream.wav
```

CustomVoice：

```bash
uv run python -m qwen3_tts_ov stream custom-voice \
  --ir-dir openvino/custom_voice \
  --device GPU \
  --realtime-profile fastest \
  --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
  --speaker Vivian \
  --instruct "用特别愤怒的语气说" \
  --language Chinese \
  --output outputs/custom_voice.wav
```

VoiceClone：

```bash
uv run python -m qwen3_tts_ov stream voice-clone \
  --ir-dir openvino/base \
  --device GPU \
  --realtime-profile fastest \
  --text "I am solving the equation, but it is a disaster." \
  --language English \
  --ref-audio /path/to/reference.wav \
  --ref-text "Reference transcript for the audio." \
  --output outputs/voice_clone.wav
```

## HTTP API

WebSocket `/v1/tts/stream` 和 HTTP NDJSON `/v1/tts/stream` 都返回 `pcm_s16le` 音频块。OpenAI-compatible endpoint：

```bash
curl -N http://127.0.0.1:17860/v1/audio/speech \
  -H "content-type: application/json" \
  -d '{"model":"qwen3-tts-openvino","voice":"default","input":"你好，这是兼容 OpenAI Speech API 的流式 PCM。","language":"Chinese","task_type":"voice_design","instructions":"A calm young female voice.","stream":true,"response_format":"pcm"}' \
  --output speech.pcm
```

## Python API

```python
from qwen3_tts_ov import OpenVINOQwen3TTS

tts = OpenVINOQwen3TTS.from_ir("openvino/voice_design", device="GPU")
for chunk in tts.stream_voice_design(
    text="你好，这是 Python 流式 API 测试。",
    instruct="A calm young female voice.",
    language="Chinese",
):
    if chunk.audio.size:
        print(chunk.index, chunk.audio.shape, chunk.is_final)
```

`from_ir()` 默认使用 `fastest` profile。需要低层实验参数时请直接调用构造函数或使用 `devtools/` 中的脚本。

## Benchmark

```bash
uv run python scripts/benchmark_streaming_realtime.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profile-set fastest-gate \
  --runs 3 \
  --warmup-generations 1
```

输出写入 `outputs/realtime_bench/streaming_profiles.json`。`serve --realtime-profile auto` 会优先读取其中已接受的 p90 summary；没有结果时回到 `fastest`。

## 常见问题

- **为什么仓库没有模型和 IR？**  
  这些文件通常很大，必须本地下载和导出。

- **为什么 native 是必需项？**  
  当前最快路径依赖 C++ pipeline 将 codegen 和 streaming decoder 的关键循环移出 Python。缺少 native 库时生产路径会直接报错。

- **还保留旧优化实验吗？**  
  保留在 `devtools/`，但不作为生产入口。主 README 只描述当前最快实现。

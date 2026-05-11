# Qwen3-TTS OpenVINO

这是一个面向 Qwen3-TTS 12Hz 模型的 OpenVINO 推理与导出项目。仓库只保存源码、文档和小型示例，不保存模型权重、OpenVINO IR、生成音频或本地虚拟环境。

## 目录说明

```text
qwen3_tts_ov/              OpenVINO runtime、导出器和 CLI
docs/                      中文补充文档
examples/                  小型输入示例
scripts/                   开发、压缩、量化和 benchmark 辅助脚本
tests/                     单元测试
```

以下目录由本地生成，默认不进入 git：

```text
models/          ModelScope/Hugging Face 下载的原始模型
openvino/        推荐的新导出目录
openvino_full/   旧的本地 VoiceDesign IR 目录
outputs/         生成音频和 benchmark 输出
.venv/           本地 Python 环境
```

## 安装

推荐继续使用当前项目的 `uv` 工作流：

```bash
uv run python -m qwen3_tts_ov --help
```

如果从干净环境开始，建议安装 runtime 依赖：

```bash
uv pip install -e .
```

安装后可以使用 console script：

```bash
uv run qwen3-tts-ov --help
```

未安装项目包时，请使用本文档默认采用的模块入口：

```bash
uv run python -m qwen3_tts_ov --help
```

需要导出模型时安装 export 依赖：

```bash
uv pip install -e ".[export]"
```

## 下载模型

示例：下载 VoiceDesign 模型到本地 `models/`。

```bash
uv run modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --local_dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

CustomVoice 和 Base/VoiceClone 需要分别下载对应模型目录。

## IR 路径约定

源码仓库不包含 OpenVINO IR。所有 `--ir-dir` 都必须指向一个已经导出的目录，并且该目录下必须存在 `manifest.json`。

推荐新导出目录：

```text
openvino/voice_design/manifest.json
openvino/custom_voice/manifest.json
openvino/base/manifest.json
```

如果你当前机器上只有旧的本地产物 `openvino_full/manifest.json`，可以先用它验证 VoiceDesign，把命令中的 `openvino/voice_design` 替换为 `openvino_full`。长期建议重新导出到 `openvino/voice_design`，这样 sidecar 的多模型目录布局更清晰。

## 导出 OpenVINO IR

VoiceDesign：

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

CustomVoice：

```bash
uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --model-type custom_voice \
  --out-dir openvino/custom_voice \
  --cache-buckets 128,192,256,320,384 \
  --cache-kernels exact,sdpa \
  --fused-cache-kernels exact \
  --decoder-tokens 64,128,256 \
  --stream-decoder-chunks 8,12,24 \
  --stream-decoder-first-chunks 6,8,12 \
  --stream-decoder-left-context 25
```

Base/VoiceClone：

```bash
uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-Base \
  --model-type base \
  --out-dir openvino/base \
  --cache-buckets 128,192,256,320,384 \
  --cache-kernels exact,sdpa \
  --fused-cache-kernels exact \
  --decoder-tokens 64,128,256 \
  --stream-decoder-chunks 8,12,24 \
  --stream-decoder-first-chunks 6,8,12 \
  --stream-decoder-left-context 25 \
  --export-clone-graphs
```

## 推理

VoiceDesign：

```bash
uv run python -m qwen3_tts_ov voice-design \
  --ir-dir openvino/voice_design \
  --device GPU \
  --text "哥哥，你回来啦，人家等了你好久好久了，要抱抱！" \
  --instruct "体现撒娇稚嫩的女声，音调偏高且起伏明显。" \
  --language Chinese \
  --output outputs/design.wav
```

CustomVoice：

```bash
uv run python -m qwen3_tts_ov custom-voice \
  --ir-dir openvino/custom_voice \
  --device GPU \
  --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
  --speaker Vivian \
  --instruct "用特别愤怒的语气说" \
  --language Chinese \
  --output outputs/custom_voice.wav
```

VoiceClone：

```bash
uv run python -m qwen3_tts_ov voice-clone \
  --ir-dir openvino/base \
  --device GPU \
  --text "I am solving the equation, but it is a disaster." \
  --language English \
  --ref-audio /path/to/reference.wav \
  --ref-text "Reference transcript for the audio." \
  --output outputs/voice_clone.wav
```

批处理：

```bash
uv run python -m qwen3_tts_ov batch \
  --ir-dir openvino/voice_design \
  --batch-jsonl examples/requests.example.jsonl \
  --output-dir outputs/batch
```

JSONL 中每行使用 `mode` 声明推理模式。实际运行时应让 `--ir-dir` 与 JSONL 中的任务类型匹配；本仓库提供的 `examples/requests.example.jsonl` 是 VoiceDesign 示例。

## 流式合成

首次部署或更新 IR 后，建议先填充 OpenVINO 编译缓存：

```bash
VOICE_DESIGN_IR=${VOICE_DESIGN_IR:-openvino/voice_design}
if [ ! -f "$VOICE_DESIGN_IR/manifest.json" ] && [ -f openvino_full/manifest.json ]; then
  VOICE_DESIGN_IR=openvino_full
fi
test -f "$VOICE_DESIGN_IR/manifest.json" || { echo "未找到 OpenVINO IR，请先导出模型。"; exit 1; }
uv run python -m qwen3_tts_ov cache-warmup \
  --ir-dir "$VOICE_DESIGN_IR" \
  --device GPU \
  --mode cache \
  --cache-step fused \
  --graphs core,stream,buckets \
  --preload-buckets warmup \
  --warmup-strategy low_latency
```

缓存默认写入用户缓存目录，而不是 IR 目录。更多细节见 [docs/cache_zh.md](docs/cache_zh.md)。
如果 `openvino/voice_design` 还不存在，但当前目录存在旧的 `openvino_full/manifest.json`，runtime 会自动回退到 `openvino_full`。

调试 CLI 会按 chunk 写出 raw PCM，并把拼接结果写为 WAV：

```bash
uv run python -m qwen3_tts_ov stream voice-design \
  --ir-dir openvino/voice_design \
  --device GPU \
  --text "你好，这是一次流式 OpenVINO 合成测试。" \
  --instruct "A calm young female voice." \
  --language Chinese \
  --chunk-strategy low_latency \
  --chunk-dir outputs/stream \
  --output outputs/stream.wav
```

本地 sidecar 服务：

```bash
uv pip install -e ".[server]"
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --host 127.0.0.1 \
  --port 17860 \
  --preload-modes voice_design \
  --preload-buckets warmup \
  --warmup-strategy low_latency
```

浏览器测试页：

```text
http://127.0.0.1:17860/
```

页面会通过 WebSocket `/v1/tts/stream` 接收 `pcm_s16le` 音频块并实时播放，也可以下载拼接后的 WAV。
服务默认会预热 VoiceDesign；如需跳过预热可加 `--no-warmup`。

如果只想临时使用单个 VoiceDesign IR 目录，也可以直接用 `--ir-dir` 指向该 IR，例如：

```bash
uv run python -m qwen3_tts_ov serve \
  --ir-dir openvino_full \
  --host 127.0.0.1 \
  --port 17860 \
  --preload-modes voice_design
```

HTTP NDJSON 流式接口：

```bash
curl -N http://127.0.0.1:17860/v1/tts/stream \
  -H "content-type: application/json" \
  -d '{"mode":"voice_design","text":"你好，这是 HTTP 流式合成。","language":"Chinese","instruct":"A calm young female voice.","stream":{"chunk_strategy":"low_latency","format":"pcm_s16le"}}'
```

OpenAI-compatible Speech API：

```bash
curl -N http://127.0.0.1:17860/v1/audio/speech \
  -H "content-type: application/json" \
  -d '{"model":"qwen3-tts-openvino","voice":"default","input":"你好，这是兼容 OpenAI Speech API 的流式 PCM。","language":"Chinese","task_type":"voice_design","instructions":"A calm young female voice.","stream":true,"response_format":"pcm","chunk_strategy":"low_latency"}' \
  --output speech.pcm
```

更多协议细节见 [docs/streaming_zh.md](docs/streaming_zh.md)。

## Python API

```python
from qwen3_tts_ov import OpenVINOQwen3TTS

tts = OpenVINOQwen3TTS.from_ir("openvino/voice_design", device="GPU")
wavs, sr = tts.generate_voice_design(
    text="你好，这是 Python API 测试。",
    instruct="A calm young female voice.",
    language="Chinese",
    max_new_tokens=128,
)
```

流式 Python API：

```python
for chunk in tts.stream_voice_design(
    text="你好，这是 Python 流式 API 测试。",
    instruct="A calm young female voice.",
    language="Chinese",
):
    if chunk.audio.size:
        print(chunk.index, chunk.audio.shape, chunk.is_final)
```

## 常见问题

- **为什么仓库没有 `openvino_full/`？**  
  OpenVINO IR 通常非常大，当前本地 VoiceDesign IR 约 58GB，不适合普通 Git 仓库。请按导出文档本地生成。

- **推理时会导入 PyTorch 吗？**  
  Runtime 入口不导入 PyTorch；导出阶段需要 PyTorch。

- **`fused-*` 模式支持采样吗？**  
  当前 fused 图是 greedy 路径。需要 `--do-sample` 时使用 `--mode no-cache`。

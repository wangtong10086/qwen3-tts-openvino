# Qwen3-TTS OpenVINO

这是一个面向 Qwen3-TTS 12Hz 模型的 OpenVINO 推理与导出项目。仓库只保存源码、文档和小型示例，不保存模型权重、OpenVINO IR、生成音频或本地虚拟环境。

## 目录说明

```text
qwen3_tts_ov/              OpenVINO runtime、导出器和 CLI
docs/                      中文补充文档
examples/                  小型输入示例
compress_openvino_weights.py  OpenVINO 权重量化/压缩辅助脚本
quantize_openvino_full.py     NNCF PTQ 实验辅助脚本
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
  --decoder-tokens 64,128,256
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
  --decoder-tokens 64,128,256
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

## 常见问题

- **为什么仓库没有 `openvino_full/`？**  
  OpenVINO IR 通常非常大，当前本地 VoiceDesign IR 约 58GB，不适合普通 Git 仓库。请按导出文档本地生成。

- **推理时会导入 PyTorch 吗？**  
  Runtime 入口不导入 PyTorch；导出阶段需要 PyTorch。

- **`fused-*` 模式支持采样吗？**  
  当前 fused 图是 greedy 路径。需要 `--do-sample` 时使用 `--mode no-cache`。

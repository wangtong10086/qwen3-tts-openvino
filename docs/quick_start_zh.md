# Quick Start

本页面向源码开发者，目标是从 PyTorch 模型导出当前生产 `fastest` OpenVINO IR，并启动 Web Demo。

普通用户请优先阅读 [Release 使用说明](release_zh.md)，不需要安装 PyTorch 或重新导出模型。

新机器先确认 [前置条件](prerequisites_zh.md)，尤其是 Python `>=3.12`、`uv`、Intel GPU 驱动和 OpenVINO 可见设备。

## 1. 安装依赖

如果尚未安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

然后安装项目依赖：

```bash
uv sync --extra native --extra server --extra export
uv run python -m qwen3_tts_ov --help
```

如果 `third_party/openvino.genai` 尚未初始化，`build-fastest` 会自动初始化 submodule。也可以手动执行：

```bash
git submodule update --init --recursive
```

## 2. 准备 PyTorch 模型

VoiceDesign：

```bash
uv run modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --local_dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

CustomVoice 和 Base/VoiceClone：

```bash
uv run modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local_dir ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice

uv run modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local_dir ./models/Qwen3-TTS-12Hz-1.7B-Base
```

`models/` 不进入 git。

## 3. 一键构建生产 IR

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU
```

该命令会：

1. 构建 native C++ backend。
2. 导出生产运行需要的 OpenVINO 图。
3. 压缩 `fastest` 和 minimal online-batching variant。
4. 执行 OpenVINO cache warmup。

从旧产物完全重来：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU \
  --clean \
  --clean-native
```

CustomVoice 和 Base/VoiceClone 使用相同入口，只换模型和输出目录：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --out-dir openvino/custom_voice \
  --device GPU

uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-Base \
  --out-dir openvino/base \
  --device GPU
```

只预览步骤：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU \
  --dry-run
```

## 4. 启动开发服务

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest \
  --host 127.0.0.1 \
  --port 17860
```

打开：

```text
http://127.0.0.1:17860/
```

快速调用：

```bash
python examples/python/http_tts_wav.py --output outputs/example_http.wav
uv run --with websockets python examples/python/websocket_stream_pcm.py --output outputs/example_ws.wav
```

## 5. 验证

```bash
uv run python -m qwen3_tts_ov build-fastest --help
uv run python -m qwen3_tts_ov serve --help
uv run python scripts/benchmark_prompt_batch_matrix.py --dry-run
python -m py_compile examples/python/*.py
```

确认 OpenVINO 能看到目标设备：

```bash
uv run python - <<'PY'
import openvino as ov
core = ov.Core()
print(core.available_devices)
for name in core.available_devices:
    print(name, core.get_property(name, "FULL_DEVICE_NAME"))
PY
```

端到端质量和架构 gate：

```bash
uv run python scripts/evaluate_single_arch_gate.py \
  --server-url http://127.0.0.1:17860 \
  --modes voice_design,custom_voice,voice_clone \
  --runs 1 \
  --concurrency 1
```

接口字段详见 [API Reference](api_reference_zh.md)，常见错误见 [Troubleshooting](troubleshooting_zh.md)。

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

Windows PowerShell 建议先启用 Python UTF-8 输出，避免 NNCF/rich 进度条在 GBK 控制台里因为 `UnicodeEncodeError` 中断：

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

Windows 源码构建 native runtime 需要 MSVC 和 CMake。确认命令：

```powershell
uv --version
cmake --version
where.exe cl
```

如果 `where.exe cl` 找不到编译器，安装 Visual Studio 2022 Build Tools，并勾选 C++ build tools / Windows SDK。

如果 `third_party/openvino.genai` 尚未初始化，`build-fastest` 会自动初始化 submodule。也可以手动执行：

```bash
git submodule update --init --recursive
```

## 2. 准备官方 Qwen3-TTS 源码

OpenVINO 导出器需要官方 Qwen3-TTS PyTorch 代码中的 `qwen_tts` 包。源码仓库不内置这部分代码；建议克隆到已被 `.gitignore` 忽略的 `.cache/`：

```bash
git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS .cache/Qwen3-TTS
export PYTHONPATH="$(pwd)/.cache/Qwen3-TTS"
```

Windows PowerShell：

```powershell
git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS .cache\Qwen3-TTS
$env:PYTHONPATH = (Resolve-Path .cache\Qwen3-TTS).Path
```

验证：

```bash
uv run python -c "import qwen_tts; print('qwen_tts ok')"
```

如果看到 `SoX could not be found`，但命令最后打印了 `qwen_tts ok`，通常可以继续导出；这是官方包在 import 时探测系统 SoX 可执行文件的提示，不是 OpenVINO IR 导出的硬依赖。

## 3. 准备 PyTorch 模型

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

## 4. 构建 native runtime

`build-fastest` 会自动构建 native runtime。Windows 上也可以先单独验证 MSVC/CMake 链路：

```powershell
uv run python scripts\build_native_codegen.py --backend cmake --config Release
```

成功后应生成：

```text
native/build/qwen3_tts_ov_genai.dll
```

Linux/macOS 可以直接跳过本节，由 `build-fastest` 自动处理。

## 5. 一键构建生产 IR

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

## 6. 启动开发服务

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
uv run python examples/python/http_tts_wav.py --output outputs/example_http.wav
uv run --with websockets python examples/python/websocket_stream_pcm.py --output outputs/example_ws.wav
```

Windows PowerShell 路径写法：

```powershell
uv run python examples\python\http_tts_wav.py --output outputs\example_http.wav --max-new-tokens 24
```

## 7. 验证

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

Windows PowerShell：

```powershell
uv run python -c "import openvino as ov; core=ov.Core(); print(core.available_devices); [print(d, core.get_property(d, 'FULL_DEVICE_NAME')) for d in core.available_devices]"
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

# 前置条件

本页用于在下载 release 包或从源码构建前确认机器环境。运行时本身不包含模型权重或 OpenVINO IR；release server 会在首次启动时按需下载公开 IR。

## 快速检查

源码开发至少需要：

- Python `>=3.12`，与 `pyproject.toml` 保持一致。
- `uv`，用于创建环境、安装 extras 和运行脚本。官方安装说明见 <https://docs.astral.sh/uv/getting-started/installation/>。
- 可访问 GitHub Release 和 Hugging Face，或已提前准备好本地 IR。

普通 release 用户需要：

- Linux x86_64 或 Windows x64。
- 可执行的 release 包：Linux 使用 `.tar.zst`，Windows 使用 `.zip`。
- 如果使用 `--device GPU`，OpenVINO 必须能看到 Intel GPU。

## 安装 uv

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

如果公司环境不允许直接执行网络脚本，可以按 uv 官方文档使用 `pipx install uv`、WinGet、Scoop 或下载独立二进制。

## Python 与 OpenVINO

源码路径通过 `uv sync` 安装 Python 依赖：

```bash
uv sync --extra native --extra server --extra export
```

`openvino`、`openvino-genai` 和 `openvino-tokenizers` 由项目依赖解析安装；通常不需要手动单独安装 OpenVINO。当前锁文件中的 OpenVINO 版本为 `2026.1.0`，但实际安装以 `uv.lock` 和平台 wheel 可用性为准。OpenVINO 官方安装入口见 <https://docs.openvino.ai/install>。

release 包会携带运行所需的 Python/OpenVINO 运行时组件；最终用户只需要安装系统 GPU/NPU 驱动和解压 release 包。

## 设备选择

服务端 `--device` 传给 OpenVINO。常用值：

| 值 | 用途 | 说明 |
| --- | --- | --- |
| `GPU` | 推荐生产路径 | Intel GPU 路径，release 示例默认使用。 |
| `CPU` | smoke/fallback | 可用于确认服务能启动和接口可用，不承诺实时性能。 |
| `AUTO` | OpenVINO 自动选择 | 可用于实验；性能和设备选择由 OpenVINO AUTO 插件决定。 |
| `GPU.0` 等 | 多 GPU 指定 | 当 OpenVINO 报告多个 GPU 时可显式选择。 |

OpenVINO 官方支持的运行设备包括 CPU、GPU、NPU；AUTO 是 OpenVINO 的自动设备选择模式。项目的 Windows GPU+NPU 路径通过 `--device GPU --npu-offload ...` 控制，不建议把主设备直接写成 `NPU`。

检查 OpenVINO 可见设备：

```bash
uv run python - <<'PY'
import openvino as ov
core = ov.Core()
print(core.available_devices)
for name in core.available_devices:
    print(name, core.get_property(name, "FULL_DEVICE_NAME"))
PY
```

release 包里没有 `uv` 时，可用系统 Python 临时检查；也可以启动服务后看 `/health` 的 `device`、`runtime` 和 `available_modes` 字段。

## Intel GPU 要求

本项目面向 OpenVINO 能识别的 Intel GPU，包括 Intel Arc 独显、Intel Core Ultra/Arc iGPU、Xe 系列 iGPU 等。不要把“显卡在系统中存在”等同于“OpenVINO 可用”；最终以 `ov.Core().available_devices` 中出现 `GPU` 为准。

Linux 上 OpenVINO GPU 推理需要 Intel OpenCL/Level Zero 运行时。OpenVINO GPU 配置文档见 <https://docs.openvino.ai/nightly/get-started/install-openvino/configurations/configurations-intel-gpu.html>。Ubuntu 22.04/24.04 常见包名包括：

```bash
sudo apt-get install -y ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero
sudo usermod -a -G render "$LOGNAME"
```

执行 `usermod` 后需要重新登录。独显机器还需要确认内核驱动、`/dev/dri/renderD*`、BIOS/直通设置和容器设备映射。

Windows 上请安装适配机器的 Intel Graphics Driver；Intel Arc/Intel Core Ultra 通用驱动入口见 <https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html>。OEM 笔记本优先确认厂商驱动说明。

## NPU 可选路径

NPU 只用于 Windows/部分机器上的异构 offload 场景：

```powershell
.\qwen3-tts-ov-server.exe --device GPU --npu-offload auto
```

`--npu-offload decoder` 会要求 OpenVINO 能看到 `NPU` 且 streaming decoder 能编译到 NPU；失败时会直接报错。需要自动回退用 `auto`。完整流程见 [Windows GPU+NPU 测试路径](windows_gpu_npu_zh.md)。

## 首次编译缓存

首次启动、首次切换设备或首次使用新 IR 时，OpenVINO 会编译图并写入用户缓存目录。这个阶段可能持续较久，尤其是 GPU/NPU、VoiceClone 或 cold cache 场景。后续命中缓存后会明显变快。

建议：

- 先访问 `http://127.0.0.1:17860/health` 看 warmup 状态。
- 需要离线预热时使用 [OpenVINO 编译缓存](cache_zh.md)。
- 不要把 OpenVINO compile cache、`outputs/`、`models/` 或 `openvino/` 提交到 git。

## 网络与模型下载

release server 默认从 Hugging Face 下载公开 IR。离线部署时先按 [Release 使用说明](release_zh.md) 下载 `openvino_realtime/`，再用：

```bash
./qwen3-tts-ov-server --device GPU --model-root /path/to/openvino_realtime
```

如需代理或缓存配置，参考 Hugging Face Hub 环境变量文档：<https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables>。


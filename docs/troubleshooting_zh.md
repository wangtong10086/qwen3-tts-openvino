# Troubleshooting / FAQ

本页按常见现象排查。先确认 [前置条件](prerequisites_zh.md)，再看具体错误。

## `--device GPU` 找不到 GPU

现象：

- 启动时报 `No device with "GPU"`、`GPU plugin`、`No OpenCL device` 等错误。
- `/health` 不可用，或 OpenVINO 可见设备只有 `CPU`。

检查：

```bash
uv run python - <<'PY'
import openvino as ov
core = ov.Core()
print(core.available_devices)
for name in core.available_devices:
    print(name, core.get_property(name, "FULL_DEVICE_NAME"))
PY
```

Linux 继续检查：

```bash
ls -l /dev/dri || true
groups
```

处理：

- Linux 安装 OpenCL/Level Zero 运行时，常见包名见 [前置条件](prerequisites_zh.md#intel-gpu-要求)。
- 将当前用户加入 `render` 组后重新登录。
- 容器/WSL/虚拟化环境要确认 `/dev/dri/renderD*` 已透传；WSL 不作为 NPU 验证环境。
- Windows 更新 Intel Graphics Driver；OEM 笔记本优先查看厂商驱动说明。
- 只想验证接口时可临时用 `--device CPU`，但不承诺实时性能。

## Hugging Face 模型下载失败

现象：

- 首次启动卡在下载或报网络、TLS、403/404、连接超时。
- `/v1/models/download` 返回下载错误。

检查：

```bash
curl -I https://huggingface.co/
```

处理：

- 在可联网环境预下载 IR，再用 `--model-root` 指向本地 `openvino_realtime`。
- 公司网络下配置 `HTTPS_PROXY`、`HTTP_PROXY` 或 Hugging Face Hub 相关环境变量。
- 使用 release 文档中的命令手动下载：

```bash
uv run --with huggingface_hub huggingface-cli download \
  waston10086/qwen3-tts-openvino-voice-design \
  --include "openvino_realtime/**" \
  --local-dir qwen3-tts-openvino-ir
```

- 如果私有镜像或私有 IR 使用不同 repo/subdir，设置 `--model-repo`、`--model-revision`、`--model-subdir` 或对应环境变量。

## `--npu-offload decoder` 报错

现象：

- Windows 上启动失败，错误包含 `NPU`、`compile`、`streaming decoder`。
- `ov.Core().available_devices` 没有 `NPU`。

处理：

- 严格验证 NPU 时继续使用 `--npu-offload decoder`；这会在 NPU 不可用时失败。
- `decoder/audio/all` 会在 GPU 主设备下自动把默认内存预算收紧到 128 个 KV/online cache blocks 和 50% VRAM，避免常见的 GPU USM allocation 失败。若使用旧版本或需要手动确认，可显式加：

```powershell
.\qwen3-tts-ov-server.exe `
  --device GPU `
  --npu-offload decoder `
  --kv-cache-max-blocks 128 `
  --online-batch-max-cache-blocks 128 `
  --max-vram-ratio 50
```

- 希望自动回退 GPU 时使用：

```powershell
.\qwen3-tts-ov-server.exe --device GPU --npu-offload auto
```

- 确认是在 Windows 原生环境，安装了 NPU/显卡驱动，且 IR 包含固定 shape streaming decoder。
- 按 [Windows GPU+NPU 测试路径](windows_gpu_npu_zh.md) 先跑 probe/smoke。

## 首次启动像是卡住

现象：

- release 包首次运行很久才可用。
- 日志显示 compile、warmup、cache 或 OpenVINO 相关信息。

原因：

- 首次使用某个 IR/设备时 OpenVINO 会编译图并写入用户缓存。
- VoiceClone、NPU、长文本和 cold cache 场景可能更慢。

处理：

- 等待首次 warmup 完成，并访问 `/health` 查看 `warmup`。
- 下次命中 compile cache 后通常会变快。
- 需要离线预热时使用 [OpenVINO 编译缓存](cache_zh.md)。
- 磁盘空间不足时 runtime 可能临时关闭 cache，清理用户缓存目录或指定 `--ov-cache-dir`。

## `uv sync` 或 torch/torchaudio 下载超时

现象：

- 源码构建时依赖下载失败。
- `torch`、`torchaudio`、`openvino` wheel 拉取超时。

处理：

- release 用户不需要源码构建，也不需要安装 PyTorch。
- 源码开发可先只安装服务端最小依赖：

```bash
uv sync --extra server --extra native
```

- 只有重新导出 IR 时才需要 `--extra export`。
- 配置公司 PyPI 镜像或代理后重试；必要时用 `uv sync -v` 看具体卡在哪个包。
- 确认 Python 是 `>=3.12`，否则依赖解析可能失败。

## 导出时报 `ModuleNotFoundError: qwen_tts`

现象：

- `uv run python -m qwen3_tts_ov build-fastest ...` 在 export 阶段失败。
- 错误包含 `No module named 'qwen_tts'`。

原因：

- 本仓库只包含 OpenVINO exporter/runtime，不内置官方 Qwen3-TTS PyTorch 源码。
- exporter 需要 import 官方源码中的 `qwen_tts` 包。

处理：

```powershell
git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS .cache\Qwen3-TTS
$env:PYTHONPATH = (Resolve-Path .cache\Qwen3-TTS).Path
uv sync --extra native --extra server --extra export
uv run python -c "import qwen_tts; print('qwen_tts ok')"
```

Linux/macOS：

```bash
git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS .cache/Qwen3-TTS
export PYTHONPATH="$(pwd)/.cache/Qwen3-TTS"
uv sync --extra native --extra server --extra export
uv run python -c "import qwen_tts; print('qwen_tts ok')"
```

`PYTHONPATH` 只对当前 shell 生效；打开新终端后需要重新设置，或把它写入自己的本地环境脚本。

## 导出时报 `ModuleNotFoundError: librosa` 或 `onnxruntime`

现象：

- 已经设置了 `PYTHONPATH`，但 import `qwen_tts` 时继续缺 `librosa` 或 `onnxruntime`。

处理：

```bash
uv sync --extra native --extra server --extra export
```

当前 `export` extra 已包含官方 `qwen_tts` 导出路径会 import 到的 `librosa` 和 `onnxruntime`。如果本地锁文件或环境来自旧版本，重新同步后再试。

## Windows 压缩时报 `UnicodeEncodeError: 'gbk' codec can't encode character`

现象：

- OpenVINO IR 已导出，但 `scripts/compress_openvino_weights.py` 在 NNCF/rich 进度条退出时报 `UnicodeEncodeError`。

处理：

- 使用当前源码；`build-fastest` 的子进程已默认设置 UTF-8。
- 如果直接运行压缩脚本，先在 PowerShell 设置：

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
uv run python scripts\compress_openvino_weights.py --ir-dir openvino\voice_design --preset fastest
```

## Windows native runtime 编译失败

现象：

- MSVC 报 `C4819`、`C2001: 常量中有换行符`，或 native CLI 编译中文字符串失败。
- `build-fastest` 每次都重建 native，或者找不到 Windows DLL。

处理：

- 使用当前源码；CMake 已给 MSVC target 加 `/utf-8`，`build-fastest` 也会识别 `native/build/qwen3_tts_ov_genai.dll`。
- 单独验证 runtime 构建：

```powershell
uv run python scripts\build_native_codegen.py --backend cmake --config Release
```

- 确认 `where.exe cl` 能找到 Visual Studio 2022 Build Tools 的 `cl.exe`。

## Windows GPU 日志出现 `onednn_verbose ... primitive,error`

现象：

- 服务已启动，`/health` 返回 `ok: true`，TTS 请求也能成功。
- `outputs/server.stdout.log` 或终端里仍出现类似 `onednn_verbose ... primitive,error ... convolution ... jit.cpp` 的日志。

处理：

- 先以服务状态为准：如果 Uvicorn 没退出、`/health` 正常并且 TTS 能生成音频，这通常是 OpenVINO/oneDNN 在 GPU kernel 探测或回退时打印的 verbose 日志，不代表本次运行失败。
- 如果本机显式设置过 `ONEDNN_VERBOSE` 或 `DNNL_VERBOSE`，在 PowerShell 中执行 `Remove-Item Env:ONEDNN_VERBOSE` / `Remove-Item Env:DNNL_VERBOSE`，或把它们设为 `0` 后重启服务。
- 如果服务退出、`/health` 失败或 TTS 请求失败，再按真实错误处理：更新 Intel Graphics Driver，清理 OpenVINO cache 后重启，或先用 `--device CPU` 做 smoke，把驱动问题和模型导出问题分开排查。

## IR 目录或 manifest 解析失败

现象：

- 错误包含 `manifest.json`、`model_type`、`VoiceClone uses Base IR`、`missing graph`。
- Web Demo 显示某些模式未就绪。

期望目录：

```text
openvino_realtime/
  voice_design/manifest.json
  custom_voice/manifest.json
  base/manifest.json
```

处理：

- `--model-root` 应指向包含 `voice_design/`、`custom_voice/`、`base/` 的根目录，而不是某个单独模式目录。
- VoiceClone 使用 `base/manifest.json`，不能用 VoiceDesign IR 代替。
- 访问 `/v1/models` 查看各模式状态。
- 若 manifest 来自旧导出，重新运行 `build-fastest` 或重新下载 release IR。

## VoiceClone 参考音频失败

现象：

- 报 `ref_text is required when x_vector_only_mode=False`。
- 报音频读取失败，或提示安装 `audio-full`。
- 生成音色不稳定。

要求和建议：

- 默认 `x_vector_only=false` 时，必须同时提供 `ref_audio` 和准确的 `ref_text`。
- `ref_audio` 支持本地路径、HTTP(S) URL、base64 或 `data:audio/...` 字符串。
- 运行时会转单声道并按模型需要重采样；推荐输入清晰、单人、低噪声、3 到 10 秒左右的参考音频。
- 常见 wav/flac 可由 `soundfile` 读取；mp3 或更多编码失败时，源码环境安装 `audio-full` extra 后会尝试 `librosa`。
- `x_vector_only=true` 只用于 speaker embedding-only 对照，通常不作为最佳效果路径。

## CustomVoice speaker 不存在

现象：

- CustomVoice 请求使用 `"speaker": "Vivian"` 以外的名字时报错或效果不符合预期。

处理：

```bash
curl http://127.0.0.1:17860/v1/audio/voices
```

用返回的 `voices` 或 `voice_details[].id` 作为 `speaker`。不同 IR 的内置 speaker 列表可能不同，文档不硬编码列表。

## 长文本或显存预算报错

现象：

- 错误包含 `effective_max_continuous_prompt_tokens`、`max_generation_tokens_available`、`USM`、`context limit`。

处理：

- 先调用 `/v1/tts/tokenize` 或看 Web Demo 的 token 预算。
- 缩短文本、降低 `generation.max_new_tokens`，或提高 `--max-vram-ratio`。
- 如果明确接受运行时 OOM 风险，可设置 `--max-continuous-prompt-tokens 0` 关闭 prompt 预算保护。
- 不要依赖固定“最多多少字”的结论；实际值由 IR、设备显存、KV profile 和 runtime planner 决定。

## CPU-only 路径很慢

现象：

- `--device CPU` 能启动，但流式播放追不上实时。

说明：

- CPU 路径主要用于 smoke、接口集成和无 GPU 环境验证。
- 生产体验优先使用 Intel GPU；如果只是验证 Web/API，可降低 `generation.max_new_tokens` 并使用短文本。


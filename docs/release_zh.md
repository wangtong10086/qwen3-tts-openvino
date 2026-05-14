# Release 使用说明

Release 面向最终调用方。调用方不需要 PyTorch、导出脚本或源码开发环境，只需要：

1. GitHub Release 中的 runtime App 包。
2. Hugging Face 上已经编译好的 OpenVINO IR。

当前正式版本：`v0.1.0`

## 下载内容

Runtime App 包在 GitHub Release：

- Linux: `qwen3-tts-ov-server-linux-x64-0.1.0-runtime-minimal.tar.zst`
- Windows: `qwen3-tts-ov-server-windows-x64-0.1.0-runtime-minimal.zip`
- Release 页面：<https://github.com/wangtong10086/qwen3-tts-openvino/releases/tag/v0.1.0>

已编译 OpenVINO IR 在 Hugging Face：

- Model repo: <https://huggingface.co/waston10086/qwen3-tts-openvino-voice-design>
- 使用目录：`openvino_realtime/`

可以从网页下载，也可以用 Hugging Face CLI：

```bash
uv run --with huggingface_hub huggingface-cli download \
  waston10086/qwen3-tts-openvino-voice-design \
  --include "openvino_realtime/**" \
  --local-dir qwen3-tts-openvino-ir
```

下载完成后，模型根目录应为：

```text
qwen3-tts-openvino-ir/openvino_realtime
```

## Linux 启动

```bash
tar --zstd -xf qwen3-tts-ov-server-linux-x64-0.1.0-runtime-minimal.tar.zst

cd qwen3-tts-ov-server-linux-x64-0.1.0-runtime-minimal
./qwen3-tts-ov-server \
  --model-root ../qwen3-tts-openvino-ir/openvino_realtime \
  --device GPU
```

打开：

```text
http://127.0.0.1:17860/
```

## Windows 启动

```powershell
Expand-Archive qwen3-tts-ov-server-windows-x64-0.1.0-runtime-minimal.zip -DestinationPath .

cd qwen3-tts-ov-server-windows-x64-0.1.0-runtime-minimal
.\qwen3-tts-ov-server.exe `
  --model-root ..\qwen3-tts-openvino-ir\openvino_realtime `
  --device GPU
```

打开：

```text
http://127.0.0.1:17860/
```

## 调用接口

Web Demo：

```text
http://127.0.0.1:17860/
```

HTTP NDJSON：

```bash
curl -N http://127.0.0.1:17860/v1/tts/stream \
  -H "content-type: application/json" \
  -d '{"mode":"voice_design","text":"你好，这是 release 包测试。","language":"Chinese","instruct":"A calm young female voice."}'
```

OpenAI-compatible Speech API：

```bash
curl -N http://127.0.0.1:17860/v1/audio/speech \
  -H "content-type: application/json" \
  -d '{"model":"qwen3-tts-openvino","voice":"default","input":"你好，这是兼容接口测试。","language":"Chinese","task_type":"voice_design","instructions":"A calm young female voice.","stream":true,"response_format":"pcm"}' \
  --output speech.pcm
```

更多 CLI、HTTP、WebSocket 和 Python API 说明见 [运行接口](runtime_zh.md)。

## 发布物边界

- GitHub Release 只包含 runtime App 包，不包含模型权重或 OpenVINO IR。
- Hugging Face model repo 存放已编译 OpenVINO IR，当前公开的是 VoiceDesign realtime IR。
- OpenVINO compile cache 会在用户机器首次运行时生成，不随 release 分发。
- 需要私有分发 IR 时，可以使用 `scripts/package_ir.py` 自行打包；当前公开分发推荐直接使用 Hugging Face。

## 系统要求

- Linux x86_64 或 Windows x64。
- Intel GPU 使用时需要目标机器已安装对应 GPU 驱动、OpenCL/Level Zero runtime。
- CPU 可作为 fallback，但实时性能不保证。
- Windows runtime 必须使用 Windows 构建产物；不要使用 Linux 交叉编译出的 DLL。

## 维护者发布流程

正式 runtime 发布由 GitHub Actions 的 `release-runtime` workflow 负责。推送 `v*` tag 会自动构建 Linux/Windows runtime-minimal App 包，完成 smoke 后上传到对应 GitHub Release：

```bash
git tag v0.1.0
git push origin v0.1.0
```

Actions 页面也可以手动运行 `release-runtime`，填写 `version`，用于重发指定版本。分平台的 `release-linux` 和 `release-windows` 只作为手动诊断入口保留。

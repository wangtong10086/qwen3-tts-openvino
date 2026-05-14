# Release 使用说明

Release 面向最终调用方，只提供本地 sidecar 服务。调用方通过浏览器、HTTP、WebSocket 或 OpenAI-compatible Speech API 使用，不需要 PyTorch、导出脚本或源码开发环境。

## 产物

- Runtime-minimal App 包：`qwen3-tts-ov-server-linux-x64-<version>-runtime-minimal.tar.zst` 或 `qwen3-tts-ov-server-windows-x64-<version>-runtime-minimal.zip`
- Full App 包：`qwen3-tts-ov-server-linux-x64-<version>.tar.zst` 或 `qwen3-tts-ov-server-windows-x64-<version>.zip`
- Runtime-minimal IR 包：`qwen3-tts-openvino-ir-voice_design-<version>-runtime-minimal.tar.zst`
- Full IR 包：`qwen3-tts-openvino-ir-voice_design-<version>.tar.zst`

App 包包含可执行入口、Python runtime、OpenVINO runtime、OpenVINO GenAI/tokenizers 依赖和 native 加速库。IR 包单独包含 `openvino/voice_design/manifest.json` 及其引用的 `.xml/.bin`。

默认推荐 `runtime-minimal`。它保留当前验证的 native paged-KV 长文本完整自回归路径，支持 VoiceDesign 和带 `base` IR 的 VoiceClone/ref audio，同时移除开发 fallback、实验图和 `librosa/scipy/numba/llvmlite/sklearn` 依赖。`runtime-minimal` 的 ref audio 支持 `soundfile/libsndfile` 可读格式，例如 WAV/FLAC/OGG；需要更宽格式兼容时使用 `full`。

也可以从公开 Hugging Face model repo 下载已验证的 realtime IR：

```bash
uv run --with huggingface_hub python scripts/download_hf_ir.py \
  --repo-id waston10086/qwen3-tts-openvino-voice-design \
  --local-dir build/hf-ir \
  --allow-pattern "openvino_realtime/**"
```

下载后启动时把 `--model-root` 指向 `build/hf-ir/openvino_realtime`。

## Linux

```bash
tar --zstd -xf qwen3-tts-ov-server-linux-x64-<version>-runtime-minimal.tar.zst
tar --zstd -xf qwen3-tts-openvino-ir-voice_design-<version>-runtime-minimal.tar.zst

cd qwen3-tts-ov-server-linux-x64-<version>-runtime-minimal
./qwen3-tts-ov-server \
  --model-root ../qwen3-tts-openvino-ir-voice_design-<version>-runtime-minimal/openvino \
  --device GPU
```

如果把 IR 包中的 `openvino/` 目录复制到 App 包目录旁边，可以省略 `--model-root`。

## Windows

```powershell
Expand-Archive qwen3-tts-ov-server-windows-x64-<version>-runtime-minimal.zip
Expand-Archive qwen3-tts-openvino-ir-voice_design-<version>-runtime-minimal.zip

cd qwen3-tts-ov-server-windows-x64-<version>-runtime-minimal
.\qwen3-tts-ov-server.exe `
  --model-root ..\qwen3-tts-openvino-ir-voice_design-<version>-runtime-minimal\openvino `
  --device GPU
```

Windows 包必须在 Windows runner 上构建。不要使用 Linux 交叉编译出的 DLL。

## 调用

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

## 系统要求

- Linux x86_64 或 Windows x64。
- Intel GPU 使用时需要目标机器已安装对应 GPU 驱动、OpenCL/Level Zero runtime。
- CPU 可作为 fallback，但实时性能不保证。
- 首次启动会在用户缓存目录生成 OpenVINO compile cache；cache 不随 release 分发。

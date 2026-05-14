# Release 使用说明

Release 面向最终调用方。调用方不需要 PyTorch、导出脚本或源码开发环境，只需要下载 GitHub Release 中的 runtime App 包。

首次启动时，如果本地没有 OpenVINO IR，release server 会自动从 Hugging Face 下载默认公开 IR 到用户缓存目录，然后继续启动。

当前正式版本：`v0.1.2`

## 下载内容

Runtime App 包在 GitHub Release：

- Linux: `qwen3-tts-ov-server-linux-x64-0.1.2-runtime-minimal.tar.zst`
- Windows: `qwen3-tts-ov-server-windows-x64-0.1.2-runtime-minimal.zip`
- Release 页面：<https://github.com/wangtong10086/qwen3-tts-openvino/releases/tag/v0.1.2>

默认自动下载的已编译 OpenVINO IR 在 Hugging Face：

- Model repo: <https://huggingface.co/waston10086/qwen3-tts-openvino-voice-design>
- 使用目录：`openvino_realtime/`

离线部署或需要预下载时，可以从网页下载，也可以用 Hugging Face CLI：

```bash
uv run --with huggingface_hub huggingface-cli download \
  waston10086/qwen3-tts-openvino-voice-design \
  --include "openvino_realtime/**" \
  --local-dir qwen3-tts-openvino-ir
```

手动下载完成后，模型根目录应为：

```text
qwen3-tts-openvino-ir/openvino_realtime
```

## Linux 启动

```bash
tar --zstd -xf qwen3-tts-ov-server-linux-x64-0.1.2-runtime-minimal.tar.zst

cd qwen3-tts-ov-server-linux-x64-0.1.2-runtime-minimal
./qwen3-tts-ov-server \
  --device GPU
```

打开：

```text
http://127.0.0.1:17860/
```

## Windows 启动

```powershell
Expand-Archive qwen3-tts-ov-server-windows-x64-0.1.2-runtime-minimal.zip -DestinationPath .

cd qwen3-tts-ov-server-windows-x64-0.1.2-runtime-minimal
.\qwen3-tts-ov-server.exe `
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
- Hugging Face model repo 存放已编译 OpenVINO IR，当前公开的是 VoiceDesign realtime IR。release server 默认在缺少本地 IR 时自动下载它。
- OpenVINO compile cache 会在用户机器首次运行时生成，不随 release 分发。
- 需要私有分发 IR 时，可以使用 `scripts/package_ir.py` 自行打包；当前公开分发推荐直接使用 Hugging Face。

## 自动下载控制

默认启动即可自动下载：

```bash
./qwen3-tts-ov-server --device GPU
```

常用控制参数：

- `--no-auto-download-model`: 禁止启动时联网下载。
- `--model-root <path>`: 使用本地已下载或私有分发的 IR。
- `--model-cache-dir <path>`: 指定自动下载缓存目录。
- `--model-repo <repo>`、`--model-revision <rev>`、`--model-subdir <dir>`: 指定下载来源。
- `--max-continuous-prompt-tokens auto|0|N`: 长文本 full-AR prompt token 预算。默认 `auto`，GPU 路径为 `2048`，CPU-only 路径为 `4096`；`0` 表示关闭该预算保护。
- `--kv-cache-profile auto|fp16|bf16|u8|u8-input|u8-all`: paged-KV cache 显存档位。默认 `auto` 跟随最快路径，即 U8 KV cache；需要保守对照时使用 `fp16`。
- `--max-vram-ratio auto|N`: Web Demo 和服务端 prompt 预算使用的最大显存占比；例如 `70` 表示按 70% 可用显存计算长文本 token 上限。

缓存目录也可以用环境变量 `QWEN3_TTS_OV_MODEL_CACHE_DIR` 指定。

## 长文本预算

长文本默认不切分输入文本，而是使用同一条 full-AR 自回归链路。为了避免过长 prompt 在部分 GPU/驱动上触发 OpenVINO USM 分配失败，server 会在推理前做 prompt token 预算检查。

默认 `--max-continuous-prompt-tokens auto` 已覆盖常见长文本：

- GPU 或默认 release 路径：`2048`
- CPU-only：`4096`

如果请求仍然超出预算，可以按机器显存/内存情况提高或关闭限制：

```bash
./qwen3-tts-ov-server --device GPU --max-continuous-prompt-tokens 4096

# 或关闭 prompt 预算保护，仅保留 OpenVINO/USM 运行时错误与重试
./qwen3-tts-ov-server --device GPU --max-continuous-prompt-tokens 0
```

服务端 `/health` 和流式 metadata 会返回 `max_continuous_prompt_tokens_config`、`effective_max_continuous_prompt_tokens` 和 `long_text_budget_policy`，用于确认实际生效值。

## KV Cache 显存档位

默认 release 路径使用 U8 paged-KV cache，降低长文本和长输出时的显存压力。需要显式指定时可以写：

```bash
./qwen3-tts-ov-server --device GPU --kv-cache-profile u8 --max-vram-ratio 70
```

可选档位：

- `fp16`: KV cache 为 FP16，保守质量对照路径。
- `bf16`: KV cache 为 BF16，用于对比硬件行为。
- `u8`: KV cache 存储为 U8，cache input 仍保持 FP32，默认生产路径。
- `u8-input`: KV cache 保持 FP16，但 cache input 改为 U8，主要用于实验。
- `u8-all`: KV cache 和 cache input 都使用 U8，显存更低，但需要额外做音质验证。

`/health` 的 `warmup`、`memory` 和 runtime metadata 会返回 `kv_cache_profile`、`native_paged_kv_precision`、`native_paged_kv_cache_input_precision`、`kv_cache_relative_to_fp16`。其中 `kv_cache_relative_to_fp16=0.5` 表示 KV cache 元素理论占用约为 FP16 的一半。这里压缩的是 paged-KV cache 存储，不等价于全部 attention 算子都以 INT8 计算。

## 系统要求

- Linux x86_64 或 Windows x64。
- Intel GPU 使用时需要目标机器已安装对应 GPU 驱动、OpenCL/Level Zero runtime。
- CPU 可作为 fallback，但实时性能不保证。
- Windows runtime 必须使用 Windows 构建产物；不要使用 Linux 交叉编译出的 DLL。

## 维护者发布流程

正式 runtime 发布由 GitHub Actions 的 `release-runtime` workflow 负责。推送 `v*` tag 会自动构建 Linux/Windows runtime-minimal App 包，完成 smoke 后上传到对应 GitHub Release：

```bash
git tag v0.1.2
git push origin v0.1.2
```

Actions 页面也可以手动运行 `release-runtime`，填写 `version`，用于重发指定版本。分平台的 `release-linux` 和 `release-windows` 只作为手动诊断入口保留。

# Qwen3-TTS OpenVINO

OpenVINO-only 的 Qwen3-TTS 12Hz 推理、导出、流式 sidecar 和 native 加速仓库。

[English README](README.md) | [文档索引](docs/README.zh-CN.md) | [示例请求](examples/README.zh-CN.md)

本仓库是源码仓库，不提交模型权重、OpenVINO IR、生成音频、OpenVINO 编译缓存或 native 编译产物。

## 我应该怎么用

| 目标 | 推荐路径 |
| --- | --- |
| 直接在 Linux/Windows 上运行 TTS 服务 | 下载 [GitHub Release runtime](https://github.com/wangtong10086/qwen3-tts-openvino/releases/tag/v0.1.2)，首次启动会自动下载公开的 [Hugging Face OpenVINO IR](https://huggingface.co/waston10086/qwen3-tts-openvino-voice-design) |
| 从 PyTorch 模型重新导出、压缩、调试 | 使用 `uv run python -m qwen3_tts_ov build-fastest ...` |
| 维护发布包 | 推送 `v*` tag，自动运行 `release-runtime` 构建 Linux/Windows 包并上传 GitHub Releases |

## 直接使用预编译包

1. 从 GitHub Release 下载 runtime 包：

```text
qwen3-tts-ov-server-linux-x64-0.1.2-runtime-minimal.tar.zst
qwen3-tts-ov-server-windows-x64-0.1.2-runtime-minimal.zip
```

2. 启动 sidecar。若本地没有 OpenVINO IR，release server 会自动下载默认公开 IR 到用户缓存目录，然后继续启动。

Linux：

```bash
tar --zstd -xf qwen3-tts-ov-server-linux-x64-0.1.2-runtime-minimal.tar.zst
cd qwen3-tts-ov-server-linux-x64-0.1.2-runtime-minimal
./qwen3-tts-ov-server \
  --device GPU
```

Windows：

```powershell
Expand-Archive qwen3-tts-ov-server-windows-x64-0.1.2-runtime-minimal.zip -DestinationPath .
cd qwen3-tts-ov-server-windows-x64-0.1.2-runtime-minimal
.\qwen3-tts-ov-server.exe `
  --device GPU
```

3. 打开浏览器：

```text
http://127.0.0.1:17860/
```

完整部署说明见 [docs/release_zh.md](docs/release_zh.md)。

离线部署或需要预下载时，可以手动下载 IR，并在启动时传入 `--model-root qwen3-tts-openvino-ir/openvino_realtime`：

```bash
uv run --with huggingface_hub huggingface-cli download \
  waston10086/qwen3-tts-openvino-voice-design \
  --include "openvino_realtime/**" \
  --local-dir qwen3-tts-openvino-ir
```

## 开发者 Quick Start

需要从 PyTorch 模型重新导出或调试图时，使用源码开发路径。

```bash
uv sync --extra native --extra server --extra export

uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU

uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest
```

完整源码构建说明见 [docs/quick_start_zh.md](docs/quick_start_zh.md)。

## 当前生产路径

- `fastest` profile 依赖 native C++ pipeline。
- codec 生成使用 OpenVINO paged-KV attention。
- talker seed 图使用 `int8_sym_paged_talker_split`。
- cached subcode 图保持 FP16，优先保证音质稳定。
- 流式输出 mono 24 kHz `pcm_s16le`。
- 长文本使用完整上下文全自回归生成；runtime 只把音频切块给播放器，不默认切分输入文本。
- 长文本 prompt 预算默认 `auto`：GPU 路径为 `2048` tokens，CPU-only 为 `4096` tokens；超长输入可通过 `--max-continuous-prompt-tokens` 调整。

## 常用入口

- 最终用户启动：`qwen3-tts-ov-server --device GPU`
- 一键构建：`uv run python -m qwen3_tts_ov build-fastest --model ...`
- 开发服务：`uv run python -m qwen3_tts_ov serve --model-root openvino --device GPU`
- 流式 CLI：`uv run python -m qwen3_tts_ov stream voice-design ...`
- 批处理：`uv run python -m qwen3_tts_ov batch --batch-jsonl examples/requests.example.jsonl`
- OpenAI-compatible API：`POST /v1/audio/speech`

运行方式、Python API 和 HTTP/WebSocket 协议见 [docs/runtime_zh.md](docs/runtime_zh.md)。

## 文档

- [Release 使用说明](docs/release_zh.md): 预编译 runtime + Hugging Face IR 的最终用户部署方式。
- [Quick Start](docs/quick_start_zh.md): 从源码仓库下载模型、构建最快 IR、启动 Web Demo。
- [开发说明](docs/development_zh.md): 源码开发、导出、压缩、release workflow 和诊断入口。
- [运行接口](docs/runtime_zh.md): CLI、Python API、sidecar、OpenAI-compatible API。
- [导出与压缩](docs/export_zh.md): 手动导出 IR 和生成 fastest variant。
- [流式与长文本](docs/streaming_zh.md): 流式协议、长文本 full-AR、质量门禁。
- [OpenVINO 编译缓存](docs/cache_zh.md): cache warmup 和缓存目录。
- [大文件与产物策略](docs/artifacts_zh.md): 模型、IR、outputs、native build 的处理规则。
- [安全说明](docs/security_zh.md): token、`.env`、凭据和提交检查。

## 仓库结构

```text
qwen3_tts_ov/    runtime、exporter、CLI、sidecar 和 web demo
native/          native C++ OpenVINO pipeline 源码
scripts/         构建、压缩、benchmark、质量评测脚本
devtools/        实验 benchmark、profiling 和诊断入口
docs/            中文文档
examples/        JSON/JSONL/长文本示例
tests/           单元测试
```

默认不进入 git 的本地产物：

```text
models/
openvino/
openvino_full/
outputs/
.venv/
.uv-cache/
native/build/
dist/release/
```

## FAQ

- **仓库为什么没有模型和 IR？**
  模型和 OpenVINO IR 文件很大，不适合提交到源码仓库。已编译好的 VoiceDesign OpenVINO IR 发布在 Hugging Face：`waston10086/qwen3-tts-openvino-voice-design`。

- **GitHub Release 里有什么？**
  Release 只包含 Linux/Windows runtime App 包，不包含模型权重或 OpenVINO IR。

- **为什么 native 是必需项？**  
  当前最快路径依赖 C++ pipeline 将 codec 自回归循环移出 Python。缺少 native 库时生产路径会直接报错。

- **如何先看一键构建会做什么？**
  使用 `uv run python -m qwen3_tts_ov build-fastest --model ... --dry-run`。

## License

Apache-2.0. See [LICENSE](LICENSE).

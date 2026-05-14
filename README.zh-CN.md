# Qwen3-TTS OpenVINO

OpenVINO-only 的 Qwen3-TTS 12Hz 推理、导出、流式 sidecar 和 native 加速仓库。

[English README](README.md) | [文档索引](docs/README.zh-CN.md) | [示例请求](examples/README.zh-CN.md)

本仓库是源码仓库，不提交模型权重、OpenVINO IR、生成音频、虚拟环境、OpenVINO 编译缓存或 native 编译产物。

## 特性

- 支持 VoiceDesign、CustomVoice、VoiceClone。
- `build-fastest` 一键构建当前验证过的最快路径，默认使用低内存 production 图集合。
- native C++ codec 生成链路，使用 OpenVINO paged-KV attention。
- 本地 sidecar 提供 WebSocket、HTTP NDJSON 和 OpenAI-compatible Speech API。
- 流式输出 mono 24 kHz `pcm_s16le`。
- 长文本默认完整上下文全自回归，不自动分段。

## Quick Start

1. 安装依赖：

```bash
uv sync --extra native --extra server --extra export
```

2. 下载 VoiceDesign 模型到本地 `models/`：

```bash
uv run modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --local_dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

3. 一键构建最快路径：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU
```

从旧导出产物完全重来时可加 `--clean --clean-native`。

4. 启动 sidecar：

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest \
  --host 127.0.0.1 \
  --port 17860
```

5. 打开浏览器：

```text
http://127.0.0.1:17860/
```

完整安装说明见 [docs/quick_start_zh.md](docs/quick_start_zh.md)。

## 当前生产路径

`fastest` profile 表示：

- 必须构建 native C++ pipeline。
- 必须导出 paged-KV seed graph。
- 必须生成 `int8_sym_paged_talker_split` 权重压缩 variant。
- 默认使用 `native paged-KV + GQA + split-subcode + block16`。
- 默认 `chunk_strategy=smooth`，降低浏览器播放块间卡顿风险。

长文本仍使用同一个 prompt 从头到尾全自回归生成。只要 IR manifest 包含最快路径需要的 paged seed 和 cached subcode 图，sidecar 会自动使用 sampled paged-KV full-AR 加速；缺图时才回退到 FP16 reference。自动分段只作为显式诊断 fallback。

## 常用入口

- 使用侧启动：`qwen3-tts-ov-server --model-root openvino --device GPU`
- 一键构建：`uv run python -m qwen3_tts_ov build-fastest --model ...`
- 启动服务：`uv run python -m qwen3_tts_ov serve --model-root openvino --device GPU`
- 流式 CLI：`uv run python -m qwen3_tts_ov stream voice-design ...`
- 批处理：`uv run python -m qwen3_tts_ov batch --batch-jsonl examples/requests.example.jsonl`
- OpenAI-compatible API：`POST /v1/audio/speech`

运行方式、Python API 和 HTTP/WebSocket 协议见 [docs/runtime_zh.md](docs/runtime_zh.md)。

正式 runtime 发布由 GitHub Actions 的 `release-runtime` 执行。推送 `v*` tag 会自动构建 Linux/Windows runtime-minimal 包并上传到 GitHub Releases。

## 文档

- [Quick Start](docs/quick_start_zh.md): 从空仓库到 Web Demo。
- [Release 使用说明](docs/release_zh.md): 预编译 Linux/Windows 包的使用方式。
- [开发说明](docs/development_zh.md): 源码开发、导出、压缩和 release 打包。
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
devtools/        实验 benchmark、profiling 和 legacy 入口
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
```

## FAQ

- **为什么仓库没有模型和 IR？**  
  模型和导出的 IR 通常很大，必须本地下载和导出。

- **为什么 native 是必需项？**  
  当前最快路径依赖 C++ pipeline 将 codec 自回归循环移出 Python。缺少 native 库时生产路径会直接报错。

- **如何先看一键构建会做什么？**
  使用 `uv run python -m qwen3_tts_ov build-fastest --model ... --dry-run`。

## License

Apache-2.0. See [LICENSE](LICENSE).

# Qwen3-TTS OpenVINO

OpenVINO-only 的 Qwen3-TTS 12Hz 推理、导出、流式 sidecar 和 native
vLLM-like 加速仓库。

[English README](README.md) | [文档索引](docs/README.zh-CN.md) | [示例请求](examples/README.zh-CN.md)

本仓库是源码仓库，不提交模型权重、OpenVINO IR、生成音频、OpenVINO 编译缓存或 native 编译产物。

## Quick Start

普通用户优先使用 GitHub Release 里的预编译 runtime。首次启动时，如果本地没有模型 IR，服务会自动从 Hugging Face 下载公开的 OpenVINO IR。

Linux：

```bash
tar --zstd -xf qwen3-tts-ov-server-linux-x64-0.1.4-runtime-minimal.tar.zst
cd qwen3-tts-ov-server-linux-x64-0.1.4-runtime-minimal
./qwen3-tts-ov-server --device GPU
```

Windows：

```powershell
Expand-Archive qwen3-tts-ov-server-windows-x64-0.1.4-runtime-minimal.zip -DestinationPath .
cd qwen3-tts-ov-server-windows-x64-0.1.4-runtime-minimal
.\qwen3-tts-ov-server.exe --device GPU
```

打开：

```text
http://127.0.0.1:17860/
```

Web Demo 支持 VoiceDesign、CustomVoice、VoiceClone、长文本 full-AR、参考音频上传、模型一键下载、请求复制、自定义请求 JSON 和同时多请求 smoke。

也可以直接使用 examples 里的客户端脚本调用本地 sidecar：

```bash
python examples/python/http_tts_wav.py --output outputs/example_http.wav
uv run --with websockets python examples/python/websocket_stream_pcm.py --output outputs/example_ws.wav
```

Windows Intel GPU+NPU 机器可尝试：

```powershell
.\qwen3-tts-ov-server.exe --device GPU --npu-offload decoder
```

## 当前生产架构

仓库已收敛为单一生产推理架构：

- 默认 profile：`fastest`。
- codec 自回归生成在 native C++ pipeline 中执行。
- attention 使用 OpenVINO paged-KV，KV cache 默认 U8。
- 服务端使用 vLLM-like online batching 调度请求。
- 长文本使用完整上下文从头到尾自回归生成，不切分输入文本。
- 默认 `runtime_residency=lazy`，按模式懒加载，避免三套模型同时占用显存。

VoiceDesign、CustomVoice、VoiceClone 都通过同一 sidecar 暴露。VoiceClone 使用 Base IR，默认走 `ref_audio + ref_text` 的 ICL 克隆路径；`x_vector_only` 只作为显式对照选项。

## 源码构建

只有需要从 PyTorch 模型重新导出 IR 或调试 backend 时，才使用源码构建路径。

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

CustomVoice 和 Base/VoiceClone 分别使用：

```text
models/Qwen3-TTS-12Hz-1.7B-CustomVoice -> openvino/custom_voice
models/Qwen3-TTS-12Hz-1.7B-Base        -> openvino/base
```

## 常用入口

- 最终用户服务：`qwen3-tts-ov-server --device GPU`
- 开发服务：`uv run python -m qwen3_tts_ov serve --model-root openvino --device GPU`
- 一键构建：`uv run python -m qwen3_tts_ov build-fastest --model ...`
- 质量门禁：`uv run python scripts/evaluate_single_arch_gate.py ...`
- 性能矩阵：`uv run python scripts/benchmark_prompt_batch_matrix.py ...`

## 文档

- [英文文档索引](docs/README.md)
- [Release 使用说明](docs/release_zh.md)
- [源码 Quick Start](docs/quick_start_zh.md)
- [运行接口](docs/runtime_zh.md)
- [导出与构建](docs/export_zh.md)
- [流式与长文本](docs/streaming_zh.md)
- [大文件与产物策略](docs/artifacts_zh.md)
- [安全说明](docs/security_zh.md)
- [示例请求与 Python 客户端](examples/README.zh-CN.md)

## 仓库结构

```text
qwen3_tts_ov/    runtime、exporter、CLI、sidecar、web demo
native/          native OpenVINO C++ codec generation backend
scripts/         生产构建、release、benchmark、质量门禁脚本
docs/            中文文档
examples/        JSON/JSONL/文本示例
tests/           单元测试
```

以下本地产物默认不进入 git：

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

## License

Apache-2.0. See [LICENSE](LICENSE).

# Quick Start

本页描述从源码仓库到浏览器 Web Demo 的最短路径。默认目标是 VoiceDesign + Intel GPU + `fastest` profile。

## 1. 准备环境

```bash
uv sync --extra native --extra server --extra export
uv run python -m qwen3_tts_ov --help
```

如果是首次 clone，并且 `third_party/openvino.genai` 还没有初始化，`build-fastest` 会自动执行 submodule 初始化。也可以手动执行：

```bash
git submodule update --init --recursive
```

## 2. 下载模型

VoiceDesign 示例：

```bash
uv run modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --local_dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

CustomVoice 和 Base/VoiceClone 需要分别下载对应模型目录：

```text
models/Qwen3-TTS-12Hz-1.7B-CustomVoice
models/Qwen3-TTS-12Hz-1.7B-Base
```

模型目录不进入 git。

## 3. 一键构建 fastest

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU
```

该命令会完成：

1. 初始化 submodule。
2. 构建 native C++ pipeline。
3. 导出低内存 production 图集合。
4. 生成 `int8_sym_paged_talker_split` variant。
5. 执行 OpenVINO cache warmup。

默认 `--graph-set production` 只导出当前 `fastest` 运行必需的图，避免 fixed-bucket/unroll 诊断图带来的额外内存压力。需要从旧产物完全重来时：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU \
  --clean \
  --clean-native
```

只有需要 legacy benchmark 或诊断 fixed-bucket/unroll graph 时，才使用 `--graph-set compat`。

先预览将执行的命令：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU \
  --dry-run
```

如果已经有本地旧 IR，例如 `openvino_full/`，可直接验证它是否满足 fastest：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --out-dir openvino_full \
  --skip-submodule \
  --dry-run
```

## 4. 启动服务

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest \
  --preload-modes voice_design \
  --preload-buckets warmup \
  --host 127.0.0.1 \
  --port 17860
```

打开：

```text
http://127.0.0.1:17860/
```

## 5. 快速命令检查

```bash
uv run python -m qwen3_tts_ov build-fastest --help
uv run python -m qwen3_tts_ov serve --help
uv run python -m qwen3_tts_ov stream voice-design --help
```

## 常见问题

- 如果 `build-fastest` 报模型不存在，先确认 `--model` 指向本地模型目录。
- 如果 fastest 报缺少 graph，不要切换到旧 profile 绕过；重新导出或重新压缩 IR。
- 如果首次请求很慢，先执行 `build-fastest` 或 [cache warmup](cache_zh.md)。

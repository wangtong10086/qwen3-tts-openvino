# Quick Start

本页面面向源码开发者，目标是从 PyTorch 模型导出当前推荐的 `fastest` OpenVINO IR，并启动浏览器 Web Demo。

如果你只是想直接运行服务，不需要重新导出模型，请先看 [Release 使用说明](release_zh.md)：runtime 在 GitHub Release，首次启动可自动下载已编译 IR。

## 1. 准备开发环境

```bash
uv sync --extra native --extra server --extra export
uv run python -m qwen3_tts_ov --help
```

首次 clone 后如果 `third_party/openvino.genai` 尚未初始化，`build-fastest` 会自动初始化 submodule。也可以手动执行：

```bash
git submodule update --init --recursive
```

## 2. 准备 PyTorch 模型

VoiceDesign 示例：

```bash
uv run modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --local_dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

CustomVoice 和 Base/VoiceClone 需要分别准备对应模型目录：

```text
models/Qwen3-TTS-12Hz-1.7B-CustomVoice
models/Qwen3-TTS-12Hz-1.7B-Base
```

`models/` 不进入 git。

## 3. 一键构建 fastest IR

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

从旧产物完全重来时：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU \
  --clean \
  --clean-native
```

只想预览将执行的命令：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU \
  --dry-run
```

默认 `--graph-set production` 只导出生产运行必需的图，降低导出内存压力。只有需要 legacy benchmark 或 fixed-bucket/unroll 诊断图时，才使用：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU \
  --graph-set compat
```

## 4. 启动开发服务

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

## 5. 常用检查

```bash
uv run python -m qwen3_tts_ov build-fastest --help
uv run python -m qwen3_tts_ov serve --help
uv run python -m qwen3_tts_ov stream voice-design --help
```

如果已经有本地旧 IR，例如 `openvino_full/`，可先用 dry-run 检查是否满足当前 fastest 路径：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --out-dir openvino_full \
  --skip-submodule \
  --dry-run
```

## 常见问题

- `build-fastest` 报模型不存在：确认 `--model` 指向本地 PyTorch 模型目录。
- `fastest` 报缺少 graph：重新导出或重新压缩 IR，不要切到旧 profile 绕过。
- 首次请求很慢：先执行 `build-fastest` 或 [cache warmup](cache_zh.md)。
- 只想部署不想导出：使用 [Release 使用说明](release_zh.md) 中的预编译包；缺少本地 IR 时会自动下载公开 Hugging Face IR。

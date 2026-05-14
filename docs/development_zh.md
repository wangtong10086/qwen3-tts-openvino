# 开发说明

开发模式用于导出、压缩、调试和性能分析。开发者需要源码仓库、PyTorch 模型目录和 export/native 依赖；最终用户不需要这些内容。

## 环境

```bash
uv sync --extra native --extra server --extra export --extra dev
uv run python -m qwen3_tts_ov --help
```

## 从 PyTorch 模型构建 IR

推荐低内存 production 构建：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU \
  --clean \
  --clean-native
```

需要 legacy fixed-bucket/unroll 诊断图时才使用：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU \
  --graph-set compat
```

## 本地开发运行

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest
```

## 构建 release 产物

正式发布使用 GitHub Actions 的 `release-runtime`：

```bash
git tag v0.1.0
git push origin v0.1.0
```

该 workflow 会自动在 Linux/Windows runner 上构建 runtime-minimal App 包，完成 smoke 后上传到 GitHub Releases。Actions 页面也可以手动运行 `release-runtime`，用于重发指定版本。

Linux 当前机器：

```bash
uv sync --extra native --extra server --extra release
uv run python scripts/package_ir.py \
  --ir-dir openvino/voice_design \
  --model-type voice_design \
  --version 0.1.0 \
  --profile runtime-minimal

uv run python scripts/package_release.py \
  --target linux-x64 \
  --version 0.1.0 \
  --profile runtime-minimal
```

Windows release 必须在 Windows runner 上执行：

```powershell
uv sync --extra native --extra server --extra release
uv run python scripts/build_native_codegen.py --backend cmake
uv run python scripts/package_release.py --target windows-x64 --version 0.1.0 --profile runtime-minimal
```

`runtime-minimal` 是最终用户推荐包，只保留当前最快稳定的 native paged-KV 长文本完整自回归路径。需要调试 fallback、实验图或更宽音频格式兼容时使用 `--profile full`，并按需安装 `audio-full` extra。

## 边界

- 开发模式可以使用 `export`、`build-fastest`、`cache-warmup`、benchmark 和质量评测脚本。
- 使用模式只运行 `qwen3-tts-ov-server`，通过 HTTP/WebSocket 调用。
- 不提交 `models/`、`openvino/`、`outputs/`、OpenVINO cache、native build 或 release archive。

# 开发说明

开发模式用于导出、压缩、调试、性能分析和维护 release。最终用户不需要源码开发环境；最终用户部署请看 [Release 使用说明](release_zh.md)。

## 环境

常规开发环境：

```bash
uv sync --extra native --extra server --extra export --extra dev
uv run python -m qwen3_tts_ov --help
```

只做 runtime 打包时：

```bash
uv sync --extra native --extra server --extra release
```

## 从 PyTorch 模型构建 IR

推荐使用低内存 production 构建：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU \
  --clean \
  --clean-native
```

需要 legacy fixed-bucket/unroll 诊断图时才使用 compat：

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU \
  --graph-set compat
```

手动导出、压缩和 cache 细节见 [导出与压缩](export_zh.md) 和 [OpenVINO 编译缓存](cache_zh.md)。

## 本地开发运行

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest
```

默认生产路径使用 native paged-KV full-AR；缺少必需图时应修复导出产物，不建议在性能或质量验收时切到旧 fallback。

## Release 维护

正式 runtime 发布使用 GitHub Actions 的 `release-runtime`：

```bash
git tag v0.1.2
git push origin v0.1.2
```

该 workflow 会自动：

1. 在 Linux runner 构建 `linux-x64` runtime-minimal 包。
2. 在 Windows runner 构建 `windows-x64` runtime-minimal 包。
3. 分别执行 smoke。
4. 将两个 runtime 包上传到 GitHub Release。

GitHub Release 只分发 runtime App。已编译 OpenVINO IR 当前发布在 Hugging Face：`waston10086/qwen3-tts-openvino-voice-design`。

## 本地打包

Linux 当前机器：

```bash
uv run python scripts/package_release.py \
  --target linux-x64 \
  --version 0.1.2 \
  --profile runtime-minimal
```

Windows release 必须在 Windows runner 上构建：

```powershell
uv run python scripts/build_native_codegen.py --backend cmake
uv run python scripts/package_release.py --target windows-x64 --version 0.1.2 --profile runtime-minimal
```

`runtime-minimal` 是最终用户推荐包，只保留当前最快稳定的 native paged-KV 长文本完整自回归路径。需要调试 fallback、实验图或更宽音频格式兼容时使用 `--profile full`，并按需安装 `audio-full` extra。

需要私有分发 IR 时，可使用：

```bash
uv run python scripts/package_ir.py \
  --ir-dir openvino/voice_design \
  --model-type voice_design \
  --version 0.1.2 \
  --profile runtime-minimal
```

当前公开分发推荐 Hugging Face，而不是把 IR 放入源码仓库。

## 边界

- 开发模式可以使用 `export`、`build-fastest`、`cache-warmup`、benchmark 和质量评测脚本。
- 使用模式只运行 `qwen3-tts-ov-server`，通过 HTTP/WebSocket/OpenAI-compatible API 调用。
- 不提交 `models/`、`openvino/`、`outputs/`、OpenVINO cache、native build 或 release archive。

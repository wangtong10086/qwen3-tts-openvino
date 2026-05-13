# 辅助脚本

正式 CLI 入口是：

```bash
uv run python -m qwen3_tts_ov --help
```

如果已执行 `uv pip install -e .`，也可以使用：

```bash
uv run qwen3-tts-ov --help
```

## 生产流程脚本

- `build_native_codegen.py`: 构建必需的 native C++ pipeline 共享库和 standalone smoke CLI。
- `compress_openvino_weights.py`: 对已导出的 OpenVINO IR 做权重压缩，并更新 manifest variants。
- `package_release.py`: 打包最终用户侧 Linux/Windows sidecar app 包。
- `package_ir.py`: 打包独立 OpenVINO IR 模型包。
- `benchmark_streaming_realtime.py`: 用子进程隔离验证 `fastest` profile 的短文本流式 RTF。
- `evaluate_long_text_quality.py`: 对长文本 full-AR 候选 profile 做客观检查和可选 Omni 质量评测。

日常不需要手动串联这些脚本，优先使用正式 CLI：

```bash
uv run python -m qwen3_tts_ov build-fastest --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

`build-fastest` 默认使用低内存 production 图集合。只有需要旧 fixed-bucket/unroll 诊断图时才加 `--graph-set compat`。

## 诊断脚本

- `verify_long_autoregressive_parity.py`: 对比 OpenVINO full-AR codec 生成和上游 PyTorch 链路。
- `verify_paged_kv_correctness.py`: 对比 paged-KV codec 结果和 reference profile。
- `audit_paged_kv_conversion.py`: 检查 `SDPAToPagedAttention` 转换覆盖率。
- `convert_paged_kv_graphs.py`: 离线转换 paged-KV seed graph，仅用于调试。
- `build_paged_kv_tool.py`: 构建离线 paged-KV 诊断工具。

历史 benchmark、PTQ 对照、profiling 和 legacy 入口放在 `devtools/`。这些脚本不属于生产主线，可能需要额外依赖或本机实验 IR。

# 辅助脚本

项目的正式入口是：

```bash
uv run python -m qwen3_tts_ov --help
uv run qwen3-tts-ov --help
```

本目录只保留开发和实验辅助脚本：

- `compress_openvino_weights.py`: 对已导出的 OpenVINO IR 做权重压缩，并更新 manifest graph variants。
- `quantize_openvino_full.py`: 基于校准样本做 NNCF PTQ 实验。
- `benchmark_fast_cache.py`: 对不同 cache/graph variant 路径做本地 benchmark。
- `env.sh`: 本地开发环境辅助变量。
- `legacy/`: 旧入口兼容包装，保留用于查历史命令，不建议新流程使用。

这些脚本可能需要额外依赖，例如 `.[export]` 或 `nncf`。普通 runtime 和 sidecar 使用不依赖它们。

# 辅助脚本

项目的正式入口是：

```bash
uv run python -m qwen3_tts_ov --help
```

如果已执行 `uv pip install -e .`，也可以使用 `uv run qwen3-tts-ov --help`。

本目录只保留生产流程会直接用到的辅助脚本：

- `compress_openvino_weights.py`: 对已导出的 OpenVINO IR 做权重压缩，并更新 manifest graph variants。
- `build_native_codegen.py`: 构建必需的 native C++ GenAI-style pipeline 共享库和 standalone smoke CLI。
- `benchmark_streaming_realtime.py`: 默认用子进程隔离验证 `fastest` profile 的流式 RTF。
- `env.sh`: 本地开发环境辅助变量。

历史 benchmark、PTQ 对照、profiling 和 legacy 入口已经移动到 `devtools/`。这些脚本可能需要额外依赖，例如 `.[export]` 或 `nncf`，不属于生产主线。

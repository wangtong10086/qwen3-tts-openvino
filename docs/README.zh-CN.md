# 文档索引

本项目已收敛为单一生产推理架构：`fastest` profile + native paged-KV + vLLM-like online batching。

English entry: [Documentation](README.md). 示例入口：[examples](../examples/README.zh-CN.md)。

## 推荐阅读顺序

| 角色 | 先读 | 内容 |
| --- | --- | --- |
| 新机器/新用户 | [前置条件](prerequisites_zh.md) | Python/uv、OpenVINO、Intel GPU/NPU 驱动和首次编译缓存 |
| 最终用户 | [Release 使用说明](release_zh.md) | 下载预编译 runtime，自动下载 Hugging Face IR，启动 Web Demo |
| 源码开发者 | [Quick Start](quick_start_zh.md) | 从 PyTorch 模型导出生产 IR，构建 native backend |
| 集成方 | [API Reference](api_reference_zh.md) | HTTP、WebSocket、OpenAI-compatible Speech API 和 Python API 字段 |
| 发布维护者 | [开发说明](development_zh.md) | 本地检查、打包、GitHub Actions release |
| 性能/质量验证 | [流式与长文本](streaming_zh.md) | full-AR 长文本、online batching、benchmark、Omni gate |

## 文档列表

- [前置条件](prerequisites_zh.md)
- [Troubleshooting / FAQ](troubleshooting_zh.md)
- [Release 使用说明](release_zh.md)
- [Release Notes](releases/README.zh-CN.md)
- [Quick Start](quick_start_zh.md)
- [运行接口](runtime_zh.md)
- [API Reference](api_reference_zh.md)
- [导出与构建](export_zh.md)
- [流式与长文本](streaming_zh.md)
- [OpenVINO 编译缓存](cache_zh.md)
- [大文件与产物策略](artifacts_zh.md)
- [Windows GPU+NPU 测试路径](windows_gpu_npu_zh.md)
- [辅助脚本](../scripts/README.zh-CN.md)
- [native backend](../native/qwen3_tts_ov_genai/README.md)
- [安全说明](security_zh.md)
- [示例请求与 Python 客户端](../examples/README.zh-CN.md)
- [贡献指南](../CONTRIBUTING.md)

## 已清理内容

历史 ONNX/XPU、旧 runtime、旧 profile sweep、分段长文本 fallback、旧 benchmark 和 PTQ 对照文档不再作为主仓库入口保留。需要新的性能实验时，应基于当前 `fastest` 生产路径新增小而明确的脚本或记录。

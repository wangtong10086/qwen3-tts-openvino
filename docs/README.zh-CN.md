# 文档索引

这是 Qwen3-TTS OpenVINO 仓库的中文文档索引。先按你的角色选择入口，再进入细节文档。

## 按角色阅读

| 角色 | 先读 | 目的 |
| --- | --- | --- |
| 最终用户 | [Release 使用说明](release_zh.md) | 下载 GitHub Release runtime，首次启动自动下载 Hugging Face OpenVINO IR |
| 源码开发者 | [Quick Start](quick_start_zh.md) | 从 PyTorch 模型导出最快 OpenVINO IR，并启动 Web Demo |
| 发布维护者 | [开发说明](development_zh.md) | 构建 native、打包 runtime、触发 `release-runtime` |
| 性能/质量调试 | [流式与长文本](streaming_zh.md) | 理解 full-AR 长文本、流式输出和质量门禁 |
| Windows 异构测试 | [Windows GPU+NPU 测试路径](windows_gpu_npu_zh.md) | 在自托管 Windows NPU 机器上验证 `GPU codegen + NPU decoder` |

## 入门与部署

- [Release 使用说明](release_zh.md): 最终用户部署路径；runtime 来自 GitHub Release，已编译 IR 可自动从 Hugging Face 下载。
- [Quick Start](quick_start_zh.md): 源码路径；下载 PyTorch 模型，一键构建 fastest IR，启动 Web Demo。
- [运行接口](runtime_zh.md): CLI、Python API、sidecar、WebSocket、OpenAI-compatible Speech API。
- [OpenVINO 编译缓存](cache_zh.md): cache warmup、缓存目录、预热策略。

## 开发与发布

- [开发说明](development_zh.md): 源码开发、native 构建、release workflow、本地打包。
- [导出与压缩](export_zh.md): 手动导出 OpenVINO IR、生成 `int8_sym_paged_talker_split`。
- [大文件与产物策略](artifacts_zh.md): 模型权重、OpenVINO IR、outputs、native build 的处理规则。
- [Windows GPU+NPU 测试路径](windows_gpu_npu_zh.md): 自托管 Windows runner 和本地 PowerShell 异构推理 smoke。

## 参考

- [流式与长文本](streaming_zh.md): 流式协议、浏览器播放、长文本 full-AR、质量门禁。
- [示例请求](../examples/README.zh-CN.md): JSON、JSONL、OpenAI-compatible 请求样例。
- [安全说明](security_zh.md): token、`.env`、凭据和提交检查。
- [辅助脚本](../scripts/README.zh-CN.md): 构建、压缩、benchmark、质量评测脚本说明。
- [native pipeline](../native/qwen3_tts_ov_genai/README.md): C++ pipeline 和 paged-KV 诊断说明。

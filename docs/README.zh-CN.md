# 文档索引

这是 Qwen3-TTS OpenVINO 仓库的中文文档索引。首次使用请从 Quick Start 开始。

## 入门与部署

- [Quick Start](quick_start_zh.md): 安装依赖、下载模型、一键构建最快路径、启动 Web Demo。
- [Release 使用说明](release_zh.md): 下载预编译 Linux/Windows app 包和独立 IR 包后直接启动 sidecar。
- [开发说明](development_zh.md): 源码开发、导出、压缩、打包 release。
- [运行接口](runtime_zh.md): CLI、Python API、sidecar、WebSocket、OpenAI-compatible Speech API。
- [OpenVINO 编译缓存](cache_zh.md): cache warmup、缓存目录、预热策略。

## 模型与产物

- [导出与压缩](export_zh.md): 手动导出 OpenVINO IR、生成 `int8_sym_paged_talker_split`。
- [大文件与产物策略](artifacts_zh.md): 模型权重、OpenVINO IR、outputs、native build 的处理规则。

## 流式与质量

- [流式与长文本](streaming_zh.md): 流式协议、浏览器播放、长文本 full-AR、质量门禁。
- [示例请求](../examples/README.zh-CN.md): JSON、JSONL、OpenAI-compatible 请求样例。

## 安全与开发

- [安全说明](security_zh.md): token、`.env`、凭据和提交检查。
- [辅助脚本](../scripts/README.zh-CN.md): 构建、压缩、benchmark、质量评测脚本说明。
- [native pipeline](../native/qwen3_tts_ov_genai/README.md): C++ pipeline 和 paged-KV 诊断说明。

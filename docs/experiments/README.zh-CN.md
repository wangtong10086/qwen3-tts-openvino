# 实验记录

本目录记录 Qwen3-TTS OpenVINO runtime 的性能、质量、打包和异构推理实验。这里的内容用于复盘和后续决策，不等同于稳定使用说明；稳定入口仍以根目录 README 和 `docs/quick_start_zh.md`、`docs/release_zh.md` 为准。

## 当前结论

- 生产默认路径仍是 `fastest` profile。
- `fastest` 使用 native C++ audio pipeline、OpenVINO paged-KV、U8 KV cache、`int8_sym_paged_talker_split` talker seed 图、FP16 cached subcode 图。
- 长文本必须保持完整上下文 full-AR 自回归；切分文本会导致音色和语气不连续，只能作为失败路径或临时降级方案。
- 最近一次 profile sweep 没有发现可以替代 `fastest` 的稳定加速配置。
- `paged_split_static_decode` 单次指标看起来快，但 native static decode 编译失败，`fast_path_ok=false`，不能采纳。
- `fastest_native_prompt_cache`、`split_next_embed_graph`、`paged_split_dq64`、`paged_split_u8_all`、`paged_split_block8` 在当前 Linux GPU 验证中没有稳定超过默认路径。

## 文档列表

- [实时推理优化历程](realtime-optimization-history.zh-CN.md): 按阶段记录从 Python/OpenVINO runtime 到 native paged-KV fastest path 的实验目标、过程和结论。
- [2026-05-15 fast-path 与 profile sweep](2026-05-15-fast-path-profile-sweep.zh-CN.md): 记录最近一次 fast-path hard metrics、native prompt cache、block size/profile sweep 的验证数据。

## 新增实验记录模板

新增实验时建议保留以下字段，避免只留下无法复现的片段化计时。

```markdown
# YYYY-MM-DD 实验名称

## 目标

- 要验证的假设
- 期望改善的指标，例如 TTFT、RTF、TPS、显存、包体积

## 环境

- 代码分支和 commit
- OS、OpenVINO 版本、硬件、device 配置
- IR 来源、profile、关键环境变量

## 过程

- 构建/导出/压缩命令
- benchmark 命令
- 质量检查或人工试听方式

## 结果

| 配置 | TTFT | first audio | RTF | TPS | fast_path_ok | 结论 |
| --- | ---: | ---: | ---: | ---: | --- | --- |

## 结论

- 是否采纳
- 为什么采纳或拒绝
- 后续动作
```

## 产物策略

实验产生的 `outputs/`、OpenVINO IR、native build、WAV 和模型文件都不进入 git。文档只记录关键命令、指标和结论；需要复核原始数据时，在本地 ignored 目录中查看对应 JSON。

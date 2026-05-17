# 实验记录

历史性能实验已经收敛到当前生产架构：

- `fastest` profile
- native C++ codec generation
- OpenVINO paged-KV
- U8 KV cache
- vLLM-like online batching
- 长文本 full autoregressive

旧 profile sweep、分段长文本 fallback、graph-fused/unroll 对照、旧 streaming benchmark 和 PTQ 对照文档已从主仓库删除。后续新增实验需要满足两个条件：

1. 基于当前生产架构，不能引入并行 fallback 架构。
2. 实验结论必须能通过 `benchmark_prompt_batch_matrix.py` 和 `evaluate_single_arch_gate.py` 复现。

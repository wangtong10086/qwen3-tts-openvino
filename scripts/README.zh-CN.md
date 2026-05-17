# scripts

本目录只保留生产构建、release、benchmark 和质量门禁脚本。

## 构建与发布

- `build_native_codegen.py`: 构建 native C++ OpenVINO backend。
- `compress_openvino_weights.py`: 生成生产 `fastest` / minimal online-batching graph variants。
- `package_ir.py`: 打包 OpenVINO IR。
- `package_release.py`: 打包 Linux/Windows runtime。
- `upload_hf_ir.py` / `download_hf_ir.py`: Hugging Face IR 上传和下载。

## 性能

- `benchmark_online_continuous_batch.py`: 测量当前 layered vLLM-like online batching。
- `benchmark_prompt_batch_matrix.py`: 按 batch、prompt 长度、离线/在线到达场景运行矩阵 benchmark。
- `benchmark_windows_gpu_npu_release.py`: Windows release 包 GPU/NPU 对照。

## 质量

- `evaluate_single_arch_gate.py`: 生产架构发布 gate，覆盖 VoiceDesign、CustomVoice、VoiceClone。
- `evaluate_prefill_quality.py`: 与原始 PyTorch 输出对照，当前只保留 runtime candidate 路径。
- `evaluate_default_policy_quality.py`: 汇总默认策略质量结果。
- `audit_prefill_quality_coverage.py`: 审计 quality summary 是否满足发布覆盖要求。
- `verify_long_autoregressive_parity.py`: 对比长文本 full-AR codec 行为。
- `verify_paged_kv_correctness.py`: 检查 paged-KV 图和 runtime 行为。

## Windows GPU+NPU

- `probe_windows_gpu_npu.py`
- `analyze_windows_gpu_npu_results.py`
- `collect_windows_accelerator_counters.ps1`
- `windows_gpu_npu_smoke.ps1`
- `windows_gpu_npu_benchmark.ps1`

脚本输出默认写入 `outputs/`，不进入 git。

# 2026-05-15 fast-path 与 profile sweep

## 目标

本次实验验证最近一组性能相关改动是否带来实际加速：

- fast-path health 和 benchmark JSON 增加硬指标。
- `/health`、benchmark、Windows release benchmark 统一输出 `fast_path_ok`、fallback、host copy、paged-KV 等状态。
- native prompt embedding cache 是否能降低 TTFT。
- full speech decoder infer request reuse 是否有可见收益。
- profile sweep 比较 `fastest`、prompt cache、block size、DQ group size、next-embed graph、static decode 等候选路径。

验收标准：

- 不能只看单次 RTF，必须同时看 `fast_path_ok`。
- 候选路径需要在短输出和更长输出上都不劣于默认 `fastest`。
- 如果出现 fallback、compile failure 或 fast-path gate 失败，即使单次 RTF 更低也不能采纳。

## 环境

- 工作目录：`/home/wt/qwen3-tts-igpu`
- IR：`openvino/voice_design`
- device：`GPU`
- chunk strategy：`smooth`
- benchmark：`scripts/benchmark_streaming_realtime.py`
- warmup：每个 worker 内 `--warmup-generations 1`
- 隔离：默认子进程隔离，每个 profile/run 独立 worker

本次测试在 Linux/WSL 环境验证 GPU 路径。Windows GPU+NPU 需要在 Windows 原生机器上另行执行 `scripts/windows_gpu_npu_benchmark.ps1`。

## 过程

### 1. 发现并修复 native prompt bug

初始运行：

```bash
uv run python scripts/benchmark_streaming_realtime.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profiles fastest,fastest_native_prompt_cache \
  --runs 3 \
  --max-new-tokens 64 \
  --warmup-generations 1 \
  --chunk-strategy smooth \
  --output-json outputs/perf_validation/prompt_cache_on.json
```

`fastest_native_prompt_cache` 失败：

```text
'NoneType' object has no attribute 'shape'
```

原因是 native prompt 分支下 `sequence=None`，但 runtime 仍无条件读取 `sequence.shape[1]`。修复后只在非 native prompt 分支读取 `sequence.shape`，native prompt 分支使用 token 计数得到的 `prompt_len`。

验证：

```bash
uv run python -m py_compile qwen3_tts_ov/runtime.py
uv run pytest tests/test_windows_gpu_npu_path.py tests/test_benchmark_script.py -q
```

结果：`52 passed`。

### 2. prompt cache on/off 对照

开启 cache：

```bash
uv run python scripts/benchmark_streaming_realtime.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profiles fastest,fastest_native_prompt_cache \
  --runs 3 \
  --max-new-tokens 64 \
  --warmup-generations 1 \
  --chunk-strategy smooth \
  --output-json outputs/perf_validation/prompt_cache_on_rerun.json
```

关闭 cache：

```bash
QWEN3_TTS_OV_NATIVE_PROMPT_EMBED_CACHE=0 \
uv run python scripts/benchmark_streaming_realtime.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profiles fastest_native_prompt_cache \
  --runs 3 \
  --max-new-tokens 64 \
  --warmup-generations 1 \
  --chunk-strategy smooth \
  --output-json outputs/perf_validation/prompt_cache_off.json
```

结果：

| profile | cache | runs | p50 RTF | p90 RTF | p50 compute RTF | p50 first audio | p50 TTFT | p50 TPS | fast_path_ok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fastest` | off | 3 | 0.936724 | 0.940356 | 0.984635 | 925.4 ms | 89.1 ms | 13.609 | true |
| `fastest_native_prompt_cache` | on | 3 | 0.944930 | 0.945211 | 0.991378 | 929.7 ms | 90.5 ms | 13.503 | true |
| `fastest_native_prompt_cache` | off | 3 | 0.939909 | 0.947220 | 0.986812 | 931.9 ms | 90.1 ms | 13.572 | true |

native timing 确认 cache 生效：

```text
prompt_embedding_cache_enabled=true
prompt_embedding_cache_size=3
prompt_embedding_cache_hits=3
prompt_embedding_cache_misses=3
```

结论：

- cache 机制工作正常。
- 当前文本和路径下没有可证明的 TTFT/RTF 收益。
- 不建议把 `fastest_native_prompt_cache` 作为默认 profile。

### 3. 短输出 profile sweep

命令：

```bash
uv run python scripts/benchmark_streaming_realtime.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profiles fastest,fastest_native_prompt_cache,paged_split_dq64,paged_split_u8_all,split_next_embed_graph,paged_split_block8,paged_split_static_decode \
  --runs 1 \
  --max-new-tokens 64 \
  --warmup-generations 1 \
  --chunk-strategy smooth \
  --output-json outputs/perf_validation/profile_sweep_short.json
```

结果：

| profile | RTF | compute RTF | first audio | TTFT | TPS | fast_path_ok | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `paged_split_static_decode` | 0.892635 | 0.938605 | 872.9 ms | 86.3 ms | 14.293 | false | 拒绝，static decode 编译失败 |
| `paged_split_block8` | 0.906748 | 0.904804 | 878.6 ms | 84.1 ms | 14.872 | true | 进入长度复测 |
| `fastest` | 0.916318 | 0.963876 | 897.9 ms | 87.2 ms | 13.900 | true | 默认基线 |
| `fastest_native_prompt_cache` | 0.938821 | 0.985694 | 917.5 ms | 95.1 ms | 13.602 | true | 不采纳 |
| `split_next_embed_graph` | 0.944487 | 0.993403 | 938.4 ms | 92.3 ms | 13.522 | true | 不采纳 |
| `paged_split_dq64` | 0.952234 | 1.001454 | 932.3 ms | 90.9 ms | 13.389 | true | 不采纳 |
| `paged_split_u8_all` | 0.961356 | 1.010519 | 946.5 ms | 91.1 ms | 13.250 | true | 不采纳 |

`paged_split_static_decode` 的失败原因：

```text
native_paged_static_decode_failure=compile_failed:
Exception from src/inference/src/cpp/core.cpp:118:
Exception from src/inference/src/dev/plugin.cpp:54:
map::at
```

结论：

- 只看 RTF 会误判 `paged_split_static_decode`。
- `fast_path_ok` gate 必须作为性能报告的一部分。
- `paged_split_block8` 单次看起来可能快，需要进一步复测。

### 4. block size 16 vs 8 长度复测

命令：

```bash
uv run python scripts/benchmark_streaming_realtime.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profiles fastest,paged_split_block8 \
  --runs 2 \
  --max-new-tokens-set 64,256 \
  --warmup-generations 1 \
  --chunk-strategy smooth \
  --worker-timeout-sec 600 \
  --output-json outputs/perf_validation/block8_length_sweep.json
```

结果：

| profile | max_new_tokens | runs | p50 RTF | p90 RTF | p50 compute RTF | p50 first audio | p50 TTFT | p50 TPS | fast_path_ok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fastest` | 64 | 2 | 0.934677 | 0.935056 | 0.982741 | 924.5 ms | 90.0 ms | 13.637 | true |
| `paged_split_block8` | 64 | 2 | 0.963237 | 0.969535 | 0.961867 | 931.6 ms | 90.5 ms | 13.894 | true |
| `fastest` | 256 | 2 | 0.930416 | 0.933297 | 0.969451 | 926.3 ms | 90.5 ms | 13.681 | true |
| `paged_split_block8` | 256 | 2 | 0.978378 | 0.993245 | 0.976992 | 945.7 ms | 92.4 ms | 13.602 | true |

结论：

- `paged_split_block8` 的 compute RTF 有时较低，但端到端 RTF、first audio、TTFT 不优于默认。
- 256 max tokens 下 `paged_split_block8` 明显更慢。
- 不采纳 block size 8 作为默认。

### 5. hidden zero-copy、OpenVINO profile 与 top1 seed 实验

本轮补充三组实验：

- `fastest + native OV profile`：建立 OpenVINO 内部算子热点基线。
- `fastest + hidden remote on/off/require`：验证 split subcode hidden tensor 是否真正直连，是否存在 copy fallback。
- `talker_top1_seed_split_subcode`：去掉 seed 图输出 logits 后的 host argmax，观察是否降低 codegen 时延。

为了支持 `talker_top1_seed_split_subcode`，本地 IR 额外导出并压缩了：

```text
talker_top1_sdpa_paged_seed.xml
talker_top1_sdpa_paged_gqa_seed.xml
talker_top1_sdpa_paged_gqa_seed_int8_sym_paged_talker_top1_split.xml
```

导出和压缩命令：

```bash
PYTHONPATH=/home/wt/Qwen3-TTS uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --model-type voice_design \
  --out-dir openvino/voice_design \
  --skip-fixed-cache-graphs \
  --export-paged-kv-seed \
  --paged-kv-unroll-steps '' \
  --stream-decoder-chunks 24 \
  --stream-decoder-first-chunks 12 \
  --subcode-attention-kernels sdpa \
  --paged-kv-subcode-attention-kernels sdpa

uv run python scripts/compress_openvino_weights.py \
  --ir-dir openvino/voice_design \
  --preset fastest-top1-seed
```

48 token 热态主实验命令：

```bash
uv run python scripts/benchmark_streaming_realtime.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profiles fastest,fastest_hidden_remote_off,fastest_hidden_remote_on,fastest_hidden_remote_require,talker_top1_seed_split_subcode \
  --runs 3 \
  --max-new-tokens 48 \
  --min-new-tokens 12 \
  --warmup-generations 1 \
  --chunk-strategy smooth \
  --worker-timeout-sec 360 \
  --output-json outputs/perf_validation/experiment_main_hot_48.json
```

结果：

| profile | runs | p50 RTF | p50 compute RTF | p50 first audio | p50 TTFT | p50 TPS | hidden direct/fallback | sampling ms | fast_path_ok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `fastest` | 3 | 0.877048 | 0.912207 | 873.1 ms | 86.2 ms | 14.632 | 48 / 0 | 0.736 | true |
| `fastest_hidden_remote_off` | 3 | 0.869894 | 0.904897 | 871.9 ms | 88.8 ms | 14.782 | 0 / 48 | 0.654 | false |
| `fastest_hidden_remote_on` | 3 | 0.874993 | 0.910093 | 874.0 ms | 86.0 ms | 14.688 | 48 / 0 | 0.808 | true |
| `fastest_hidden_remote_require` | 3 | 0.877461 | 0.913076 | 875.2 ms | 86.4 ms | 14.644 | 48 / 0 | 0.837 | true |
| `talker_top1_seed_split_subcode` | 3 | 0.882548 | 0.916882 | 889.9 ms | 91.8 ms | 14.575 | 48 / 0 | 0.046 | true |

结论：

- `hidden_remote_on/require` 均确认 direct bind，fallback 为 0，可以保留为 fast-path 约束和诊断指标。
- 强制 `hidden_remote_off` 会触发 `split_subcode_hidden_bind_fallback_count=48`，`fast_path_ok=false`；即使单次 RTF 接近，也不能作为默认路径。
- `talker_top1_seed_split_subcode` 确实把 `sampling_ms` 从约 0.7 ms 降到约 0.05 ms，但端到端 RTF 没有改善，因为主耗时仍在 decode/subcode OpenVINO 图。

OpenVINO perf count 命令：

```bash
uv run python scripts/benchmark_streaming_realtime.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profiles fastest \
  --runs 1 \
  --max-new-tokens 48 \
  --min-new-tokens 12 \
  --warmup-generations 1 \
  --chunk-strategy smooth \
  --native-ov-profile \
  --worker-timeout-sec 360 \
  --output-json outputs/perf_validation/experiment_native_ov_profile_hot_48.json
```

该结果只用于热点归因，perf count 本身会显著增加运行开销。内部热点如下：

| scope/type | real time | 占比/说明 |
| --- | ---: | --- |
| `codegen_paged_kv_decode_dynamic` | 982.162 ms | 最大图级热点 |
| `codegen_paged_kv_subcode` | 368.190 ms | 第二图级热点 |
| `stream_decoder_steady` | 142.153 ms | audio decoder 稳态 |
| `FullyConnected` | 859.417 ms | 55.7%，主要算子热点 |
| `Rms` | 175.499 ms | 11.4%，小算子数量多 |
| `ScaledDotProductAttention` | 82.778 ms | 5.4% |
| `ArgMaxMin` | 78.320 ms | 5.1%，主要来自 subcode TopK |
| `PagedAttention` | 54.800 ms | 3.5% |

192 token 上限补充测试命令：

```bash
uv run python scripts/benchmark_streaming_realtime.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profiles fastest,talker_top1_seed_split_subcode \
  --runs 1 \
  --max-new-tokens 192 \
  --min-new-tokens 48 \
  --warmup-generations 1 \
  --chunk-strategy smooth \
  --worker-timeout-sec 480 \
  --output-json outputs/perf_validation/experiment_long_hot_192.json
```

实际两组都在 80 frames 处 EOS：

| profile | frames | RTF | compute RTF | first audio | TPS | codegen ms | subcode ms | sampling ms | fast_path_ok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fastest` | 80 | 0.878503 | 0.913864 | 877.6 ms | 14.520 | 5538.360 | 3489.569 | 1.118 | true |
| `talker_top1_seed_split_subcode` | 80 | 0.863991 | 0.899662 | 864.7 ms | 14.764 | 5450.899 | 3409.661 | 0.079 | true |

结论：

- top1 seed 在更长一点的输出上有小幅正收益，但幅度仍很有限。
- 可保留为实验 profile，不建议立即替换默认 `fastest`；是否进入默认需要 Windows 真机重复验证和音频一致性验证。

## 最终结论

本次实验没有发现可以替代 `fastest` 的稳定加速配置。

保留的改动：

- fast-path hard metrics。
- benchmark JSON 中的 fast-path/fallback/host-copy 指标。
- `/health` 中的当前 profile 和 fast-path 状态。
- Windows release benchmark 的 profile sweep 能力。
- full decoder infer request reuse。
- native prompt cache 代码可以保留为实验开关，但不作为默认加速点。

不采纳为默认的路径：

- `fastest_native_prompt_cache`
- `paged_split_block8`
- `paged_split_dq64`
- `paged_split_u8_all`
- `split_next_embed_graph`
- `paged_split_static_decode`
- `talker_top1_seed_split_subcode` 暂不采纳为默认，只保留实验 profile

当前默认仍应保持：

```text
profile=fastest
native_paged_kv_precision=u8
native_paged_kv_cache_input_precision=f32
native_paged_kv_block_size=16
native_dynamic_quantization_group_size=32
native_codegen_fusion=split
```

## 原始结果文件

这些文件是本地 ignored 产物，不进入 git：

```text
outputs/perf_validation/prompt_cache_on.json
outputs/perf_validation/prompt_cache_on_rerun.json
outputs/perf_validation/prompt_cache_off.json
outputs/perf_validation/profile_sweep_short.json
outputs/perf_validation/block8_length_sweep.json
outputs/perf_validation/experiment_main_hot_48.json
outputs/perf_validation/experiment_native_ov_profile_hot_48.json
outputs/perf_validation/experiment_long_hot_192.json
```

## 后续建议

- 下一轮优化应继续集中在 `codegen_paged_kv_decode_dynamic` 和 `codegen_paged_kv_subcode`，尤其是 FullyConnected、Rms、ArgMaxMin/TopK，而不是 prompt embedding cache。
- 所有性能报告必须带上 `fast_path_ok`、`fallback`、`host_copy_fallback_count`、`subcode_host_copy_fallback_count`。
- 任何单次更快但 `fast_path_ok=false` 的结果都不能进入默认路径。

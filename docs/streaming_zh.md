# 流式合成与长文本

本页只说明流式协议、浏览器播放策略和长文本 full-AR 策略。具体 CLI、Python API、HTTP 和 WebSocket 调用方式见 [运行接口](runtime_zh.md)。

## 当前生产路径

短文本和长文本共用 `fastest` profile：

- native C++ pipeline
- paged-KV seed graph
- cached standalone subcode graph
- streaming decoder graph
- `int8_sym_paged_talker_split` graph variant
- OpenVINO GPU 编译默认设置 `DYNAMIC_QUANTIZATION_GROUP_SIZE=32`

缺少这些图时，`fastest` 应直接报错。不要在生产路径中静默切到 legacy profile，否则 RTF 和音质结论会失真。

## WebSocket 消息顺序

客户端连接 `/v1/tts/stream` 后先发送请求 JSON。服务端返回顺序固定为：

1. metadata JSON
2. 多个 binary `pcm_s16le` 音频块
3. final JSON

metadata 会包含 `realtime_profile`、`chunk_strategy`、`recommended_playback_buffer_ms`、`graph_variant`、`paged_kv`、`continuous_long_output`、`effective_max_continuous_prompt_tokens` 等运行状态。前端必须以 final JSON 作为本次合成结束信号，不要用连接关闭或最后一个音频块判断结束。

## 播放缓冲

浏览器 demo 使用 jitter buffer 播放连续 PCM chunk。建议调用方也采用类似策略：

- 首次播放前先累积 `recommended_playback_buffer_ms`。
- 如果队列低于安全水位，短暂提高目标缓冲，而不是立刻中断播放。
- 不要对每个 chunk 单独创建 `<audio>` 元素；应把 PCM 连续推入同一条播放队列。
- final JSON 到达后，等播放队列消耗完再结束 UI 计时。

如果服务端 `stream_rtf < 1.0` 但浏览器仍卡顿，优先检查播放队列是否归零；如果 `stream_rtf >= 1.0`，瓶颈在生成速度。

## 长文本 full-AR

长文本必须保持同一条自回归链路。默认策略是：

- `long_text_mode=full_ar`
- `segmented=false`
- `continuous_long_output=true`
- codec 生成来自同一个 prompt 和同一条 autoregressive 序列
- decoder 可以按 chunk 输出音频，但不能把文本拆成多个独立 prompt

这样可以避免分段生成导致的音色、语气和韵律不连续。

## VoiceClone 流式

VoiceClone 默认使用 ICL 克隆路径，也就是 `x_vector_only=false`。请求必须提供参考音频和对应的 `ref_text`，服务端会先从参考音频提取 codec prompt 和 speaker embedding，再从同一条自回归链路生成目标文本；流式 decoder 只输出目标文本对应的新音频，不播放参考音频。

`x_vector_only=true` 只适合做 speaker embedding-only 的对照实验。它不会把参考音频 codec prompt 拼入生成上下文，因此通常不如默认 ICL 路径稳定地保留参考音频的韵律和说话风格。

## 长文本 prompt 预算

服务端默认使用 `--max-continuous-prompt-tokens auto` 做推理前保护，避免极长 prompt 在部分 GPU/驱动上触发 OpenVINO USM 分配失败。GPU 路径会根据模型上下文、KV/cache-input 精度、block size、GPU 总显存、`--max-vram-ratio` 和保留显存计算可用 KV blocks；CPU-only 路径使用保守固定预算。

长文本 full-AR 不再根据文本长度预估 `max_new_tokens`。服务端会先计算精确 prompt tokens，再将运行时 `max_new_tokens` 设置为 `effective_max_total_tokens - prompt_len - 1`，让模型生成直到 EOS 或上下文/KV 上限。metadata 中的 `max_generation_tokens_available` 和 `generation_stop_condition=eos_or_context_limit` 可用于确认当前请求的真实生成上限。

如果 metadata 或错误信息显示文本超过预算，可以先提高显存比例或降低保留显存：

```bash
qwen3-tts-ov-server --device GPU --max-vram-ratio 90 --kv-cache-reserve-mb 1024
```

也可以显式设置 `--max-continuous-prompt-tokens N`，或设置为 `0` 关闭 prompt 预算保护。关闭后仍不会自动分段，后续如果显存不足会由 OpenVINO/USM 错误和 retry 机制处理。

显存压力主要来自长 prompt 的 paged-KV cache 和运行时中间 buffer。默认生产路径使用 `u8` KV cache；如果需要显式指定或配合更低 prompt 预算，可以启动时使用：

```bash
qwen3-tts-ov-server --device GPU --kv-cache-profile u8 --max-vram-ratio 70
```

`u8` 会把 KV cache 存储元素从 FP16 的 2 bytes 降为 1 byte，`/health` 中 `kv_cache_relative_to_fp16` 应显示为 `0.5`。但默认 `u8` 的 cache input 仍为 FP32，planner 会按实际 cache input 做保守预算；需要保守对照时使用 `--kv-cache-profile fp16`，切换到 `u8-all` 前需要重新做长文本质量评测。

流式 metadata 会返回 `effective_max_total_tokens`、`effective_max_continuous_prompt_tokens`、`preallocated_kv_blocks`、`kv_cache_limit_source`。Web Demo 会用 tokenizer 实时计算 prompt token，并用这些字段判断当前文本是否超过预算。

## 长文本采样参数

默认生成参数跟随上游习惯：

- `do_sample=true`
- `top_k=50`
- `top_p=1.0`
- `temperature=0.9`
- `repetition_penalty>=1.05`

服务端会优先使用 manifest 中的 sampled paged-KV full-AR 加速图。缺图时才回退到 FP16 no-cache sampled reference。

## 质量门禁

部署新 IR、切换硬件或修改生成逻辑后，先跑长文本质量评测：

```bash
uv pip install -e ".[quality]"

uv run python scripts/evaluate_long_text_quality.py \
  --ir-dir auto \
  --device GPU \
  --text-file examples/long_text_zh.example.txt \
  --profiles quality \
  --runs 1
```

只做本地客观检查时可跳过外部 Omni 评测：

```bash
uv run python scripts/evaluate_long_text_quality.py \
  --ir-dir auto \
  --device GPU \
  --text-file examples/long_text_zh.example.txt \
  --profiles quality \
  --runs 1 \
  --skip-omni
```

结果写入 `outputs/long_text_quality/quality_summary.json`。sidecar 会在可用时读取该结果，用于选择长文本 runtime/env 配置。

## 自动分段

自动分段只用于诊断或非常规 fallback，不是默认路径。请求中必须同时传：

```json
{
  "auto_segment_text": true,
  "allow_auto_segment_text": true
}
```

或设置 `QWEN3_TTS_OV_ENABLE_AUTO_SEGMENT=1`。分段路径会启动多个短 prompt，不能保证长文本音色和语气连续。

## 性能门禁

短文本实时性能使用 isolated benchmark：

```bash
uv run python scripts/benchmark_streaming_realtime.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profile-set fastest-gate \
  --runs 3 \
  --warmup-generations 1
```

不要用 `PERF_COUNT` 下的结果作为线上 RTF。OpenVINO operator profiling 会显著放慢小图自回归循环，只适合定位算子热点。

长度维度验证使用统一脚本，同时覆盖短文本和长文本，并输出 TTFT/TPS：

```bash
uv run python scripts/benchmark_tts_length_scaling.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profiles fastest \
  --cases short,long \
  --short-max-new-tokens-set 48,128 \
  --long-max-new-tokens-set 256,768 \
  --warmup-generations 1
```

关键指标定义：

- `ttft_ms`：从开始测量到首个 codec frame 生成完成的时间，来自 native timing。
- `first_audio_ms`：从开始测量到首个可播放音频块返回 Python 的时间。
- `codec_tps_post_ttft`：首 token 后的 codec frame/s，更接近 LLM 的 steady-state TPS。
- `codec_tps_codegen`：仅按 native codegen infer 时间计算的 codec frame/s，用于隔离 decoder 影响。
- `stream_rtf`：端到端流式耗时 / 输出音频时长，真实播放是否追得上主要看它是否小于 1。

当前性能热点仍在 codegen：standalone subcode 约占 codegen 时间 60% 以上，talker decode 约占 35%。`fastest` 已默认启用 `DYNAMIC_QUANTIZATION_GROUP_SIZE=32`，在本机隔离 benchmark 中相对旧默认带来约 1-2% 的稳定收益；`LATENCY`/`NUM_STREAMS=1`/subcode CPU/exact subcode/graph-fused subcode 都没有通过默认路径验收。

native timing 现在会拆出 `subcode_bind_ms`、`subcode_output_read_ms`、`subcode_next_embed_ms` 和 `subcode_host_copy_*`。如果要验证 standalone subcode 的 host 侧开销是否仍值得优化，优先看这些字段，而不是只看总的 `tensor_bind_ms`。

进一步定位单步 decoding 时，使用 paged-KV ablation profile set：

```bash
uv run python scripts/benchmark_streaming_realtime.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profile-set paged-kv-step-ablation \
  --runs 1 \
  --max-new-tokens 64 \
  --warmup-generations 1
```

关注 `codegen_decode_step_mean_ms`、`subcode_infer_step_mean_ms`、`codegen_bind_step_mean_ms` 和 `native_paged_static_decode_failure`。当前验证结果是：`cachedsub` 在短文本上可能降低几毫秒 subcode 时间，但长文本不稳定；`split_next_embed_graph` 基本没有收益；static decode reshape 已可进入编译阶段，但当前 OpenVINO GPU 会在编译时报 `map::at` 并自动回退到 dynamic decode。因此这些路径都保持为实验项，不替换 `fastest`。

## Codegen 融合实验

默认 `fastest` 仍使用已经验证的 split-subcode 路径。若导出的 IR 包含
`graphs.paged_kv_seed.fused_cache_step_gqa`，可以测试 talker+subcode 单图路径：

旧 IR 可能只包含 `talker_stateful_gqa`。这种情况下需要用当前 exporter 重新导出，
并保留 `--export-paged-kv-seed`，否则 graph-fused 路径会明确报缺图。

```bash
uv run python scripts/compress_openvino_weights.py \
  --ir-dir openvino/voice_design \
  --preset fastest-fused-seed-selective

uv run python scripts/verify_codegen_fusion_correctness.py \
  --ir-dir openvino/voice_design \
  --graph-variant-split int8_sym_paged_talker_split \
  --graph-variant-graph int8_sym_paged_fused_seed_selective \
  --max-new-tokens 48 \
  --trace-frames 16

uv run python scripts/benchmark_streaming_realtime.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profiles fastest,graph_fused_fp16,graph_fused_int8_selective,graph_fused_int8_full \
  --runs 3
```

`verify_codegen_fusion_correctness.py` 默认会先跑 `fp16 split` vs `fp16 graph`
作为结构基线，再跑目标 variant。`classification=passed` 才能用于性能对比；
若显示 `quantization_mismatch`，说明图结构正确但目标量化 variant 改变了 codec 序列。

如果导出了 `subcode_greedy_cached_next_embed.xml`，可以单独测试把
`sum_embed + tts_pad_embed` 下沉到 standalone subcode 图内：

```bash
uv run python scripts/benchmark_streaming_realtime.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profiles fastest,split_next_embed_graph \
  --runs 3
```

该路径默认不替换 `fastest`。只有 codes/质量一致且 RTF 稳定改善后，才应把
`QWEN3_TTS_OV_NATIVE_SUBCODE_NEXT_EMBED_GRAPH=1` 用于生产 profile。

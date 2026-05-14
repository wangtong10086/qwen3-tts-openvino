# 流式合成与长文本

本页只说明流式协议、浏览器播放策略和长文本 full-AR 策略。具体 CLI、Python API、HTTP 和 WebSocket 调用方式见 [运行接口](runtime_zh.md)。

## 当前生产路径

短文本和长文本共用 `fastest` profile：

- native C++ pipeline
- paged-KV seed graph
- cached standalone subcode graph
- streaming decoder graph
- `int8_sym_paged_talker_split` graph variant

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

## 长文本 prompt 预算

服务端默认使用 `--max-continuous-prompt-tokens auto` 做推理前保护，避免极长 prompt 在部分 GPU/驱动上触发 OpenVINO USM 分配失败。`auto` 的生效值为：

- GPU 或默认 release 路径：`2048`
- CPU-only：`4096`

如果 metadata 或错误信息显示文本超过预算，可以启动时提高上限：

```bash
qwen3-tts-ov-server --device GPU --max-continuous-prompt-tokens 4096
```

也可以设置为 `0` 关闭 prompt 预算保护。关闭后仍不会自动分段，后续如果显存不足会由 OpenVINO/USM 错误和 retry 机制处理。

显存压力主要来自长 prompt 的 paged-KV cache 和运行时中间 buffer。默认生产路径使用 `u8` KV cache；如果需要显式指定或配合更低 prompt 预算，可以启动时使用：

```bash
qwen3-tts-ov-server --device GPU --kv-cache-profile u8 --max-vram-ratio 70
```

`u8` 会把 KV cache 存储元素从 FP16 的 2 bytes 降为 1 byte，`/health` 中 `kv_cache_relative_to_fp16` 应显示为 `0.5`。需要保守对照时使用 `--kv-cache-profile fp16`；切换到 `u8-all` 前需要重新做长文本质量评测。

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

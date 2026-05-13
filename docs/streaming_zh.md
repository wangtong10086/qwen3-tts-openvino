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

metadata 会包含 `realtime_profile`、`chunk_strategy`、`recommended_playback_buffer_ms`、`graph_variant`、`paged_kv`、`continuous_long_output` 等运行状态。前端必须以 final JSON 作为本次合成结束信号，不要用连接关闭或最后一个音频块判断结束。

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

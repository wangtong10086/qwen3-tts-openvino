# 流式与长文本

当前实现只保留生产流式路径：完整上下文自回归生成 codec token，音频 decoder 按 chunk 输出给播放器。输入文本不会被自动切段。

## 流式输出

默认输出：

- `pcm_s16le`
- mono
- 24 kHz
- WebSocket binary audio chunk
- final JSON 包含总耗时、RTF、online batching 统计和 fallback counter

推荐参数：

```json
{
  "stream": {
    "chunk_strategy": "smooth",
    "initial_chunk_frames": 8,
    "chunk_frames": 24,
    "left_context_frames": 25,
    "format": "pcm_s16le"
  }
}
```

服务端会根据短文本/长文本和显存预算调整生成上限，但不会通过切分文本绕过上下文限制。

## 长文本策略

长文本路径必须满足：

- `full_context_text=true`
- `segmented=false`
- `auto_segment_text=false`
- 生成直到 EOS 或达到显存/上下文预算

Web Demo 会实时显示 tokenizer 统计、上下文使用量、KV cache 预算和已生成 token。用户看到的最大 token 不是“语音必然长度”，只是当前模型上下文和显存允许的上限。

## Online Batching

sidecar 默认启用 native online batching：

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest \
  --online-batching on \
  --online-batch-scheduler layered \
  --online-batch-max-num-batched-tokens 32
```

调度层采用 vLLM-like 分层：

- prefill 和 decode 分开调度。
- decode 使用 active batch bucket。
- `max_num_batched_tokens` 控制每轮 decode token budget。
- 并发大于 1 时关注 aggregate TPS/RTF；不要求每个请求单路 RTF 都小于 1。

## 质量与性能 Gate

三模式统一 gate：

```bash
uv run python scripts/evaluate_single_arch_gate.py \
  --server-url http://127.0.0.1:17860 \
  --modes voice_design,custom_voice,voice_clone \
  --runs 3 \
  --concurrency 1,2,4,8 \
  --require-omni
```

性能矩阵：

```bash
uv run python scripts/benchmark_prompt_batch_matrix.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profile-set baseline \
  --batch-sizes 1,2,4,8,16 \
  --prompt-lengths short,medium,long,xlong \
  --scenarios offline,online \
  --runs 3
```

验收重点：

- 并发 1 下三模式 RTF 达标。
- 长文本仍为 full-AR。
- metadata 中无 generation fallback、无自动分段。
- online batching 命中 `scheduler=layered`。
- Omni gate 通过。

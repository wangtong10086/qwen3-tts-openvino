# 流式合成

生产流式路径固定为 `fastest` profile：native C++ pipeline 负责 codec autoregressive codegen 和 streaming decoder，Python 侧只负责请求编排、prompt 构造和 API 输出。

## Python Iterator

```python
from qwen3_tts_ov import OpenVINOQwen3TTS

tts = OpenVINOQwen3TTS.from_ir("openvino/voice_design", device="GPU")
for chunk in tts.stream_voice_design(
    text="你好，这是 Python 流式 API 测试。",
    instruct="A calm young female voice.",
    language="Chinese",
):
    if chunk.audio.size:
        print(chunk.index, chunk.audio.shape, chunk.timings["stream_rtf"])
```

`from_ir()` 默认使用 `fastest`。缺少 native 库或必需 graph 时会直接报错。

## CLI

```bash
uv run python -m qwen3_tts_ov stream voice-design \
  --ir-dir openvino/voice_design \
  --device GPU \
  --realtime-profile fastest \
  --text "你好，这是一次流式 OpenVINO 合成测试。" \
  --instruct "A calm young female voice." \
  --language Chinese \
  --chunk-dir outputs/stream \
  --output outputs/stream.wav
```

CustomVoice 和 VoiceClone 使用同一 profile，只需替换子命令和对应参数。

## Sidecar

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest \
  --preload-modes voice_design \
  --preload-buckets warmup \
  --host 127.0.0.1 \
  --port 17860
```

浏览器测试页：

```text
http://127.0.0.1:17860/
```

## WebSocket 协议

连接 `/v1/tts/stream` 后先发送请求 JSON：

```json
{
  "mode": "voice_design",
  "text": "你好，这是 WebSocket 流式合成。",
  "language": "Chinese",
  "instruct": "A calm young female voice."
}
```

服务端发送顺序：

1. metadata JSON
2. 多个 binary `pcm_s16le` 音频块
3. final JSON

metadata 中包含 `realtime_profile=fastest`、`chunk_strategy=smooth`、`recommended_playback_buffer_ms` 和当前 graph/native 状态。

## HTTP / OpenAI-compatible API

NDJSON：

```bash
curl -N http://127.0.0.1:17860/v1/tts/stream \
  -H "content-type: application/json" \
  -d '{"mode":"voice_design","text":"你好，这是 HTTP 流式合成。","language":"Chinese","instruct":"A calm young female voice."}'
```

OpenAI-compatible streaming PCM：

```bash
curl -N http://127.0.0.1:17860/v1/audio/speech \
  -H "content-type: application/json" \
  -d '{"model":"qwen3-tts-openvino","voice":"default","input":"你好，这是兼容 OpenAI Speech API 的流式 PCM。","language":"Chinese","task_type":"voice_design","instructions":"A calm young female voice.","stream":true,"response_format":"pcm"}' \
  --output speech.pcm
```

## 性能指标

每个音频块的 `timings` 会包含：

- `stream_rtf`
- `stream_compute_rtf`
- `decode_path`
- `native_audio_pipeline`
- `native_timing`
- `selected_bucket`
- `active_codegen_unroll`

生产验收使用：

```bash
uv run python scripts/benchmark_streaming_realtime.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profile-set fastest-gate \
  --runs 3 \
  --warmup-generations 1
```

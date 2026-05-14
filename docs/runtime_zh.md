# 运行接口

本页汇总当前生产接口。默认使用 `fastest` profile。

## CLI

VoiceDesign 流式：

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

CustomVoice：

```bash
uv run python -m qwen3_tts_ov stream custom-voice \
  --ir-dir openvino/custom_voice \
  --device GPU \
  --realtime-profile fastest \
  --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" \
  --speaker Vivian \
  --instruct "用特别愤怒的语气说" \
  --language Chinese \
  --output outputs/custom_voice.wav
```

VoiceClone：

```bash
uv run python -m qwen3_tts_ov stream voice-clone \
  --ir-dir openvino/base \
  --device GPU \
  --realtime-profile fastest \
  --text "I am solving the equation, but it is a disaster." \
  --language English \
  --ref-audio /path/to/reference.wav \
  --ref-text "Reference transcript for the audio." \
  --output outputs/voice_clone.wav
```

Batch JSONL：

```bash
uv run python -m qwen3_tts_ov batch \
  --ir-dir openvino/voice_design \
  --realtime-profile fastest \
  --batch-jsonl examples/requests.example.jsonl \
  --output-dir outputs/batch
```

## Python API

```python
from qwen3_tts_ov import OpenVINOQwen3TTS

tts = OpenVINOQwen3TTS.from_ir("openvino/voice_design", device="GPU")

for chunk in tts.stream_voice_design(
    text="你好，这是 Python 流式 API 测试。",
    instruct="A calm young female voice.",
    language="Chinese",
):
    if chunk.audio.size:
        print(chunk.index, chunk.audio.shape, chunk.is_final)
```

`from_ir()` 默认使用 `fastest`。低层实验参数请直接调用构造函数或使用 `devtools/`。

## Sidecar

启动：

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest \
  --max-continuous-prompt-tokens auto \
  --host 127.0.0.1 \
  --port 17860
```

健康检查：

```bash
curl http://127.0.0.1:17860/health
```

Web Demo：

```text
http://127.0.0.1:17860/
```

## WebSocket

Endpoint:

```text
ws://127.0.0.1:17860/v1/tts/stream
```

请求示例：

```json
{
  "mode": "voice_design",
  "text": "你好，这是 WebSocket 流式合成。",
  "language": "Chinese",
  "instruct": "A calm young female voice.",
  "generation": {
    "max_new_tokens": 48
  },
  "stream": {
    "chunk_strategy": "smooth",
    "format": "pcm_s16le"
  }
}
```

返回顺序：

1. metadata JSON
2. 多个 binary `pcm_s16le` 音频块
3. final JSON

## HTTP NDJSON

```bash
curl -N http://127.0.0.1:17860/v1/tts/stream \
  -H "content-type: application/json" \
  -d @examples/stream_request.example.json
```

HTTP NDJSON 会把音频块用 base64 放在 JSON 行中，适合不方便处理 WebSocket binary frame 的客户端。

## OpenAI-Compatible Speech API

流式 PCM：

```bash
curl -N http://127.0.0.1:17860/v1/audio/speech \
  -H "content-type: application/json" \
  -d @examples/openai_speech_request.example.json \
  --output outputs/openai_speech.pcm
```

非流式 WAV：

```bash
curl http://127.0.0.1:17860/v1/audio/speech \
  -H "content-type: application/json" \
  -d '{"model":"qwen3-tts-openvino","voice":"default","input":"你好，这是一次 WAV 输出测试。","language":"Chinese","task_type":"voice_design","instructions":"A calm young female voice.","stream":false,"response_format":"wav"}' \
  --output outputs/speech.wav
```

## 长文本

长文本仍使用同一个 prompt 做完整自回归。服务端可以分块发送音频，但不会默认把文本切成多段独立生成。详细策略见 [streaming_zh.md](streaming_zh.md)。

`--max-continuous-prompt-tokens` 默认值为 `auto`：GPU 路径生效为 `2048`，CPU-only 生效为 `4096`。更长输入可启动时改为具体数值，或设为 `0` 关闭推理前 prompt 预算保护。

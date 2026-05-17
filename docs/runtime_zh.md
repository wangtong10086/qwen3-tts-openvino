# 运行接口

当前生产路径固定为 `fastest` profile：native C++ codec generation、OpenVINO paged-KV、vLLM-like online batching、长文本 full autoregressive。

## Sidecar

推荐所有桌面应用和 Web 应用通过本地 HTTP/WebSocket sidecar 集成：

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest \
  --runtime-residency lazy \
  --host 127.0.0.1 \
  --port 17860
```

默认行为：

- `online_batching=on`
- `online_batch_scheduler=layered`
- `runtime_residency=lazy`
- `kv_cache_profile=auto`，当前等价于 U8 paged-KV
- 长文本 `full_context_text=true`，不分段生成

健康检查：

```bash
curl http://127.0.0.1:17860/health
```

`/health` 会报告模型状态、warmup 状态、KV cache 精度、预分配块数、最大 token 预算、online batching 状态和当前设备。

## Web Demo

打开 `/` 或 `/web` 可以进入本地控制台。当前 demo 覆盖：

- VoiceDesign、CustomVoice、VoiceClone 三种模式。
- VoiceClone 参考音频上传、路径/URL 和 `ref_text`。
- 长文本 full-AR token 预算、上下文使用率和显存占比调节。
- WebSocket 流式播放、WAV 下载、请求 JSON/curl 复制。
- `/v1/audio/speech` OpenAI-compatible 请求预览。
- 自定义最终请求 JSON，用于完全客制化调试。
- `同时请求数`，用于快速观察 online batching 是否工作。

多请求 smoke 只用于交互式排查，正式性能数据仍使用 `scripts/benchmark_prompt_batch_matrix.py`。

## Endpoint 总览

| Endpoint | 方法 | 用途 |
| --- | --- | --- |
| `/`、`/web` | GET | Web Demo |
| `/health` | GET | 服务状态、模型可用性、warmup、KV/cache 预算 |
| `/v1/models` | GET | 查看 VoiceDesign、CustomVoice、VoiceClone IR 是否就绪和下载状态 |
| `/v1/models/download` | POST | 触发指定模式 IR 下载 |
| `/v1/tts/tokenize` | POST | 计算 prompt token 和上下文预算，用于 Web Demo 实时提示 |
| `/v1/tts` | POST | 返回整段 WAV |
| `/v1/tts/stream` | POST | HTTP NDJSON 流式返回 metadata/audio/final |
| `/v1/tts/stream` | WebSocket | 先发请求 JSON，再接收 metadata、PCM binary chunk、final |
| `/v1/audio/voices` | GET | 返回 CustomVoice speaker 列表和模式可用性 |
| `/v1/audio/speech` | POST | OpenAI-compatible Speech API |

模型状态与一键下载：

```bash
curl http://127.0.0.1:17860/v1/models

curl -X POST http://127.0.0.1:17860/v1/models/download \
  -H 'Content-Type: application/json' \
  -d '{"mode":"voice_clone","sync":false}'
```

## WebSocket Streaming

Endpoint:

```text
ws://127.0.0.1:17860/v1/tts/stream
```

VoiceDesign：

```json
{
  "mode": "voice_design",
  "text": "你好，这是 WebSocket 流式合成。",
  "language": "Chinese",
  "instruct": "A calm young female voice.",
  "generation": {
    "max_new_tokens": 128
  },
  "stream": {
    "chunk_strategy": "smooth",
    "format": "pcm_s16le"
  }
}
```

CustomVoice：

```json
{
  "mode": "custom_voice",
  "text": "这是一段自定义说话人测试。",
  "language": "Chinese",
  "speaker": "Vivian",
  "instruct": "自然朗读。",
  "generation": {
    "max_new_tokens": 128
  }
}
```

VoiceClone：

```json
{
  "mode": "voice_clone",
  "text": "This is generated with the reference voice.",
  "language": "English",
  "ref_audio": "/path/to/reference.wav",
  "ref_text": "Reference transcript for the audio.",
  "x_vector_only": false,
  "generation": {
    "max_new_tokens": 128
  }
}
```

VoiceClone 使用 Base IR，不能用 VoiceDesign IR 代替。默认 `x_vector_only=false`，即使用参考音频 codec prompt 和 speaker embedding。

## HTTP

整段 WAV：

```bash
curl -X POST http://127.0.0.1:17860/v1/tts \
  -H 'Content-Type: application/json' \
  -o out.wav \
  -d '{"mode":"voice_design","text":"你好。","language":"Chinese","instruct":"自然朗读。"}'
```

OpenAI-compatible Speech API：

```bash
curl -X POST http://127.0.0.1:17860/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o out.wav \
  -d '{"model":"qwen3-tts-openvino","voice":"voice_design","input":"你好。","language":"Chinese","instructions":"自然朗读。"}'
```

Python examples:

```bash
python examples/python/http_tts_wav.py --output outputs/example_http.wav
python examples/python/openai_speech.py --output outputs/example_openai.pcm
uv run --with websockets python examples/python/websocket_stream_pcm.py --output outputs/example_ws.wav
```

## CLI

CLI 保留用于开发验证和批处理。生产集成优先使用 sidecar。

```bash
uv run python -m qwen3_tts_ov stream voice-design \
  --ir-dir openvino/voice_design \
  --device GPU \
  --realtime-profile fastest \
  --text "你好，这是一次流式 OpenVINO 合成测试。" \
  --instruct "A calm young female voice." \
  --language Chinese \
  --output outputs/voice_design.wav
```

批处理：

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

## 性能验证

不同 batch、prompt 长度、在线/离线到达场景：

```bash
uv run python scripts/benchmark_prompt_batch_matrix.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profile-set baseline \
  --batch-sizes 1,2,4,8,16 \
  --prompt-lengths short,medium,long,xlong \
  --scenarios offline,online \
  --runs 3 \
  --max-new-tokens 96 \
  --output outputs/online_batch/prompt_batch_matrix.json
```

发布前 gate：

```bash
uv run python scripts/evaluate_single_arch_gate.py \
  --server-url http://127.0.0.1:17860 \
  --modes voice_design,custom_voice,voice_clone \
  --runs 3 \
  --concurrency 1,2,4,8
```

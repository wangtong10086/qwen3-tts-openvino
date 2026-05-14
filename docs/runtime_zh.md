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

注意：VoiceClone 不能使用 VoiceDesign IR 代替。它依赖 Base/VoiceClone 导出的参考音频 encoder、speaker encoder、speech tokenizer decoder 和主自回归图，服务端需要能找到 `openvino/base/manifest.json`。公开 Hugging Face IR 已提供 `openvino_realtime/base`，Web Demo 可在缺失时直接下载。

VoiceClone 默认是 ICL 克隆：`x_vector_only=false`，需要同时提供 `--ref-audio` 和准确的 `--ref-text`。这条路径会使用参考音频 codec prompt，更适合保留参考语音的音色、语气和韵律。只有需要做 speaker embedding-only 对照实验时，才显式添加 `--x-vector-only`。

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

默认 `fastest` 路径会启用 U8 paged-KV cache。需要显式确认或降低长文本 token 预算使用的显存比例时：

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest \
  --kv-cache-profile u8 \
  --max-vram-ratio 70
```

`/health` 中的 `warmup.kv_cache_profile`、`memory.native_paged_kv_precision`、`memory.kv_cache_relative_to_fp16`、`memory.effective_max_total_tokens` 和 `memory.preallocated_kv_blocks` 会显示实际生效配置。

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

VoiceClone 请求示例：

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
  },
  "stream": {
    "chunk_strategy": "smooth",
    "format": "pcm_s16le"
  }
}
```

`x_vector_only` 省略时也按 `false` 处理。Web Demo 会把历史保存过的旧设置迁移回默认关闭，避免误跳过参考音频 codec prompt。

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

`--max-continuous-prompt-tokens` 默认值为 `auto`。GPU 路径会使用 KV-cache planner，根据 GPU 总显存、`--max-vram-ratio`、KV/cache-input 精度、block size、模型层数和上下文长度计算 `effective_max_total_tokens`。普通长输出会按请求的 `max_new_tokens` 预留输出空间；full-context 长文本不会按文本长度预估语音 token，而是在精确 tokenizer 后使用 `effective_max_total_tokens - prompt_len - 1` 作为运行时 `max_new_tokens`，生成直到 EOS 或上下文/KV 上限。CPU-only 路径仍使用保守固定预算。

因此 full-context 路径不再依赖 `字数 -> 语音 token` 的估算公式；前端显示的“运行上限”是当前 prompt 后真实可生成的剩余上下文容量。

相关参数：

- `--max-vram-ratio auto|N`: planner 可使用的最大显存比例，例如 `70` 表示 70%。
- `--kv-cache-reserve-mb auto|N`: 为模型权重、中间 buffer 和驱动保留的显存；默认按 GPU 显存自动保留。
- `--kv-cache-max-blocks auto|N`: 手动限制 KV block 数，用于调试或规避特定驱动上的 USM 压力。
- `--kv-cache-preallocation auto|off|static`: `auto` 只计算预算；`static` 同时把计划出的 block 数传给 native static decode 路径。

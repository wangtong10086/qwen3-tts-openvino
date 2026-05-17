# API Reference

This page documents the public sidecar HTTP, WebSocket, OpenAI-compatible, and
Python APIs. Runtime startup notes are in [Runtime APIs](runtime_zh.md), and
common failures are in [Troubleshooting](troubleshooting.md).

Default service URL:

```text
http://127.0.0.1:17860
```

## Common TTS Request

`/v1/tts`, `/v1/tts/stream`, and WebSocket `/v1/tts/stream` share the same
request shape.

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `mode` | string | yes | none | `voice_design`, `custom_voice`, or `voice_clone`. Hyphenated spellings are normalized. |
| `text` | string | yes | none | Text to synthesize. |
| `language` | string | no | `Auto` | Supported keys come from the IR manifest. Common values are `Auto`, `Chinese`, and `English`. |
| `instruct` | string | common for VoiceDesign | `""` | Style or reading instruction for VoiceDesign/CustomVoice. Ignored by VoiceClone. |
| `speaker` | string | CustomVoice yes | none | Query `GET /v1/audio/voices` first. |
| `ref_audio` | string | VoiceClone yes | none | Local path, HTTP(S) URL, base64, or `data:audio/...`. |
| `ref_text` | string | VoiceClone default path yes | none | Required when `x_vector_only=false`; should be an accurate reference transcript. |
| `x_vector_only` | bool | no | `false` | `false` uses ICL cloning with `ref_audio + ref_text`; `true` is speaker-embedding-only comparison. |
| `generation` | object | no | `{}` | Generation fields below. Same names may also be top-level. |
| `stream` | object | no | `{}` | Streaming fields below. Chunk fields may also be top-level. |
| `full_context_text` | bool | no | `false` | Used by the Web Demo long-text full-AR path. |
| `allow_auto_segment_text` / `auto_segment_text` | bool | no | `false` | Debug fallback only; production long text is full-AR. |

### `generation`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `max_new_tokens` | int | `512` | Maximum codec token count. For short interactive requests, set `48` or `128` explicitly. |
| `min_new_tokens` | int | `2` | Minimum token count. |
| `do_sample` | bool | policy-driven | Default is resolved from policy and long-text state. VoiceClone `x_vector_only=true` defaults to no sampling. |
| `top_k` | int | `50` | Sampling top-k. |
| `top_p` | float | `1.0` | Sampling top-p. |
| `temperature` | float | `0.9` | Sampling temperature. |
| `repetition_penalty` | float | usually `1.0` in `fastest` | Long text or CustomVoice may raise the automatic default; explicit values win. |
| `max_prompt_tokens` | int | `512` | Prompt protection. Long text is also constrained by KV planner settings. |
| `progress_interval` | int | `0` in sidecar | Mostly for development diagnostics. |
| `seed` | int | `0` | Used by sampled generation paths. |

### `stream`

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `format` | string | `pcm_s16le` | The only supported streaming format today. |
| `chunk_strategy` | string | `smooth` in the `fastest` sidecar | Supports `auto`, `short_compute`, `realtime`, `low_latency`, `smooth`, `balanced`, `stable`. |
| `initial_chunk_frames` | int | strategy default | First audio chunk codec frames; must be positive. |
| `chunk_frames` | int | strategy default | Later audio chunk codec frames; must be positive. |
| `left_context_frames` | int | strategy default | Streaming decoder left context; must be non-negative. |
| `include_chunk_metadata` | bool | `false` | WebSocket only: send a JSON metadata message before each binary chunk. |

Strategy defaults:

| Strategy | `initial_chunk_frames` | `chunk_frames` | `left_context_frames` |
| --- | ---: | ---: | ---: |
| `short_compute` | 12 | 24 | 25 |
| `realtime` | 8 | 12 | 25 |
| `low_latency` | 8 | 12 | 25 |
| `smooth` | 8 | 24 | 25 |
| `balanced` | 12 | 12 | 25 |
| `stable` | 12 | 24 | 25 |

`auto` resolves to `short_compute` or `stable` based on text length and
full-context state.

## Endpoints

| Endpoint | Method | Request | Response |
| --- | --- | --- | --- |
| `/health` | GET | none | Service, warmup, device, KV/cache budget, and online batching status. |
| `/v1/models` | GET | none | Per-mode IR availability and download status. |
| `/v1/models/download` | POST | `{"mode":"voice_clone","sync":false}` | Starts or runs a mode download; `sync`/`wait` waits for completion. |
| `/v1/tts/tokenize` | POST | Common TTS request | Token and prompt budget metadata. |
| `/v1/tts` | POST | Common TTS request | Complete `audio/wav`. |
| `/v1/tts/stream` | POST | Common TTS request | `application/x-ndjson` events: `metadata`, `audio`, `final`, or `error`. |
| `/v1/tts/stream` | WebSocket | Send Common TTS JSON after connect | `metadata` JSON, binary PCM chunks, then `final` JSON. |
| `/v1/audio/voices` | GET | none | CustomVoice speakers and mode availability. |
| `/v1/audio/speech` | POST | OpenAI-compatible request | WAV, raw PCM, or streaming PCM. |

## HTTP Examples

Complete WAV:

```bash
curl -X POST http://127.0.0.1:17860/v1/tts \
  -H 'Content-Type: application/json' \
  -o out.wav \
  -d '{"mode":"voice_design","text":"Hello.","language":"English","instruct":"A clear narrator voice.","generation":{"max_new_tokens":48}}'
```

HTTP NDJSON stream:

```bash
curl -N http://127.0.0.1:17860/v1/tts/stream \
  -H 'Content-Type: application/json' \
  -d @examples/stream_request.example.json
```

NDJSON events:

- `metadata`: sample rate, chunk strategy, token/budget metadata, online batching, recommended playback buffer.
- `audio`: base64 `pcm_s16le` with `index`, `sample_rate`, and `timings`.
- `final`: final index, elapsed time, and final timings.
- `error`: error message.

## WebSocket Example

Connect:

```text
ws://127.0.0.1:17860/v1/tts/stream
```

Send:

```json
{
  "mode": "voice_design",
  "text": "Hello from WebSocket streaming.",
  "language": "English",
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

Receive:

1. `metadata` JSON.
2. One or more binary PCM chunks.
3. `final` JSON.

With `stream.include_chunk_metadata=true`, the server sends an `audio` JSON
message before each binary chunk.

## OpenAI-Compatible Speech API

Endpoint:

```text
POST /v1/audio/speech
```

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `model` | string | none | Compatibility field; mode is mainly selected by `task_type`, `voice`, and reference fields. |
| `input` | string | required | Text to synthesize. |
| `voice` | string | `default` | Used as CustomVoice speaker when `task_type=custom_voice`; a non-default voice can imply CustomVoice. |
| `task_type` / `mode` | string | inferred | `voice_design`, `custom_voice`, `voice_clone`, or `base`. |
| `instructions` / `instruct` | string | `""` | VoiceDesign/CustomVoice instruction. |
| `language` | string | `Auto` | Language. |
| `response_format` | string | `wav` | Supports `wav`, `pcm`, `pcm_s16le`. |
| `stream` | bool or object | `false` | Enables streaming when true/object. Streaming requires `response_format=pcm`. |
| `ref_audio` / `ref_text` / `x_vector_only` | mixed | none | VoiceClone fields. |
| `generation` or top-level generation fields | object | `{}` | Same as common `generation`. |

Example:

```bash
curl -X POST http://127.0.0.1:17860/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o out.wav \
  -d '{"model":"qwen3-tts-openvino","voice":"voice_design","input":"Hello.","language":"English","instructions":"Natural reading."}'
```

## VoiceClone Reference Audio

Default production request:

```json
{
  "mode": "voice_clone",
  "text": "This is generated with the reference voice.",
  "language": "English",
  "ref_audio": "/path/to/reference.wav",
  "ref_text": "Reference transcript for the audio.",
  "x_vector_only": false
}
```

Notes:

- VoiceClone uses Base IR: `base/manifest.json`.
- `ref_text` is required when `x_vector_only=false`.
- `ref_audio` can be a local path, HTTP(S) URL, base64, or `data:audio/...`.
- Runtime converts to mono and resamples as needed. Use clear, single-speaker,
  low-noise audio; 3 to 10 seconds is a good starting point.

## Python API

```python
from qwen3_tts_ov import OpenVINOQwen3TTS

tts = OpenVINOQwen3TTS.from_ir(
    "openvino/voice_design",
    device="GPU",
    realtime_profile="fastest",
)

for chunk in tts.stream_voice_design(
    text="Hello from the Python streaming API.",
    instruct="A calm young female voice.",
    language="English",
    max_new_tokens=48,
    chunk_strategy="smooth",
):
    if chunk.audio.size:
        print(chunk.index, chunk.audio.shape, chunk.sample_rate, chunk.is_final)
```

Common constructor arguments:

| Argument | Default | Notes |
| --- | --- | --- |
| `ir_dir` | required | Single-mode IR directory, for example `openvino/voice_design`. |
| `device` | required | OpenVINO device such as `GPU`, `CPU`, or `AUTO`. |
| `realtime_profile` | `fastest` | `from_ir` supports `fastest`; lower-level kwargs are for development experiments. |
| `decoder_device` | `device` | Optional separate audio decoder device. Prefer sidecar flags for NPU paths. |
| `ov_cache_dir` / `disable_ov_cache` | automatic | OpenVINO compile cache controls. |

Main methods:

- `stream_voice_design(text, instruct, language="Auto", ...) -> Iterator[StreamChunk]`
- `stream_custom_voice(text, speaker, language="Auto", instruct="", ...) -> Iterator[StreamChunk]`
- `stream_voice_clone(text, language="Auto", ref_audio=..., ref_text=..., x_vector_only_mode=False, ...) -> Iterator[StreamChunk]`
- `generate_voice_design(...) -> (list[np.ndarray], sample_rate)`
- `generate_custom_voice(...) -> (list[np.ndarray], sample_rate)`
- `generate_voice_clone(...) -> (list[np.ndarray], sample_rate)`
- `create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode=False)` caches reusable reference prompts.

`StreamChunk` fields:

| Field | Type | Notes |
| --- | --- | --- |
| `index` | int | Chunk index. |
| `audio` | `np.ndarray` | float32 mono audio. |
| `sample_rate` | int | Usually 24000. |
| `codes` | `np.ndarray` | Codec tokens/code frames. |
| `is_final` | bool | Whether this is the final chunk. |
| `timings` | dict | Decode, RTF, context, and online batching metadata. |
| `pcm_s16le` | bytes or None | Some paths may carry ready-to-send PCM bytes. |


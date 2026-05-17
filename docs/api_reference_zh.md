# API Reference

本页描述 sidecar 的公开 HTTP、WebSocket、OpenAI-compatible 和 Python 调用接口。运行和启动说明见 [运行接口](runtime_zh.md)，排障见 [Troubleshooting](troubleshooting_zh.md)。

默认服务地址：

```text
http://127.0.0.1:17860
```

## 通用 TTS 请求

`/v1/tts`、`/v1/tts/stream` 和 WebSocket `/v1/tts/stream` 使用同一套请求结构。

| 字段 | 类型 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `mode` | string | 是 | 无 | `voice_design`、`custom_voice`、`voice_clone`。连字符写法也会归一化。 |
| `text` | string | 是 | 无 | 要合成的文本。 |
| `language` | string | 否 | `Auto` | 语言由 IR manifest 决定；常用 `Auto`、`Chinese`、`English`。不支持值会在运行时报错并列出 manifest 中的语言键。 |
| `instruct` | string | VoiceDesign 常用 | `""` | VoiceDesign/CustomVoice 的风格或朗读指令。VoiceClone 忽略该字段。 |
| `speaker` | string | CustomVoice 是 | 无 | CustomVoice 说话人。先调用 `GET /v1/audio/voices` 获取列表。 |
| `ref_audio` | string | VoiceClone 是 | 无 | 本地路径、HTTP(S) URL、base64 或 `data:audio/...`。 |
| `ref_text` | string | VoiceClone 默认路径是 | 无 | 默认 `x_vector_only=false` 时必填，应为参考音频准确转写。 |
| `x_vector_only` | bool | 否 | `false` | `false` 使用 `ref_audio + ref_text` ICL 克隆路径；`true` 只做 speaker embedding-only 对照。 |
| `generation` | object | 否 | `{}` | 生成参数，见下表。也可把同名字段放在顶层。 |
| `stream` | object | 否 | `{}` | 流式参数，见下表。也可把同名 chunk 字段放在顶层。 |
| `full_context_text` | bool | 否 | `false` | Web Demo 长文本 full-AR 路径使用；开启后不自动分段。 |
| `allow_auto_segment_text` / `auto_segment_text` | bool | 否 | `false` | 只作为调试 fallback；生产长文本默认 full-AR。 |

### `generation`

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `max_new_tokens` | int | `512` | 最大 codec token 数。短交互建议显式设为 `48` 或 `128`；长文本可由 full-context 预算调整。 |
| `min_new_tokens` | int | `2` | 最少生成 token。 |
| `do_sample` | bool | 策略决定 | 未显式设置时由默认策略和长文本状态决定；`x_vector_only=true` 的 VoiceClone 默认关闭采样。 |
| `top_k` | int | `50` | 采样 top-k。 |
| `top_p` | float | `1.0` | 采样 top-p。 |
| `temperature` | float | `0.9` | 采样温度。 |
| `repetition_penalty` | float | `fastest` 通常 `1.0` | 长文本或 CustomVoice 可能提高默认值；显式传入会覆盖策略。 |
| `max_prompt_tokens` | int | `512` | prompt token 保护值。长文本还会受 `--max-continuous-prompt-tokens` 和 KV planner 影响。 |
| `progress_interval` | int | `0` in sidecar | 运行时进度间隔，主要用于开发调试。 |
| `seed` | int | `0` | 生成种子；当前主要用于采样路径。 |

### `stream`

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `format` | string | `pcm_s16le` | 当前唯一支持值。 |
| `chunk_strategy` | string | `smooth` in `fastest` sidecar | 支持 `auto`、`short_compute`、`realtime`、`low_latency`、`smooth`、`balanced`、`stable`。 |
| `initial_chunk_frames` | int | 策略决定 | 首个音频 chunk 的 codec frame 数，必须大于 0。 |
| `chunk_frames` | int | 策略决定 | 后续 chunk 的 codec frame 数，必须大于 0。 |
| `left_context_frames` | int | 策略决定 | 流式 decoder 左上下文 frame 数，必须非负。 |
| `include_chunk_metadata` | bool | `false` | WebSocket 下为每个 binary PCM chunk 前额外发送 JSON metadata。 |

策略默认值：

| 策略 | `initial_chunk_frames` | `chunk_frames` | `left_context_frames` |
| --- | ---: | ---: | ---: |
| `short_compute` | 12 | 24 | 25 |
| `realtime` | 8 | 12 | 25 |
| `low_latency` | 8 | 12 | 25 |
| `smooth` | 8 | 24 | 25 |
| `balanced` | 12 | 12 | 25 |
| `stable` | 12 | 24 | 25 |

`auto` 会按文本长度和 full-context 状态解析为 `short_compute` 或 `stable`。

## Endpoints

| Endpoint | 方法 | 请求 | 响应 |
| --- | --- | --- | --- |
| `/health` | GET | 无 | 服务、warmup、设备、KV/cache 预算、online batching 状态。 |
| `/v1/models` | GET | 无 | 三种模式 IR 是否可用、下载状态。 |
| `/v1/models/download` | POST | `{"mode":"voice_clone","sync":false}` | 触发指定模式下载；`sync`/`wait` 为 `true` 时同步等待。 |
| `/v1/tts/tokenize` | POST | 通用 TTS 请求 | token 预算、prompt 估算、长文本预算。 |
| `/v1/tts` | POST | 通用 TTS 请求 | 完整 `audio/wav`。 |
| `/v1/tts/stream` | POST | 通用 TTS 请求 | `application/x-ndjson`，逐行返回 `metadata`、`audio`、`final` 或 `error`。 |
| `/v1/tts/stream` | WebSocket | 连接后先发送通用 TTS JSON | 先收 `metadata` JSON，再收 PCM binary chunk，最后收 `final` JSON。 |
| `/v1/audio/voices` | GET | 无 | CustomVoice speaker 列表和模式可用性。 |
| `/v1/audio/speech` | POST | OpenAI-compatible 请求 | `audio/wav`、`audio/L16` 或 streaming PCM。 |

## HTTP 示例

完整 WAV：

```bash
curl -X POST http://127.0.0.1:17860/v1/tts \
  -H 'Content-Type: application/json' \
  -o out.wav \
  -d '{"mode":"voice_design","text":"你好。","language":"Chinese","instruct":"自然朗读。","generation":{"max_new_tokens":48}}'
```

HTTP NDJSON 流：

```bash
curl -N http://127.0.0.1:17860/v1/tts/stream \
  -H 'Content-Type: application/json' \
  -d @examples/stream_request.example.json
```

NDJSON 事件：

- `metadata`: 采样率、流式策略、token/预算、online batching、推荐播放缓冲。
- `audio`: base64 编码的 `pcm_s16le`，包含 `index`、`sample_rate`、`timings`。
- `final`: 最终 index、耗时、最后 chunk timings。
- `error`: 错误消息。

## WebSocket 示例

连接：

```text
ws://127.0.0.1:17860/v1/tts/stream
```

发送：

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

接收顺序：

1. `metadata` JSON。
2. 一个或多个 binary PCM chunk。
3. `final` JSON。

如果请求中设置 `stream.include_chunk_metadata=true`，每个 binary chunk 前会多一个 `audio` JSON，包含 byte length、index 和 timings。

## OpenAI-Compatible Speech API

Endpoint:

```text
POST /v1/audio/speech
```

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `model` | string | 无 | 用于兼容；模式主要由 `task_type`、`voice`、`ref_audio` 推断。 |
| `input` | string | 必填 | 要合成的文本。 |
| `voice` | string | `default` | `task_type=custom_voice` 时作为 speaker；非 `default` 且未指定 task 时会推断为 CustomVoice。 |
| `task_type` / `mode` | string | 推断 | `voice_design`、`custom_voice`、`voice_clone` 或 `base`。 |
| `instructions` / `instruct` | string | `""` | VoiceDesign/CustomVoice 指令。 |
| `language` | string | `Auto` | 语言。 |
| `response_format` | string | `wav` | 支持 `wav`、`pcm`、`pcm_s16le`。 |
| `stream` | bool 或 object | `false` | `true` 或 object 时启用 streaming；streaming 要求 `response_format=pcm`。 |
| `ref_audio` / `ref_text` / `x_vector_only` | mixed | 无 | VoiceClone 字段。 |
| `generation` 或顶层生成字段 | object | `{}` | 同通用 `generation`。 |

示例：

```bash
curl -X POST http://127.0.0.1:17860/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o out.wav \
  -d '{"model":"qwen3-tts-openvino","voice":"voice_design","input":"你好。","language":"Chinese","instructions":"自然朗读。"}'
```

## VoiceClone 参考音频

默认生产路径：

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

要点：

- VoiceClone 使用 Base IR，即 `base/manifest.json`。
- `x_vector_only=false` 时 `ref_text` 必填。
- `ref_audio` 可以是本地路径、HTTP(S) URL、base64 或 `data:audio/...`。
- 运行时会转单声道并重采样；推荐清晰、单人、低噪声、约 3 到 10 秒的参考音频。

## Python API

```python
from qwen3_tts_ov import OpenVINOQwen3TTS

tts = OpenVINOQwen3TTS.from_ir(
    "openvino/voice_design",
    device="GPU",
    realtime_profile="fastest",
)

for chunk in tts.stream_voice_design(
    text="你好，这是 Python 流式 API 测试。",
    instruct="A calm young female voice.",
    language="Chinese",
    max_new_tokens=48,
    chunk_strategy="smooth",
):
    if chunk.audio.size:
        print(chunk.index, chunk.audio.shape, chunk.sample_rate, chunk.is_final)
```

构造参数常用项：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `ir_dir` | 必填 | 单个模式 IR 目录，例如 `openvino/voice_design`。 |
| `device` | 必填 | `GPU`、`CPU`、`AUTO` 等 OpenVINO device。 |
| `realtime_profile` | `fastest` | `from_ir` 支持 `fastest`；其它 profile 属于开发实验。 |
| `decoder_device` | `device` | 可把音频 decoder 放到另一设备；NPU 路径通常用 sidecar 参数。 |
| `ov_cache_dir` / `disable_ov_cache` | 自动 | 控制 OpenVINO compile cache。 |

主要方法：

- `stream_voice_design(text, instruct, language="Auto", ...) -> Iterator[StreamChunk]`
- `stream_custom_voice(text, speaker, language="Auto", instruct="", ...) -> Iterator[StreamChunk]`
- `stream_voice_clone(text, language="Auto", ref_audio=..., ref_text=..., x_vector_only_mode=False, ...) -> Iterator[StreamChunk]`
- `generate_voice_design(...) -> (list[np.ndarray], sample_rate)`
- `generate_custom_voice(...) -> (list[np.ndarray], sample_rate)`
- `generate_voice_clone(...) -> (list[np.ndarray], sample_rate)`
- `create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode=False)` 可缓存参考音频 prompt，便于重复使用。

`StreamChunk` 字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `index` | int | chunk 序号。 |
| `audio` | `np.ndarray` | float32 mono audio。 |
| `sample_rate` | int | 通常 24000。 |
| `codes` | `np.ndarray` | codec token/code frame。 |
| `is_final` | bool | 是否最终 chunk。 |
| `timings` | dict | decode、RTF、上下文、online batching 等 metadata。 |
| `pcm_s16le` | bytes 或 None | 某些路径可直接携带 PCM bytes。 |


# 流式合成说明

当前实现提供两层流式能力：

- `generate_codes_iter()`：逐 codec frame 生成。
- `stream_*()`：每累计一段 codec frame 就解码并返回音频块。

默认 `chunk_strategy=low_latency`：首块 `initial_chunk_frames=8`，后续 `chunk_frames=12`，12Hz codec 下首块约 0.67 秒音频；`left_context_frames=25` 用于减少 chunk 边界不连续。`balanced` 使用 12/12，`stable` 使用 12/24，适合弱设备或更稳定的播放缓冲。

## Python API

```python
from qwen3_tts_ov import OpenVINOQwen3TTS

# 这里的路径必须是包含 manifest.json 的导出目录。
# 如果只使用本机旧 VoiceDesign IR，可临时改成 "openvino_full"。
tts = OpenVINOQwen3TTS.from_ir("openvino/voice_design", device="GPU", mode="cache", cache_step="fused")

for chunk in tts.stream_voice_design(
    text="你好，这是一次流式合成测试。",
    instruct="A calm young female voice.",
    language="Chinese",
    chunk_strategy="low_latency",
):
    if chunk.audio.size:
        # chunk.audio 是 float32 mono PCM，采样率 chunk.sample_rate
        pass
```

`StreamChunk` 字段：

```text
index        chunk 序号
audio        float32 mono PCM
sample_rate  采样率，通常是 24000
codes        本 chunk 新生成的 codec frames
is_final     是否为最终 chunk 或最终空标记
timings      profile 信息和累计帧数
```

## CLI 调试

```bash
uv run python -m qwen3_tts_ov stream voice-design \
  --ir-dir openvino/voice_design \
  --device GPU \
  --text "你好，这是流式 CLI 测试。" \
  --instruct "A calm young female voice." \
  --language Chinese \
  --chunk-strategy low_latency \
  --chunk-dir outputs/stream \
  --output outputs/stream.wav
```

`chunk-dir` 中的 `chunk_*.pcm` 是 raw `pcm_s16le`，`output` 是拼接后的 WAV，便于快速试听。

## 本地服务

安装服务依赖：

```bash
uv pip install -e ".[server]"
```

首次启动前建议先填充 OpenVINO 编译缓存：

```bash
export VOICE_DESIGN_IR=openvino/voice_design
test -f "$VOICE_DESIGN_IR/manifest.json"
uv run python -m qwen3_tts_ov cache-warmup \
  --ir-dir "$VOICE_DESIGN_IR" \
  --device GPU \
  --mode cache \
  --cache-step fused \
  --graphs core,stream,buckets \
  --preload-buckets warmup \
  --warmup-strategy low_latency
```

启动：

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --host 127.0.0.1 \
  --port 17860 \
  --preload-modes voice_design \
  --preload-buckets warmup \
  --warmup-strategy low_latency
```

服务默认会预热当前策略所需的首块/稳态 decoder 和最小 cache bucket。`--preload-buckets all` 会尝试常驻编译所有 cache bucket，iGPU 显存/共享内存较小时不建议作为默认值。如果只想快速启动服务，可加 `--no-warmup`。
如果需要提前编译所有 bucket 到磁盘，请使用 `cache-warmup --preload-buckets all --stream-decoders all`，不要把它作为 sidecar resident warmup。

目录约定：

```text
openvino/
  voice_design/manifest.json
  custom_voice/manifest.json
  base/manifest.json
```

如果当前只有单个旧 VoiceDesign IR，例如 `openvino_full/manifest.json`，可以临时使用：

```bash
uv run python -m qwen3_tts_ov serve \
  --ir-dir openvino_full \
  --host 127.0.0.1 \
  --port 17860 \
  --preload-modes voice_design \
  --preload-buckets warmup \
  --warmup-strategy low_latency
```

浏览器测试页：

```text
http://127.0.0.1:17860/
```

该页面使用 WebSocket `/v1/tts/stream`，收到每个 `pcm_s16le` chunk 后立即用 Web Audio 排队播放，并支持把已收到的 chunk 合并下载为 WAV。它适合快速验证桌面应用 sidecar 的可用性。

## HTTP NDJSON

```bash
curl -N http://127.0.0.1:17860/v1/tts/stream \
  -H "content-type: application/json" \
  -d '{"mode":"voice_design","text":"你好，这是 HTTP 流式合成。","language":"Chinese","instruct":"A calm young female voice.","stream":{"chunk_strategy":"low_latency","format":"pcm_s16le"}}'
```

返回事件：

```json
{"type":"metadata","sample_rate":24000,"format":"pcm_s16le","chunk_strategy":"low_latency","initial_chunk_frames":8,"chunk_frames":12}
{"type":"audio","index":0,"audio":"base64 pcm_s16le ..."}
{"type":"final","index":1}
```

## WebSocket

连接 `ws://127.0.0.1:17860/v1/tts/stream`，先发送同样的请求 JSON。服务会先发 metadata JSON，然后发送二进制 `pcm_s16le` 音频块，最后发送 final JSON。

## OpenAI-Compatible Speech API

新增 `POST /v1/audio/speech`，保留 OpenAI Speech API 的 `input`、`voice`、`model`、`response_format`、`stream` 字段，并扩展 `task_type`、`language`、`instructions`、`ref_audio`、`ref_text`、`x_vector_only_mode`、`chunk_strategy` 等参数。

流式请求返回连续 raw `pcm_s16le`：

```bash
curl -N http://127.0.0.1:17860/v1/audio/speech \
  -H "content-type: application/json" \
  -d '{"model":"qwen3-tts-openvino","voice":"default","input":"你好，这是兼容 Speech API 的流式输出。","language":"Chinese","task_type":"voice_design","instructions":"A calm young female voice.","stream":true,"response_format":"pcm","chunk_strategy":"low_latency"}' \
  --output speech.pcm
```

`GET /v1/audio/voices` 会从 CustomVoice manifest 中列出可用 speaker。

## Decoder 路径

导出时建议保留：

```bash
--stream-decoder-first-chunks 6,8,12 --stream-decoder-chunks 8,12,24 --stream-decoder-left-context 25
```

Runtime 首块优先使用 `speech_decoder_stream_c0_t8.xml`，后续低延迟默认使用 `speech_decoder_stream_c25_t12.xml`。如果旧 IR 没有精确匹配图，会自动 fallback 到已有流式 decoder；仍然缺失时才使用普通 `speech_decoder_t*.xml` 分块解码。

## 实时播放建议

浏览器测试页默认等待约 `250ms` 播放缓冲再出声。如果播放队列低于 `100ms`，页面会自动增加 jitter buffer，上限为 `500ms`，以避免后续 chunk 轻微抖动造成卡顿。

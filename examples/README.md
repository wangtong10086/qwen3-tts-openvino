# Examples

These examples contain only small JSON, JSONL, text, and Python client files.
They do not include model weights, OpenVINO IR, generated audio, or private
paths.

## Prerequisites

Start the sidecar first. For release users:

```bash
./qwen3-tts-ov-server --device GPU
```

For source developers:

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest
```

The default service URL is `http://127.0.0.1:17860`.

## HTTP WAV

```bash
python examples/python/http_tts_wav.py \
  --text "Hello from the Qwen3-TTS OpenVINO sidecar." \
  --output outputs/example_http.wav
```

## HTTP NDJSON Streaming

`stream_request.example.json` can be posted to `/v1/tts/stream`:

```bash
curl -N http://127.0.0.1:17860/v1/tts/stream \
  -H "content-type: application/json" \
  -d @examples/stream_request.example.json
```

## WebSocket Streaming

The WebSocket example requires the optional `websockets` package:

```bash
uv run --with websockets python examples/python/websocket_stream_pcm.py \
  --request examples/stream_request.example.json \
  --output outputs/example_ws.wav
```

## OpenAI-Compatible Speech API

```bash
python examples/python/openai_speech.py \
  --request examples/openai_speech_request.example.json \
  --output outputs/example_openai.pcm
```

## Batch JSONL

```bash
uv run python -m qwen3_tts_ov batch \
  --ir-dir auto \
  --realtime-profile fastest \
  --batch-jsonl examples/requests.example.jsonl \
  --output-dir outputs/batch
```

## VoiceClone

`voice_clone_request.example.json` uses a placeholder reference audio path.
Replace `ref_audio` with a local wav/flac/mp3 file or URL, and set `ref_text`
to the accurate transcript of that reference audio.

`x_vector_only` is explicitly set to `false`, matching the production default.
That uses the ICL cloning path with `ref_audio + ref_text`; only set it to
`true` when intentionally testing speaker-embedding-only behavior.

中文说明见 [README.zh-CN.md](README.zh-CN.md)。

# 示例请求

这些示例只包含小型文本 JSON，不包含真实模型、IR、音频或私有路径。

English version: [README.md](README.md)。

## 前置条件

先确认 [前置条件](../docs/prerequisites_zh.md)。Release 用户直接启动预编译服务；源码开发者先按 [Quick Start](../docs/quick_start_zh.md) 构建 `fastest` 所需 IR。

Release 包：

```bash
./qwen3-tts-ov-server --device GPU
```

源码服务：

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest
```

默认服务地址是 `http://127.0.0.1:17860`。

## Python 客户端

整段 WAV：

```bash
python examples/python/http_tts_wav.py \
  --text "你好，这是一个 HTTP WAV 示例。" \
  --output outputs/example_http.wav
```

WebSocket 流式 PCM，写为 WAV：

```bash
uv run --with websockets python examples/python/websocket_stream_pcm.py \
  --request examples/stream_request.example.json \
  --output outputs/example_ws.wav
```

OpenAI-compatible Speech API：

```bash
python examples/python/openai_speech.py \
  --request examples/openai_speech_request.example.json \
  --output outputs/example_openai.pcm
```

## 批处理 JSONL

`requests.example.jsonl` 展示 batch CLI 的三种 mode 字段写法。

```bash
uv run python -m qwen3_tts_ov batch \
  --ir-dir auto \
  --realtime-profile fastest \
  --batch-jsonl examples/requests.example.jsonl \
  --output-dir outputs/batch
```

## Sidecar WebSocket / HTTP NDJSON

`stream_request.example.json` 可作为 `/v1/tts/stream` 的请求体：

```bash
curl -N http://127.0.0.1:17860/v1/tts/stream \
  -H "content-type: application/json" \
  -d @examples/stream_request.example.json
```

## OpenAI-Compatible Speech API

`openai_speech_request.example.json` 对应兼容 OpenAI 风格的 `/v1/audio/speech` endpoint：

```bash
curl -N http://127.0.0.1:17860/v1/audio/speech \
  -H "content-type: application/json" \
  -d @examples/openai_speech_request.example.json \
  --output outputs/openai_speech.pcm
```

## 长文本质量评测

`long_text_zh.example.txt` 是中文长文本样例，可用于 full-AR 质量门禁：

```bash
uv run python scripts/evaluate_long_text_quality.py \
  --ir-dir auto \
  --device GPU \
  --text-file examples/long_text_zh.example.txt \
  --profiles quality \
  --runs 1
```

## VoiceClone

`voice_clone_request.example.json` 使用占位参考音频路径。运行前需要替换为本机可访问的 wav/flac/mp3 文件或 URL，并把 `ref_text` 改成参考音频的准确转写。

示例中 `x_vector_only` 显式写为 `false`，这也是服务端和 Web Demo 的默认行为。该模式会使用 `ref_audio + ref_text` 的 ICL 克隆路径；只有需要只测试 speaker embedding 时才改为 `true`。

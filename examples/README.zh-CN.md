# 示例请求

这些示例只包含小型文本 JSON，不包含真实模型、IR、音频或私有路径。

## 批处理 JSONL

运行前请先按主 README 导出并压缩 `fastest` 所需 IR，确认 `openvino/voice_design/manifest.json` 存在。

```bash
uv run python -m qwen3_tts_ov batch \
  --ir-dir openvino/voice_design \
  --realtime-profile fastest \
  --batch-jsonl examples/requests.example.jsonl \
  --output-dir outputs/batch
```

## Sidecar WebSocket / HTTP NDJSON

`stream_request.example.json` 可直接作为 `/v1/tts/stream` 的请求体：

```bash
curl -N http://127.0.0.1:17860/v1/tts/stream \
  -H "content-type: application/json" \
  -d @examples/stream_request.example.json
```

## OpenAI-Compatible Speech API

```bash
curl -N http://127.0.0.1:17860/v1/audio/speech \
  -H "content-type: application/json" \
  -d @examples/openai_speech_request.example.json \
  --output outputs/openai_speech.pcm
```

## VoiceClone

`voice_clone_request.example.json` 使用占位参考音频路径。运行前需要替换为本机可访问的 wav/flac/mp3 文件或 URL。

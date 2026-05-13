# 示例请求

这些示例只包含小型文本 JSON，不包含真实模型、IR、音频或私有路径。

## 前置条件

先按 [Quick Start](../docs/quick_start_zh.md) 构建 `fastest` 所需 IR，确认 `openvino/voice_design/manifest.json` 存在。

## 批处理 JSONL

`requests.example.jsonl` 展示 batch CLI 的三种 mode 字段写法。

```bash
uv run python -m qwen3_tts_ov batch \
  --ir-dir openvino/voice_design \
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

`voice_clone_request.example.json` 使用占位参考音频路径。运行前需要替换为本机可访问的 wav/flac/mp3 文件或 URL。

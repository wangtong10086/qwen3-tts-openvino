# Qwen3-TTS OpenVINO

OpenVINO-only runtime, exporter, sidecar server, and native acceleration path for
Qwen3-TTS 12 Hz models.

[中文文档](README.zh-CN.md) | [Documentation Index](docs/README.zh-CN.md) | [Examples](examples/README.zh-CN.md)

This is a source-only repository. It does not include model weights, exported
OpenVINO IR, generated audio, virtual environments, compile caches, or native
build artifacts.

## Highlights

- OpenVINO runtime for Qwen3-TTS VoiceDesign, CustomVoice, and VoiceClone.
- One-command fastest-path build with a low-memory production graph set.
- Native C++ codec generation pipeline with paged-KV attention.
- Local sidecar with WebSocket, HTTP NDJSON, and OpenAI-compatible Speech API.
- Streaming playback with `pcm_s16le` chunks.
- Full autoregressive long-text generation by default, without text segmentation.

## Quick Start

```bash
uv sync --extra native --extra server --extra export

uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU

uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest \
  --host 127.0.0.1 \
  --port 17860
```

Open `http://127.0.0.1:17860/`.

Add `--clean --clean-native` when rebuilding from scratch.

See the full setup guide in [docs/quick_start_zh.md](docs/quick_start_zh.md).

## Runtime Profile

The production profile is `fastest`:

- native C++ pipeline is required
- code generation uses OpenVINO paged-KV attention
- paged talker seed graph uses `int8_sym_paged_talker_split`
- cached subcode graph remains FP16 for stability
- streaming decoder emits mono 24 kHz `pcm_s16le`

Long VoiceDesign requests are generated as one continuous autoregressive
sequence. The runtime only chunks decoded audio for playback; it does not split
the input text by default.

For end users, release packages expose only the sidecar executable:

```bash
qwen3-tts-ov-server --model-root openvino --device GPU
```

## Documentation

- [Quick Start](docs/quick_start_zh.md): install, prepare model, build fastest IR, start web demo.
- [Release Usage](docs/release_zh.md): prebuilt Linux/Windows app packages plus standalone IR packages.
- [Development Guide](docs/development_zh.md): source build, export, compression, release packaging.
- [Runtime Usage](docs/runtime_zh.md): CLI, Python API, sidecar, WebSocket, OpenAI-compatible API.
- [Export Guide](docs/export_zh.md): manual export and compression.
- [Streaming and Long Text](docs/streaming_zh.md): streaming protocol, full-AR long text, quality gate.
- [OpenVINO Cache](docs/cache_zh.md): cache warmup and cache location.
- [Artifacts Policy](docs/artifacts_zh.md): model/IR/output handling.
- [Security](docs/security_zh.md): credentials and commit checks.

## Repository Layout

```text
qwen3_tts_ov/  runtime, exporter, CLI, sidecar server, and web client
native/        native OpenVINO C++ codec generation pipeline
scripts/       build, compression, benchmark, and quality helper scripts
devtools/      legacy scripts and experimental profiling tools
docs/          Chinese documentation
examples/      small JSON/JSONL/text examples
tests/         unit tests
```

## License

Apache-2.0. See [LICENSE](LICENSE).

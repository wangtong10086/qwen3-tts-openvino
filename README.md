# Qwen3-TTS OpenVINO

OpenVINO-only runtime and exporter for Qwen3-TTS 12 Hz models.

This repository is source-only. It does not include model weights, exported
OpenVINO IR, generated audio, local virtual environments, or native build
artifacts.

The production path is the `fastest` profile:

- native C++ pipeline is required
- INT8_SYM cached-subcode fused graphs are required
- no-repeat unroll4 + decode-unroll graphs are required
- streaming uses the `smooth` chunk strategy by default

See the full Chinese guide: [README.zh-CN.md](README.zh-CN.md).

## Quick Start

```bash
git submodule update --init --recursive
uv sync --extra native --extra server --extra export
uv run python scripts/build_native_codegen.py
uv run python -m qwen3_tts_ov --help
```

Export and compress the required VoiceDesign IR locally:

```bash
uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --model-type voice_design \
  --out-dir openvino/voice_design \
  --cache-buckets 96,128,192,256,320,384 \
  --cache-kernels exact \
  --fused-cache-kernels exact \
  --fused-subcode-mode cached \
  --fused-cache-unroll-steps 4 \
  --fused-cache-norepeat-steps 4 \
  --decoder-tokens 64,128,256 \
  --stream-decoder-chunks 8,12,24 \
  --stream-decoder-first-chunks 8 \
  --stream-decoder-left-context 25

uv run python scripts/compress_openvino_weights.py \
  --ir-dir openvino/voice_design \
  --source-variant fp16_fused_cachedsub \
  --variant int8_sym_fused_cachedsub \
  --mode int8_sym \
  --fused-cache-unroll-steps 4
```

Warm the OpenVINO cache and start the sidecar:

```bash
uv run python -m qwen3_tts_ov cache-warmup \
  --ir-dir openvino/voice_design \
  --device GPU \
  --realtime-profile fastest \
  --graphs core,stream,buckets \
  --preload-buckets warmup

uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest \
  --host 127.0.0.1 \
  --port 17860
```

Open the web demo at `http://127.0.0.1:17860/`.

## Supported Modes

- VoiceDesign: `text + instruct + language`
- CustomVoice: `text + speaker + optional instruct + language`
- VoiceClone: `text + ref_audio/ref_text or reusable prompt + language`

The public runtime, CLI, HTTP/WebSocket sidecar, and OpenAI-compatible speech
endpoint all use the same `fastest` production profile by default. Experimental
profiles and historical scripts live under `devtools/`.

## Layout

```text
qwen3_tts_ov/  production runtime, exporter, sidecar, web client, and CLI
native/        required C++ OpenVINO GenAI-style pipeline source
scripts/       production helper scripts
devtools/      experimental benchmarks, legacy scripts, and profiling tools
docs/          Chinese guides
examples/      small JSON/JSONL request examples
tests/         unit tests
```

## License

Apache-2.0. See [LICENSE](LICENSE).

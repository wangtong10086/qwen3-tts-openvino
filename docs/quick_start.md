# Source Quick Start

This is the English source-development path. The detailed Chinese guide is
[quick_start_zh.md](quick_start_zh.md). End users should prefer
[Release usage](release.md).

Review [Prerequisites](prerequisites.md) before running source commands.

## 1. Install Dependencies

```bash
uv sync --extra native --extra server --extra export
uv run python -m qwen3_tts_ov --help
```

If `third_party/openvino.genai` has not been initialized, `build-fastest` will
initialize it. You may also run:

```bash
git submodule update --init --recursive
```

## 2. Prepare PyTorch Models

VoiceDesign:

```bash
uv run modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --local_dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

CustomVoice and Base/VoiceClone:

```bash
uv run modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local_dir ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice

uv run modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local_dir ./models/Qwen3-TTS-12Hz-1.7B-Base
```

`models/` is intentionally ignored by git.

## 3. Build Production IR

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU
```

For CustomVoice and VoiceClone:

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --out-dir openvino/custom_voice \
  --device GPU

uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-Base \
  --out-dir openvino/base \
  --device GPU
```

Preview without mutating outputs:

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU \
  --dry-run
```

## 4. Start the Development Sidecar

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest \
  --host 127.0.0.1 \
  --port 17860
```

Open `http://127.0.0.1:17860/`.

## 5. Verify

```bash
uv run python -m qwen3_tts_ov build-fastest --help
uv run python -m qwen3_tts_ov serve --help
uv run python scripts/benchmark_prompt_batch_matrix.py --dry-run
python -m py_compile examples/python/*.py
```

For API fields, see [API Reference](api_reference.md). For failures, see
[Troubleshooting](troubleshooting.md).


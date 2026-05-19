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

On Windows source builds, use MSVC/CMake and enable UTF-8 output in PowerShell:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
cmake --version
where.exe cl
```

If `cl.exe` is not found, install Visual Studio 2022 Build Tools with the C++
toolchain and Windows SDK.

If `third_party/openvino.genai` has not been initialized, `build-fastest` will
initialize it. You may also run:

```bash
git submodule update --init --recursive
```

## 2. Prepare Official Qwen3-TTS Sources

The exporter imports the upstream `qwen_tts` Python package. Clone the official
Qwen3-TTS sources into ignored local cache space and add them to `PYTHONPATH`:

```bash
git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS .cache/Qwen3-TTS
export PYTHONPATH="$(pwd)/.cache/Qwen3-TTS"
uv run python -c "import qwen_tts; print('qwen_tts ok')"
```

Windows PowerShell:

```powershell
git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS .cache\Qwen3-TTS
$env:PYTHONPATH = (Resolve-Path .cache\Qwen3-TTS).Path
uv run python -c "import qwen_tts; print('qwen_tts ok')"
```

An import-time `SoX could not be found` warning from upstream Qwen3-TTS is not
fatal for OpenVINO IR export if the command still prints `qwen_tts ok`.

## 3. Prepare PyTorch Models

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

## 4. Build Native Runtime

`build-fastest` builds the native runtime automatically. To verify the Windows
MSVC/CMake path before export:

```powershell
uv run python scripts\build_native_codegen.py --backend cmake --config Release
```

The expected runtime library is `native/build/qwen3_tts_ov_genai.dll`.

## 5. Build Production IR

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU
```

To rebuild from stale or partially generated outputs:

```bash
uv run python -m qwen3_tts_ov build-fastest \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --out-dir openvino/voice_design \
  --device GPU \
  --clean \
  --clean-native
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

## 6. Start the Development Sidecar

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest \
  --host 127.0.0.1 \
  --port 17860
```

Open `http://127.0.0.1:17860/`.

Quick HTTP smoke:

```bash
uv run python examples/python/http_tts_wav.py --output outputs/example_http.wav --max-new-tokens 24
```

## 7. Verify

```bash
uv run python -m qwen3_tts_ov build-fastest --help
uv run python -m qwen3_tts_ov serve --help
uv run python scripts/benchmark_prompt_batch_matrix.py --dry-run
python -m py_compile examples/python/*.py
```

For API fields, see [API Reference](api_reference.md). For failures, see
[Troubleshooting](troubleshooting.md).


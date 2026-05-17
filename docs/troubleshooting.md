# Troubleshooting / FAQ

Check [Prerequisites](prerequisites.md) first, then match the symptom below.

## `--device GPU` Cannot Find a GPU

Symptoms:

- Startup fails with `No device with "GPU"`, `GPU plugin`, or `No OpenCL device`.
- `/health` is unavailable, or OpenVINO only reports `CPU`.

Check:

```bash
uv run python - <<'PY'
import openvino as ov
core = ov.Core()
print(core.available_devices)
for name in core.available_devices:
    print(name, core.get_property(name, "FULL_DEVICE_NAME"))
PY
```

On Linux, also check:

```bash
ls -l /dev/dri || true
groups
```

Fix:

- Install Intel OpenCL/Level Zero runtime packages; see [Prerequisites](prerequisites.md#intel-gpu-requirements).
- Add the user to the `render` group and log in again.
- For containers, WSL, and VM passthrough, verify `/dev/dri/renderD*` is exposed. WSL is not the NPU validation path.
- On Windows, update the Intel Graphics Driver. OEM laptops may require vendor-specific drivers.
- Use `--device CPU` only for API/startup smoke tests; realtime performance is not promised.

## Hugging Face Download Fails

Symptoms:

- First start stalls or fails with network, TLS, 403/404, or timeout errors.
- `/v1/models/download` returns a download error.

Check:

```bash
curl -I https://huggingface.co/
```

Fix:

- Pre-download IR in a networked environment and start with `--model-root`.
- Configure `HTTPS_PROXY`, `HTTP_PROXY`, or Hugging Face Hub environment variables.
- Manual download:

```bash
uv run --with huggingface_hub huggingface-cli download \
  waston10086/qwen3-tts-openvino-voice-design \
  --include "openvino_realtime/**" \
  --local-dir qwen3-tts-openvino-ir
```

- For private IR, set `--model-repo`, `--model-revision`, `--model-subdir`, or the per-mode environment variables documented in [Release usage](release_zh.md).

## `--npu-offload decoder` Fails

Symptoms:

- Windows startup fails with `NPU`, `compile`, or `streaming decoder`.
- `ov.Core().available_devices` does not contain `NPU`.

Fix:

- Keep `--npu-offload decoder` for strict validation; it fails when NPU cannot be used.
- Use auto fallback when desired:

```powershell
.\qwen3-tts-ov-server.exe --device GPU --npu-offload auto
```

- Validate on native Windows with the proper drivers and fixed-shape streaming decoder IR.
- Follow [Windows GPU+NPU path](windows_gpu_npu_zh.md) for probe and smoke scripts.

## First Start Looks Stuck

Symptoms:

- First release run takes a long time.
- Logs mention compile, warmup, cache, or OpenVINO.

Cause:

- OpenVINO compiles graphs and writes a user cache on first use of an IR/device.
- VoiceClone, NPU, long text, and cold caches can take longer.

Fix:

- Wait for warmup and inspect `/health`.
- Subsequent cache hits should be faster.
- Use [OpenVINO compile cache](cache_zh.md) for offline warmup.
- If disk space is low, clean the user cache or set `--ov-cache-dir`.

## `uv sync` or torch/torchaudio Times Out

Symptoms:

- Source dependency installation fails.
- `torch`, `torchaudio`, or `openvino` wheels time out.

Fix:

- Release users do not need PyTorch or source export dependencies.
- For source server work without export:

```bash
uv sync --extra server --extra native
```

- Use `--extra export` only when rebuilding IR from PyTorch weights.
- Configure a PyPI mirror/proxy and retry; `uv sync -v` can show the slow package.
- Confirm Python is `>=3.12`.

## IR Directory or Manifest Fails

Symptoms:

- Errors mention `manifest.json`, `model_type`, `VoiceClone uses Base IR`, or `missing graph`.
- The Web Demo marks some modes unavailable.

Expected layout:

```text
openvino_realtime/
  voice_design/manifest.json
  custom_voice/manifest.json
  base/manifest.json
```

Fix:

- `--model-root` should point to the root containing `voice_design/`, `custom_voice/`, and `base/`.
- VoiceClone uses `base/manifest.json`; VoiceDesign IR is not interchangeable.
- Query `/v1/models` for per-mode status.
- Re-run `build-fastest` or re-download release IR if the manifest is stale.

## VoiceClone Reference Audio Fails

Symptoms:

- `ref_text is required when x_vector_only_mode=False`.
- Audio loading fails, or the error suggests installing `audio-full`.
- Voice quality is unstable.

Guidance:

- With the default `x_vector_only=false`, provide both `ref_audio` and accurate `ref_text`.
- `ref_audio` may be a local path, HTTP(S) URL, base64 string, or `data:audio/...` string.
- Runtime converts to mono and resamples as needed. Use clear, single-speaker, low-noise audio; around 3 to 10 seconds is a good starting point.
- wav/flac usually work through `soundfile`; for mp3 and broader codecs in source environments, install the `audio-full` extra so `librosa` can be tried.
- `x_vector_only=true` is for speaker-embedding-only comparison, not the best default path.

## CustomVoice Speaker Is Unknown

Check the available speakers:

```bash
curl http://127.0.0.1:17860/v1/audio/voices
```

Use `voices` or `voice_details[].id` as the `speaker`. The built-in speaker list
comes from the active IR, so this documentation does not hard-code it.

## Long Text or VRAM Budget Error

Symptoms:

- Errors mention `effective_max_continuous_prompt_tokens`,
  `max_generation_tokens_available`, `USM`, or `context limit`.

Fix:

- Use `/v1/tts/tokenize` or the Web Demo budget panel.
- Shorten text, reduce `generation.max_new_tokens`, or raise `--max-vram-ratio`.
- If you accept runtime allocation failures, set `--max-continuous-prompt-tokens 0`.
- Do not rely on a fixed character count; the real limit depends on IR, device memory, KV profile, and runtime planning.

## CPU-only Is Slow

`--device CPU` is a smoke/fallback path for startup and API integration. For
production interaction, use Intel GPU. For CPU smoke tests, keep text short and
lower `generation.max_new_tokens`.


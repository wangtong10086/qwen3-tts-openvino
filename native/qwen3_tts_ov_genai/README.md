# Native OpenVINO GenAI C++ codegen pipeline

This directory contains the native C++ migration path for the Qwen3-TTS
streaming inference pipeline. The shared library links both OpenVINO Runtime and
`libopenvino_genai`, and wraps the Qwen3-specific graphs in a
`Qwen3TTSGenAIPipeline` object modeled after OpenVINO GenAI speech pipelines.
The native pipeline uses GenAI `SpeechGenerationConfig` validation and
`SpeechGenerationPerfMetrics`/`PerfMetrics` timing helpers, while keeping the
Qwen3 codec generation loop specialized to the exported TTS graphs.
It can optionally chain `frame_embed` between autoregressive steps with OpenVINO
remote tensors, avoiding a GPU-to-CPU-to-GPU copy on devices that support remote
tensors. This path is disabled by default because it has not shown stable speedup
on the current iGPU setup.
It can also optionally build VoiceDesign prompts in C++ with OpenVINO GenAI
Tokenizer plus the exported text/codec embedding graphs. That path is useful for
validating a fuller C++ pipeline but is disabled by default because it is slower
than Python prompt construction on the current machine.

The production `fastest` profile requires this native pipeline. Set
`QWEN3_TTS_OV_NATIVE_PIPELINE=require` in verification runs to fail fast when
the shared library or required graphs are missing:

- `fused_cache_step_unrollN_*`
- `fused_cache_decode_unrollN_*_statefulmask_*`
- `speech_decoder_stream_c0_t8.xml`
- `speech_decoder_stream_c25_t12.xml`

The C++ loop is specialized for Qwen3-TTS codec generation instead of directly
using the stock GenAI `Text2SpeechPipeline` / `LLMPipeline`, because those
pipelines do not model Qwen3-TTS' multi-codebook codec autoregression:

- prompt is already provided as `inputs_embeds`
- output is codec frames, not text token ids
- `frame_embed` is fed back into the next autoregressive step
- `repeated_mask` is moved into the stateful decode-unroll graph
- codec frames or decoded audio chunks are emitted back to Python through native
  streaming callbacks

Build:

```bash
uv sync --extra native
uv run python scripts/build_native_codegen.py
```

The build creates both:

- `native/build/libqwen3_tts_ov_genai.so`
- `native/build/qwen3_tts_ov_native_cli`

Run:

```bash
uv run python -m qwen3_tts_ov stream voice-design --realtime-profile fastest ...
```

`fastest` sets the required native environment automatically through the CLI and
sidecar. Low-level environment variables remain available for devtools and
operator profiling.

`QWEN3_TTS_OV_NATIVE_ASYNC_DECODE=1` enables an experimental C++ decoder worker
that overlaps codegen and audio decode. It is disabled by default because some
iGPU/OpenVINO runtime combinations serialize or contend heavily when two GPU
infer requests are submitted concurrently.

`QWEN3_TTS_OV_NATIVE_REMOTE_EMBED=1` enables remote `frame_embed` chaining. The
runtime reports `native_remote_embed=true` when the remote path is actually
active.

`QWEN3_TTS_OV_NATIVE_PROMPT=1` enables native VoiceDesign prompt construction.
It requires `openvino_tokenizer.xml` and `openvino_detokenizer.xml` in the IR
directory. Generate them with `python -m qwen3_tts_ov.exporter --tokenizer-only`.

Standalone C++ smoke:

```bash
native/build/qwen3_tts_ov_native_cli \
  --ir-dir openvino_full \
  --device GPU \
  --decoder-device GPU \
  --prompt-device CPU \
  --text "你好" \
  --instruct "自然朗读" \
  --max-new-tokens 8 \
  --min-new-tokens 1 \
  --warmup-generations 1 \
  --ov-profile \
  --output outputs/native_cpp_smoke.wav \
  --profile-json outputs/native_cpp_profile.json
```

The standalone CLI runs VoiceDesign prompt construction, codegen, streaming
decode, and WAV writing in C++ through the native pipeline. Use `--cache-dir` to
point it at a warmed OpenVINO cache directory when measuring startup latency.
`--warmup-generations` runs unmeasured requests on the same compiled requests so
that `--profile-json` can separate compile/setup cost from hot generation cost.
The profile records compile/setup, prompt setup, warmup, first audio latency,
generation, WAV writing, per-chunk codegen/decode timings, and native RTF
metrics. `--ov-profile` enables OpenVINO `PERF_COUNT` in the native C++
compiled models and adds `native_ov_profile.by_type/by_label/top` to the JSON,
which is the preferred way to inspect codegen operator hotspots.

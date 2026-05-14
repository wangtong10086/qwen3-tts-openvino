# Native OpenVINO Pipeline

This directory contains the C++ runtime used by the production `fastest`
profile. It links OpenVINO Runtime and OpenVINO GenAI utilities, but the codec
generation loop is Qwen3-TTS-specific.

The stock GenAI `LLMPipeline` / `Text2SpeechPipeline` is not used directly
because Qwen3-TTS generation differs from normal text generation:

- the prompt is already represented as `inputs_embeds`
- autoregressive output is codec frames, not token ids
- generated codec frames are embedded and fed back as `frame_embed`
- subcode prediction uses multiple codebooks per frame
- audio chunks are emitted through a streaming speech decoder

## Production Path

`fastest` uses:

- paged-KV talker seed graph converted in memory with OpenVINO `SDPAToPagedAttention`
- GQA KV cache
- `KV_CACHE_PRECISION=f16`
- `block_size=16`
- split-subcode mode
- `int8_sym_paged_talker_split` for the paged talker seed
- FP16 cached subcode graph
- `speech_decoder_stream_c0_t8.xml` for the first chunk when available; older IR may use `c0_t12`
- `speech_decoder_stream_c25_t24.xml` for steady chunks

The runtime requires these graphs to be present. Missing required graphs should
fail fast instead of silently falling back to a slower or lower-quality path.

## Build

```bash
uv sync --extra native
uv run python scripts/build_native_codegen.py
```

Build outputs are ignored by git:

```text
native/build/libqwen3_tts_ov_genai.so
native/build/qwen3_tts_ov_native_cli
```

## Runtime Usage

The Python CLI and sidecar set the required native environment automatically
when `--realtime-profile fastest` is used:

```bash
uv run python -m qwen3_tts_ov stream voice-design \
  --ir-dir openvino/voice_design \
  --device GPU \
  --realtime-profile fastest \
  --text "你好，这是 native pipeline 测试。" \
  --instruct "用自然清晰的中文朗读。" \
  --language Chinese
```

Low-level flags remain available for diagnostics:

- `--native-paged-kv require`
- `--native-paged-kv-gqa on|off`
- `--native-paged-kv-split-subcode on|off`
- `--native-paged-kv-block-size N`
- `--native-paged-kv-precision f16|bf16|u8`
- `--kv-cache-profile fp16|bf16|u8|u8-input|u8-all` from the Python CLI is the
  preferred public switch for memory experiments. The production default is
  `u8`: it stores paged-KV cache in 8-bit form while keeping cache inputs at
  FP32; it reduces KV cache storage to roughly half of FP16 but does not mean
  every attention operator runs as INT8.
- `--native-paged-kv-score-aggregation on|off`

Use these flags only for A/B testing. Production should use `--realtime-profile fastest`.

## Long Text

Long text still uses one full autoregressive sequence. The native pipeline
supports sampled paged-KV split-subcode generation for quality-gated long text
profiles. The default sidecar only enables that path after
`scripts/evaluate_long_text_quality.py` writes a passing
`outputs/long_text_quality/quality_summary.json`, when the IR manifest exposes
the built-in `int8_sym_paged_talker_split` long-text graph set, or when
`QWEN3_TTS_OV_LONG_AR_PROFILE=paged-sample-fp16|paged-sample-int8` is set.

## Diagnostics

Use these scripts before accepting a new native optimization:

```bash
uv run python scripts/verify_paged_kv_correctness.py --help
uv run python scripts/verify_long_autoregressive_parity.py --help
uv run python scripts/audit_paged_kv_conversion.py --help
```

`PERF_COUNT` and operator profiling are diagnostic-only. They slow down the
small-graph autoregressive loop and should not be used for production RTF
measurements.

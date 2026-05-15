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
- `KV_CACHE_PRECISION=u8` by default, with FP32 cache input tensors
- `block_size=16`
- split-subcode mode
- `int8_sym_paged_talker_split` for the paged talker seed
- FP16 cached subcode graph
- `DYNAMIC_QUANTIZATION_GROUP_SIZE=32` for OpenVINO GPU compilation
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
- `--native-dynamic-quantization-group-size N`; production uses `32` after
  isolated benchmark validation on the current Intel GPU.
- `--native-codegen-fusion split|graph|auto|off`. The production default is
  `split`. `graph` is an experimental correctness-gated path that requires a
  fused paged seed graph such as `graphs.paged_kv_seed.fused_cache_step_gqa`.
- `QWEN3_TTS_OV_NATIVE_TRACE_CODEGEN_FRAMES=N` adds per-frame codegen trace
  records to `native_timing.codegen_trace`; use it only with short diagnostic
  runs.
- `QWEN3_TTS_OV_NATIVE_SUBCODE_NEXT_EMBED_GRAPH=1` enables the optional
  `subcode_greedy_cached_next_embed.xml` split-subcode graph, when present,
  to move `sum_embed + tts_pad_embed` into OpenVINO.

Use these flags only for A/B testing. Production should use `--realtime-profile fastest`.

## Single-Step Profiling

The native timing JSON reports per-step distributions for the paged-KV codegen
loop:

- `codegen_decode_step_stats`: one talker decode step after prefill
- `subcode_infer_step_stats`: one standalone subcode graph call
- `codegen_bind_step_stats` and `codegen_sampling_step_stats`: host-side
  overhead around each step
- `subcode_output_read_step_stats` and `subcode_next_embed_step_stats`: output
  readback and frame-embed preparation overhead
- `paged_static_decode_requested`, `paged_static_decode_enabled`, and
  `paged_static_decode_failure`: whether static decode actually compiled

Run the focused ablation set before changing the production profile:

```bash
uv run python scripts/benchmark_streaming_realtime.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --profile-set paged-kv-step-ablation \
  --runs 1 \
  --max-new-tokens 64 \
  --warmup-generations 1
```

On the current validation machine, `cachedsub` can improve short requests but is
not stable on longer requests, and `split_next_embed_graph` does not materially
reduce total RTF. Static decode currently reaches GPU compile and then falls
back with an OpenVINO `map::at` error, which is now surfaced in
`paged_static_decode_failure`.

## Experimental Fusion Hooks

The runtime can load optional OpenVINO extension libraries through:

```bash
export QWEN3_TTS_OV_OPENVINO_EXTENSIONS=/path/to/custom_extension.so
export QWEN3_TTS_OV_OPENVINO_EXTENSIONS_REQUIRED=1
```

`QWEN3_TTS_OV_RMS_EXTENSION_PATH` is accepted as a narrower alias for future
RMSNorm kernel experiments. No custom RMSNorm kernel is required for the
production package; if the extension path is absent and `REQUIRED` is not set,
the runtime logs a warning and keeps the normal OpenVINO graph.

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
uv run python scripts/verify_codegen_fusion_correctness.py --help
uv run python scripts/verify_paged_kv_correctness.py --help
uv run python scripts/verify_long_autoregressive_parity.py --help
uv run python scripts/audit_paged_kv_conversion.py --help
```

`verify_codegen_fusion_correctness.py --trace-frames N` compares the generated
codes and emits the first split-vs-graph trace mismatch, including first code,
all codec groups, and hidden/embed norms. By default it first runs an FP16
split-vs-graph structural baseline and then classifies the requested target as
`passed`, `structural_mismatch`, or `quantization_mismatch`. Use the
`fastest-fused-seed-selective` compression preset when testing graph-fused INT8:
it keeps subcode-sensitive nodes in FP precision so correctness failures can be
separated from full subcode quantization drift. A graph-fused path must pass this
gate before it can be considered for the fastest profile.

`PERF_COUNT` and operator profiling are diagnostic-only. They slow down the
small-graph autoregressive loop and should not be used for production RTF
measurements.

Current hotspot split for `fastest` on the validation machine is codegen-bound:
standalone subcode inference is roughly 60% of codegen time, talker decode is
roughly 35%, and streaming decoder is secondary. `DYNAMIC_QUANTIZATION_GROUP_SIZE=32`
is a small but repeatable compile-time improvement; graph-fused subcode remains
disabled because the correctness gate still detects codec mismatches.

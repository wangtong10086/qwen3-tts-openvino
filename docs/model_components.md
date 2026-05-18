# Model Components And Optimization Design

This page is a concise English entry point for the Chinese-first design note:
[模型组件与优化设计](model_components_zh.md).

The public release uses the `runtime-minimal` IR profile. It keeps only the
validated production graph set for the `fastest` runtime path:

- text and codec embedding graphs
- paged-KV talker seed graph from the production graph variant, usually
  `int8_sym_batch_fused_gqa` in public `runtime-minimal` IR
- `subcode_greedy_cached.xml`
- streaming speech decoders `c0_t8`, `c25_t12`, and `c25_t24`
- VoiceClone/Base-only `speech_encoder`, `speaker_encoder`, and
  `code_frame_embedding`

High-level pipeline:

```text
request
  -> prompt builder
  -> text_embedding / codec_embedding
  -> native codec generation
       -> paged-KV talker seed graph
       -> subcode_greedy_cached
  -> speech_decoder_stream
  -> PCM / WAV
```

The split between the talker graph and `subcode_greedy_cached` is intentional.
Qwen3-TTS generates a multi-codebook codec frame at each 12 Hz audio step. The
talker graph handles long-context autoregressive attention and produces the
first codebook plus hidden state; the cached subcode graph fills the remaining
codebooks and returns the next frame embedding.

OpenVINO paged-KV is used to avoid fixed cache buckets for long text, reduce
compile/package complexity, and let generation continue until EOS or the
configured context/memory budget. The default KV cache storage precision is U8
to reduce memory pressure. Online batching lives in the scheduler/backend layer,
not in a separate model file, so single-user and multi-user requests can reuse
the same IR set.

See [模型组件与优化设计](model_components_zh.md) for the full component table,
release-minimal rationale, and production-path verification checklist.

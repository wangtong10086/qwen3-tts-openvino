# qwen3_tts_ov_genai

Native C++ backend for the Qwen3-TTS OpenVINO runtime.

The production path is intentionally narrow:

- OpenVINO paged-KV codec generation.
- `layered` online scheduler.
- `layered_vllm` continuous batching policy.
- U8 KV cache by default.
- Streaming decoder chunk output for the Python sidecar.

## Build

```bash
uv run python scripts/build_native_codegen.py
```

Output goes to `native/build/` and is ignored by git.

## Used By

- `qwen3_tts_ov.native_codegen`
- `qwen3_tts_ov.online_batch`
- `qwen3_tts_ov.server`
- release server packages

## Validation

Use the Python-level gates:

```bash
uv run python scripts/benchmark_prompt_batch_matrix.py --dry-run
uv run python scripts/evaluate_single_arch_gate.py --help
```

Low-level native diagnostics should be added only when they validate the current
production backend. Historical graph-fused, unroll-only, and non-layered
experiments are no longer part of the maintained path.

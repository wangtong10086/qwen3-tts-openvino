# Contributing

Thanks for helping improve Qwen3-TTS OpenVINO. The project keeps one production
runtime path: `fastest` profile, native C++ codec generation, paged-KV, online
batching, and full-context autoregressive long-text generation.

## Local Setup

Install prerequisites from [docs/prerequisites.md](docs/prerequisites.md), then:

```bash
uv sync --extra native --extra server --extra export --extra dev
```

If you only work on the sidecar and do not rebuild IR, `--extra export` is not
required:

```bash
uv sync --extra native --extra server --extra dev
```

Do not commit local artifacts such as `models/`, `openvino/`, `openvino_full/`,
`outputs/`, `.venv/`, `.uv-cache/`, `native/build/`, or `dist/release/`.

## Checks

Before opening a PR, run the focused checks that match your change:

```bash
uv run ruff check .
uv run ty check .
uv run python -m py_compile qwen3_tts_ov/*.py scripts/*.py examples/python/*.py
uv run pytest -q
```

For documentation-only changes, at minimum run:

```bash
uv run pytest tests/test_docs_examples.py -q
```

## Runtime and Quality Validation

For source runtime smoke:

```bash
uv run python -m qwen3_tts_ov serve \
  --model-root openvino \
  --device GPU \
  --realtime-profile fastest
```

For release-quality validation:

```bash
uv run python scripts/evaluate_single_arch_gate.py \
  --server-url http://127.0.0.1:17860 \
  --modes voice_design,custom_voice,voice_clone \
  --runs 3 \
  --concurrency 1,2,4,8
```

Performance matrix:

```bash
uv run python scripts/benchmark_prompt_batch_matrix.py \
  --ir-dir openvino/voice_design \
  --device GPU \
  --batch-sizes 1,2,4,8,16 \
  --prompt-lengths short,medium,long,xlong \
  --scenarios offline,online \
  --runs 3
```

## Documentation

Keep user-facing docs close to the code behavior. For API fields, verify against
`qwen3_tts_ov/server.py` and `qwen3_tts_ov/runtime.py`.

When adding a Markdown page, link it from the appropriate index and run the docs
example test. Prefer documenting verified behavior over hardware promises; for
driver or platform requirements, link to official OpenVINO or Intel docs.

## Pull Requests

PRs should include:

- A concise summary of what changed and why.
- Tests or checks run, including skipped checks with reasons.
- Any user-facing docs updates for changed CLI/API behavior.
- Notes on model/IR compatibility if manifests, export profiles, or release
  packaging changed.

Keep unrelated refactors separate from behavior changes.


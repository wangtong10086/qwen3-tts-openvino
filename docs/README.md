# Documentation

This project uses one production inference path: `fastest` profile, native
paged-KV codec generation, vLLM-like online batching, and full-context
autoregressive long-text generation.

## Recommended Reading Order

| Role | Start Here | Purpose |
| --- | --- | --- |
| End user | [Release usage](release_zh.md) | Download the prebuilt runtime, auto-download the public OpenVINO IR, and open the Web Demo |
| Source developer | [Source quick start](quick_start_zh.md) | Export production IR from PyTorch weights and build the native backend |
| Integrator | [Runtime APIs](runtime_zh.md) | Use HTTP, WebSocket, and OpenAI-compatible Speech APIs |
| Maintainer | [Development](development_zh.md) | Run checks, package releases, and understand CI workflows |
| Performance/quality | [Streaming and long text](streaming_zh.md) | Validate full-AR long text, online batching, benchmarks, and quality gates |

Most detailed documents are currently written in Chinese because the active
development and validation notes are Chinese-first. The root [README.md](../README.md)
and [examples](../examples/README.md) provide English entry points.

## Documents

- [Release usage](release_zh.md)
- [Release notes](releases/v0.1.4.md)
- [Source quick start](quick_start_zh.md)
- [Runtime APIs](runtime_zh.md)
- [Export and build](export_zh.md)
- [Streaming and long text](streaming_zh.md)
- [OpenVINO compile cache](cache_zh.md)
- [Artifacts policy](artifacts_zh.md)
- [Windows GPU+NPU path](windows_gpu_npu_zh.md)
- [Scripts](../scripts/README.zh-CN.md)
- [Native backend](../native/qwen3_tts_ov_genai/README.md)
- [Security](security_zh.md)

## Current Scope

The repository does not store model weights, exported OpenVINO IR, generated
audio, compile caches, or native build outputs. Runtime application packages
are published through GitHub Releases; compiled OpenVINO IR is published through
the linked Hugging Face model repository.

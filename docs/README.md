# Documentation

This project uses one production inference path: `fastest` profile, native
paged-KV codec generation, vLLM-like online batching, and full-context
autoregressive long-text generation.

## Recommended Reading Order

| Role | Start Here | Purpose |
| --- | --- | --- |
| New machine | [Prerequisites](prerequisites.md) | Install uv, verify Python/OpenVINO, and confirm Intel GPU visibility |
| End user | [Release usage](release.md) | Download the prebuilt runtime, auto-download the public OpenVINO IR, and open the Web Demo |
| Source developer | [Source quick start](quick_start.md) | Export production IR from PyTorch weights and build the native backend |
| Integrator | [API Reference](api_reference.md) | Use HTTP, WebSocket, OpenAI-compatible Speech, and Python APIs |
| Maintainer | [Contributing](../CONTRIBUTING.md) | Run checks, update docs, and prepare PRs |
| Performance/quality | [Streaming and long text](streaming_zh.md) | Validate full-AR long text, online batching, benchmarks, and quality gates |

Core setup, troubleshooting, API, release, and source quick-start docs have
English entry points. Deeper development and validation notes remain
Chinese-first for now and are marked with `_zh`.

## Documents

- [Prerequisites](prerequisites.md) / [前置条件](prerequisites_zh.md)
- [Troubleshooting](troubleshooting.md) / [Troubleshooting / FAQ](troubleshooting_zh.md)
- [Release usage](release.md) / [Release 使用说明](release_zh.md)
- [Release notes](releases/v0.1.4.md)
- [Source quick start](quick_start.md) / [Quick Start](quick_start_zh.md)
- [API Reference](api_reference.md) / [API Reference 中文](api_reference_zh.md)
- [Runtime APIs](runtime_zh.md)
- [Export and build](export_zh.md)
- [Streaming and long text](streaming_zh.md)
- [OpenVINO compile cache](cache_zh.md)
- [Artifacts policy](artifacts_zh.md)
- [Windows GPU+NPU path](windows_gpu_npu_zh.md)
- [Scripts](../scripts/README.zh-CN.md)
- [Native backend](../native/qwen3_tts_ov_genai/README.md)
- [Security](security_zh.md)
- [Contributing](../CONTRIBUTING.md)

## Current Scope

The repository does not store model weights, exported OpenVINO IR, generated
audio, compile caches, or native build outputs. Runtime application packages
are published through GitHub Releases; compiled OpenVINO IR is published through
the linked Hugging Face model repository.

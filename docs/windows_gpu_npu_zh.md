# Windows GPU+NPU 测试路径

该路径用于验证 Windows 原生环境下的异构推理：

```text
GPU: native codegen / paged-KV
NPU: streaming speech decoder
```

WSL 当前不作为 NPU 验证环境。测试需要 Windows 原生 Intel GPU/NPU 驱动，并且 OpenVINO `Core().available_devices` 能看到 `GPU` 和 `NPU`。

## 本地 PowerShell

在 Windows PowerShell 中运行：

```powershell
.\scripts\windows_gpu_npu_smoke.ps1
```

脚本会执行：

1. 构建 native C++ pipeline。
2. 打包 Windows runtime-minimal release。
3. 下载 Hugging Face `openvino_realtime/**` IR。
4. 运行 `probe_windows_gpu_npu.py`。
5. 如果 probe 成功，启动 release server 并执行真实 streaming TTS。

常用参数：

```powershell
.\scripts\windows_gpu_npu_smoke.ps1 `
  -Device GPU `
  -DecoderDevice NPU `
  -NpuOffload decoder `
  -MaxNewTokens 8
```

如果机器没有 NPU，默认会输出 skipped summary 并正常退出。需要强制失败时加：

```powershell
.\scripts\windows_gpu_npu_smoke.ps1 -Strict
```

测试产物：

```text
build/windows-gpu-npu-smoke/probe.json
build/windows-gpu-npu-smoke/summary.json
build/windows-gpu-npu-smoke/server.log
```

## 服务端 NPU Offload 参数

release server 和源码 CLI 都支持同一组参数：

```powershell
qwen3-tts-ov-server.exe --device GPU --npu-offload auto
qwen3-tts-ov-server.exe --device GPU --npu-offload decoder
qwen3-tts-ov-server.exe --device GPU --npu-offload audio
```

```bash
uv run python -m qwen3_tts_ov serve --device GPU --npu-offload auto
```

含义：

- `off`: 默认兼容模式，不自动使用 NPU；如果显式传入 `--decoder-device NPU`，仍按显式设备运行。
- `auto`: 主设备是 GPU、且 OpenVINO 能看到 `NPU` 时，自动把 streaming decoder 放到 NPU；没有 NPU 时退回 GPU。
- `decoder`: 强制把 streaming decoder 放到 NPU；缺少 NPU 或同时传入非 NPU 的 `--decoder-device` 会启动失败。
- `audio`: 强制把 streaming decoder 和 VoiceClone 参考音频侧 encoder 放到 NPU，用于进一步降低 GPU 负载。
- `require`: 当前等价于 `decoder` 的严格模式，用于 CI 或部署验收。

`/health` 和流式 metadata 会返回 `decoder_device`、`encoder_device`、`speech_encoder_device`、`speaker_encoder_device`、`npu_offload_requested`、`npu_offload_effective`、`npu_offload_reason`，用于确认实际是否命中 GPU codegen + NPU audio path。

## GPU-only 与 GPU+NPU 对比

功能 smoke 只能证明路径可运行，不能证明性能收益。要对比 NPU 是否降低 decoder 对 GPU 的占用或改善端到端指标，运行：

```powershell
.\scripts\windows_gpu_npu_benchmark.ps1 `
  -Device GPU `
  -NpuOffload audio `
  -Runs 2 `
  -MaxNewTokens 48
```

该脚本会使用同一个 release 包和同一份 IR 依次启动两组服务：

```text
gpu_only:       --device GPU --npu-offload off
gpu_npu_audio:  --device GPU --npu-offload audio
```

输出文件：

```text
build/windows-gpu-npu-benchmark/benchmark-summary.json
build/windows-gpu-npu-benchmark/gpu_only/server.log
build/windows-gpu-npu-benchmark/gpu_npu_audio/server.log
```

重点看 `comparison.computed_rtf_speedup`、每组 `decoder_device`、`speaker_encoder_device`、`npu_offload_effective` 和 `median_computed_rtf`。如果 `decoder_device=NPU` 但 RTF 没有改善，这说明当前瓶颈仍主要在 GPU codegen/paged-KV，而 NPU offload 主要价值是降低 GPU 音频侧负载。

## GitHub Actions

测试 workflow 位于：

```text
.github/workflows/windows-gpu-npu.yml
```

默认面向自托管 Windows runner：

```text
self-hosted, Windows, X64, npu
```

触发方式：

- push 到 `test/windows-gpu-npu-path` 或 `test/windows-gpu-npu-*`
- 手动运行 `windows-gpu-npu`

无 NPU 或 NPU decoder 编译失败时，默认标记为 skipped 并上传 artifact；手动运行时可设置 `strict=true`，使缺设备或编译失败直接失败。

## 零拷贝 Probe

`probe_windows_gpu_npu.py` 会记录 OpenVINO Python remote-context API 的可见性：

```bash
uv run python scripts/probe_windows_gpu_npu.py \
  --model-root build/hf-ir/openvino_realtime \
  --device GPU \
  --decoder-device NPU \
  --skip-if-missing-devices \
  --output-json build/gpu-npu-probe/probe.json
```

当前零拷贝 probe 只作为诊断信息。真正的 GPU/NPU shared-handle zero-copy 需要 native handle 和 RemoteTensor 集成，未作为默认推理路径启用。需要把 remote-context 可用性作为硬性要求时，传入 `--require-zero-copy`。

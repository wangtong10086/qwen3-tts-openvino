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

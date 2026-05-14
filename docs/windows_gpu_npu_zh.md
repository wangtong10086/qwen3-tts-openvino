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

`-NpuOffload audio` 会额外断言 `encoder_device`、`speech_encoder_device`、`speaker_encoder_device` 都为 NPU；`-NpuOffload all` 还会断言 `prompt_device` 和 `text_embedding_device` 为 NPU。这样 smoke 不只验证“能跑”，也验证实际 offload 范围。

如果要让 smoke 真正执行 VoiceClone 参考音频 encoder，使用：

```powershell
.\scripts\windows_gpu_npu_smoke.ps1 `
  -Mode voice_clone `
  -NpuOffload audio `
  -RefAudio C:\path\ref.wav `
  -RefText "参考音频对应文本"
```

VoiceDesign 不会执行 `speech_encoder/speaker_encoder`，因此 audio encoder 的真实性能收益必须通过 VoiceClone smoke 或 `-Mode voice_clone` benchmark 判断。

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
qwen3-tts-ov-server.exe --device GPU --npu-offload all
```

```bash
uv run python -m qwen3_tts_ov serve --device GPU --npu-offload auto
```

含义：

- `off`: 默认兼容模式，不自动使用 NPU；如果显式传入 `--decoder-device NPU`，仍按显式设备运行。
- `auto`: 主设备是 GPU、且 OpenVINO 能看到 `NPU` 时，自动把 streaming decoder 放到 NPU；没有 NPU 时退回 GPU。
- `decoder`: 强制把 streaming decoder 放到 NPU；缺少 NPU 或同时传入非 NPU 的 `--decoder-device` 会启动失败。
- `audio`: 强制把 streaming decoder 和 VoiceClone 参考音频侧 encoder 放到 NPU，用于进一步降低 GPU 负载。
- `all`: 在 `audio` 基础上，把 prompt/text embedding 也放到 NPU，用于测试更激进的 GPU 卸载路径。
- `require`: 当前等价于 `decoder` 的严格模式，用于 CI 或部署验收。

`/health` 和流式 metadata 会返回 `decoder_device`、`encoder_device`、`prompt_device`、`speech_encoder_device`、`speaker_encoder_device`、`npu_offload_requested`、`npu_offload_effective`、`npu_offload_reason`，用于确认实际是否命中 GPU codegen + NPU audio/prompt path。

## GPU-only 与 GPU+NPU 对比

功能 smoke 只能证明路径可运行，不能证明性能收益。要对比 NPU 是否降低 decoder 对 GPU 的占用或改善端到端指标，运行：

```powershell
.\scripts\windows_gpu_npu_benchmark.ps1 `
  -Device GPU `
  -Scenarios gpu_only,npu_decoder,npu_audio `
  -Runs 2 `
  -MaxNewTokens 48
```

该脚本会先运行 NPU probe，再使用同一个 release 包和同一份 IR 依次启动多组服务，最后生成 `analysis.json`：

```text
gpu_only:    --device GPU --npu-offload off
npu_decoder: --device GPU --npu-offload decoder
npu_audio:   --device GPU --npu-offload audio
npu_all:     --device GPU --npu-offload all
```

输出文件：

```text
build/windows-gpu-npu-benchmark/benchmark-summary.json
build/windows-gpu-npu-benchmark/probe.json
build/windows-gpu-npu-benchmark/analysis.json
build/windows-gpu-npu-benchmark/gpu_only/server.log
build/windows-gpu-npu-benchmark/npu_decoder/server.log
build/windows-gpu-npu-benchmark/npu_audio/server.log
```

重点看 `recommendation.recommended_npu_offload`、`comparison.npu_decoder.computed_rtf_speedup`、`comparison.npu_audio.computed_rtf_speedup`，以及每组 `decoder_device`、`speaker_encoder_device`、`npu_offload_effective` 和 `median_computed_rtf`。如果 `decoder_device=NPU` 但 RTF 没有改善，这说明当前瓶颈仍主要在 GPU codegen/paged-KV，而 NPU offload 主要价值是降低 GPU 音频侧负载。

`benchmark-summary.json` 的每个场景还会记录 `summary.exercised_runtime_stages` 和 `summary.npu_offload_coverage`。例如 VoiceDesign 请求不会实际执行 `speech_encoder/speaker_encoder`，所以 `npu_audio` 会显示这些 stage 位于 `unexercised_npu_stages`。要真实覆盖 audio encoder，需要用 `-Mode voice_clone -RefAudio ...` 跑 benchmark。

需要把“声明 NPU offload 的 stage 必须被本次请求实际执行”作为硬门禁时，加：

```powershell
.\scripts\windows_gpu_npu_benchmark.ps1 `
  -Mode voice_clone `
  -RefAudio C:\path\ref.wav `
  -RefText "参考音频对应文本" `
  -Scenarios gpu_only,npu_decoder,npu_audio `
  -RequireExercisedNpuStages
```

`benchmark-summary.json` 会给出三类推荐：

- `recommendation.fastest`: RTF 最低的 NPU 场景。
- `recommendation.lowest_gpu_utilization`: GPU utilization 降幅最大的 NPU 场景，需要 `-CollectCounters` 才有完整依据。
- `recommendation.balanced` / `recommended_npu_offload`: 在允许的 RTF 回退范围内优先降低 GPU 负载的部署建议。

release server 可以直接读取 benchmark 结果并应用推荐配置：

```powershell
qwen3-tts-ov-server.exe `
  --model-root build\hf-ir\openvino_realtime `
  --device GPU `
  --npu-offload-summary build\windows-gpu-npu-benchmark\benchmark-summary.json `
  --npu-offload-policy balanced
```

`--npu-offload-policy` 可选 `balanced`、`fastest`、`lowest-gpu` 或 `recommended`。生产部署默认建议 `balanced`：在不明显牺牲 RTF 的前提下优先降低 GPU 负载。

源码开发入口也支持同一份 summary：

```powershell
uv run python -m qwen3_tts_ov serve `
  --model-root build\hf-ir\openvino_realtime `
  --device GPU `
  --npu-offload-summary build\windows-gpu-npu-benchmark\benchmark-summary.json

uv run python -m qwen3_tts_ov cache-warmup `
  --ir-dir build\hf-ir\openvino_realtime\voice_design `
  --device GPU `
  --npu-offload-summary build\windows-gpu-npu-benchmark\benchmark-summary.json
```

要额外测试 prompt/text embedding 是否适合放到 NPU，显式加入 `npu_all`：

```powershell
.\scripts\windows_gpu_npu_benchmark.ps1 `
  -Scenarios gpu_only,npu_decoder,npu_audio,npu_all `
  -Runs 2 `
  -MaxNewTokens 48
```

如果 `npu_all` 比 `npu_audio` 更慢或 NPU 编译失败，保持生产路径使用 `audio`；如果 `npu_all` 的 GPU utilization 降幅更明显且 RTF 不回退，再考虑把它作为特定机器的部署配置。

要直接观察 GPU 负载是否下降，在 Windows 上启用性能计数器采样：

```powershell
.\scripts\windows_gpu_npu_benchmark.ps1 `
  -Scenarios gpu_only,npu_decoder,npu_audio `
  -CollectCounters
```

每个场景会额外生成：

```text
build/windows-gpu-npu-benchmark/<scenario>/accelerator-counters.json
build/windows-gpu-npu-benchmark/<scenario>/accelerator-counters.log
```

`benchmark-summary.json` 中的 `summary.accelerator_counters.gpu.utilization_average` 和 `summary.accelerator_counters.npu.utilization_average` 来自 Windows `Get-Counter` 自动发现的 GPU/NPU utilization/usage/busy/load 计数器。默认 `-CounterScope server` 会优先按 release server PID 过滤进程级 GPU/NPU engine counter，拿不到进程级路径时回退到系统级并标记 `selected_scope=system_fallback`。需要观察整机负载时使用：

```powershell
.\scripts\windows_gpu_npu_benchmark.ps1 `
  -Scenarios gpu_only,npu_decoder,npu_audio `
  -CollectCounters `
  -CounterScope system
```

计数器名称会因驱动和 Windows 版本不同而变化；如果机器没有暴露相关计数器，结果会标记为 `no_counters`，不会影响普通 benchmark。

需要把性能收益变成硬性门禁时，增加阈值参数：

```powershell
.\scripts\windows_gpu_npu_benchmark.ps1 `
  -Scenarios gpu_only,npu_decoder,npu_audio `
  -MinSpeedup 1.02 `
  -MaxRtfRegression 0.03 `
  -CollectCounters `
  -MinGpuUtilizationReduction 0.05
```

`-MinSpeedup` 要求每个 NPU 场景的 RTF speedup 不低于阈值；`-MaxRtfRegression` 允许 NPU 场景比 GPU-only 慢的最大 RTF 差值；`-MinGpuUtilizationReduction` 要求平均 GPU utilization 相对 GPU-only 至少下降指定比例。失败会使脚本退出非零，并把失败原因写入 `analysis.json`。

需要把 prompt 或 VoiceClone audio encoder 的 NPU 编译也作为硬门禁时，增加：

```powershell
.\scripts\windows_gpu_npu_benchmark.ps1 `
  -Scenarios gpu_only,npu_decoder,npu_audio,npu_all `
  -Strict `
  -RequirePromptCompile `
  -RequireAudioCompile
```

如果已经有 workflow artifact 或本地 benchmark 结果，可以离线审计：

```powershell
uv run python scripts/analyze_windows_gpu_npu_results.py `
  --benchmark-summary build/windows-gpu-npu-benchmark/benchmark-summary.json `
  --probe-json build/gpu-npu-probe/probe.json `
  --require-scenarios gpu_only,npu_decoder,npu_audio,npu_all `
  --require-probe-ok `
  --require-counters `
  --min-speedup 1.02 `
  --max-rtf-regression 0.03 `
  --min-gpu-utilization-reduction 0.05 `
  --output-json build/windows-gpu-npu-benchmark/analysis.json
```

该报告会检查实际 `npu_offload_effective`、`decoder_device`、`encoder_device`、`prompt_device/text_embedding_device`、NPU probe 编译结果、RTF 和 GPU utilization 降幅。失败时脚本退出非零，适合做发布前验收。

要实际触发 VoiceClone 的参考音频 encoder，可在含 `base/` IR 的模型根目录上增加：

```powershell
.\scripts\windows_gpu_npu_benchmark.ps1 `
  -Mode voice_clone `
  -RefAudio C:\path\ref.wav `
  -RefText "参考音频对应文本" `
  -Scenarios gpu_only,npu_decoder,npu_audio
```

## GitHub Actions

该分支的 GitHub workflow 位于：

```text
.github/workflows/windows-gpu-npu.yml
```

它只负责在 hosted runner 上构建 runtime 包：

```text
ubuntu-latest  -> linux-x64 runtime-minimal
windows-latest -> windows-x64 runtime-minimal
```

触发后会执行：

- push 到 `test/windows-gpu-npu-path` 或 `test/windows-gpu-npu-*`
- 手动运行 `windows-gpu-npu`
- Python 编译和单元测试
- Linux/Windows native runtime 构建
- `runtime-minimal` 打包
- `smoke_release_package.py` 基础启动检查
- 上传 runtime archive 和 smoke log artifact

GitHub Actions 不下载模型 IR、不探测 NPU、不运行真实 TTS、不做 GPU-only/GPU+NPU benchmark。Windows NPU 的正确性和性能必须在本地 Windows 原生机器上通过 `windows_gpu_npu_smoke.ps1` 和 `windows_gpu_npu_benchmark.ps1` 验证。

## 零拷贝 Probe

`probe_windows_gpu_npu.py` 会先编译 VoiceDesign streaming decoder 到 NPU，并额外尝试把 VoiceDesign prompt 相关的 `text_embedding`、`codec_embedding` 编译到 NPU，用于判断 `--npu-offload all` 是否可行。如果模型根目录还包含 `base/manifest.json`，它也会尝试把 VoiceClone 需要的 `speech_encoder` 和 `speaker_encoder` 编译到 NPU。最后会记录 OpenVINO Python remote-context API 的可见性：

```bash
uv run python scripts/probe_windows_gpu_npu.py \
  --model-root build/hf-ir/openvino_realtime \
  --device GPU \
  --decoder-device NPU \
  --skip-if-missing-devices \
  --output-json build/gpu-npu-probe/probe.json
```

当前零拷贝 probe 只作为诊断信息。真正的 GPU/NPU shared-handle zero-copy 需要 native handle 和 RemoteTensor 集成，未作为默认推理路径启用。需要把 remote-context 可用性作为硬性要求时，传入 `--require-zero-copy`。

只想验证 streaming decoder、不验证 prompt 或 Base/VoiceClone 音频 encoder 时，传入 `--skip-prompt-graphs` 或 `--skip-audio-encoders`。

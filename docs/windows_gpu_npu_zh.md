# Windows GPU+NPU 测试路径

该路径用于验证 Windows 原生环境下的异构推理：

```text
GPU: native codegen / paged-KV
NPU: streaming speech decoder
```

WSL 当前不作为 NPU 验证环境。测试需要 Windows 原生 Intel GPU/NPU 驱动，并且 OpenVINO `Core().available_devices` 能看到 `GPU` 和 `NPU`。

## Windows 原生源码完整流程

先用 GPU-only 路径跑通下载、native runtime 构建、导出和服务，再验证 NPU offload。下面命令假设在仓库根目录执行。

1. 准备 PowerShell 环境：

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
uv --version
cmake --version
where.exe cl
```

`where.exe cl` 找不到时，安装 Visual Studio 2022 Build Tools，并勾选 C++ build tools / Windows SDK。

2. 安装依赖并准备官方 Qwen3-TTS 源码：

```powershell
uv sync --extra native --extra server --extra export
git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS .cache\Qwen3-TTS
$env:PYTHONPATH = (Resolve-Path .cache\Qwen3-TTS).Path
uv run python -c "import qwen_tts; print('qwen_tts ok')"
```

3. 下载 VoiceDesign PyTorch 模型：

```powershell
uv run modelscope download `
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign `
  --local_dir .\models\Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

4. 构建 native runtime：

```powershell
uv run python scripts\build_native_codegen.py --backend cmake --config Release
```

成功产物是：

```text
native/build/qwen3_tts_ov_genai.dll
```

5. 导出并预热 OpenVINO IR：

```powershell
uv run python -m qwen3_tts_ov build-fastest `
  --model-type voice_design `
  --model models\Qwen3-TTS-12Hz-1.7B-VoiceDesign `
  --out-dir openvino\voice_design `
  --device GPU
```

6. 启动服务并 smoke：

```powershell
uv run python -m qwen3_tts_ov serve `
  --model-root openvino `
  --device GPU `
  --realtime-profile fastest `
  --host 127.0.0.1 `
  --port 17860
```

另开一个 PowerShell：

```powershell
Invoke-RestMethod http://127.0.0.1:17860/health
uv run python examples\python\http_tts_wav.py `
  --server http://127.0.0.1:17860 `
  --output outputs\windows_smoke.wav `
  --max-new-tokens 24
```

确认 `outputs/windows_smoke.wav` 非空后，再继续下面的 GPU+NPU 验证。

## WSL 导出 NPU 静态 IR

Windows NPU 编译要求 streaming decoder 使用固定输入 shape。可以在 WSL 中导出 IR，再复制到 Windows 原生 runtime 验证。下面用 `D:\qwen3-tts-ov-npu-build` 作为临时构建目录示例。WSL 中同样需要先准备官方 Qwen3-TTS 源码，并把它加入当前 shell 的 `PYTHONPATH`：

```bash
uv sync --extra native --extra server --extra export
git clone --depth 1 https://github.com/QwenLM/Qwen3-TTS .cache/Qwen3-TTS
export PYTHONPATH="$(pwd)/.cache/Qwen3-TTS"
uv run python -c "import qwen_tts; print('qwen_tts ok')"

uv run python -m qwen3_tts_ov export \
  --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --model-type voice_design \
  --out-dir /mnt/d/qwen3-tts-ov-npu-build/ir/openvino/voice_design \
  --skip-fixed-cache-graphs \
  --cache-buckets 96 \
  --cache-kernels exact \
  --fused-cache-kernels exact \
  --fused-subcode-mode cached \
  --fused-cache-unroll-steps "" \
  --fused-cache-decode-unroll-steps "" \
  --fused-cache-stateful-mask-steps "" \
  --fused-cache-norepeat-steps "" \
  --export-paged-kv-seed \
  --paged-kv-unroll-steps "" \
  --paged-kv-subcode-attention-kernels sdpa \
  --decoder-tokens 256 \
  --stream-decoder-first-chunks 8,12 \
  --stream-decoder-chunks 12,24 \
  --stream-decoder-left-context 25 \
  --stream-decoder-input-shape static

uv run python scripts/compress_openvino_weights.py \
  --ir-dir /mnt/d/qwen3-tts-ov-npu-build/ir/openvino/voice_design \
  --preset fastest
```

导出器会把 `vocab.json`、`merges.txt`、`tokenizer_config.json` 复制到 IR 目录，并把 manifest 中的 `model_dir` 写为 `.`。这样从 WSL 导出的目录可以直接在 Windows runtime 中使用，不会依赖 `/home/...` 这类 Linux 绝对路径。

如果要按 release 方式分发，建议再打包成 runtime-minimal IR：

```bash
uv run python scripts/package_ir.py \
  --ir-dir /mnt/d/qwen3-tts-ov-npu-build/ir/openvino/voice_design \
  --model-type voice_design \
  --version npu-static \
  --profile runtime-minimal \
  --format zip \
  --out-dir /mnt/d/qwen3-tts-ov-npu-build
```

导出的 streaming decoder 关键 shape 应为：

```text
speech_decoder_stream_c0_t8.xml    [1,8,16]
speech_decoder_stream_c0_t12.xml   [1,12,16]
speech_decoder_stream_c25_t12.xml  [1,37,16]
speech_decoder_stream_c25_t24.xml  [1,49,16]
```

Windows 严格验证使用 `--npu-offload decoder`；该模式不会静默回退 GPU，适合判断 NPU decoder 是否真正可编译。

## 干净 Windows 测试目录

建议把 Windows runtime 和已打包的 IR 解压到一个单独目录，避免旧 OpenVINO compile cache、旧 raw IR 或多份 CI 产物影响判断：

```text
D:\qwen3-tts-ov-clean-test\
  runtime\                  # qwen3-tts-ov-server.exe 和 _internal/
  model\openvino\
    voice_design\manifest.json
  ov-cache\                 # 首次运行时生成，可随时删除重建
  logs\                     # 本地测试日志
```

Windows PowerShell 启动 GPU-only 路径：

```powershell
cd D:\qwen3-tts-ov-clean-test\runtime

.\qwen3-tts-ov-server.exe `
  --model-root D:\qwen3-tts-ov-clean-test\model\openvino `
  --host 127.0.0.1 `
  --port 17860 `
  --device GPU `
  --npu-offload off `
  --ov-cache-dir D:\qwen3-tts-ov-clean-test\ov-cache
```

严格验证 NPU decoder 路径：

```powershell
.\qwen3-tts-ov-server.exe `
  --model-root D:\qwen3-tts-ov-clean-test\model\openvino `
  --host 127.0.0.1 `
  --port 17860 `
  --device GPU `
  --npu-offload decoder `
  --ov-cache-dir D:\qwen3-tts-ov-clean-test\ov-cache
```

`--npu-offload decoder/audio/all` 在主设备为 GPU 且缓存参数保持 `auto/default` 时，会自动使用更保守的内存默认值：`kv_cache_max_blocks=128`、`online_batch_max_cache_blocks=128`、`max_vram_ratio=50%`。这是为了避免部分 Windows Intel GPU+NPU 组合在 decoder 放到 NPU 后仍因 GPU 侧 paged-KV/online batching 预留过大而出现 USM allocation 或 output memory 错误。显式传入的 `--kv-cache-max-blocks`、`--online-batch-max-cache-blocks`、`--max-vram-ratio` 会覆盖这些默认值。

打开 Web Demo：

```text
http://127.0.0.1:17860/
```

验收时重点看 `/health` 或 Web Demo 日志中的 metadata：

```text
decoder_device=NPU
npu_offload_effective=decoder
```

如果要从空 cache 重新验证编译行为，先停止服务并删除：

```powershell
Remove-Item -Recurse -Force D:\qwen3-tts-ov-clean-test\ov-cache
New-Item -ItemType Directory D:\qwen3-tts-ov-clean-test\ov-cache | Out-Null
```

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

`decoder/audio/all` 会在缓存参数保持 `auto/default` 时自动收紧内存预算。`/health` 和流式 metadata 会返回 `decoder_device`、`encoder_device`、`prompt_device`、`speech_encoder_device`、`speaker_encoder_device`、`npu_offload_requested`、`npu_offload_effective`、`npu_offload_reason`、`npu_decoder_memory_defaults_applied`、`npu_decoder_memory_defaults`，用于确认实际是否命中 GPU codegen + NPU audio/prompt path，以及是否应用了保守内存默认值。

如果要手动复现同一套保守配置，可以显式传入：

```powershell
uv run python -m qwen3_tts_ov serve `
  --device GPU `
  --npu-offload decoder `
  --kv-cache-max-blocks 128 `
  --online-batch-max-cache-blocks 128 `
  --max-vram-ratio 50
```

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

`probe_windows_gpu_npu.py` 默认只编译 VoiceDesign streaming decoder 到 NPU，这是 `--npu-offload decoder` 的必要图。最后还会记录 OpenVINO Python remote-context API 的可见性：

```bash
uv run python scripts/probe_windows_gpu_npu.py \
  --model-root build/hf-ir/openvino_realtime \
  --device GPU \
  --decoder-device NPU \
  --skip-if-missing-devices \
  --output-json build/gpu-npu-probe/probe.json
```

当前零拷贝 probe 只作为诊断信息。真正的 GPU/NPU shared-handle zero-copy 需要 native handle 和 RemoteTensor 集成，未作为默认推理路径启用。需要把 remote-context 可用性作为硬性要求时，传入 `--require-zero-copy`。

需要额外判断 `--npu-offload all` 是否可行时，加 `--check-prompt-graphs`，它会尝试把 VoiceDesign prompt 相关的 `text_embedding`、`codec_embedding` 编译到 NPU。需要额外判断 `--npu-offload audio` 是否可行时，加 `--check-audio-encoders`，它会在模型根目录包含 `base/manifest.json` 时尝试把 VoiceClone 需要的 `speech_encoder` 和 `speaker_encoder` 编译到 NPU。

这些额外图不是 decoder offload 的必要条件。部分 OpenVINO/NPU driver 组合会因为 prompt 或 audio encoder 的动态 shape 上界缺失而拒绝编译，甚至在底层 NPU compiler 中断；此时保持生产路径使用 `--npu-offload decoder` 或 `auto`，不要把 `audio/all` 作为该机器的部署配置。

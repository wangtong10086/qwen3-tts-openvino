param(
  [string]$Archive = "",
  [string]$ModelRoot = "build/hf-ir/openvino_realtime",
  [string]$Version = "gpu-npu-smoke",
  [string]$WorkDir = "build/windows-gpu-npu-benchmark",
  [string]$Device = "GPU",
  [string]$NpuOffload = "decoder",
  [string]$HfRepo = "waston10086/qwen3-tts-openvino-voice-design",
  [string]$HfRevision = "main",
  [string]$Text = "你好，这是 Windows GPU 加 NPU 推理性能对比测试。",
  [int]$MaxNewTokens = 48,
  [int]$Runs = 2,
  [int]$BasePort = 17990,
  [switch]$SkipBuild,
  [switch]$NoWarmup,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = "utf-8"

if (-not $SkipBuild) {
  uv sync --extra native --extra server --extra release
  uv run python scripts/build_native_codegen.py --backend cmake
  uv run python scripts/package_release.py `
    --target windows-x64 `
    --version $Version `
    --profile runtime-minimal
}

if (-not (Test-Path $ModelRoot)) {
  uv run --with huggingface_hub python scripts/download_hf_ir.py `
    --repo-id $HfRepo `
    --revision $HfRevision `
    --local-dir build/hf-ir `
    --allow-pattern "openvino_realtime/**"
}

if (-not $Archive) {
  $Archive = "dist/release/qwen3-tts-ov-server-windows-x64-$Version-runtime-minimal.zip"
}

$argsList = @(
  "scripts/benchmark_windows_gpu_npu_release.py",
  "--archive", $Archive,
  "--model-root", $ModelRoot,
  "--work-dir", $WorkDir,
  "--base-port", "$BasePort",
  "--device", $Device,
  "--npu-offload", $NpuOffload,
  "--require-devices", "$Device,NPU",
  "--text", $Text,
  "--max-new-tokens", "$MaxNewTokens",
  "--runs", "$Runs",
  "--chunk-strategy", "smooth",
  "--summary-out", "$WorkDir/benchmark-summary.json"
)
if (-not $Strict) {
  $argsList += "--skip-if-missing-devices"
}
if ($NoWarmup) {
  $argsList += "--no-warmup"
}

uv run python @argsList

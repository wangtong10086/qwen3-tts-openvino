param(
  [string]$Archive = "",
  [string]$ModelRoot = "build/hf-ir/openvino_realtime",
  [string]$Version = "gpu-npu-smoke",
  [string]$WorkDir = "build/windows-gpu-npu-smoke",
  [string]$Device = "GPU",
  [string]$DecoderDevice = "NPU",
  [string]$HfRepo = "waston10086/qwen3-tts-openvino-voice-design",
  [string]$HfRevision = "main",
  [string]$Text = "你好，这是 Windows GPU 加 NPU 推理测试。",
  [int]$MaxNewTokens = 8,
  [int]$Port = 17981,
  [switch]$Strict,
  [switch]$RequireZeroCopy,
  [switch]$SkipBuild
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

$probeArgs = @(
  "scripts/probe_windows_gpu_npu.py",
  "--model-root", $ModelRoot,
  "--device", $Device,
  "--decoder-device", $DecoderDevice,
  "--cache-dir", "$WorkDir/probe-ov-cache",
  "--output-json", "$WorkDir/probe.json"
)
if (-not $Strict) {
  $probeArgs += "--skip-if-missing-devices"
}
if ($RequireZeroCopy) {
  $probeArgs += "--require-zero-copy"
}

uv run python @probeArgs
$probe = Get-Content "$WorkDir/probe.json" -Raw | ConvertFrom-Json
if ($probe.status -ne "ok") {
  Write-Host "GPU+NPU smoke skipped: $($probe.skip_reason)"
  exit 0
}

uv run python scripts/smoke_release_tts.py `
  --archive $Archive `
  --model-root $ModelRoot `
  --work-dir $WorkDir `
  --port $Port `
  --device $Device `
  --decoder-device $DecoderDevice `
  --require-devices "$Device,$DecoderDevice" `
  --skip-if-missing-devices `
  --expect-native-codegen-device $Device `
  --expect-decoder-device $DecoderDevice `
  --text $Text `
  --max-new-tokens $MaxNewTokens `
  --do-sample false `
  --chunk-strategy smooth `
  --summary-out "$WorkDir/summary.json"

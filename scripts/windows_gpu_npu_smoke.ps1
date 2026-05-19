param(
  [string]$Archive = "",
  [string]$ModelRoot = "build/hf-ir/openvino_realtime",
  [string]$Version = "gpu-npu-smoke",
  [string]$WorkDir = "build/windows-gpu-npu-smoke",
  [string]$Device = "GPU",
  [string]$DecoderDevice = "NPU",
  [ValidateSet("decoder", "audio", "all", "auto", "require")]
  [string]$NpuOffload = "decoder",
  [string]$HfRepo = "waston10086/qwen3-tts-openvino-voice-design",
  [string]$HfRevision = "main",
  [ValidateSet("voice_design", "voice_clone")]
  [string]$Mode = "voice_design",
  [string]$Text = "你好，这是 Windows GPU 加 NPU 推理测试。",
  [string]$RefAudio = "",
  [string]$RefText = "",
  [switch]$XVectorOnly,
  [int]$MaxNewTokens = 8,
  [int]$Port = 17981,
  [switch]$Strict,
  [switch]$RequireZeroCopy,
  [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = "utf-8"

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,
    [Parameter(Mandatory = $true)]
    [string[]]$Arguments
  )
  & $FilePath @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "$FilePath failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
  }
}

if (-not $SkipBuild) {
  Invoke-Checked "uv" @("sync", "--extra", "native", "--extra", "server", "--extra", "release")
  Invoke-Checked "uv" @("run", "python", "scripts/build_native_codegen.py", "--backend", "cmake")
  Invoke-Checked "uv" @(
    "run", "python", "scripts/package_release.py",
    "--target", "windows-x64",
    "--version", $Version,
    "--profile", "runtime-minimal"
  )
}

if (-not (Test-Path $ModelRoot)) {
  Invoke-Checked "uv" @(
    "run", "--with", "huggingface_hub", "python", "scripts/download_hf_ir.py",
    "--repo-id", $HfRepo,
    "--revision", $HfRevision,
    "--local-dir", "build/hf-ir",
    "--allow-pattern", "openvino_realtime/**"
  )
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
if ($NpuOffload -eq "audio" -or $NpuOffload -eq "all") {
  $probeArgs += "--check-audio-encoders"
}
if ($NpuOffload -eq "all") {
  $probeArgs += "--check-prompt-graphs"
}

Invoke-Checked "uv" (@("run", "python") + $probeArgs)
$probe = Get-Content "$WorkDir/probe.json" -Raw | ConvertFrom-Json
if ($probe.status -ne "ok") {
  Write-Host "GPU+NPU smoke skipped: $($probe.skip_reason)"
  exit 0
}

$expectedNpuOffload = if ($NpuOffload -eq "require" -or $NpuOffload -eq "auto") { "decoder" } else { $NpuOffload }
$smokeArgs = @(
  "run", "python", "scripts/smoke_release_tts.py",
  "--archive", $Archive,
  "--model-root", $ModelRoot,
  "--work-dir", $WorkDir,
  "--port", "$Port",
  "--device", $Device,
  "--npu-offload", $NpuOffload,
  "--require-devices", "$Device,$DecoderDevice",
  "--skip-if-missing-devices",
  "--expect-native-codegen-device", $Device,
  "--expect-decoder-device", $DecoderDevice,
  "--expect-npu-offload-effective", $expectedNpuOffload,
  "--mode", $Mode,
  "--text", $Text,
  "--max-new-tokens", "$MaxNewTokens",
  "--do-sample", "false",
  "--chunk-strategy", "smooth",
  "--summary-out", "$WorkDir/summary.json"
)
if ($Mode -eq "voice_clone") {
  if (-not $RefAudio) {
    throw "-Mode voice_clone requires -RefAudio"
  }
  $smokeArgs += @("--ref-audio", $RefAudio)
  if ($RefText) {
    $smokeArgs += @("--ref-text", $RefText)
  }
  if ($XVectorOnly) {
    $smokeArgs += "--x-vector-only"
  }
}
if (($NpuOffload -eq "audio" -or $NpuOffload -eq "all") -and $Mode -eq "voice_clone") {
  $smokeArgs += @(
    "--expect-encoder-device", $DecoderDevice,
    "--expect-speaker-encoder-device", $DecoderDevice
  )
  if (-not $XVectorOnly) {
    $smokeArgs += @("--expect-speech-encoder-device", $DecoderDevice)
  }
}
if ($NpuOffload -eq "all") {
  $smokeArgs += @(
    "--expect-prompt-device", $DecoderDevice,
    "--expect-text-embedding-device", $DecoderDevice
  )
}

Invoke-Checked "uv" $smokeArgs

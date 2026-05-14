param(
  [string]$Archive = "",
  [string]$ModelRoot = "build/hf-ir/openvino_realtime",
  [string]$Version = "gpu-npu-smoke",
  [string]$WorkDir = "build/windows-gpu-npu-benchmark",
  [string]$Device = "GPU",
  [string]$Scenarios = "gpu_only,npu_decoder,npu_audio",
  [string]$HfRepo = "waston10086/qwen3-tts-openvino-voice-design",
  [string]$HfRevision = "main",
  [string]$Mode = "voice_design",
  [string]$Text = "你好，这是 Windows GPU 加 NPU 推理性能对比测试。",
  [string]$RefAudio = "",
  [string]$RefText = "",
  [int]$MaxNewTokens = 48,
  [int]$Runs = 2,
  [int]$BasePort = 17990,
  [ValidateSet("server", "system")]
  [string]$CounterScope = "server",
  [double]$MinSpeedup = -1,
  [double]$MaxRtfRegression = -1,
  [double]$MinGpuUtilizationReduction = -1,
  [switch]$SkipBuild,
  [switch]$SkipProbe,
  [switch]$NoWarmup,
  [switch]$CollectCounters,
  [switch]$RequirePromptCompile,
  [switch]$RequireAudioCompile,
  [switch]$RequireZeroCopy,
  [switch]$Strict
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

$probeJson = "$WorkDir/probe.json"
if (-not $SkipProbe) {
  $probeArgs = @(
    "scripts/probe_windows_gpu_npu.py",
    "--model-root", $ModelRoot,
    "--device", $Device,
    "--decoder-device", "NPU",
    "--cache-dir", "$WorkDir/probe-ov-cache",
    "--output-json", $probeJson
  )
  if (-not $Strict) {
    $probeArgs += "--skip-if-missing-devices"
  }
  if ($RequireZeroCopy) {
    $probeArgs += "--require-zero-copy"
  }
  Invoke-Checked "uv" (@("run", "python") + $probeArgs)

  $probe = Get-Content $probeJson -Raw | ConvertFrom-Json
  if ($probe.status -eq "skipped") {
    Write-Host "GPU+NPU benchmark skipped: $($probe.skip_reason)"
    exit 0
  }
}

$argsList = @(
  "scripts/benchmark_windows_gpu_npu_release.py",
  "--archive", $Archive,
  "--model-root", $ModelRoot,
  "--work-dir", $WorkDir,
  "--base-port", "$BasePort",
  "--device", $Device,
  "--scenarios", $Scenarios,
  "--require-devices", "$Device,NPU",
  "--mode", $Mode,
  "--text", $Text,
  "--max-new-tokens", "$MaxNewTokens",
  "--runs", "$Runs",
  "--chunk-strategy", "smooth",
  "--summary-out", "$WorkDir/benchmark-summary.json"
)
if ($RefAudio) {
  $argsList += @("--ref-audio", $RefAudio)
}
if ($RefText) {
  $argsList += @("--ref-text", $RefText)
}
if ($MinSpeedup -ge 0) {
  $argsList += @("--min-speedup", "$MinSpeedup")
}
if ($MaxRtfRegression -ge 0) {
  $argsList += @("--max-rtf-regression", "$MaxRtfRegression")
}
if ($MinGpuUtilizationReduction -ge 0) {
  $argsList += @("--min-gpu-utilization-reduction", "$MinGpuUtilizationReduction")
}
if ($CollectCounters) {
  $argsList += @("--collect-accelerator-counters", "--counter-scope", $CounterScope)
}
if (-not $Strict) {
  $argsList += "--skip-if-missing-devices"
}
if ($NoWarmup) {
  $argsList += "--no-warmup"
}

Invoke-Checked "uv" (@("run", "python") + $argsList)

$benchmarkSummary = Get-Content "$WorkDir/benchmark-summary.json" -Raw | ConvertFrom-Json
if ($benchmarkSummary.status -eq "skipped") {
  Write-Host "GPU+NPU benchmark skipped: $($benchmarkSummary.skip_reason)"
  exit 0
}

$analysisArgs = @(
  "scripts/analyze_windows_gpu_npu_results.py",
  "--benchmark-summary", "$WorkDir/benchmark-summary.json",
  "--require-scenarios", $Scenarios,
  "--output-json", "$WorkDir/analysis.json"
)
if ((-not $SkipProbe) -and (Test-Path $probeJson)) {
  $analysisArgs += @("--probe-json", $probeJson)
  if ($Strict) {
    $analysisArgs += "--require-probe-ok"
  }
  if ($RequirePromptCompile) {
    $analysisArgs += "--require-prompt-compile"
  }
  if ($RequireAudioCompile) {
    $analysisArgs += "--require-audio-compile"
  }
}
if ($CollectCounters) {
  $analysisArgs += "--require-counters"
}
if ($MinSpeedup -ge 0) {
  $analysisArgs += @("--min-speedup", "$MinSpeedup")
}
if ($MaxRtfRegression -ge 0) {
  $analysisArgs += @("--max-rtf-regression", "$MaxRtfRegression")
}
if ($MinGpuUtilizationReduction -ge 0) {
  $analysisArgs += @("--min-gpu-utilization-reduction", "$MinGpuUtilizationReduction")
}

Invoke-Checked "uv" (@("run", "python") + $analysisArgs)

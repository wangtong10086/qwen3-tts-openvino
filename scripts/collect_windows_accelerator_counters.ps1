param(
  [Parameter(Mandatory = $true)]
  [string]$OutputJson,
  [string]$StopFile = "",
  [int]$IntervalMs = 500,
  [int]$MaxSamples = 0,
  [int]$ProcessId = 0,
  [ValidateSet("server", "system")]
  [string]$CounterScope = "server",
  [switch]$NoFallbackToSystem
)

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

function Get-UniqueCounterSets {
  $setsByName = @{}
  $exactNames = @(
    "GPU Engine",
    "GPU Adapter Memory",
    "GPU Process Memory",
    "NPU Compute Adapter Engine",
    "NPU Compute Adapter",
    "NPU Engine",
    "NPU Adapter Memory"
  )
  foreach ($name in $exactNames) {
    try {
      $set = Get-Counter -ListSet $name -ErrorAction SilentlyContinue
      if ($set -and -not $setsByName.ContainsKey($set.CounterSetName)) {
        $setsByName[$set.CounterSetName] = $set
      }
    } catch {
    }
  }
  try {
    $sets = Get-Counter -ListSet * -ErrorAction SilentlyContinue |
      Where-Object { $_.CounterSetName -match "(?i)(gpu|npu|compute adapter)" }
    foreach ($set in $sets) {
      if ($set -and -not $setsByName.ContainsKey($set.CounterSetName)) {
        $setsByName[$set.CounterSetName] = $set
      }
    }
  } catch {
  }
  return @($setsByName.Values)
}

function Select-UtilizationPaths {
  param([array]$CounterSets)

  $pathsByName = @{}
  foreach ($set in $CounterSets) {
    $paths = @()
    if ($set.PathsWithInstances) {
      $paths += $set.PathsWithInstances
    }
    if ($set.Paths) {
      $paths += $set.Paths
    }
    foreach ($path in $paths) {
      if ($path -match "(?i)(utilization|usage|busy|load)" -and -not $pathsByName.ContainsKey($path)) {
        $pathsByName[$path] = $true
      }
    }
  }
  return @($pathsByName.Keys | Sort-Object)
}

function Select-CounterScope {
  param(
    [array]$Paths,
    [int]$TargetProcessId,
    [string]$Scope,
    [bool]$FallbackToSystem
  )

  $allPaths = @($Paths)
  if ($Scope -ne "server" -or $TargetProcessId -le 0) {
    return @{
      paths = $allPaths
      selected_scope = "system"
      requested_process_id = $TargetProcessId
      system_path_count = $allPaths.Count
      process_path_count = 0
    }
  }

  $pidPattern = "(?i)(pid[_ -]*$TargetProcessId|processid[_ -]*$TargetProcessId)"
  $processPaths = @($allPaths | Where-Object { $_ -match $pidPattern })
  if ($processPaths.Count -gt 0) {
    return @{
      paths = $processPaths
      selected_scope = "server"
      requested_process_id = $TargetProcessId
      system_path_count = $allPaths.Count
      process_path_count = $processPaths.Count
    }
  }

  if ($FallbackToSystem) {
    return @{
      paths = $allPaths
      selected_scope = "system_fallback"
      requested_process_id = $TargetProcessId
      system_path_count = $allPaths.Count
      process_path_count = 0
    }
  }

  return @{
    paths = @()
    selected_scope = "server_unavailable"
    requested_process_id = $TargetProcessId
    system_path_count = $allPaths.Count
    process_path_count = 0
  }
}

function Get-Category {
  param([string]$Path)
  if ($Path -match "(?i)npu") {
    return "npu"
  }
  if ($Path -match "(?i)gpu") {
    return "gpu"
  }
  return "other"
}

function Summarize-Category {
  param(
    [array]$Rows,
    [string]$Category
  )
  $categoryRows = @($Rows | Where-Object { $_.category -eq $Category })
  $sampleSums = @()
  foreach ($group in ($categoryRows | Group-Object sample_index)) {
    $sum = 0.0
    foreach ($row in $group.Group) {
      $sum += [double]$row.value
    }
    $sampleSums += $sum
  }
  $avg = $null
  $max = $null
  if ($sampleSums.Count -gt 0) {
    $avg = ($sampleSums | Measure-Object -Average).Average
    $max = ($sampleSums | Measure-Object -Maximum).Maximum
  }
  return @{
    sample_count = $sampleSums.Count
    path_count = @($categoryRows | Select-Object -ExpandProperty path -Unique).Count
    utilization_average = $avg
    utilization_max = $max
  }
}

$startedAt = Get-Date
$rows = @()
$errors = @()
$counterSets = Get-UniqueCounterSets
$allPaths = Select-UtilizationPaths -CounterSets $counterSets
$selection = Select-CounterScope `
  -Paths $allPaths `
  -TargetProcessId $ProcessId `
  -Scope $CounterScope `
  -FallbackToSystem (-not $NoFallbackToSystem)
$paths = @($selection.paths)

try {
  $sampleIndex = 0
  while ($true) {
    if ($StopFile -and (Test-Path $StopFile)) {
      break
    }
    if ($MaxSamples -gt 0 -and $sampleIndex -ge $MaxSamples) {
      break
    }
    if ($paths.Count -eq 0) {
      break
    }
    $timestamp = (Get-Date).ToString("o")
    try {
      $sample = Get-Counter -Counter $paths -ErrorAction SilentlyContinue
      foreach ($counterSample in $sample.CounterSamples) {
        $rows += @{
          sample_index = $sampleIndex
          timestamp = $timestamp
          path = [string]$counterSample.Path
          value = [double]$counterSample.CookedValue
          category = Get-Category -Path ([string]$counterSample.Path)
        }
      }
    } catch {
      $errors += $_.Exception.Message
    }
    $sampleIndex += 1
    Start-Sleep -Milliseconds ([Math]::Max(100, $IntervalMs))
  }

  $endedAt = Get-Date
  $status = "ok"
  if ($paths.Count -eq 0) {
    $status = "no_counters"
  } elseif ($rows.Count -eq 0) {
    $status = "no_samples"
  }
  $payload = @{
    status = $status
    started_at = $startedAt.ToString("o")
    ended_at = $endedAt.ToString("o")
    duration_seconds = ($endedAt - $startedAt).TotalSeconds
    interval_ms = $IntervalMs
    requested_scope = $CounterScope
    selected_scope = $selection.selected_scope
    requested_process_id = $ProcessId
    sample_count = @($rows | Select-Object -ExpandProperty sample_index -Unique).Count
    selected_path_count = $paths.Count
    system_path_count = $selection.system_path_count
    process_path_count = $selection.process_path_count
    selected_paths = $paths
    counter_sets = @($counterSets | ForEach-Object { $_.CounterSetName } | Sort-Object -Unique)
    gpu = Summarize-Category -Rows $rows -Category "gpu"
    npu = Summarize-Category -Rows $rows -Category "npu"
    other = Summarize-Category -Rows $rows -Category "other"
    errors = $errors
  }
} catch {
  $endedAt = Get-Date
  $payload = @{
    status = "failed"
    started_at = $startedAt.ToString("o")
    ended_at = $endedAt.ToString("o")
    duration_seconds = ($endedAt - $startedAt).TotalSeconds
    interval_ms = $IntervalMs
    requested_scope = $CounterScope
    selected_scope = $selection.selected_scope
    requested_process_id = $ProcessId
    sample_count = 0
    selected_path_count = $paths.Count
    system_path_count = $selection.system_path_count
    process_path_count = $selection.process_path_count
    selected_paths = $paths
    counter_sets = @($counterSets | ForEach-Object { $_.CounterSetName } | Sort-Object -Unique)
    gpu = $null
    npu = $null
    other = $null
    errors = @($_.Exception.Message)
  }
}

$outputPath = [System.IO.Path]::GetFullPath($OutputJson)
$outputDir = [System.IO.Path]::GetDirectoryName($outputPath)
if ($outputDir -and -not (Test-Path $outputDir)) {
  New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}
$payload | ConvertTo-Json -Depth 8 | Set-Content -Path $outputPath -Encoding UTF8

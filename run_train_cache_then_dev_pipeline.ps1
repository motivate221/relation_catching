param(
    [string]$PythonExe = "D:\Asoftwares\anaconda\envs\llama3\python.exe",
    [string]$ProjectRoot = "D:\Asoftwares\new_bishe\EP-RSR-main",
    [int]$TrainDocStart = 0,
    [int]$TrainDocEnd = -1,
    [int]$DevDocStart = 0,
    [int]$DevDocEnd = 998,
    [bool]$UseRerank = $true,
    [int]$EntityPairSampleCount = 1,
    [string]$MethodTag = "rerank_full",
    [switch]$Resume,
    [switch]$ResetAllCheckpoints
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-PathExists {
    param(
        [string]$PathValue,
        [string]$Description
    )

    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Description not found: $PathValue"
    }
}

function Invoke-PowerShellScript {
    param(
        [string]$ScriptPath,
        [string[]]$Arguments
    )

    $cmd = @("-ExecutionPolicy", "Bypass", "-File", $ScriptPath) + $Arguments
    Write-Host ""
    Write-Host ">> powershell $($cmd -join ' ')"
    & powershell @cmd
    if ($LASTEXITCODE -ne 0) {
        throw "Sub-script failed: $ScriptPath"
    }
}

Assert-PathExists -PathValue $ProjectRoot -Description "Project root"
Assert-PathExists -PathValue $PythonExe -Description "Python executable"

$startServiceScript = Join-Path $ProjectRoot "start_model_service.ps1"
$trainCacheScript = Join-Path $ProjectRoot "run_train_annotated_cache_pipeline.ps1"
$devPipelineScript = Join-Path $ProjectRoot "run_dev_0_20_pipeline.ps1"

Assert-PathExists -PathValue $startServiceScript -Description "Model service start script"
Assert-PathExists -PathValue $trainCacheScript -Description "Training cache pipeline script"
Assert-PathExists -PathValue $devPipelineScript -Description "Dev pipeline script"

Write-Host "Project root: $ProjectRoot"
Write-Host "Python: $PythonExe"
if ($TrainDocEnd -ge 0) {
    Write-Host "Train cache range: train_annotated $TrainDocStart-$TrainDocEnd"
} else {
    Write-Host "Train cache range: train_annotated from $TrainDocStart to full dataset"
}
Write-Host "Dev range: dev $DevDocStart-$DevDocEnd"
Write-Host "USE_RERANK: $UseRerank"
Write-Host "ENTITY_PAIR_SAMPLE_COUNT: $EntityPairSampleCount"
Write-Host "METHOD_TAG: $MethodTag"
Write-Host "Resume mode: $Resume"
Write-Host "Reset all checkpoints: $ResetAllCheckpoints"

$serviceArgs = @(
    "-PythonExe", $PythonExe,
    "-ProjectRoot", $ProjectRoot
)
Invoke-PowerShellScript -ScriptPath $startServiceScript -Arguments $serviceArgs

$trainArgs = @(
    "-PythonExe", $PythonExe,
    "-ProjectRoot", $ProjectRoot,
    "-DocStart", [string]$TrainDocStart,
    "-DocEnd", [string]$TrainDocEnd
)
if ($ResetAllCheckpoints) {
    $trainArgs += "-ResetCheckpoint"
}
if ($Resume) {
    $trainArgs += "-Resume"
}
Invoke-PowerShellScript -ScriptPath $trainCacheScript -Arguments $trainArgs

$devArgs = @(
    "-PythonExe", $PythonExe,
    "-ProjectRoot", $ProjectRoot,
    "-DataName", "dev",
    "-DocStart", [string]$DevDocStart,
    "-DocEnd", [string]$DevDocEnd,
    "-EntityPairSampleCount", [string]$EntityPairSampleCount,
    "-UseRerank", (if ($UseRerank) { "true" } else { "false" }),
    "-MethodTag", $MethodTag
)
if ($ResetAllCheckpoints) {
    $devArgs += "-ResetCheckpoint"
}
if ($Resume) {
    $devArgs += "-Resume"
}
Invoke-PowerShellScript -ScriptPath $devPipelineScript -Arguments $devArgs

Write-Host ""
Write-Host "Training-side cache + dev pipeline completed successfully."

param(
    [string]$PythonExe = "D:\Asoftwares\anaconda\envs\llama3\python.exe",
    [string]$ProjectRoot = "D:\Asoftwares\new_bishe\EP-RSR-main",
    [string]$DataName = "train_annotated",
    [int]$DocStart = 0,
    [int]$DocEnd = -1,
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 6006,
    [switch]$Resume,
    [switch]$ResetCheckpoint,
    [string]$CheckpointFile = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($null -ne (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue)) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$script:StepResults = @()
$script:CompletedSteps = @()

function Normalize-CompletedSteps {
    param(
        [object]$RawSteps
    )

    $normalized = @()
    foreach ($raw in @($RawSteps)) {
        if ($null -eq $raw) {
            continue
        }
        $text = $raw.ToString().Trim()
        if (-not $text) {
            continue
        }

        $parts = @([regex]::Split($text, '(?=\d{2}_)') | Where-Object { $_ -and $_.Trim() })
        if (@($parts).Count -eq 0) {
            $parts = @($text)
        }

        foreach ($part in $parts) {
            $stepName = $part.Trim()
            if ($stepName -and $normalized -notcontains $stepName) {
                $normalized += $stepName
            }
        }
    }
    return @($normalized)
}

function Assert-PathExists {
    param(
        [string]$PathValue,
        [string]$Description
    )

    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Description not found: $PathValue"
    }
}

function Ensure-Directory {
    param(
        [string]$PathValue
    )

    New-Item -ItemType Directory -Force -Path $PathValue | Out-Null
}

function Test-PortOpen {
    param(
        [string]$HostName,
        [int]$PortNumber
    )

    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($HostName, $PortNumber, $null, $null)
        $wait = $async.AsyncWaitHandle.WaitOne(1500, $false)
        if (-not $wait) {
            return $false
        }
        $client.EndConnect($async)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Assert-ModelServiceAvailable {
    if (-not (Test-PortOpen -HostName $HostAddress -PortNumber $Port)) {
        throw "Model service is not reachable on ${HostAddress}:$Port. Start it first with start_model_service.ps1."
    }
}

function Load-Checkpoint {
    param(
        [string]$PathValue
    )

    $candidatePaths = @($PathValue, "$PathValue.bak")
    $existingCandidates = @($candidatePaths | Where-Object { Test-Path -LiteralPath $_ })
    if ($existingCandidates.Count -eq 0) {
        return @()
    }

    foreach ($candidatePath in $existingCandidates) {
        try {
            $raw = Get-Content -LiteralPath $candidatePath -Raw
            if (-not $raw.Trim()) {
                continue
            }
            $data = $raw | ConvertFrom-Json
            if ($null -eq $data.completed_steps) {
                return @()
            }
            if ($candidatePath -ne $PathValue) {
                Write-Host "Checkpoint fallback loaded: $candidatePath"
            }
            return Normalize-CompletedSteps -RawSteps $data.completed_steps
        } catch {
            continue
        }
    }

    throw "Failed to load checkpoint file: $PathValue"
}

function Save-Checkpoint {
    param(
        [string]$PathValue,
        [string[]]$CompletedSteps,
        [string]$LatestLogDir
    )

    $normalizedSteps = Normalize-CompletedSteps -RawSteps $CompletedSteps
    $payload = [pscustomobject]@{
        data_name = $DataName
        doc_start = $DocStart
        doc_end = $DocEnd
        latest_log_dir = $LatestLogDir
        updated_at = (Get-Date).ToString("s")
        completed_steps = $normalizedSteps
    }
    $tmpPath = "$PathValue.tmp"
    $bakPath = "$PathValue.bak"
    $jsonText = $payload | ConvertTo-Json -Depth 4
    Set-Content -LiteralPath $tmpPath -Value $jsonText
    if (Test-Path -LiteralPath $PathValue) {
        Copy-Item -LiteralPath $PathValue -Destination $bakPath -Force
    }
    Move-Item -LiteralPath $tmpPath -Destination $PathValue -Force
}

function Invoke-Step {
    param(
        [string]$Name,
        [string]$WorkingDirectory,
        [string]$ScriptName,
        [string]$LogDir,
        [bool]$RequiresModelService = $false
    )

    $safeName = $Name -replace "[^A-Za-z0-9\-_]", "_"
    $logFile = Join-Path $LogDir "${safeName}.log"
    $stdoutLog = Join-Path $LogDir "${safeName}.stdout.log"
    $stderrLog = Join-Path $LogDir "${safeName}.stderr.log"
    $scriptPath = Join-Path $WorkingDirectory $ScriptName

    Assert-PathExists -PathValue $WorkingDirectory -Description "Working directory for $Name"
    Assert-PathExists -PathValue $scriptPath -Description "Script for $Name"

    if ($script:CompletedSteps -contains $Name) {
        Write-Host ""
        Write-Host "==== $Name ===="
        Write-Host "Skipping completed step from checkpoint."
        $script:StepResults += [pscustomobject]@{
            Step = $Name
            Status = "SKIPPED"
            Seconds = 0
            Log = $logFile
        }
        return
    }

    if ($RequiresModelService) {
        Assert-ModelServiceAvailable
    }

    Write-Host ""
    Write-Host "==== $Name ===="
    Write-Host "Working directory: $WorkingDirectory"
    Write-Host "Command: $PythonExe $ScriptName"
    Write-Host "Log: $logFile"

    $startedAt = Get-Date
    try {
        if (Test-Path -LiteralPath $stdoutLog) {
            Remove-Item -LiteralPath $stdoutLog -Force
        }
        if (Test-Path -LiteralPath $stderrLog) {
            Remove-Item -LiteralPath $stderrLog -Force
        }

        $process = Start-Process `
            -FilePath $PythonExe `
            -ArgumentList $scriptPath `
            -WorkingDirectory $WorkingDirectory `
            -RedirectStandardOutput $stdoutLog `
            -RedirectStandardError $stderrLog `
            -NoNewWindow `
            -Wait `
            -PassThru

        $stdoutText = if (Test-Path -LiteralPath $stdoutLog) { Get-Content -LiteralPath $stdoutLog -Raw } else { "" }
        $stderrText = if (Test-Path -LiteralPath $stderrLog) { Get-Content -LiteralPath $stderrLog -Raw } else { "" }
        $combinedText = ($stdoutText, $stderrText | Where-Object { $_ -ne "" }) -join [Environment]::NewLine
        Set-Content -LiteralPath $logFile -Value $combinedText

        if ($stdoutText) {
            Write-Host $stdoutText.TrimEnd()
        }
        if ($stderrText) {
            Write-Host $stderrText.TrimEnd()
        }

        $exitCode = $process.ExitCode
        if ($exitCode -ne 0) {
            Write-Host "Step failed with exit code: $exitCode"
            throw "Step failed: $Name"
        }

        $duration = [math]::Round(((Get-Date) - $startedAt).TotalSeconds, 1)
        $script:StepResults += [pscustomobject]@{
            Step = $Name
            Status = "OK"
            Seconds = $duration
            Log = $logFile
        }
        $script:CompletedSteps = Normalize-CompletedSteps -RawSteps @($script:CompletedSteps + $Name)
        Save-Checkpoint -PathValue $script:CheckpointFile -CompletedSteps $script:CompletedSteps -LatestLogDir $LogDir
        Write-Host "Step completed: $Name (${duration}s)"
    } catch {
        $duration = [math]::Round(((Get-Date) - $startedAt).TotalSeconds, 1)
        $script:StepResults += [pscustomobject]@{
            Step = $Name
            Status = "FAILED"
            Seconds = $duration
            Log = $logFile
        }
        throw
    }
}

if ($DataName -notin @("train_annotated", "train")) {
    throw "This script is intended for training-side cache generation only. Use DataName=train_annotated or train."
}

Assert-PathExists -PathValue $ProjectRoot -Description "Project root"
Assert-PathExists -PathValue $PythonExe -Description "Python executable"

$pythonVersionOutput = & $PythonExe --version 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Unable to execute Python with: $PythonExe"
}

$env:DATA_NAME = $DataName
$env:DOC_START = [string]$DocStart
if ($DocEnd -ge 0) {
    $env:DOC_END = [string]$DocEnd
    $rangeTag = "${DocStart}-${DocEnd}"
} else {
    Remove-Item Env:DOC_END -ErrorAction SilentlyContinue
    $rangeTag = "${DocStart}-full"
}
Remove-Item Env:USE_RERANK -ErrorAction SilentlyContinue
Remove-Item Env:METHOD_TAG -ErrorAction SilentlyContinue

$logsRoot = Join-Path $ProjectRoot "logs"
Ensure-Directory -PathValue $logsRoot

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runLogDir = Join-Path $logsRoot "pipeline_${DataName}_cache_${rangeTag}_$timestamp"
Ensure-Directory -PathValue $runLogDir

if ([string]::IsNullOrWhiteSpace($CheckpointFile)) {
    $CheckpointFile = Join-Path $logsRoot "pipeline_${DataName}_cache_${rangeTag}.checkpoint.json"
}

$script:CheckpointFile = $CheckpointFile

if ($ResetCheckpoint -and (Test-Path -LiteralPath $CheckpointFile)) {
    Remove-Item -LiteralPath $CheckpointFile -Force
}

if ($Resume) {
    $script:CompletedSteps = Load-Checkpoint -PathValue $CheckpointFile
} else {
    $script:CompletedSteps = @()
    Save-Checkpoint -PathValue $CheckpointFile -CompletedSteps $script:CompletedSteps -LatestLogDir $runLogDir
}

Assert-ModelServiceAvailable

$requiredDirs = @(
    (Join-Path $ProjectRoot "data\entity_information_prompt\$DataName"),
    (Join-Path $ProjectRoot "data\entity_information_run\$DataName"),
    (Join-Path $ProjectRoot "data\entity_information\$DataName"),
    (Join-Path $ProjectRoot "data\relation_summary_prompt\$DataName"),
    (Join-Path $ProjectRoot "data\relation_summary_run\$DataName"),
    (Join-Path $ProjectRoot "data\check_result_relation_summary_jsonl\$DataName"),
    (Join-Path $ProjectRoot "data\get_embeddings")
)
foreach ($dir in $requiredDirs) {
    Ensure-Directory -PathValue $dir
}

Write-Host "Python: $PythonExe"
Write-Host "Python version: $pythonVersionOutput"
Write-Host "Project root: $ProjectRoot"
if ($DocEnd -ge 0) {
    Write-Host "Data range: $DataName $DocStart-$DocEnd"
} else {
    Write-Host "Data range: $DataName from $DocStart to full dataset"
}
Write-Host "Logs directory: $runLogDir"
Write-Host "Checkpoint file: $CheckpointFile"
if ($Resume -and @($script:CompletedSteps).Count -gt 0) {
    Write-Host "Resume mode: skipping $(@($script:CompletedSteps).Count) completed step(s)"
}

try {
    Invoke-Step -Name "01_entity_information_prompt" -WorkingDirectory (Join-Path $ProjectRoot "2.entity_information") -ScriptName "entity_information_prompt_new.py" -LogDir $runLogDir
    Invoke-Step -Name "02_entity_information_run" -WorkingDirectory (Join-Path $ProjectRoot "2.entity_information") -ScriptName "entity_information_run.py" -LogDir $runLogDir -RequiresModelService $true
    Invoke-Step -Name "03_entity_information_check" -WorkingDirectory (Join-Path $ProjectRoot "2.entity_information") -ScriptName "check_result_entity_information_jsonl.py" -LogDir $runLogDir

    Invoke-Step -Name "04_relation_summary_prompt" -WorkingDirectory (Join-Path $ProjectRoot "3.relation_summary") -ScriptName "relation_summary_prompt.py" -LogDir $runLogDir
    Invoke-Step -Name "05_relation_summary_run" -WorkingDirectory (Join-Path $ProjectRoot "3.relation_summary") -ScriptName "relation_summary_run.py" -LogDir $runLogDir -RequiresModelService $true
    Invoke-Step -Name "06_relation_summary_check" -WorkingDirectory (Join-Path $ProjectRoot "3.relation_summary") -ScriptName "check_result_relation_summary_jsonl.py" -LogDir $runLogDir

    Invoke-Step -Name "07_get_embeddings" -WorkingDirectory (Join-Path $ProjectRoot "4.retrieval") -ScriptName "get_embeddings.py" -LogDir $runLogDir

    Write-Host ""
    Write-Host "Training-side cache pipeline completed successfully."
} finally {
    Write-Host "Logs directory: $runLogDir"
    if (@($script:StepResults).Count -gt 0) {
        Write-Host ""
        Write-Host "Step summary:"
        $script:StepResults | Format-Table -AutoSize
    }
}

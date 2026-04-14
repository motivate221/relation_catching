param(
    [string]$PythonExe = "D:\Asoftwares\anaconda\envs\llama3\python.exe",
    [string]$ProjectRoot = "D:\Asoftwares\new_bishe\EP-RSR-main",
    [string]$DataName = "dev",
    [int]$DocStart = 0,
    [int]$DocEnd = 100,
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 6006,
    [object]$UseRerank = $true,
    [int]$EntityPairSampleCount = 1,
    [string]$MethodTag = "",
    [switch]$Resume,
    [switch]$ResetCheckpoint,
    [string]$CheckpointFile = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Prevent PowerShell from treating stderr text emitted by native commands
# as terminating errors. We rely on $LASTEXITCODE for step success/failure.
if ($null -ne (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue)) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$script:StepResults = @()
$script:CompletedSteps = @()

function ConvertTo-BoolValue {
    param(
        [object]$Value,
        [bool]$Default = $true
    )

    if ($null -eq $Value) {
        return $Default
    }

    if ($Value -is [bool]) {
        return [bool]$Value
    }

    $normalized = $Value.ToString().Trim().ToLowerInvariant()
    switch ($normalized) {
        "1" { return $true }
        "true" { return $true }
        "yes" { return $true }
        "y" { return $true }
        "on" { return $true }
        "0" { return $false }
        "false" { return $false }
        "no" { return $false }
        "n" { return $false }
        "off" { return $false }
        default { throw "Invalid UseRerank value: $Value" }
    }
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

function Assert-AtLeastOneMatch {
    param(
        [string]$Pattern,
        [string]$Description
    )

    $matches = @(Get-ChildItem -Path $Pattern -ErrorAction SilentlyContinue)
    if (-not $matches -or $matches.Count -eq 0) {
        throw "$Description not found. Expected pattern: $Pattern"
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
            return @($data.completed_steps)
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

    $payload = [pscustomobject]@{
        data_name = $DataName
        doc_start = $DocStart
        doc_end = $DocEnd
        latest_log_dir = $LatestLogDir
        updated_at = (Get-Date).ToString("s")
        completed_steps = $CompletedSteps
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
        $script:CompletedSteps = @($script:CompletedSteps + $Name)
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

Assert-PathExists -PathValue $ProjectRoot -Description "Project root"
Assert-PathExists -PathValue $PythonExe -Description "Python executable"

$pythonVersionOutput = & $PythonExe --version 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Unable to execute Python with: $PythonExe"
}

$UseRerank = ConvertTo-BoolValue -Value $UseRerank -Default $true
if ($EntityPairSampleCount -lt 1 -or $EntityPairSampleCount -gt 20) {
    throw "EntityPairSampleCount must be between 1 and 20."
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
$env:USE_RERANK = if ($UseRerank) { "true" } else { "false" }
if ([string]::IsNullOrWhiteSpace($MethodTag)) {
    $env:METHOD_TAG = if ($UseRerank) { "rerank" } else { "baseline" }
} else {
    $env:METHOD_TAG = $MethodTag
}

$logsRoot = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $logsRoot | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runLogDir = Join-Path $logsRoot "pipeline_${DataName}_${rangeTag}_$timestamp"
New-Item -ItemType Directory -Force -Path $runLogDir | Out-Null

if ([string]::IsNullOrWhiteSpace($CheckpointFile)) {
    $CheckpointFile = Join-Path $logsRoot "pipeline_${DataName}_${rangeTag}.checkpoint.json"
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
Assert-AtLeastOneMatch -Pattern (Join-Path $ProjectRoot "data\check_result_relation_summary_jsonl\train_annotated\result_docred_train_annotated_relation_summary_*.jsonl") -Description "Train-side relation summary cache"
Assert-AtLeastOneMatch -Pattern (Join-Path $ProjectRoot "data\get_embeddings\docred_train_annotated_embeddings_*.npy") -Description "Train-side embedding cache"

$requiredDirs = @(
    (Join-Path $ProjectRoot "data\entity_pair_selection_prompt\$DataName"),
    (Join-Path $ProjectRoot "data\entity_pair_selection_run\$DataName"),
    (Join-Path $ProjectRoot "data\check_result_entity_pair_selection_jsonl\$DataName"),
    (Join-Path $ProjectRoot "data\get_entity_pair_selection_label\$DataName"),
    (Join-Path $ProjectRoot "data\entity_information_prompt\$DataName"),
    (Join-Path $ProjectRoot "data\entity_information_run\$DataName"),
    (Join-Path $ProjectRoot "data\entity_information\$DataName"),
    (Join-Path $ProjectRoot "data\relation_summary_prompt\$DataName"),
    (Join-Path $ProjectRoot "data\relation_summary_run\$DataName"),
    (Join-Path $ProjectRoot "data\check_result_relation_summary_jsonl\$DataName"),
    (Join-Path $ProjectRoot "data\get_embeddings"),
    (Join-Path $ProjectRoot "data\retrieval_from_train\$DataName"),
    (Join-Path $ProjectRoot "data\retrieval_rerank\$DataName"),
    (Join-Path $ProjectRoot "data\multiple_choice_prompt\$DataName"),
    (Join-Path $ProjectRoot "data\multiple_choice_run\$DataName"),
    (Join-Path $ProjectRoot "data\check_result_multiple_choice_jsonl\$DataName"),
    (Join-Path $ProjectRoot "data\get_multiple_choice_label\$DataName"),
    (Join-Path $ProjectRoot "data\triplet_fact_judgement_prompt\$DataName"),
    (Join-Path $ProjectRoot "data\triplet_fact_judgement_run\$DataName"),
    (Join-Path $ProjectRoot "data\check_result_triplet_fact_judgement_jsonl\$DataName"),
    (Join-Path $ProjectRoot "data\get_triplet_fact_judgement_label\$DataName")
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
Write-Host "USE_RERANK: $env:USE_RERANK"
Write-Host "METHOD_TAG: $env:METHOD_TAG"
Write-Host "ENTITY_PAIR_SAMPLE_COUNT: $EntityPairSampleCount"
Write-Host "Logs directory: $runLogDir"
Write-Host "Checkpoint file: $CheckpointFile"
if ($Resume -and @($script:CompletedSteps).Count -gt 0) {
    Write-Host "Resume mode: skipping $(@($script:CompletedSteps).Count) completed step(s)"
}

try {
    Invoke-Step -Name "01_entity_pair_selection_prompt" -WorkingDirectory (Join-Path $ProjectRoot "1.entity_pair_selection") -ScriptName "entity_pair_selection_prompt.py" -LogDir $runLogDir
    for ($sampleId = 1; $sampleId -le $EntityPairSampleCount; $sampleId++) {
        $sampleTag = "{0:D2}" -f $sampleId
        $env:SAMPLE_TAG = $sampleTag
        Invoke-Step -Name ("02_entity_pair_selection_run_{0}" -f $sampleTag) -WorkingDirectory (Join-Path $ProjectRoot "1.entity_pair_selection") -ScriptName "entity_pair_selection_run.py" -LogDir $runLogDir -RequiresModelService $true
        Invoke-Step -Name ("03_entity_pair_selection_check_{0}" -f $sampleTag) -WorkingDirectory (Join-Path $ProjectRoot "1.entity_pair_selection") -ScriptName "check_result_entity_pair_selection_jsonl.py" -LogDir $runLogDir
        Invoke-Step -Name ("04_entity_pair_selection_label_{0}" -f $sampleTag) -WorkingDirectory (Join-Path $ProjectRoot "1.entity_pair_selection") -ScriptName "get_entity_pair_selection_label.py" -LogDir $runLogDir
    }
    Remove-Item Env:SAMPLE_TAG -ErrorAction SilentlyContinue

    Invoke-Step -Name "05_entity_information_prompt" -WorkingDirectory (Join-Path $ProjectRoot "2.entity_information") -ScriptName "entity_information_prompt_new.py" -LogDir $runLogDir
    Invoke-Step -Name "06_entity_information_run" -WorkingDirectory (Join-Path $ProjectRoot "2.entity_information") -ScriptName "entity_information_run.py" -LogDir $runLogDir -RequiresModelService $true
    Invoke-Step -Name "07_entity_information_check" -WorkingDirectory (Join-Path $ProjectRoot "2.entity_information") -ScriptName "check_result_entity_information_jsonl.py" -LogDir $runLogDir

    Invoke-Step -Name "08_relation_summary_prompt" -WorkingDirectory (Join-Path $ProjectRoot "3.relation_summary") -ScriptName "relation_summary_prompt.py" -LogDir $runLogDir
    Invoke-Step -Name "09_relation_summary_run" -WorkingDirectory (Join-Path $ProjectRoot "3.relation_summary") -ScriptName "relation_summary_run.py" -LogDir $runLogDir -RequiresModelService $true
    Invoke-Step -Name "10_relation_summary_check" -WorkingDirectory (Join-Path $ProjectRoot "3.relation_summary") -ScriptName "check_result_relation_summary_jsonl.py" -LogDir $runLogDir

    Invoke-Step -Name "11_get_embeddings" -WorkingDirectory (Join-Path $ProjectRoot "4.retrieval") -ScriptName "get_embeddings.py" -LogDir $runLogDir
    Invoke-Step -Name "12_retrieval_from_train" -WorkingDirectory (Join-Path $ProjectRoot "4.retrieval") -ScriptName "retrieval_from_train-few.py" -LogDir $runLogDir
    if ($UseRerank) {
        Invoke-Step -Name "13_evidence_relation_rerank" -WorkingDirectory (Join-Path $ProjectRoot "4.retrieval") -ScriptName "evidence_relation_rerank.py" -LogDir $runLogDir
    }

    Invoke-Step -Name "14_multiple_choice_prompt" -WorkingDirectory (Join-Path $ProjectRoot "5.multiple_choice") -ScriptName "multiple_choice_prompt.py" -LogDir $runLogDir
    Invoke-Step -Name "15_multiple_choice_run" -WorkingDirectory (Join-Path $ProjectRoot "5.multiple_choice") -ScriptName "multiple_choice_run.py" -LogDir $runLogDir -RequiresModelService $true
    Invoke-Step -Name "16_multiple_choice_check" -WorkingDirectory (Join-Path $ProjectRoot "5.multiple_choice") -ScriptName "check_result_multiple_choice_jsonl.py" -LogDir $runLogDir
    Invoke-Step -Name "17_multiple_choice_label" -WorkingDirectory (Join-Path $ProjectRoot "5.multiple_choice") -ScriptName "get_multiple_choice_label.py" -LogDir $runLogDir

    Invoke-Step -Name "18_triplet_fact_judgement_prompt" -WorkingDirectory (Join-Path $ProjectRoot "6.triplet_fact_judgement") -ScriptName "triplet_fact_judgement_prompt.py" -LogDir $runLogDir
    Invoke-Step -Name "19_triplet_fact_judgement_run" -WorkingDirectory (Join-Path $ProjectRoot "6.triplet_fact_judgement") -ScriptName "triplet_fact_judgement_run.py" -LogDir $runLogDir -RequiresModelService $true
    Invoke-Step -Name "20_triplet_fact_judgement_check" -WorkingDirectory (Join-Path $ProjectRoot "6.triplet_fact_judgement") -ScriptName "check_result_triplet_fact_judgement_jsonl.py" -LogDir $runLogDir
    Invoke-Step -Name "21_triplet_fact_judgement_label" -WorkingDirectory (Join-Path $ProjectRoot "6.triplet_fact_judgement") -ScriptName "get_triplet_fact_judgement_label.py" -LogDir $runLogDir

    Write-Host ""
    Write-Host "Pipeline completed successfully."
} finally {
    Write-Host "Logs directory: $runLogDir"
    if ($script:StepResults.Count -gt 0) {
        Write-Host ""
        Write-Host "Step summary:"
        $script:StepResults | Format-Table -AutoSize
    }
}

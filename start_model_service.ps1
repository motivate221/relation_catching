param(
    [string]$PythonExe = "D:\Asoftwares\anaconda\envs\llama3\python.exe",
    [string]$ModelPath = "D:\Asoftwares\new_bishe\EP-RSR-main\models\Qwen2.5-3B-Instruct",
    [string]$ProjectRoot = "D:\Asoftwares\new_bishe\EP-RSR-main",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 6006,
    [int]$StartupTimeoutSeconds = 120
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

if (Test-PortOpen -HostName $HostAddress -PortNumber $Port) {
    Write-Host "Model service is already running on ${HostAddress}:$Port"
    exit 0
}

Assert-PathExists -PathValue $ProjectRoot -Description "Project root"
Assert-PathExists -PathValue $PythonExe -Description "Python executable"
Assert-PathExists -PathValue $ModelPath -Description "Model path"

$serviceScript = Join-Path $ProjectRoot "0.pre_model\llama3-api.py"
Assert-PathExists -PathValue $serviceScript -Description "Model service script"

$logsDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutLog = Join-Path $logsDir "model_service_${timestamp}.out.log"
$stderrLog = Join-Path $logsDir "model_service_${timestamp}.err.log"
$pidLog = Join-Path $logsDir "model_service_${timestamp}.pid.txt"

$command = @"
`$env:MODEL_PATH='$ModelPath'
`$env:MODEL_LOAD_IN_4BIT='true'
`$env:MODEL_LOAD_IN_8BIT='false'
Set-Location '$ProjectRoot'
& '$PythonExe' '0.pre_model\llama3-api.py'
"@

$process = Start-Process `
    -FilePath (Join-Path $PSHOME "powershell.exe") `
    -ArgumentList "-NoProfile", "-Command", $command `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path $pidLog -Value $process.Id

$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    if (Test-PortOpen -HostName $HostAddress -PortNumber $Port) {
        Write-Host "Model service started successfully on ${HostAddress}:$Port"
        Write-Host "Process ID: $($process.Id)"
        Write-Host "stdout log: $stdoutLog"
        Write-Host "stderr log: $stderrLog"
        Write-Host "pid file: $pidLog"
        exit 0
    }

    if ($process.HasExited) {
        break
    }

    Start-Sleep -Seconds 2
}

Write-Error "Model service did not start successfully within ${StartupTimeoutSeconds}s. Check logs:`n$stdoutLog`n$stderrLog"

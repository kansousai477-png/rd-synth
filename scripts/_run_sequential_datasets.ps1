# Sequential dataset runner - avoids GPU contention
param(
    [string]$Datasets = "2017,2018,iot23",
    [string]$Profile = "paper"
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot | Split-Path -Parent
Set-Location $repoRoot

$env:PYTHONPATH = "src"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:NUMEXPR_NUM_THREADS = "4"

$datasetsList = $Datasets -split ","
$pythonExe = "venv\Scripts\python.exe"

foreach ($ds in $datasetsList) {
    $ds = $ds.Trim()
    $timestamp = Get-Date -Format "yyyy-MM-ddTHH-mm-ssZ"
    $logFile = "outputs\reviewer_suite\logs\seq_${ds}_${timestamp}.log"
    Write-Output "[$(Get-Date -Format 'HH:mm:ss')] Starting $ds..."

    $proc = Start-Process -FilePath $pythonExe `
        -ArgumentList "scripts\run_reviewer_suite.py --datasets $ds --profile $Profile --execution-mode inline" `
        -NoNewWindow -PassThru `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError "$logFile.err"

    Write-Output "[$(Get-Date -Format 'HH:mm:ss')] $ds PID=$($proc.Id), log=$logFile"
    $proc.WaitForExit()

    if ($proc.ExitCode -eq 0) {
        Write-Output "[$(Get-Date -Format 'HH:mm:ss')] $ds COMPLETED (exit 0)"
    } else {
        Write-Output "[$(Get-Date -Format 'HH:mm:ss')] $ds FAILED (exit $($proc.ExitCode))"
    }
}

Write-Output "[$(Get-Date -Format 'HH:mm:ss')] All datasets done."

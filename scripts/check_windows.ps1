param(
    [string]$VenvDir = "venv",
    [switch]$WithLint,
    [switch]$SkipTests,
    [switch]$Smoke,
    [int]$Threads = 1
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PythonExe = Join-Path (Join-Path $RepoRoot $VenvDir) "Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment not found. Run scripts\bootstrap_windows.ps1 first."
}

if ($WithLint) {
    & $PythonExe -m ruff check src tests
    & $PythonExe -m ruff format --check src tests
}

if (-not $SkipTests) {
    $ThreadCount = [Math]::Max(1, $Threads)
    $env:OMP_NUM_THREADS = "$ThreadCount"
    $env:MKL_NUM_THREADS = "$ThreadCount"
    $env:OPENBLAS_NUM_THREADS = "$ThreadCount"
    $env:NUMEXPR_NUM_THREADS = "$ThreadCount"
    if ($Smoke) {
        & $PythonExe -m pytest -q --no-cov `
            tests\test_runtime.py `
            tests\test_oracle.py `
            tests\test_reviewer_suite.py `
            tests\test_pipeline_reporting.py
    }
    else {
        & $PythonExe -m pytest -q
    }
}

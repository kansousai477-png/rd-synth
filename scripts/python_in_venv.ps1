param(
    [string]$VenvDir = "venv",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PythonExe = Join-Path (Join-Path $RepoRoot $VenvDir) "Scripts\\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment not found at '$VenvDir'. Run scripts\\bootstrap_windows.ps1 first."
}

if (-not $PythonArgs -or $PythonArgs.Count -eq 0) {
    & $PythonExe --version
    exit $LASTEXITCODE
}

& $PythonExe @PythonArgs
exit $LASTEXITCODE

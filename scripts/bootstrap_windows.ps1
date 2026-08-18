param(
    [string]$PythonSpec = "3.9",
    [string]$VenvDir = "venv",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$VenvPath = Join-Path $RepoRoot $VenvDir
if ((Test-Path $VenvPath) -and $Recreate) {
    Remove-Item -LiteralPath $VenvPath -Recurse -Force
}

if (-not (Test-Path $VenvPath)) {
    py -$PythonSpec -m venv $VenvDir
}

$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found in virtual environment: $PythonExe"
}

& $PythonExe -m pip install --upgrade pip setuptools wheel
& $PythonExe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
& $PythonExe -m pip install -r requirements.txt pytest ruff
& $PythonExe -c "import torch; print(torch.__version__)"

param(
    [ValidateSet("start", "resume", "monitor", "status", "report-only")]
    [string]$Action = "status",
    [string]$Datasets = "unsw,2017,2018,iot23",
    [string]$Profile = "paper",
    [string]$OutRoot = "outputs/reviewer_suite",
    [int]$ComboJobs = 1,
    [int]$AblationJobs = 1,
    [string]$RunTag = "",
    [switch]$SkipTransfer,
    [switch]$PrebuildData,
    [switch]$RequirePrebuiltData,
    [switch]$TwoPhaseStage3,
    [switch]$EstimateOnly,
    [double]$MonitorInterval = 3.0
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PythonExe = Join-Path (Join-Path $RepoRoot "venv") "Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment not found. Expected: $PythonExe"
}

$LogsRoot = Join-Path $RepoRoot "outputs\reviewer_suite\logs"
New-Item -ItemType Directory -Path $LogsRoot -Force | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogsRoot "full_run_${Action}_$Timestamp.log"

if ($Action -eq "monitor") {
    & $PythonExe scripts\run_process_monitor.py --interval $MonitorInterval
    exit $LASTEXITCODE
}

if ($Action -eq "status") {
    & $PythonExe scripts\inspect_full_run_status.py --root $OutRoot
    exit $LASTEXITCODE
}

$Command = @(
    $PythonExe,
    "scripts\run_cross_dataset_suite.py",
    "--datasets", $Datasets,
    "--out-root", $OutRoot,
    "--profile", $Profile,
    "--combo-jobs", ([Math]::Max(1, $ComboJobs)).ToString(),
    "--ablation-jobs", ([Math]::Max(1, $AblationJobs)).ToString()
)

if ($Action -eq "resume") {
    $Command += "--skip-existing"
}
elseif ($Action -eq "report-only") {
    $Command = @(
        $PythonExe,
        "scripts\run_cross_dataset_suite.py",
        "--datasets", $Datasets,
        "--out-root", $OutRoot,
        "--report-only"
    )
}

if ($RunTag.Trim()) {
    $Command += @("--run-tag", $RunTag.Trim())
}
if ($SkipTransfer) {
    $Command += "--skip-transfer"
}
if ($PrebuildData) {
    $Command += "--prebuild-data"
}
if ($RequirePrebuiltData) {
    $Command += "--require-prebuilt-data"
}
if ($TwoPhaseStage3) {
    $Command += "--two-phase-stage3"
}
if ($EstimateOnly) {
    $Command += "--estimate-only"
}

"[FullRunControl] repo=$RepoRoot" | Tee-Object -FilePath $LogPath
"[FullRunControl] action=$Action" | Tee-Object -FilePath $LogPath -Append
"[FullRunControl] log=$LogPath" | Tee-Object -FilePath $LogPath -Append
"[FullRunControl] command=$($Command -join ' ')" | Tee-Object -FilePath $LogPath -Append
& $Command[0] $Command[1..($Command.Length - 1)] 2>&1 | Tee-Object -FilePath $LogPath -Append
exit $LASTEXITCODE

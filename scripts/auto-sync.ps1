$ErrorActionPreference = 'SilentlyContinue'

$repoRoot = Split-Path -Parent $PSScriptRoot
$logFile = Join-Path $env:TEMP 'complete-data-science-auto-sync.log'

Set-Location -LiteralPath $repoRoot

git add -A
if ($LASTEXITCODE -ne 0) {
    "auto-sync: git add failed on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -Append -FilePath $logFile
    exit 1
}

git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    exit 0
}

$ts = Get-Date -Format 'yyyy-MM-dd HH:mm'
$files = (git diff --cached --name-only | Measure-Object).Count
git commit -m "chore: auto-sync WIP ($files files) $ts"
if ($LASTEXITCODE -ne 0) {
    "auto-sync: commit failed on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -Append -FilePath $logFile
    exit 1
}

git push origin main
if ($LASTEXITCODE -ne 0) {
    "auto-sync: push failed on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -Append -FilePath $logFile
}
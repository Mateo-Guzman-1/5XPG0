[CmdletBinding()]
param([switch]$Recreate)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if ($Recreate -and (Test-Path -LiteralPath ".venv")) {
    Remove-Item -LiteralPath ".venv" -Recurse -Force
}

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv .venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv .venv
    } else {
        throw "Python 3 was not found. Install it from python.org and enable the Python Launcher."
    }
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the virtual environment." }
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Failed to install requirements." }

Write-Host ""
Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1"

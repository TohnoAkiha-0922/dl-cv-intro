param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & $Python demo.py --output outputs/demo
    if ($LASTEXITCODE -ne 0) { throw 'Demo failed. See the Python error above.' }
} finally { Pop-Location }

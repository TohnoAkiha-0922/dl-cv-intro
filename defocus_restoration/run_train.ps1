param([string]$Python = '')
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $localEnvironment = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $localEnvironment) { $Python = $localEnvironment }
    elseif (Test-Path -LiteralPath 'D:\Anaconda3\envs\cv_pytorch\python.exe') {
        $Python = 'D:\Anaconda3\envs\cv_pytorch\python.exe'
    } else { $Python = 'python' }
}
Push-Location $PSScriptRoot
try {
    & $Python train.py --output outputs/network --epochs 12 --train-count 96 --val-count 16 --size 128 --batch-size 8 --device auto
    if ($LASTEXITCODE -ne 0) { throw 'Training failed. See the Python error above.' }
    & $Python evaluate.py --checkpoint outputs/network/best.pt --output outputs/network-evaluation --test-count 12 --device auto
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed. See the Python error above.' }
} finally { Pop-Location }

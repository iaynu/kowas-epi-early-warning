# 공식 백테스트 실행 (Windows PowerShell).
# 사용: .\scripts\run_backtest.ps1 -Kowas C:\data\KOWAS-EPI [-Folds 2024-Q2]
param([Parameter(Mandatory=$true)][string]$Kowas, [string[]]$Folds = @())
$ErrorActionPreference = "Stop"
$Here = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Py = Join-Path $Here ".venv\Scripts\python.exe"; if (-not (Test-Path $Py)) { $Py = "python" }
New-Item -ItemType Directory -Force (Join-Path $Here "results") | Out-Null
$env:PYTHONUTF8 = "1"; $env:OMP_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
$FoldArgs = @(); if ($Folds.Count -gt 0) { $FoldArgs = @("--folds") + $Folds }
Set-Location $Kowas
& $Py evaluation.py backtest --model (Join-Path $Here "model.py") @FoldArgs --out (Join-Path $Here "results\predictions.csv") --json (Join-Path $Here "results\score.json")
& $Py evaluation.py score (Join-Path $Here "results\predictions.csv") @FoldArgs

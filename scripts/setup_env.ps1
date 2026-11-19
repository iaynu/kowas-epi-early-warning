# 실행환경 구성 (Windows PowerShell). 사용: .\scripts\setup_env.ps1 [-Lock]
param([switch]$Lock)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip | Out-Null
if ($Lock) { .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt }
else       { .\.venv\Scripts\python.exe -m pip install -r requirements.txt }
.\.venv\Scripts\python.exe -c "import sys, numpy, sklearn; print('Python', sys.version.split()[0], '| NumPy', numpy.__version__, '| scikit-learn', sklearn.__version__)"

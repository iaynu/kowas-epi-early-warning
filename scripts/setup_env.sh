#!/usr/bin/env bash
# 실행환경 구성 (Linux·macOS). 사용: bash scripts/setup_env.sh [lock]
#   기본: requirements.txt (버전 범위, Python 3.11 이상)
#   lock: requirements-lock.txt (검증 환경의 상세 버전, Python 3.12 이상)
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip >/dev/null
if [ "${1:-}" = "lock" ]; then
  .venv/bin/python -m pip install -r requirements-lock.txt
else
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python -c "import sys, numpy, sklearn; print('Python', sys.version.split()[0], '| NumPy', numpy.__version__, '| scikit-learn', sklearn.__version__)"

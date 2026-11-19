#!/usr/bin/env bash
# 공식 백테스트 실행 (Linux·macOS).
# 사용: bash scripts/run_backtest.sh <KOWAS-EPI 배포 폴더> [평가창ID ...]
#   예) bash scripts/run_backtest.sh ~/KOWAS-EPI              # 10개 창 전체 (약 20~35분)
#       bash scripts/run_backtest.sh ~/KOWAS-EPI 2024-Q2      # 한 창만 (약 1.5~3분, 동작 확인용)
# 결과는 results/ 에 저장한다(제출본 predictions.csv·score.json은 건드리지 않는다).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
KOWAS="${1:?KOWAS-EPI 배포 폴더 경로를 첫 인자로 주십시오}"; shift || true
PY="$HERE/.venv/bin/python"; [ -x "$PY" ] || PY="python3"
mkdir -p "$HERE/results"
FOLDS=(); if [ "$#" -gt 0 ]; then FOLDS=(--folds "$@"); fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1
cd "$KOWAS"
"$PY" evaluation.py backtest --model "$HERE/model.py" "${FOLDS[@]}" \
      --out "$HERE/results/predictions.csv" --json "$HERE/results/score.json"
"$PY" evaluation.py score "$HERE/results/predictions.csv" "${FOLDS[@]}"

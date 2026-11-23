"""predictions.csv 형식 검증과 두 예측 파일 비교 (표준 라이브러리만 사용).

사용:
  python scripts/check_predictions.py predictions.csv --keys <KOWAS-EPI>/eval_keys.csv
  python scripts/check_predictions.py results/predictions.csv --compare predictions.csv
"""
import argparse, csv, math, sys

def load(path):
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        header = next(rd, [])
        rows = list(rd)
    return header, rows

ap = argparse.ArgumentParser()
ap.add_argument("pred")
ap.add_argument("--keys", help="배포본의 eval_keys.csv (있으면 행 키까지 검증)")
ap.add_argument("--compare", help="비교할 다른 predictions.csv (예: 제출본)")
ap.add_argument("--folds", action="store_true", help="--folds 로 만든 일부 창 결과일 때 키 누락 검사를 건너뜀")
a = ap.parse_args()

header, rows = load(a.pred)
bad = []
if header != ["region", "date_week", "pred_conc", "pred_alert"]:
    bad.append(f"헤더가 규정과 다릅니다: {header}")
seen = {}
for i, r in enumerate(rows, 2):
    if len(r) != 4:
        bad.append(f"{i}행: 열 수 {len(r)}"); continue
    k = (r[0], r[1])
    if k in seen: bad.append(f"{i}행: 중복 키 {k}")
    seen[k] = r
    try:
        c = float(r[2])
        if not math.isfinite(c) or c < 0: bad.append(f"{i}행: pred_conc 범위 {r[2]}")
    except ValueError:
        bad.append(f"{i}행: pred_conc 숫자 아님 {r[2]!r}")
    if r[3] not in ("0", "1"):
        bad.append(f"{i}행: pred_alert 는 정수 0/1 이어야 함 {r[3]!r}")
print(f"{a.pred}: {len(rows)}행")
if a.keys:
    with open(a.keys, encoding="utf-8", newline="") as f:
        want = {(r["region"], r["date_week"]) for r in csv.DictReader(f)}
    if not a.folds:
        miss = want - set(seen)
        if miss: bad.append(f"평가 키 {len(miss)}개 누락")
    extra = set(seen) - want
    if extra: bad.append(f"평가 키에 없는 행 {len(extra)}개")
    print(f"평가 키 {len(want)}개 대비 확인 완료")
if a.compare:
    _, rows2 = load(a.compare)
    other = {(r[0], r[1]): r for r in rows2}
    common = [k for k in seen if k in other]
    rel = max((abs(float(seen[k][2]) / float(other[k][2]) - 1) for k in common if float(other[k][2]) > 0), default=0.0)
    diff = sum(seen[k][3] != other[k][3] for k in common)
    print(f"비교 {a.compare}: 공통 {len(common)}행, 농도 최대 상대오차 {rel:.2e}, 경보 차이 {diff}행")
if bad:
    print("문제", len(bad), "건")
    for b in bad[:20]: print(" -", b)
    sys.exit(1)
print("형식 검증 통과")

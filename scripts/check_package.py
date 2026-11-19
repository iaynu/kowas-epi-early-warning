"""제출 패키지 자체 점검 (표준 라이브러리만 사용, 네트워크 불필요).

점검 항목
  1. 학습된 가중치·모델 파일, KOWAS 자료 파일이 패키지에 들어 있지 않은가 (대회 README §9-1)
  2. API Key·비밀번호·토큰·이메일·개인 경로 같은 민감정보 흔적이 없는가
  3. model.py와 보조 모듈이 규정 위치(최상위)에 있고 requirements.txt가 있는가

사용:  python scripts/check_package.py
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBID_EXT = {".pkl", ".pickle", ".npy", ".npz", ".joblib", ".pt", ".pth", ".h5", ".hdf5", ".onnx", ".bin", ".ckpt", ".safetensors", ".parquet", ".xlsx", ".gpkg", ".nc"}
ALLOWED_CSV = {"predictions.csv"}          # 공식 백테스트 결과(제출물 ②)만 허용
SAMPLE_DIR = "dashboard/sample/sample_data/"      # 난수로 만든 합성 샘플(dashboard/make_sample_data.py). 실제 자료 아님
KOWAS_NAMES = {"kowas_epi_panel.csv", "national_weekly.csv", "eval_keys.csv", "folds.csv", "plant_meta.csv",
               "asos_stations.csv", "weather_asos_daily.csv", "weather_quality_flags_v0.1.csv"}
SECRET_PATTERNS = {
    "API Key/토큰 형태": re.compile(r"(?i)\b(api[_-]?key|secret|token|passwd|password)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}"),
    "키 접두사": re.compile(r"\b(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12,}|xox[bap]-[A-Za-z0-9-]{10,})"),
    "이메일": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.(com|net|org|edu|gov|io|kr|jp|cn|uk|de)\b"),
    "개인·작업 경로": re.compile(r"(/tmp/claude|/home/claude|/root/|/Users/[A-Za-z]|C:\\Users\\|scratchpad)"),
}
TEXT_EXT = {".py", ".md", ".txt", ".json", ".ipynb", ".sh", ".ps1", ".csv", ".html"}
problems, notes = [], []

files = []
for d, dirs, fs in os.walk(ROOT):
    dirs[:] = [x for x in dirs if x not in {".git", "__pycache__", ".venv", "results"}]
    for f in fs:
        files.append(os.path.relpath(os.path.join(d, f), ROOT))
files.sort()

for rel in files:
    base, ext = os.path.basename(rel), os.path.splitext(rel)[1].lower()
    if ext in FORBID_EXT:
        problems.append(f"가중치·자료 형식 의심 파일: {rel}")
    if ext == ".csv" and base not in ALLOWED_CSV and not rel.replace(os.sep, "/").startswith(SAMPLE_DIR):
        problems.append(f"허용되지 않은 CSV: {rel}")
    if base in KOWAS_NAMES:
        problems.append(f"KOWAS 배포 자료 파일: {rel}")
    if ext in TEXT_EXT and base != "predictions.csv" and os.path.abspath(os.path.join(ROOT, rel)) != os.path.abspath(__file__):
        txt = open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace").read()
        for name, pat in SECRET_PATTERNS.items():
            m = pat.search(txt)
            if m:
                problems.append(f"{name} 흔적: {rel} → {m.group(0)[:60]!r}")

for need in ["model.py", "requirements.txt", "README.md", "predictions.csv", "score.json"]:
    if not os.path.exists(os.path.join(ROOT, need)):
        problems.append(f"필수 파일 없음: {need}")
if not any(f.endswith(".ipynb") for f in files):
    notes.append("Jupyter Notebook 없음")
if not any(f.startswith("demo" + os.sep) and f.endswith((".mp4", ".webm", ".mov")) for f in files):
    notes.append("데모영상 없음")

print(f"점검 대상 파일 {len(files)}개")
for f in files:
    print("  ", f)
for n in notes:
    print("참고:", n)
if problems:
    print("\n문제", len(problems), "건")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("\n문제 없음: 가중치·KOWAS 자료·민감정보 흔적이 발견되지 않았습니다.")

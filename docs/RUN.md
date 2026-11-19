# KOWAS-EPI 제출 모델 r2_plant_v7_safe (r2_plant_v7 + 학습 이력 안전장치) · The사랑해팀

공식 v2.5 전체 10개 평가 창을 위한 실행 가능한 모델입니다.
검증 점수: 0.3789432154787439 (Linux, Python 3.13, NumPy 2.5.3, scikit-learn 1.9.1). 이 폴더의 predictions.csv와 score.json은 이 코드로 한 번 실행한 공식 전체 백테스트 결과다. 같은 원본 코드는 Windows 11·NumPy 2.5.2 환경에서 0.3789599330이었다(환경 차이 0.00002, 경보 동일).

## 실행

간단히 실행하려면 `scripts/`의 `setup_env`와 `run_backtest`(README 0·1절)를 쓰면 됩니다. 아래는 같은 일을 직접 입력하는 방법입니다.

검증 환경: Windows 11, Python 3.13.14, NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.1, CPU. Linux(Python 3.13, NumPy 2.5.3, scikit-learn 1.9.1)에서 다시 실행한 결과 최종 점수 0.3789432155(차이 0.00002)로 재현되었고 경보 2,159행은 모두 같았다. requirements.txt는 버전 범위(numpy>=1.26, scikit-learn>=1.4)로 적었다. NumPy 2.5는 Python 3.12 이상만 지원하므로 버전을 고정하면 Python 3.11 채점 환경에서 설치가 실패할 수 있기 때문이며, SciPy는 코드에서 쓰지 않아 뺐다.
운영기관이 확정한 Python/CPU/메모리/시간 상한은 별도로 확인해야 합니다. 이 환경 외의 실행은 검증하지 않았습니다.

1. 공식 KOWAS-EPI v2.5 데이터 ZIP을 별도 폴더에 풉니다(아래 예시는 `C:/data/KOWAS-EPI`).
2. 이 저장소를 `KOWAS-EPI_model` 폴더로 받습니다(`git clone <저장소 주소> KOWAS-EPI_model`, 또는 제출용 모델자료 zip을 풉니다). 코드 파일들이 이 폴더 안에 함께 있어야 합니다(아래 예시는 `C:/KOWAS-EPI_model`).
3. `KOWAS-EPI_model` 폴더로 이동한 뒤 다음 명령을 실행합니다. 데이터 폴더 경로는 자신의 위치로 바꿉니다. 결과는 이 폴더의 `predictions.csv`·`score.json`에 저장됩니다.

Windows (PowerShell):

```powershell
cd C:/KOWAS-EPI_model
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONUTF8="1"
$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
.venv\Scripts\python.exe "C:/data/KOWAS-EPI/evaluation.py" backtest --model "C:/KOWAS-EPI_model/model.py" --out predictions.csv --json score.json
.venv\Scripts\python.exe "C:/data/KOWAS-EPI/evaluation.py" score predictions.csv
```

Linux·macOS:

```bash
cd /path/to/KOWAS-EPI_model
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python /path/to/KOWAS-EPI/evaluation.py backtest --model "$PWD/model.py" \
    --out predictions.csv --json score.json
.venv/bin/python /path/to/KOWAS-EPI/evaluation.py score predictions.csv
```

`--folds`와 `--keep-going`을 붙이지 않습니다. 평가기가 매 평가 창마다 독립 프로세스를 만들고 정적 파일 위치를 KOWAS_STATIC_DIR로 제공합니다.
패키지 설치 뒤 모델 실행에는 네트워크를 사용하지 않습니다. 모델은 fit/predict 인자로 받은 rows, national, aux만 시계열 입력으로 사용합니다.
직접 읽는 파일은 허용된 asos_stations.csv와 plant_meta.csv뿐입니다. 패널, 정답, 보조 관측 CSV를 직접 읽지 않습니다.

## 모델과 시점 조건

- 보존된 v8의 최근 Huber 농도 예측과 장기 선형/HGB 다양성 예측을 기반으로 합니다.
- 과거에 발행한 예측을 그 당시 정보만으로 재구성하고, 정답이 관측된 뒤에만 최근 오차 보정과 경보 임계값 학습에 사용합니다.
- n_sites와 pop_sampled를 허용된 정적 처리장 정보와 대조해 당시 관측 구성을 추론합니다. first_sample_week가 해당 주보다 늦은 처리장은 제외하며, last_sample_week와 in_current_registry는 사용하지 않습니다.
- 관측 구성의 농도 편향은 과거 관측 변화로부터 실행 중 학습합니다. 특정 연도·주차·지역별 정답 수치나 사전 학습 가중치를 포함하지 않습니다.
- 결측 주에 기존 3주 평균이 있으면 관측된 과거 값으로 재구성해 알려진 미래 평균 예외를 차단합니다.
- t+2 라벨은 실제 관측 시점이 원점 이하일 때만, t+3 보조 정답도 해당 관측이 원점 이하일 때만 사용합니다.
- state_space.py는 처리장 편향과 지역 상태를 분리하는 인과적 상태 공간 예측을 현재 n_sites에 따라 2.5~20% 결합합니다. 전국 변화를 추정할 관측 지역이 부족하면 기존 최종 예측을 유지합니다.

## 한계

공개 10개 평가 창을 반복해서 확인하며 선택했습니다. 추가 과거 워크포워드도 이미 관찰된 자료이며 미사용 시험 데이터가 아닙니다.
점수 상승은 이 공식 백테스트에서 확인한 결과입니다. 평가 기간 이후의 일반화 성능이나 실시간 공표 지연을 반영한 성능을 보장하지 않습니다.

## 과적합 점검과 학습 이력 안전장치 (이 버전에서 추가)

- **추가한 규칙(`HISTORY_MIN`=26, `HISTORY_FULL`=50, model.py)**: 원점까지 쌓인 주차 수가 26주 이하이면 공식 기준 모델(원점 3주 평균, 현재 경보 지속)을 그대로 쓰고, 50주 이상이면 모델 예측을 그대로 쓰며, 그 사이는 로그 농도를 선형으로 섞는다(경보는 비중이 큰 쪽을 따른다). 52주 수준·계절항 같은 특징은 약 1년 이력이 있어야 의미가 생기기 때문이다. 특정 주차·연도·지역을 가리키지 않는 일반 규칙이다.
- **공식 10개 평가 창에는 영향이 없다**: 공식 창의 가장 이른 원점도 이력이 51주라 규칙이 작동하지 않으며, 예측 파일이 원본 r2_plant_v7 재실행 결과와 바이트 단위로 같다.
- **조정에 쓰지 않은 기간 점검(2023-Q3·2023-Q4·2024-Q1, 같은 평가기·같은 규칙으로 추가 정의)**:

| 모델 | 공식 10개 창 | 추가 3개 창 |
|---|---|---|
| 공식 기준 모델 | 0.2212 | 0.2226 |
| r2_plant_v7 원본 | 0.3789 | 0.1076 |
| **이 폴더 (안전장치 추가)** | **0.3789** | **0.1774** |

  추가 3개 창은 학습 이력이 10~50주뿐이라 모든 학습 모델이 기준 모델보다 낮았다. 안전장치의 26·50주 값은 이 기간을 본 뒤 정했으므로 위 0.1774는 효과의 크기를 가늠하는 참고치다.
- **누수 점검**: 실행 중 열린 파일은 `plant_meta.csv`·`asos_stations.csv`(허용 정적 파일)와 라이브러리 시스템 파일뿐이었다. `last_sample_week`·`in_current_registry`를 무작위로 바꾸거나, 마지막 원점 이후의 패널·전국·기상 값과 정답을 바꿔도 예측이 하나도 변하지 않았다(2025-Q2 창).
- **일반화 성능**: 공개 10개 창은 설정 선택에 반복 사용되었다. 한 창 제외 진단상 선택 단계마다 0.002~0.004가 부풀려졌을 수 있어, 학습 이력이 1년 이상인 새 기간의 기대 성능은 약 0.37로 본다(추정치).

## 자료 출처

- 하수 농도·전국 집계·임상 신고수·처리장 정보: 질병관리청 KOWAS 하수 감시 공개 통계, 감염병포털 표본감시(대회 배포본 v2.5).
- 패널 기상(`precip_mm`·`temp_avg`): Weather data by Open-Meteo.com (CC BY 4.0), KOWAS 주차별로 집계·가공.
- ASOS 일자료·관측소 정보: 출처: 기상청 종관기상관측(ASOS) 일자료(공공데이터포털)·지점정보(기상자료개방포털).
- 대회 배포 자료 밖의 외부 자료는 쓰지 않았고, 패키지에 자료 파일을 넣지 않았다.

## 제출 전 점검 (README §9-4 권장 절차)

- 제출 zip을 새 폴더에 그대로 풀고, Python 수준에서 네트워크 연결을 막은 상태로 `evaluation.py backtest`를 `--folds`·`--keep-going` 없이 처음부터 실행했다.
- 결과: 10개 평가 창 모두 `"status": "ok"`, 최종 정량점수 0.3789432155이고, 패키지의 `predictions.csv`와 바이트 단위로 같았다. `evaluation.py score` 형식 검사도 통과했다.
- 이 점검은 Python 수준 차단이며 OS 보안 샌드박스 검증은 아니다. `requirements.txt`만 설치한 새 가상환경 점검은 공식 채점 환경(Python 버전 등)이 공지되면 그 환경에서 다시 하는 것을 권한다.

## 검토 이력

- 2026-11-05: 개선 문서(P0) 순서로 검토했다. 모듈별 기여도 표, 취약 창(2025-Q1·2026-Q1) 분석, 수정안 1개를 시험했다. 수정안(결측 원점 경보를 기준 모델 규칙으로 대체)은 점수가 0.3789에서 0.3775로 낮아져 기각했다.
- 2026-11-05: 확장성 점검을 했다. 농도 ×100 자료에서 0.3789(예측이 100배로 따라감, 경보 동일), 임상 신고수가 없는 자료에서 0.3560으로 코드 수정 없이 실행됐다(README 8절).
- 모델 코드·조정값·예측 파일은 바뀌지 않았다. 최종 정량점수는 0.3789(0.3789432155) 그대로다. 자세한 내용은 `REVIEW.md`와 README 7절에 있다.

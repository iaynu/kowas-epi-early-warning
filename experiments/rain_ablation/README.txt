강수 비교 실험 근거 (설명서 4.8절, 모델자료 README 7.4절)

목적
  제출 모델의 코드와 설정은 그대로 두고, 입력에서 기상 열만 빼서 공식 10개 평가 창을 다시 돌려
  강수 입력의 예측 기여를 측정했다. 원점마다 처음부터 다시 학습하므로 각 조건에서 모든 학습기가 다시 학습된다.

조건                     최종 점수   평균 sMAPE   Macro-F1
  R0  기상 입력 없음         0.3597     53.38       0.6273
  R1  패널 시·도 주간 기상만  0.3782     52.70       0.6594
  RA  ASOS 일자료 강수만     0.3612     53.21       0.6273
  RT  패널 기온만            0.3597     53.38       0.6273   (R0와 예측이 바이트 단위로 같음: 기온 변수는 학습기 입력 목록에 없음)
  R2  둘 다 (제출 모델)       0.3789     52.61       0.6594   (모델자료의 score.json)

파일
  wrap.py               입력 필터. 제출 모델(model_base.Model)을 감싸 rows의 precip_mm·temp_avg를 비우거나,
                        ASOS를 뺄 때는 aux를 넘기지 않는다(남는 품질주의표는 제출 코드가 쓰지 않음).
  R0~RT/model.py        조건별 진입점
  R0~RT/score.json      공식 평가기 결과
  R0~RT/predictions.csv 공식 평가기 예측

재현 방법
  1. 조건 폴더(예: R0)에 모델자료의 코드 파일을 모두 복사하고, 제출본 model.py의 이름을 model_base.py로 바꾼다.
  2. wrap.py를 같은 폴더에 둔다(R0~RT/model.py는 그대로 둔다).
  3. KOWAS-EPI v2.5 배포 폴더에서:
     OMP_NUM_THREADS=1 python evaluation.py backtest --model <조건 폴더>/model.py --out <조건 폴더>/predictions.csv --json <조건 폴더>/score.json

주의
  - 예측 기여를 측정한 것이지 희석 원리를 입증한 것이 아니다.
  - 모델 설정은 강수가 있는 상태에서 골랐고 다른 조건에서 다시 고르지 않았다.
  - 이미 여러 번 본 공개 창에서의 비교이며, 실행 환경은 Linux, Python 3.13, NumPy 2.5.3, scikit-learn 1.9.1이다.

공개 저장소 안내
  각 조건 폴더의 predictions.csv는 대회 배포 자료에서 만들어진 결과라 이 저장소에서는 뺐다. 위 재현 방법으로 다시 만들 수 있고, 점수는 score.json에 남아 있다.

# KOWAS SIGNAL 대시보드

모델이 낸 2주 뒤 하수 농도·급증 경보 예측을 지도, 시계열, 평가표로 살펴보는 단일 HTML 대시보드입니다.
브라우저로 열면 바로 보이고 인터넷 연결이 필요 없습니다. 시연 영상(`../demo/KOWAS_demo.mp4`)에 나오는 화면이 이 파일입니다.

| 파일 | 내용 |
|---|---|
| `KOWAS_signal.html` | 최종 대시보드. 10개 평가 구간의 시·도별 관측값, 예측값, 점수가 내장되어 있습니다 |

## 데이터에 관해

`KOWAS_signal.html`에는 대회 배포 자료에서 온 시·도별 주간 농도값과 모델 예측값이 파일 안에 들어 있습니다.
대회 배포 원자료 파일(CSV 등)은 저장소에 없습니다. 이 HTML의 공개 가능 여부는 팀이 확인했습니다.

## 화면 구성

1. 감시 현황: 급증 경보 지역 수, 지역별 경보 지도, 지역 상세 시계열, 관측·예측 표
2. 경보 지도: 지역별 경보 상태와 예측 농도
3. 전국·임상 흐름: 전국 예측 농도와 임상 신고수 비교
4. 평가 결과: 종합 점수, sMAPE, Macro-F1, 오류 분석, 평가 구간별 성능

## 내 자료로 보려면

화면 오른쪽 위 '자료 불러오기'에서 CSV를 고릅니다. 필수 열은 아래와 같고, 그 밖의 열은 있어도 됩니다.

- 패널(`panel.csv`): `region, date_week, target_week_t2, conc_mean, conc_base_avg, n_sites, pop_sampled, target_conc_t2, target_alert_t2` (선택: `pathogen, conc_3wk_avg, missing_reason`)
- 예측(`predictions.csv`): `region, date_week(목표 주), pred_conc, pred_alert`
- 전국(`national.csv`, 선택): `date_week, conc_mean_national, covid_cases_national`
- 점수(`score.json`, 선택): `final`, `folds[]`(`score, smape, macro_f1`)

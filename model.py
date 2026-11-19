# [개요] 평가기가 부르는 진입점. Model.fit(rows, national, aux)이 자료를 받고, Model.predict(...)가 t+2주 농도와 경보를 낸다.
# 처리 순서: core(최근 학습기) -> diversity(장기 학습기 혼합) -> online(오차 보정) -> calibration(경보 임계값)
#            -> plant_bias(처리장 구성 보정) -> state_space(상태 공간 결합) -> 학습 이력 안전장치.
# 입력은 원점 주차까지의 공개 자료뿐이며, 저장된 가중치나 KOWAS 수치는 쓰지 않는다.
"""KOWAS-EPI v8 (r2_plant_v7) + 학습 이력 안전장치: causal rolling forecasts with observation-weighted diversity."""
import math
import numpy as np
from core import Model as Core
from calibration import CalibratedCorrector
from plant_bias import PlantBias
from state_space import StateSpace
from diversity import LongForecaster as Diversity

BLEND_WEIGHT = 0.3
# 학습 이력 안전장치: 원점까지 쌓인 주차 수가 HISTORY_MIN 이하이면 공식 기준 모델(3주 평균·현재 경보 지속)을 그대로 쓰고,
# HISTORY_FULL 이상이면 모델 예측을 그대로 쓰며, 그 사이는 로그 농도를 선형으로 섞는다(경보는 비중이 큰 쪽).
# 52주 수준·계절항 같은 특징은 약 1년 이력이 있어야 의미가 생기므로, 이력이 짧은 시기의 과적합 위험을 줄이는 일반 규칙이다.
HISTORY_MIN, HISTORY_FULL = 26, 50

def baseline(rows, origin_week):
    """공식 나이브 기준 모델과 같은 규칙."""
    latest, current = {}, {}
    for r in sorted(rows, key=lambda r: r["date_week"]):
        if r["conc_3wk_avg"] != "" or r["conc_mean"] != "":
            latest[r["region"]] = r
        if r["date_week"] == origin_week:
            current[r["region"]] = r
    out = {}
    for region in sorted({r["region"] for r in rows}):
        cur = current.get(region)
        ref = cur if cur and (cur["conc_3wk_avg"] or cur["conc_mean"]) else latest.get(region)
        conc = float(ref["conc_3wk_avg"] or ref["conc_mean"]) if ref else 0.0
        alert = int(cur is not None and cur["conc_mean"] != "" and cur["conc_base_avg"] != ""
                    and float(cur["conc_mean"]) >= 1.3 * float(cur["conc_base_avg"]))
        out[region] = (conc, alert)
    return out

def sanitize(rows):
    """Reconstruct an existing rolling average only when observed past values permit it.
    Never fill a missing average or fabricate concentrations; no region/date switches.
    """
    by={}
    for r in rows:by.setdefault(r["region"],[]).append(r)
    changed={}
    for reg,rs in by.items():
        rs.sort(key=lambda r:r["date_week"])
        for j,r in enumerate(rs):
            if not r["conc_mean"] and r["conc_3wk_avg"]:
                values=[float(x["conc_mean"]) for x in rs[max(0,j-2):j+1] if x["conc_mean"]]
                if values:changed[(reg,r["date_week"])]=str(sum(values)/len(values))
    return [dict(r,conc_3wk_avg=changed[(r["region"],r["date_week"])]) if (r["region"],r["date_week"]) in changed else r for r in rows]

class Model:
    def fit(self, rows, national, aux=None):
        rows=sanitize(rows)
        self.core=Core()
        self.corrector=CalibratedCorrector()
        self.plant_bias=PlantBias()
        self.state_space=StateSpace()
        self.diversity=Diversity()
        self.core.fit(rows,national)
        self.diversity.fit(rows,national,aux)

    def predict(self, rows, origin_week, national, aux=None):
        # Explicit guards supplement the official evaluator's masking contract.
        rows=[dict(r, target_conc_t2="",target_alert_t2="") if r["target_week_t2"]>origin_week else r
              for r in rows if r["date_week"]<=origin_week]
        national=[r for r in national if r["date_week"]<=origin_week]
        if aux:aux={k:[r for r in v if r["date_week"]<=origin_week] for k,v in aux.items()}
        raw_rows=rows
        rows=sanitize(rows)
        a=self.core.predict(rows,origin_week,national)
        long=self.diversity.predict(rows,origin_week,national,aux)
        result={}
        counts={}
        for row in rows:
            counts.setdefault(row["region"],[]).append((row["date_week"],float(row["n_sites"]) if row["n_sites"] else 0.0))
        for reg in counts:counts[reg].sort()
        for reg in a:
            diag=self.core.diagnostics.get(reg)
            if diag is None:
                result[reg]=a[reg];continue
            other=long[reg]
            history=counts[reg]
            site_change=abs(history[-1][1]-history[-3][1]) if len(history)>=3 else 0.0
            weight=BLEND_WEIGHT*float(np.clip(diag["features"]["ns"]/10,0,1))/(1+.25*site_change)
            lp=(1-weight)*math.log10(max(a[reg][0],1e-25))+weight*other
            result[reg]=(float(10**np.clip(lp,-25,30)),int(a[reg][1]))
        result=self.corrector.correct(result, self.core._previous_data, origin_week, rows, self.core.diagnostics)
        result=self.plant_bias.correct(result, rows, self.core._previous_data, origin_week)
        o=self.core._previous_data.widx[origin_week]
        for g,reg in enumerate(self.core._previous_data.regions):
            m=self.corrector.margins[o][g]
            if np.isfinite(m):
                adj=self.plant_bias.audit['details'][g]['alert_correction']
                result[reg]=(result[reg][0],int(m+adj>=self.corrector.audit['threshold']))
        final=self.state_space.correct(result, rows, self.core._previous_data, origin_week, self.plant_bias, a)
        history=len({r["date_week"] for r in raw_rows})
        w=float(np.clip((history-HISTORY_MIN)/(HISTORY_FULL-HISTORY_MIN),0.0,1.0))
        if w<1.0:
            base=baseline(raw_rows,origin_week)
            for reg,(c,al) in list(final.items()):
                bc,bal=base.get(reg,(0.0,0))
                if bc>0 and c>0:
                    c=10**(w*math.log10(c)+(1-w)*math.log10(bc))
                elif bc>0 and w<0.5:
                    c=bc
                final[reg]=(float(c), int(al if w>=0.5 else bal))
        return final

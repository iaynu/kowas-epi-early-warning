# [개요] 장기 학습기. 평가 창 시작 때 한 번 학습한다(HistGradientBoosting 25% + Huber 선형 75%).
# ASOS 일자료에서 주 강수 합, 일 최대, 20mm 이상 일수를 만들어 입력에 더한다. 최근 학습기와 최대 30%까지만 섞는다.
"""KOWAS-EPI 통합 모델 (v5.1) — t+2주 시·도 하수 농도(pred_conc)와 급증 경보(pred_alert).

v4.1의 두 구성 요소(A: kowas_v31, B: kowas_split)를 한 파일·한 자료 계층으로 합치고, 제거 실험(ablation)에서
기여가 확인되지 않은 부품을 걷어 냈다. 남은 구조는 '서로 다른 두 학습 방식'이다.

  자료 계층 Data      : rows·national·aux를 한 번만 읽어 시·도 × 주차 행렬로 만든다(두 학습기가 공유).
                         농도 결측은 '혼합4'(최근 관측 8개 로그 지수가중 α=0.5 + 전국 변화, 최대 8주)로 채운다.
  장기 학습기 Long    : 평가 창 시작 때(fit) 지금까지의 모든 라벨로 한 번 학습.
                         HistGradientBoosting 3종 평균 50% + Huber 선형 50%. 목표 = log10(t+2 농도) − log10(3주 평균).
  최근 학습기 Recent  : 원점마다(predict) 최근 104주 라벨로 다시 학습하는 Huber 릿지 선형(23개 특징, 1년·반년 계절항).
                         (선택) numpy LSTM·GRU 변화량을 30% 섞는다 — CFG["seq"].
  결합               : 예측 로그 농도 = 0.4·장기 + 0.6·최근 (가중 기하평균).
  경보               : 장기 학습기의 예측 농도 ≥ 1.3 × conc_base_avg 이면 1 (README 경보 정의와 같은 배수).

v4.1에서 뺀 것(제거 실험 근거는 보고서): ExtraTrees·kNN, 묶음 A/B/C 3중 평균, 처리장 커버리지 묶음(plant_meta),
B의 경보 전용 선형 모델, 1위 강제 경보, Transformer 코드(미사용), 장기 학습기의 1주 변화(mom) 특징.
결합 비중은 장기 0.4 : 최근 0.6.
v5.1(팀원 제안 특징 반영): 최근 학습기에 '시·도 간 2주 변화 분산(disp2)', 장기 학습기에 '결측 보정 변화량(impd)'을 더하고,
최근 학습기 선형 모델의 Huber 경계를 0.15 → 0.3으로 바꿨다(공식 0.3414 → 0.3432).

규칙(README §9-1)
  - 자료는 evaluation.py가 넘기는 rows·national·aux 인자만 쓴다. auxdata 파일을 직접 읽지 않는다.
  - 정적 파일은 KOWAS_STATIC_DIR의 asos_stations.csv(관측소→시·도)만 읽는다.
  - 모든 특징은 그 행의 주차까지 공개된 값만 쓴다. 라벨은 목표주 ≤ 기준 주인 것만 쓴다.
  - 가중치를 저장하지 않고 평가 창마다 새로 학습한다. 난수 시드 고정, 네트워크 없음, 특정 주차·연도 조건문 없음.
"""
import os, csv, math
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor

# ---------------------------------------------------------------- 설정(제출본은 이 기본값 그대로)
CFG = dict(
    w_long=0.4,          # 결합 비중(장기 학습기). 0.35~0.45에서 평평(0.3408~0.3412)
    seq=True,            # 최근 학습기에 LSTM·GRU 변화량 섞기. 실행 시간 상한이 빠듯하면 False(약 4.5배 빠름, 점수 −0.0008)
)
NAN = float("nan")
ALPHA, NOBS, GATE, MAXSTALE = 0.5, 8, 12, 8          # 혼합4 결측 채움
COVMIN, COVLAG = 45, 1                                # 장기: 확진자 특징은 학습 자료에 45주 이상일 때, t−1주까지
TAU = 1.3                                             # 경보 기준 배수
HGB_CFGS = [dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40),
            dict(max_iter=400, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=30),
            dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=7, min_samples_leaf=60)]
WINDOW, HUBER_LIN, LAM = 104, 0.3, 3.0               # 최근 학습기 선형: 학습 주 수, Huber 경계(log10), 릿지 강도
HUBER = 0.15                                          # 순환망 학습의 Huber 경계(log10)
SEQ_W, SEQ_H, SEQ_EPOCHS, SEQ_LR, SEQ_WD, SEQ_SEEDS = 0.3, 8, 80, 0.01, 0.01, 3
SEQ_KINDS = ("lstm", "gru")
RECENT_FEATS = ["x_last", "x_med", "x_d4", "age", "miss", "comp4", "comp1", "d1", "pr0", "pr2", "cs1", "cs4",
                "l52", "ns", "sin1", "cos1", "sin2", "cos2", "vol", "vxl", "vxm", "ald", "disp2"]


def _f(x):
    return float(x) if x not in ("", None) else np.nan


def _static(name):
    return Path(os.environ["KOWAS_STATIC_DIR"]) / name


def _ffill(a):
    """앞으로 채운 값과 각 위치의 마지막 관측 위치(없으면 -1)"""
    out = a.copy(); idx = np.full(len(a), -1); last = -1
    for i in range(len(a)):
        if a[i] == a[i]: last = i
        else: out[i] = out[i - 1] if i > 0 else NAN
        idx[i] = last
    return out, idx


def _roll_median(a, i, win=26, min_obs=8):
    seg = a[max(0, i - win + 1): i + 1]; seg = seg[seg == seg]
    return float(np.median(seg)) if len(seg) >= min_obs else NAN


# ================================================================ 1) 자료 계층(공유)
class Data:
    """rows(문자열 dict) → 시·도 × 주차 행렬. 두 학습기가 같은 객체를 쓴다."""
    def __init__(self, rows, national):
        self.weeks = sorted({r["date_week"] for r in rows}); self.regs = sorted({r["region"] for r in rows})
        self.widx = {w: k for k, w in enumerate(self.weeks)}; ri = {g: k for k, g in enumerate(self.regs)}
        R, W = len(self.regs), len(self.weeks)
        z = lambda: np.full((R, W), np.nan)
        self.C, self.C3, self.B, self.T, self.PR = z(), z(), z(), z(), z()
        self.NS = np.zeros((R, W)); self.code = {}
        for r in rows:
            i, j = ri[r["region"]], self.widx[r["date_week"]]
            self.C[i, j] = _f(r["conc_mean"]); self.C3[i, j] = _f(r["conc_3wk_avg"]); self.B[i, j] = _f(r["conc_base_avg"])
            self.NS[i, j] = _f(r["n_sites"]) if r["n_sites"] != "" else 0
            self.T[i, j] = _f(r.get("target_conc_t2", "")); self.PR[i, j] = _f(r.get("precip_mm", ""))
            self.code[r["region"]] = r["region_code"]
        nat = {r["date_week"]: _f(r["conc_mean_national"]) for r in national}
        cov = {r["date_week"]: _f(r.get("covid_cases_national", "")) for r in national}
        with np.errstate(divide="ignore", invalid="ignore"):
            self.N = np.log10(np.array([nat.get(w, np.nan) for w in self.weeks]))     # 전국 로그 농도
            self.CV = np.log10(1 + np.array([cov.get(w, np.nan) for w in self.weeks]))  # 전국 신고수 log10(1+x)
            self.L = np.log10(self.C); self.L3 = np.log10(self.C3); self.TG = np.log10(self.T)
        self.woy = np.array([int(w[-2:]) for w in self.weeks])
        self.Limp, self.filled, self.stale = self._impute()
        self.A3 = self._three_week()
        ff = [_ffill(self.L[g]) for g in range(R)]
        self.F = np.vstack([f[0] for f in ff]); last_idx = np.vstack([f[1] for f in ff])
        self.age = np.where(last_idx < 0, 8.0, np.arange(W)[None, :] - last_idx).astype(float)
        self.comp = self._comp()
        R_, W_ = self.L.shape
        chg2 = np.full((R_, W_), np.nan); chg2[:, 2:] = self.F[:, 2:] - self.F[:, :-2]     # 직전값 유지 로그 농도의 2주 변화
        self.disp2 = np.full(W_, np.nan)                         # 그 주 시·도 간 2주 변화의 표준편차(6개 시·도 이상)
        for j in range(W_):
            v = chg2[:, j][~np.isnan(chg2[:, j])]
            if len(v) >= 6: self.disp2[j] = v.std()
        self.impd = np.where(self.filled, np.abs(np.nan_to_num(self.Limp - self.F)), 0.0)  # 혼합4 채움과 직전값의 차이

    def _impute(self):
        """혼합4: 각 칸은 그 주까지의 값만 쓴다. 반환: 채운 로그 농도, 채움 여부, 마지막 관측 이후 주 수"""
        L = self.L; R, W = L.shape; out = L.copy(); filled = np.zeros((R, W), bool); stale = np.zeros((R, W))
        nobs = (~np.isnan(L)).sum(0)
        for i in range(R):
            last = None
            for j in range(W):
                if not np.isnan(L[i, j]):
                    last = j; continue
                stale[i, j] = (j - last) if last is not None else 99
                if last is None or j - last > MAXSTALE: continue
                ks = [k for k in range(j - 1, -1, -1) if not np.isnan(L[i, k])][:NOBS]
                w = (1 - ALPHA) ** np.arange(len(ks)); v = float(np.dot(w, L[i, ks]) / w.sum()); k0 = ks[0]
                if nobs[j] >= GATE and nobs[k0] >= GATE and not np.isnan(self.N[j]) and not np.isnan(self.N[k0]):
                    v += self.N[j] - self.N[k0]
                out[i, j] = v; filled[i, j] = True
        return out, filled, stale

    def _three_week(self):
        """3주 평균(로그): 제공값 → 비면 t−2~t 관측 주 평균 → 그것도 없으면 채운 값"""
        A = self.L3.copy(); R, W = A.shape
        for i in range(R):
            for j in range(W):
                if np.isnan(A[i, j]):
                    w = self.C[i, max(0, j - 2):j + 1]; w = w[~np.isnan(w)]
                    A[i, j] = math.log10(w.mean()) if len(w) else self.Limp[i, j]
        return A

    def _comp(self):
        """전국 상대 수준: 관측된 시·도의 (농도 − 자기 직전 26주 중앙값) 평균, 6개 시·도 미만이면 직전 값 유지"""
        n, m = self.L.shape; comp = np.full(m, NAN)
        for j in range(1, m):
            vals = [self.L[g, j] - md for g in range(n) if self.L[g, j] == self.L[g, j]
                    for md in [_roll_median(self.L[g], j - 1)] if md == md]
            if len(vals) >= 6: comp[j] = float(np.mean(vals))
        return _ffill(comp)[0]

    # 최근 학습기의 기준값: 제공 3주 평균, 없으면 직전값 유지
    def rbase(self, g, j):
        v = self.L3[g, j]
        return v if v == v else self.F[g, j]

    # 장기 학습기의 기준값(앵커): 3주 평균 → 직전 관측 → 그 주 시·도 중앙값
    def anchor(self, i, j):
        a = self.A3[i, j]
        if np.isnan(a):
            h = self.L[i, :j + 1][~np.isnan(self.L[i, :j + 1])]
            a = h[-1] if len(h) else (np.nanmedian(self.L[:, j]) if (~np.isnan(self.L[:, j])).any() else 3.5)
        return a

    def alert_base(self, i, j):
        b = self.B[i, j]
        if np.isnan(b):
            ww = self.C[i, max(0, j - 3):j + 1]; ww = ww[~np.isnan(ww)]; b = ww.mean() if len(ww) else 10 ** self.anchor(i, j)
        return b


class Rain:
    """ASOS 일강수(aux 인자). 빈칸 0 가정. 관측소-주(7일)마다 합·최대·20mm 이상 일수 → 시·도 관측소 평균"""
    def __init__(self):
        with open(_static("asos_stations.csv"), encoding="utf-8", newline="") as f:
            self.reg = {s["stn_id"]: s["region_code"] for s in csv.DictReader(f)}
        self.stats, self.done = {}, set()

    def update(self, aux):
        rows = aux.get("weather_asos_daily", []); buf = {}
        for r in reversed(rows):
            if r["date_week"] in self.done: break
            buf.setdefault((r["date_week"], r["stn_id"]), []).append(float(r["precip_mm"]) if r["precip_mm"] != "" else 0.0)
        acc = {}
        for (wk, s), v in buf.items():
            if len(v) == 7: acc.setdefault((self.reg[s], wk), []).append((sum(v), max(v), sum(x >= 20 for x in v)))
        for k, v in acc.items(): self.stats[k] = tuple(np.mean(np.array(v), axis=0))
        self.done |= {wk for wk, _ in buf}
        return self.stats


# ================================================================ 2) 장기 학습기
def long_features(D, rain, usecov):
    R, W = D.L.shape; L = D.Limp; obs = D.L; N = D.N; A3 = D.A3
    med = np.full(W, np.nan)                                   # 그 주 시·도 간 전주 대비 변화의 중앙값
    for j in range(1, W):
        b = ~np.isnan(obs[:, j]) & ~np.isnan(obs[:, j - 1])
        if b.sum() >= 3: med[j] = np.median(obs[b, j] - obs[b, j - 1])
    F = []
    for i in range(R):
        for j in range(W):
            a3 = A3[i, j]; g = lambda q: L[i, j - q] if j - q >= 0 else np.nan
            b = math.log10(D.B[i, j]) if not np.isnan(D.B[i, j]) else np.nan
            f = [i, math.sin(2 * math.pi * D.woy[j] / 52), math.cos(2 * math.pi * D.woy[j] / 52),
                 g(0) - a3, g(1) - a3, g(2) - a3, g(3) - a3, b - a3, a3,
                 D.stale[i, j], float(D.filled[i, max(0, j - 3):j + 1].sum()), D.NS[i, j],
                 N[j] - N[j - 1] if j >= 1 else np.nan, N[j] - N[j - 2] if j >= 2 else np.nan, med[j],
                 np.nansum(med[max(0, j - 1):j + 1])]
            if usecov:
                jc = j - COVLAG
                c0 = D.CV[jc] if jc >= 0 else np.nan
                c1 = D.CV[jc - 1] if jc - 1 >= 0 else np.nan
                c3 = D.CV[jc - 3] if jc - 3 >= 0 else np.nan
                f += [c0, c0 - c1, c0 - c3, (c0 - N[jc]) if (jc >= 0 and not np.isnan(N[jc])) else np.nan]
            if rain is not None:
                rc = D.code[D.regs[i]]; cur = rain.get((rc, D.weeks[j])); prv = rain.get((rc, D.weeks[j - 1])) if j >= 1 else None
                f += [math.log1p(cur[0]) if cur else np.nan, math.log1p(prv[0]) if prv else np.nan,
                      cur[1] if cur else np.nan, cur[2] if cur else np.nan]
            f.append(D.impd[i, j])                             # 결측 보정 변화량(관측 주는 0)
            F.append(f)
    return np.array(F, float).reshape(R, W, -1)


def design(X):
    """선형 입력: 결측은 0 + 결측 표시, 시·도는 원-핫"""
    Z = X[:, 1:].copy(); M = np.isnan(Z).astype(float); Z = np.where(np.isnan(Z), 0.0, Z)
    oh = np.zeros((len(X), 17)); oh[np.arange(len(X)), X[:, 0].astype(int)] = 1
    return np.hstack([Z, M, oh])


class Long:
    """HGB 1종 25% + Huber 선형 75%. 평가 창 시작 때 한 번 학습"""
    def fit(self, X, y):
        w = np.ones(len(y))
        self.hgb = [HistGradientBoostingRegressor(loss="absolute_error", l2_regularization=1.0, categorical_features=[0],
                                                  random_state=0, **c).fit(X, y, sample_weight=w) for c in HGB_CFGS[:1]]
        Z = design(X); self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        self.lin = HuberRegressor(epsilon=1.35, alpha=1e-2, max_iter=500).fit((Z - self.mu) / self.sd, y, sample_weight=w)
        return self

    def predict(self, X):
        return 0.25 * self.hgb[0].predict(X) + 0.75 * self.lin.predict((design(X) - self.mu) / self.sd)



class LongForecaster:
    def _prep(self, rows, national, aux):
        d=Data(rows,national)
        if self.usecov is None:self.usecov=int(np.isfinite(d.CV).sum())>=COVMIN
        rain=self.rain.update(aux) if aux else None
        return d,long_features(d,rain,self.usecov)

    def fit(self, rows, national, aux=None):
        self.rain,self.usecov=Rain(),None
        d,F=self._prep(rows,national,aux)
        X=[];y=[]
        cutoff=len(d.weeks)-1
        for g in range(len(d.regs)):
            for j in range(len(d.weeks)):
                if j+2>cutoff or not np.isfinite(d.T[g,j]) or d.T[g,j]<=0 or not np.isfinite(d.A3[g,j]):continue
                X.append(F[g,j]);y.append(math.log10(d.T[g,j])-d.A3[g,j])
        self.long=Long().fit(np.array(X),np.array(y))

    def predict(self, rows, origin_week, national, aux=None):
        d,F=self._prep(rows,national,aux);j=d.widx[origin_week]
        residual=self.long.predict(F[:,j,:])
        return {reg:float(d.anchor(g,j)+residual[g]) for g,reg in enumerate(d.regs)}

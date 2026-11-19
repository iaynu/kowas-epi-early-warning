# [개요] 최근 학습기. 원점 주차마다 그 시점까지의 자료로 처음부터 다시 학습한다.
# 입력: 시.도 주간 하수 농도, 임상 신고수, 패널 기상 등. 출력: 시.도별 t+2주 농도 예측과 경보 여유값.
# 구성: 결측 보정, 특징 생성, Huber 릿지 혼합, LSTM.GRU(numpy 구현).
"""KOWAS-EPI 팀 최종 후보 v4b — 팀원 회귀 안정 후보(stable candidate 1)를 바탕으로 모델 혼합·보정값·경보 규칙을 다시 고른 버전. numpy 하나만 쓴다.

구조
- 타깃: log10(t+2주 농도) - log10(기준값). 기준값 = 원점 주차 conc_3wk_avg, 없으면 그 시·도의 마지막 관측값.
- 변수: 기존 22개 + 지역 간 변화 분산·지역 특이 변화·결측 신뢰도·혼합4 보정 정보·전국 확산(breadth) 변수.
- 농도: Huber 릿지 6종의 가중 평균(CONC_MIX) — 104주(기본·약한 릿지·t+1주 정답), 78주(t+1주 정답),
  26주·39주(sMAPE 모양 손실). 여기에 공통 보정(CONC_POST: 상승/하락 비대칭·전국 확산·두 학습기간 불일치 감쇠·
  작년 같은 시기 변화)을 적용한다.
  이어서 변화 폭을 현재 상태(전국 상대 수준 변화·전국 신고수 변화·지역 변동성·급증 시·도 비율)에 따라 조정한다(AMP_TERMS 5개).
- 경보: 기존 팀 혼합과 '잡음 줄인 정답'(t+2주 값에 원점 이하로 관측된 t+1·t+3주 값을 섞은 학습 정답) 모델의 평균에
  경보 전용 보정(ALERT_POST)을 한 농도를 conc_base_avg 와 비교한다. 기준은 약 0.97배에서 출발해 모델끼리 의견이
  엇갈리거나 LSTM·GRU 예측이 흩어질수록 높이고 전국 확산이 넓을수록 낮춘다. 정답 라벨은 대회 정의(1.3배) 그대로이며,
  경보는 농도 제출값과 따로 정해도 된다는 운영기관 답변을 따른다.
- 모든 학습 정답은 원점 이하 관측값으로만 만든다(t+1주·t+3주 정답도 그 주가 원점 이하인 표본만).
- 조정값은 10개 평가 창 전체에서 고르게 잘 맞도록 고른 공통 값이다(README §12 교차검증). 창·주차·연도별 분기 없음.
저장된 가중치·조회표 없음. 난수는 시드 고정. 외부 파일은 읽지 않고 rows·national 인자만 쓴다.
"""
import math

import numpy as np

SANITIZE = 'available'
WINDOW = 104        # 학습에 쓰는 최근 목표주 수
HUBER = 0.15        # Huber 경계(log10 단위)
LAM = 3.0           # 릿지 강도
ALERT_C = 1.0       # 기준값이 없는 시·도(다른 시·도 평균으로 예측)의 경보 임계 배수
BASE8 = ["x_last", "x_med", "x_d4", "age", "miss", "comp4", "comp1", "d1"]
# 경보 모델(v3 그대로): 기본 8개 + 시·도 강수, 전국 코로나 신고수 변화
ALERT_FEATS = BASE8 + ["pr0", "pr2", "cs1", "cs4"]
# 농도 모델: 경보 변수 + 52주 평균 대비 수준, 측정 처리장 수, 1년·반년 주기, 최근 변동성과 그 상호작용, 경보기준 대비 수준
CONC_FEATS = ALERT_FEATS + ["l52", "ns", "sin1", "cos1", "sin2", "cos2", "vol", "vxl", "vxm", "ald"] + ["region_disp2", "local_minus_regmedian", "miss_recent4"] + ["imp_adjustment", "imp_confidence", "observed_share8", "gap_length"] + ["breadth2", "breadth_med2"]
NAN = float("nan")
SEQ_H = 8            # 은닉 크기
SEQ_EPOCHS = 80
SEQ_LR = 0.01
SEQ_WD = 0.01
SEQ_SEEDS = 3
SEQ_KINDS = ("lstm", "gru")


def _num(s):
    if s is None or s == "":
        return NAN
    try:
        return float(s)
    except ValueError:
        return NAN


def _log(x):
    return math.log10(x) if x == x and x > 0 else NAN


def _ffill(a):
    """앞으로 채운 값과, 각 위치의 마지막 관측 위치(없으면 -1)."""
    out = a.copy()
    idx = np.full(len(a), -1)
    last = -1
    for i in range(len(a)):
        if a[i] == a[i]:
            last = i
        else:
            out[i] = out[i - 1] if i > 0 else NAN
        idx[i] = last
    return out, idx


def _roll_median(a, i, win=26, min_obs=8):
    seg = a[max(0, i - win + 1): i + 1]
    seg = seg[seg == seg]
    return float(np.median(seg)) if len(seg) >= min_obs else NAN


class _Data:
    """rows 를 (시·도 × 주차) 배열로 바꾼다. 격자가 완전하므로 배열 위치 = 달력 주차."""

    def __init__(self, rows, national=None):
        self._feature_cache = {}
        self._seq_cache = {}
        self.weeks = sorted({r["date_week"] for r in rows})
        self.widx = {w: i for i, w in enumerate(self.weeks)}
        self.regions = sorted({r["region"] for r in rows})
        n, m = len(self.regions), len(self.weeks)
        ridx = {g: i for i, g in enumerate(self.regions)}
        self.L = np.full((n, m), NAN)       # log10 conc_mean
        self.L3 = np.full((n, m), NAN)      # log10 conc_3wk_avg
        self.TG = np.full((n, m), NAN)      # log10 target_conc_t2 (공개된 라벨만)
        self.BASE = np.full((n, m), NAN)    # conc_base_avg
        self.PR = np.full((n, m), NAN); self.TP = np.full((n, m), NAN); self.NS = np.full((n, m), NAN)
        self.CS = np.full(m, NAN); self.NC = np.full(m, NAN)
        for r in (national or []):
            j = self.widx.get(r["date_week"])
            if j is not None:
                v = _num(r.get("covid_cases_national")); self.CS[j] = math.log10(v + 1) if v == v else NAN
                self.NC[j] = _log(_num(r.get("conc_mean_national")))
        for r in rows:
            g, j = ridx[r["region"]], self.widx[r["date_week"]]
            self.L[g, j] = _log(_num(r["conc_mean"]))
            self.L3[g, j] = _log(_num(r["conc_3wk_avg"]))
            self.TG[g, j] = _log(_num(r["target_conc_t2"]))
            self.BASE[g, j] = _num(r["conc_base_avg"])
            self.PR[g, j] = _num(r.get("precip_mm")); self.TP[g, j] = _num(r.get("temp_avg"))
            self.NS[g, j] = _num(r.get("n_sites"))
        if SANITIZE:
            for g in range(n):
                for j in range(m):
                    if not np.isfinite(self.L[g, j]):
                        obs = self.L[g, max(0, j-2):j+1]
                        obs = obs[np.isfinite(obs)]
                        if SANITIZE == 'available' and (not np.isfinite(self.L3[g,j]) or not len(obs)):
                            continue
                        self.L3[g,j] = math.log10(float(np.mean(10.0**obs))) if len(obs) else NAN
        ff = [_ffill(self.L[g]) for g in range(n)]
        self.F = np.vstack([f[0] for f in ff])
        last_idx = np.vstack([f[1] for f in ff])
        self.age = np.where(last_idx < 0, 8.0, np.arange(m)[None, :] - last_idx).astype(float)
        self._comp()
        self.F_IMPUTED = self._impute_hybrid4()

    def _comp(self):
        """전국 상대 수준: 관측된 시·도의 (농도 - 자기 직전 26주 중앙값) 평균. 6개 시·도 미만이면 직전 값을 유지."""
        n, m = self.L.shape
        comp = np.full(m, NAN)
        for j in range(1, m):
            vals = []
            for g in range(n):
                if self.L[g, j] == self.L[g, j]:
                    med = _roll_median(self.L[g], j - 1)
                    if med == med:
                        vals.append(self.L[g, j] - med)
            if len(vals) >= 6:
                comp[j] = float(np.mean(vals))
        self.comp, _ = _ffill(comp)


    def _impute_hybrid4(self):
        # Causal interpolation on original observed concentrations only.
        # No future point, no cached KOWAS values, no imputed targets.
        raw = self.L
        out = self.F.copy()
        n,m = raw.shape
        n_valid = np.sum(np.isfinite(raw), axis=0)
        for g in range(n):
            seen = []
            for j in range(m):
                if np.isfinite(raw[g,j]):
                    seen.append(j)
                    continue
                if not seen or j-seen[-1] > 8:
                    continue
                prev = seen[-8:][::-1]
                weights = 0.5 ** np.arange(len(prev))
                value = float(np.dot(weights, raw[g,prev]) / weights.sum())
                k0 = seen[-1]
                if (n_valid[j] >= 12 and n_valid[k0] >= 12
                    and np.isfinite(self.NC[j]) and np.isfinite(self.NC[k0])):
                    value += float(self.NC[j] - self.NC[k0])
                out[g,j] = value
        return out

    def base(self, g, j):
        v = self.L3[g, j]
        return v if v == v else self.F[g, j]

    def feats(self, g, j, keys):
        key = (g, j, tuple(keys))
        if key not in self._feature_cache:
            self._feature_cache[key] = self._calculate_feats(g, j, keys)
        return self._feature_cache[key]

    def _calculate_feats(self, g, j, keys):
        Fx = self.F
        b = self.base(g, j)
        if b != b:
            return None
        last = Fx[g, j]
        med = _roll_median(self.L[g], j)
        d4 = (Fx[g, j] - Fx[g, j - 4]) if j >= 4 else NAN
        d1 = (Fx[g, j] - Fx[g, j - 1]) if j >= 1 else NAN
        c4 = (self.comp[j] - self.comp[j - 4]) if j >= 4 else NAN
        c1 = (self.comp[j] - self.comp[j - 1]) if j >= 1 else NAN
        z = lambda v: v if v == v else 0.0
        f = {
            "x_last": z(last - b),
            "x_med": z(med - b),
            "x_d4": z(d4),
            "age": min(self.age[g, j], 6.0),
            "miss": 1.0 if self.L[g, j] != self.L[g, j] else 0.0,
            "comp4": z(c4),
            "comp1": z(c1),
            "d1": z(d1),
        }
        def df(a, k):
            return (a[j] - a[j - k]) if j >= k else NAN
        pr = self.PR[g]; tp = self.TP[g]
        f["pr0"] = z(math.log1p(pr[j])) if pr[j] == pr[j] else 0.0
        f["pr2"] = z(float(np.nanmean(np.log1p(pr[max(0, j - 1): j + 1])))) if j >= 1 and np.any(pr[max(0, j - 1): j + 1] == pr[max(0, j - 1): j + 1]) else 0.0
        f["tp4"] = z(df(tp, 4)); f["tp0"] = z(tp[j])
        f["cs1"] = z(df(self.CS, 1)); f["cs4"] = z(df(self.CS, 4)); f["cs2"] = z(df(self.CS, 2))
        f["nc1"] = z(df(self.NC, 1)); f["nc4"] = z(df(self.NC, 4))
        Fg = Fx[g]
        dd = np.diff(Fg[max(0, j - 8): j + 1])
        f["vol"] = float(np.std(dd)) if len(dd) >= 4 and np.all(dd == dd) else 0.0
        bv = self.BASE[g, j]
        f["ald"] = z(Fg[j] - math.log10(bv)) if bv == bv and bv > 0 else 0.0
        wk = int(self.weeks[j][6:]) if len(self.weeks[j]) >= 7 else 1
        f["sin1"] = math.sin(2 * math.pi * wk / 52.0); f["cos1"] = math.cos(2 * math.pi * wk / 52.0)
        f["sin2"] = math.sin(4 * math.pi * wk / 52.0); f["cos2"] = math.cos(4 * math.pi * wk / 52.0)
        f["ns"] = z(self.NS[g, j])
        seg52 = Fg[max(0, j - 51): j + 1]
        f["l52"] = float(Fg[j] - np.nanmean(seg52)) if np.any(seg52 == seg52) else 0.0
        f["vxl"] = f["vol"] * f["x_last"]; f["vxm"] = f["vol"] * f["x_med"]
        # New concentration-only features: leave alert model unchanged.
        # Cross-sectional rates use only current and prior-week observations.
        if any(k in keys for k in ("region_disp2", "local_minus_regmedian", "miss_recent4")):
            rates=[]
            for h in range(self.L.shape[0]):
                if j>=2 and self.age[h,j] <= 1 and Fx[h,j]==Fx[h,j] and Fx[h,j-2]==Fx[h,j-2]:
                    rates.append(float(Fx[h,j]-Fx[h,j-2]))
            local_rate=float(Fx[g,j]-Fx[g,j-2]) if j>=2 and Fx[g,j]==Fx[g,j] and Fx[g,j-2]==Fx[g,j-2] else 0.0
            f["region_disp2"]=float(np.std(rates)) if len(rates)>1 else 0.0
            f["local_minus_regmedian"]=local_rate-float(np.median(rates)) if rates else 0.0
            window=self.L[g,max(0,j-3):j+1]
            f["miss_recent4"]=float(np.mean(np.isnan(window)))
        
        # Reliability of causal imputation; 0 when current concentration observed.
        is_imputed = float(self.L[g,j] != self.L[g,j] and np.isfinite(self.F_IMPUTED[g,j]))
        imp_shift = (self.F_IMPUTED[g,j]-self.F[g,j]) if is_imputed and np.isfinite(self.F[g,j]) else 0.0
        obs_seg = self.L[g,max(0,j-7):j+1]
        f["imp_adjustment"] = float(np.clip(imp_shift,-1.5,1.5))
        f["imp_confidence"] = float(is_imputed * np.exp(-self.age[g,j]/2.0))
        f["observed_share8"] = float(np.mean(np.isfinite(obs_seg)))
        f["gap_length"] = float(min(self.age[g,j], 8))
        # Concurrent broad-based regional growth from origin and earlier only.
        if j>=2:
            moving=[]
            for h in range(self.L.shape[0]):
                if np.isfinite(self.L[h,j]) and np.isfinite(self.L[h,j-2]):
                    moving.append(float(self.L[h,j]-self.L[h,j-2]))
            breadth=float(np.mean(np.array(moving)>math.log10(2.0))) if len(moving)>=8 else 0.0
        else:
            breadth=0.0
        f["breadth2"]=breadth
        f["breadth_local2"]=breadth * z(d4/2.0)
        # Ancillary diffusion features: only actual regional observations at or before j.
        def _cross_lag(lag, up=None):
            if j<lag: return 0.0
            valid=np.isfinite(self.L[:,j]) & np.isfinite(self.L[:,j-lag])
            if valid.sum()<8: return 0.0
            dd=self.L[valid,j]-self.L[valid,j-lag]
            if up=='rise': return float(np.mean(dd>math.log10(2.0)))
            if up=='fall': return float(np.mean(dd< -math.log10(2.0)))
            if up=='half_rise': return float(np.mean(dd>math.log10(2.0)/2.0))
            if up=='median': return float(np.clip(np.median(dd),-0.8,0.8))
            return 0.0
        f["breadth_fall2"] = _cross_lag(2,'fall')
        f["breadth_rise4"] = _cross_lag(4,'rise')
        f["breadth_med2"] = _cross_lag(2,'median')
        f["breadth_rise1"] = _cross_lag(1,'half_rise')
        if j>=3:
            prev_valid=np.isfinite(self.L[:,j-1])&np.isfinite(self.L[:,j-3])
            prev_b=float(np.mean((self.L[prev_valid,j-1]-self.L[prev_valid,j-3])>math.log10(2.0))) if prev_valid.sum()>=8 else 0.0
        else: prev_b=0.0
        f["breadth_onset"] = float(np.clip(breadth-prev_b,-1.0,1.0))
        if j>=2:
            valid=np.isfinite(self.L[:,j])&np.isfinite(self.L[:,j-2]);own=float(valid[g])
            numerator=float(np.sum((self.L[valid,j]-self.L[valid,j-2])>math.log10(2.0))) if valid.sum()>=9 else 0.0
            own_rise=float(valid[g] and (self.L[g,j]-self.L[g,j-2]>math.log10(2.0)))
            f["peer_breadth2"] = float((numerator-own_rise)/max(1.0,valid.sum()-own)) if valid.sum()>=9 else 0.0
        else: f["peer_breadth2"]=0.0
        return [f[k] for k in keys]


def _huber_ridge(X, y, delta=HUBER, lam=LAM, iters=25):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    pen = np.eye(Z.shape[1]) * lam
    pen[0, 0] = 0.0
    w = np.ones(len(y))
    beta = np.zeros(Z.shape[1])
    for _ in range(iters):
        W = Z.T * w
        beta = np.linalg.solve(W @ Z + pen, W @ y)
        r = np.abs(y - Z @ beta)
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-12))
    return beta, mu, sd


_LN10H = 0.5*math.log(10.0)

def _bounded_refit(X, y, fit, core, lam, iters=15):
    """sMAPE 모양 손실(|tanh(c·r)|, 오차가 커도 상한이 있음)에 맞춘 재추정. Huber 해에서 출발하는 IRLS."""
    beta, mu, sd = fit
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    pen = np.eye(Z.shape[1]) * lam; pen[0, 0] = 0.0
    wc = (1.0/np.cosh(_LN10H*core))**2/core
    for _ in range(iters):
        r = np.abs(y - Z @ beta); mm = np.maximum(r, core)
        w = ((1.0/np.cosh(_LN10H*mm))**2/mm)/wc
        W = Z.T * w
        beta = np.linalg.solve(W @ Z + pen, W @ y)
    return beta, mu, sd

def _alert_base(d, g, j):
    """conc_base_avg(원점): 공개값, 없으면 최근 4주 중 관측 2개 이상의 평균으로 자체 계산."""
    v = d.BASE[g, j]
    if v == v and v > 0:
        return v
    seg = d.L[g, max(0, j - 3): j + 1]
    seg = seg[seg == seg]
    return float(np.mean(10.0 ** seg)) if len(seg) >= 2 else NAN


sig = lambda x: 1 / (1 + np.exp(-x))
class Net:
    def __init__(s,kind,D,H,seed,T=8):
        r=np.random.RandomState(seed);s.kind=kind;s.H=H;s.T=T;P={}
        g=lambda *sh:r.randn(*sh)*0.3/np.sqrt(sh[0])
        if kind=='gru':
            P['W']=g(D,3*H);P['U']=g(H,3*H);P['b']=np.zeros(3*H)
        elif kind=='lstm':
            P['W']=g(D,4*H);P['U']=g(H,4*H);b=np.zeros(4*H);b[H:2*H]=1;P['b']=b
        else: # tiny transformer, d=H
            P['We']=g(D,H);P['pos']=r.randn(T,H)*0.1
            for k in 'qkvo':P['W'+k]=g(H,H)
            P['W1']=g(H,2*H);P['b1']=np.zeros(2*H);P['W2']=g(2*H,H);P['b2']=np.zeros(H)
        P['Wh']=g(H,1);P['bh']=np.zeros(1);s.P=P
    def fwd(s,X):
        P=s.P;B,T,D=X.shape;H=s.H;C=[]
        if s.kind=='gru':
            h=np.zeros((B,H))
            for t in range(T):
                x=X[:,t];a=x@P['W']+P['b'];g=h@P['U']
                z=sig(a[:,:H]+g[:,:H]);r=sig(a[:,H:2*H]+g[:,H:2*H]);n=np.tanh(a[:,2*H:]+r*g[:,2*H:])
                C.append((x,h,g,z,r,n));h=(1-z)*n+z*h
            return h@P['Wh']+P['bh'],(C,h)
        if s.kind=='lstm':
            h=np.zeros((B,H));c=np.zeros((B,H))
            for t in range(T):
                x=X[:,t];a=x@P['W']+h@P['U']+P['b']
                i=sig(a[:,:H]);f=sig(a[:,H:2*H]);g=np.tanh(a[:,2*H:3*H]);o=sig(a[:,3*H:])
                c2=f*c+i*g;tc=np.tanh(c2);C.append((x,h,c,i,f,g,o,tc));h=o*tc;c=c2
            return h@P['Wh']+P['bh'],(C,h)
        E=X@P['We']+P['pos'];Q=E@P['Wq'];K=E@P['Wk'];V=E@P['Wv']
        S=Q@K.transpose(0,2,1)/np.sqrt(H);S=S-S.max(2,keepdims=True);A=np.exp(S);A/=A.sum(2,keepdims=True)
        AV=A@V;E2=E+AV@P['Wo'];Z=E2@P['W1']+P['b1'];R=np.maximum(Z,0);E3=E2+R@P['W2']+P['b2'];m=E3.mean(1)
        return m@P['Wh']+P['bh'],(X,E,Q,K,V,A,AV,E2,Z,R,m)
    def grad(s,X,y,delta,wd):
        P=s.P;p,cache=s.fwd(X);p=p[:,0];B=len(y);e=p-y
        L=np.where(abs(e)<=delta,0.5*e*e,delta*(abs(e)-0.5*delta)).mean()
        dp=np.clip(e,-delta,delta)[:,None]/B;G={k:np.zeros_like(v) for k,v in P.items()};H=s.H
        if s.kind in('gru','lstm'):
            C,h=cache;G['Wh']=h.T@dp;G['bh']=dp.sum(0);dh=dp@P['Wh'].T
            if s.kind=='gru':
                for (x,hp,g,z,r,n) in reversed(C):
                    dn=dh*(1-z);dz=dh*(hp-n);dhp=dh*z;dnp=dn*(1-n*n)
                    dr=dnp*g[:,2*H:];dgn=dnp*r;dzp=dz*z*(1-z);drp=dr*r*(1-r)
                    da=np.hstack([dzp,drp,dnp]);dg=np.hstack([dzp,drp,dgn])
                    G['W']+=x.T@da;G['b']+=da.sum(0);G['U']+=hp.T@dg;dh=dhp+dg@P['U'].T
            else:
                dc=np.zeros_like(dh)
                for (x,hp,cp,i,f,g,o,tc) in reversed(C):
                    do=dh*tc;dc=dc+dh*o*(1-tc*tc);di=dc*g;dg=dc*i;df=dc*cp;dcn=dc*f
                    da=np.hstack([di*i*(1-i),df*f*(1-f),dg*(1-g*g),do*o*(1-o)])
                    G['W']+=x.T@da;G['b']+=da.sum(0);G['U']+=hp.T@da;dh=da@P['U'].T;dc=dcn
        else:
            X,E,Q,K,V,A,AV,E2,Z,R,m=cache;T=X.shape[1]
            G['Wh']=m.T@dp;G['bh']=dp.sum(0);dm=dp@P['Wh'].T;dE3=np.repeat(dm[:,None,:]/T,T,1)
            dE2=dE3.copy();G['W2']=np.einsum('btk,btd->kd',R,dE3);G['b2']=dE3.sum((0,1))
            dR=dE3@P['W2'].T;dZ=dR*(Z>0);G['W1']=np.einsum('btd,btk->dk',E2,dZ);G['b1']=dZ.sum((0,1));dE2+=dZ@P['W1'].T
            dE=dE2.copy();G['Wo']=np.einsum('btd,bte->de',AV,dE2);dAV=dE2@P['Wo'].T
            dA=dAV@V.transpose(0,2,1);dV=A.transpose(0,2,1)@dAV
            dS=A*(dA-(dA*A).sum(2,keepdims=True))/np.sqrt(H)
            dQ=dS@K;dK=dS.transpose(0,2,1)@Q
            for nm,dd in(('Wq',dQ),('Wk',dK),('Wv',dV)):
                G[nm]=np.einsum('btd,bte->de',E,dd);dE+=dd@P[nm].T
            G['We']=np.einsum('btd,bte->de',X,dE);G['pos']=dE.sum(0)
        for k in P:
            if k[0]=='W' or k=='pos':G[k]+=wd*P[k]
        return L,G
    def fit(s,X,y,delta,ep,lr,wd):
        m={k:np.zeros_like(v) for k,v in s.P.items()};v={k:np.zeros_like(x) for k,x in s.P.items()}
        for t in range(1,ep+1):
            _,G=s.grad(X,y,delta,wd)
            for k in s.P:
                m[k]=0.9*m[k]+0.1*G[k];v[k]=0.999*v[k]+0.001*G[k]**2
                s.P[k]-=lr*(m[k]/(1-0.9**t))/(np.sqrt(v[k]/(1-0.999**t))+1e-8)
        return s
    def predict(s,X):return s.fwd(X)[0][:,0]


def _seq(d, g, j, T=8):
    key=(g,j,T)
    if key not in d._seq_cache:
        d._seq_cache[key]=_calculate_seq(d,g,j,T)
    return d._seq_cache[key]

def _calculate_seq(d, g, j, T=8):
    b = d.base(g, j)
    out = np.zeros((T, 5))
    for k in range(T):
        s_ = j - T + 1 + k
        if s_ < 0:
            continue
        f = d.F[g, s_]
        out[k, 0] = (f - b) if f == f else 0.0
        out[k, 1] = 1.0 if d.L[g, s_] != d.L[g, s_] else 0.0
        p = d.PR[g, s_]
        out[k, 2] = math.log1p(p) if p == p else 0.0
        c = (d.CS[s_] - d.CS[s_ - 1]) if s_ >= 1 else NAN
        out[k, 3] = c if c == c else 0.0
        cj = d.comp[s_] - d.comp[j]
        out[k, 4] = cj if cj == cj else 0.0
    return out




# Universal causal forecast: no year-, week-, or evaluation-fold-specific branches.
# Two Huber feature variants, trained on rolling windows 78/104, and original RNN/alert.
CONC_DELTA = 0.24
CONC_LAM = 1.5
CONC_WINDOWS = (78, 104, 26, 39)
# 10개 평가 창 전체에서 고르게 잘 맞도록 고른 공통 조정값(창·주차·연도별 분기 없음).
CONC_MIX = (((1,0),15),(("lA104",0),45),(("m1104",0),13),(("m178",1),20),(("b26",2),2),(("b39",2),1))
CONC_POST = dict(sw=0.0, sc=1.00, pos=1.45, neg=.80, br=.06, bc=.30, bk=6.0, dg=.90, ds=.325, sa=.05)
ALERT_POST = dict(sw=0.0, sc=.90, pos=1.10, neg=.80, br=.06, bc=.40, bk=2.0, dg=.35, ds=.15, sa=.02)
ALERT_K_DIS, ALERT_K_SEQ, ALERT_K_BR = 0.10, 0.60, -0.05
# 농도 예측 변화 폭을 현재 상태에 따라 키우거나 줄이는 공통 조정값(5개): 1 + AMP_C0 + Σ 계수 × 변수값.
# 전국 상대 수준 4주 변화·전국 신고수 변화·지역 변동성이 크면 키우고, 전국 1주 변화·급증 시·도 비율이 크면 줄인다.
AMP_C0 = -0.051714
AMP_TERMS = (("breadth2", -0.293328), ("comp4", 0.156654), ("vol", 0.329542), ("comp1", -0.936700), ("cs1", 0.775799))
ALERT_LOG_THR = math.log10(0.97) - ALERT_K_DIS*0.0305 - ALERT_K_SEQ*0.0471

def _network_cross_features(d, j):
    """Only contemporaneous+historical *observed* regional concentrations."""
    if j < 2:
        return 0.0, 0.0, 0.0, 0.0
    available = np.isfinite(d.L[:, j]) & np.isfinite(d.L[:, j-2])
    if int(available.sum()) < 8:
        return 0.0, 0.0, 0.0, 0.0
    changes = d.L[available, j] - d.L[available, j-2]
    typical = float(np.median(changes))
    agreement = float(np.mean(np.sign(changes)==np.sign(typical)))
    lower_quartile = float(np.quantile(changes, 0.25))
    dispersion = float(np.std(changes))
    up_fraction = float(np.mean(changes > math.log10(2.0)))
    return agreement, lower_quartile, dispersion, up_fraction

class Model:
    def __init__(self, use_sequence=True):
        self.use_sequence = use_sequence

    def fit(self, rows, national, aux=None):
        self._previous_data = None

    def predict(self, rows, origin_week, national, aux=None):
        d = _Data(rows, national)
        return self.predict_data(d, origin_week)

    def predict_data(self, d, origin_week):
        previous = getattr(self, "_previous_data", None)
        if previous is not None:
            width = len(previous.weeks)
            consistent = (d.regions == previous.regions and d.weeks[:width] == previous.weeks
                and all(np.array_equal(getattr(d,key)[...,:width], getattr(previous,key), equal_nan=True)
                        for key in ("L","L3","PR","TP","NS","CS","NC")))
            if consistent:
                d._feature_cache = previous._feature_cache
                d._seq_cache = previous._seq_cache
        self._previous_data = d
        n, m = d.L.shape
        o = d.widx.get(origin_week, m - 1)

        # Original 104-week reference training set. Do not modify alerts or sequence training.
        XA, y, IDX = [], [], []
        for j in range(max(0, o - 2 - WINDOW + 1), o - 1):
            for g in range(n):
                t = d.TG[g, j]
                if t != t:
                    continue
                f = d.feats(g, j, CONC_FEATS)
                if f is None:
                    continue
                XA.append(d.feats(g, j, ALERT_FEATS))
                IDX.append((g,j))
                y.append(t - d.base(g,j))

        coefA = None
        seqn = None
        if len(y) >= 50 and self.use_sequence:
            coefA = _huber_ridge(np.array(XA,float), np.array(y,float))
            XS = np.array([_seq(d,g,j) for (g,j) in IDX])
            smu = XS.reshape(-1,5).mean(0)
            ssd = XS.reshape(-1,5).std(0) + 1e-6
            XSn = (XS - smu)/ssd
            ys = np.array(y,float)
            seqn = (smu, ssd, [Net(kd, 5, SEQ_H, sd0).fit(XSn, ys, HUBER, SEQ_EPOCHS, SEQ_LR, SEQ_WD) for kd in SEQ_KINDS for sd0 in range(SEQ_SEEDS)])

        # Each iteration trains from past-labelled examples only (j+2 <= origin).
        # Two suffixes carry distinct across-region signal without using any later observations.
        xcache = {}
        for j in range(max(0, o - 2 - max(CONC_WINDOWS)+1), o+1):
            if j >= 0:
                xcache[j] = _network_cross_features(d,j)
        h = {}
        for wi, win in enumerate(CONC_WINDOWS):
            train = [[], [], []]
            yy = []; ysm = []; y1 = []
            for j in range(max(0, o - 2 - win + 1), o - 1):
                extra = xcache[j]
                for g in range(n):
                    t = d.TG[g,j]
                    if t != t:
                        continue
                    ff = d.feats(g,j,CONC_FEATS)
                    if ff is None:
                        continue
                    xx = np.asarray(ff,float)
                    train[0].append(np.concatenate([xx,extra[:2]]))
                    train[1].append(np.concatenate([xx,(extra[0],extra[2])]))
                    train[2].append(np.concatenate([xx,(extra[0],extra[3])]))
                    bj = d.base(g,j)
                    yy.append(t - bj)
                    # 경보용 보조 모델의 학습 정답: t+2주 값에 t+1·t+3주 관측(모두 원점 이하일 때만)을 섞어 잡음을 줄인다.
                    a_ = d.L[g,j+1]; c_ = d.L[g,j+3] if j+3 <= o else NAN
                    ysm.append((0.5*t + 0.25*(a_+c_) if (a_==a_ and c_==c_) else t) - bj)
                    # 보조 정답: t+1주 관측(원점 이하) 변화량. 없으면 t+2주 값.
                    y1.append((a_ if a_==a_ else t) - bj)
            if len(yy) < 50:
                continue
            yy = np.asarray(yy,float); ysm = np.asarray(ysm,float); y1 = np.asarray(y1,float)
            X0,X1,X2 = (np.asarray(train[v],float) for v in (0,1,2))
            fits = []
            if win in (78,104):
                for variant,Xv in ((0,X0),(1,X1)):
                    fits.append((wi,variant,_huber_ridge(Xv,yy,delta=CONC_DELTA,lam=CONC_LAM)))
            if win == 104:
                fits.append(("lA104",0,_huber_ridge(X0,yy,delta=CONC_DELTA,lam=0.5)))     # 약한 릿지
                fits.append(("m1104",0,_huber_ridge(X0,y1,delta=CONC_DELTA,lam=CONC_LAM)))  # t+1주 정답
                fits.append(("s104",2,_huber_ridge(X2,ysm,delta=CONC_DELTA,lam=CONC_LAM)))  # 경보용: 잡음 줄인 정답
            if win == 78:
                fits.append(("m178",1,_huber_ridge(X1,y1,delta=CONC_DELTA,lam=CONC_LAM)))
            if win in (26,39):                   # 짧은 학습기간 + sMAPE 모양 손실
                fits.append((f"b{win}",2,_bounded_refit(X2,yy,_huber_ridge(X2,yy,delta=CONC_DELTA,lam=CONC_LAM),CONC_DELTA,CONC_LAM)))
            ex=xcache[o]
            for key,variant,(beta,mu,sd) in fits:
                extra=ex[:2] if variant==0 else ((ex[0],ex[2]) if variant==1 else (ex[0],ex[3]))
                # use per-region directly to preserve fallback for regions without a base.
                for g in range(n):
                    bb=d.base(g,o)
                    if bb!=bb: continue
                    xx=np.asarray(d.feats(g,o,CONC_FEATS),float)
                    aug=np.concatenate([xx,extra])
                    h[(key,variant,g)]=float(beta[0]+((aug-mu)/sd)@beta[1:])
        known=[d.base(g,o) for g in range(n) if d.base(g,o)==d.base(g,o)]
        fallback=float(np.mean(known)) if known else 3.0
        out={}
        self.diagnostics={}
        for g,region in enumerate(d.regions):
            b=d.base(g,o)
            base_avg=_alert_base(d,g,o)
            if b!=b:
                conc=10.0**fallback
                alert=int(base_avg==base_avg and conc>=ALERT_C*base_avg)
                out[region]=(float(conc),alert)
                continue
            delta_a=0.0
            if coefA is not None:
                beta,mu,sd=coefA
                xx=np.asarray(d.feats(g,o,ALERT_FEATS),float)
                delta_a=float(beta[0]+((xx-mu)/sd)@beta[1:])
            # Huber 78/104 mixture uses *only current-origin observation completeness*.
            share8=float(np.mean(np.isfinite(d.L[g,max(0,o-7):o+1])))
            weight=float(np.clip(.5+.4*(share8-.8),.1,.9))
            hh=[]
            for variant in (0,1):
                v78=h.get((0,variant,g),0.0)
                v104=h.get((1,variant,g),0.0)
                hh.append(weight*v78+(1-weight)*v104)
            hpred=.5*(hh[0]+hh[1])
            seq_delta=0.0; seq_sd=0.0
            if seqn is not None:
                xs=((_seq(d,g,o)-seqn[0])/seqn[1])[None]
                seq_parts=[float(nt.predict(xs)[0]) for nt in seqn[2]]
                seq_delta=float(np.mean(seq_parts)); seq_sd=float(np.std(seq_parts))
            bshare=xcache[o][3]
            # Universal disagreement-aware forecast attenuation, not date/fold-switching.
            # Two independently fitted past-only Huber windows must disagree to activate it.
            disagreement=0.5*((h.get((1,0,g),0.0)-h.get((0,0,g),0.0))
                              +(h.get((1,1,g),0.0)-h.get((0,1,g),0.0)))
            # Causal seasonal analogue: last year's +2-week growth is already observed (origin o-52 -> o-50).
            seasonal_delta=0.0
            if o >= 52 and np.isfinite(d.F[g,o-50]) and np.isfinite(d.F[g,o-52]):
                seasonal_delta=float(np.clip(d.F[g,o-50]-d.F[g,o-52],-1.0,1.0))
            def _finish(hp, P):
                pre=P["sc"]*((1-P["sw"])*hp+P["sw"]*seq_delta)
                adj=pre*(P["pos"] if pre>=0 else P["neg"])
                adj+=P["br"]*math.tanh((bshare-P["bc"])*P["bk"])
                adj*=(1.0-P["dg"]*math.tanh(abs(disagreement)/P["ds"]))
                adj=(1-P["sa"])*adj+P["sa"]*.90*seasonal_delta
                return float(np.clip(b+adj,-25.,30.))
            # 농도(pred_conc): 여러 Huber 모델 가중 평균 + LSTM·GRU, 공통 보정값(10개 창 전체 교차검증으로 선택)
            hc=sum(wgt*(hpred if key=="team" else h.get((*key,g),hpred)) for key,wgt in CONC_MIX)/sum(w_ for _,w_ in CONC_MIX)
            lc=_finish(hc, CONC_POST)
            fo=dict(zip(CONC_FEATS, d.feats(g,o,CONC_FEATS)))
            amp=1.0+AMP_C0+sum(c*fo[f] for f,c in AMP_TERMS)
            conc=10.0**float(np.clip(b+(lc-b)*amp,-25.,30.))
            # 경보(pred_alert): 경보 전용 혼합(기존 혼합 + 잡음 줄인 정답 모델)과 전용 보정값, 순환망 제외.
            ha=0.5*(hpred+h.get(("s104",2,g),hpred))
            la=_finish(ha, ALERT_POST)
            # 모델끼리 의견이 엇갈리거나 순환망 예측이 흩어지면 기준을 높이고, 전국 확산이 넓으면 낮춘다.
            thr=ALERT_LOG_THR+ALERT_K_DIS*abs(disagreement)+ALERT_K_SEQ*seq_sd+ALERT_K_BR*(bshare-.4)
            alert=int(base_avg==base_avg and base_avg>0 and la-math.log10(base_avg)>=thr)
            self.diagnostics[region] = dict(
                b=b, base_avg=base_avg, hpred=hpred, hc=hc, ha=ha, lc=lc, la=la,
                seq_delta=seq_delta, seq_sd=seq_sd, disagreement=disagreement,
                seasonal_delta=seasonal_delta, bshare=bshare, amp=amp, threshold=thr,
                features=fo,
                components={str((k,v)):h.get((k,v,g),0.0) for k,v in
                            [(0,0),(0,1),(1,0),(1,1),('lA104',0),('m1104',0),
                             ('m178',1),('s104',2),('b26',2),('b39',2)]})
            out[region]=(float(max(conc,0.)),alert)
        return out

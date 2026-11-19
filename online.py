# [개요] 온라인 오차 보정. 정답이 이미 관측된 과거 예측만 써서 최근 편향을 보정한다.
# 입력: 최근 학습기 예측과 과거 예측 오차. 출력: 보정된 농도 예측.
"""Learn only from forecasts whose outcomes have already been observed."""
import numpy as np
from core import Model as Core, CONC_FEATS

def fit_huber(X,y,lam=500.,delta=.2):
    mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1.
    Z=np.column_stack([np.ones(len(y)),(X-mu)/sd])
    pen=np.eye(Z.shape[1])*lam;pen[0,0]=0
    w=np.ones(len(y))
    for _ in range(25):
        W=Z.T*w;beta=np.linalg.solve(W@Z+pen,W@y)
        residual=np.abs(y-Z@beta);w=np.minimum(1.,delta/np.maximum(residual,1e-12))
    return beta,mu,sd

class OnlineCorrector:
    def __init__(self):
        self.model=Core(use_sequence=False);self.forecasts={};self.previous=None
    def correct(self,pred,d,origin_week):
        o=d.widx[origin_week];n=len(d.regions)
        previous=self.previous
        if previous is not None:
            width=len(previous.weeks)
            consistent=(d.regions==previous.regions and d.weeks[:width]==previous.weeks
                and all(np.array_equal(getattr(d,k)[...,:width],getattr(previous,k),equal_nan=True)
                        for k in ("L","L3","PR","TP","NS","CS","NC"))
                and np.array_equal(d.TG[...,:max(0,width-2)],previous.TG[...,:max(0,width-2)],equal_nan=True))
            if not consistent:
                self.forecasts={};self.model=Core(use_sequence=False)
        self.previous=d
        # 26 labelled origin weeks plus the 4-week error feature's required history.
        for j in range(max(11,o-32),o-1):
            if j not in self.forecasts:
                raw=self.model.predict_data(d,d.weeks[j])
                self.forecasts[j]=np.array([np.log10(max(raw[g][0],1e-25)) for g in d.regions])
        error=np.full((o+1,n),np.nan)
        for j,p in self.forecasts.items():
            if j+2<=o:error[j+2]=d.L[:,j+2]-p
        def feature(g,j):
            vals=error[max(0,j-3):j+1,g];vals=vals[np.isfinite(vals)]
            nation=error[j];nation=nation[np.isfinite(nation)]
            return [error[j,g] if np.isfinite(error[j,g]) else 0.,
                    error[j-1,g] if j and np.isfinite(error[j-1,g]) else 0.,
                    float(vals.mean()) if len(vals) else 0.,
                    float(nation.mean()) if len(nation) else 0.,
                    d.feats(g,j,CONC_FEATS)[CONC_FEATS.index('vol')]]
        X=[];y=[]
        for j in range(max(0,o-27),o-1):
            if j not in self.forecasts:continue
            for g in range(n):
                if np.isfinite(d.L[g,j+2]) and np.isfinite(self.forecasts[j][g]) and d.base(g,j)==d.base(g,j):
                    X.append(feature(g,j));y.append(d.L[g,j+2]-self.forecasts[j][g])
        if len(y)<50:return pred
        beta,mu,sd=fit_huber(np.array(X),np.array(y))
        out={}
        for g,reg in enumerate(d.regions):
            if d.base(g,o)!=d.base(g,o):out[reg]=pred[reg];continue
            correction=float(np.clip(beta[0]+((np.array(feature(g,o))-mu)/sd)@beta[1:],-.4,.4))
            lp=np.log10(max(pred[reg][0],1e-25))+.25*correction
            out[reg]=(float(10**np.clip(lp,-25,30)),pred[reg][1])
        return out

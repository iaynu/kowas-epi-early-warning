# [개요] 경보 임계값 학습. 과거(표본 밖) 경보 여유값으로 기대 Macro-F1이 큰 임계값을 고른다.
# 입력: 과거 경보 여유값과 실제 경보. 출력: 경보(0/1)를 정하는 임계값.
"""Smoothed expected Macro-F1 calibration on past out-of-sample margins."""
import math
import numpy as np
from core import Model as Core
from online import OnlineCorrector

def consistent(a,b):
    if a is None:return True
    width=len(a.weeks)
    return (a.regions==b.regions and b.weeks[:width]==a.weeks
        and all(np.array_equal(getattr(b,k)[...,:width],getattr(a,k),equal_nan=True)
                for k in ('L','L3','PR','TP','NS','CS','NC'))
        and np.array_equal(b.TG[...,:max(0,width-2)],a.TG[...,:max(0,width-2)],equal_nan=True))

def margin(diag,regions):
    values=[]
    for reg in regions:
        z=diag.get(reg)
        values.append(z['la']-math.log10(z['base_avg'])-z['threshold']
                      if z is not None and z['base_avg']>0 and math.isfinite(z['base_avg']) else np.nan)
    return np.array(values)

def macro(y,p):
    out=0.
    for c in (0,1):
        tp=np.sum((y==c)&(p==c));den=2*tp+np.sum((y!=c)&(p==c))+np.sum((y==c)&(p!=c))
        out+=(2*tp/den if den else 1.)/2
    return out

class CalibratedCorrector(OnlineCorrector):
    def __init__(self):
        super().__init__()
        self.alert_model=Core();self.margins={};self.audit={}

    def correct(self,pred,d,origin_week,rows,diagnostics):
        if not consistent(self.previous,d):
            self.alert_model=Core();self.margins={}
        pred=super().correct(pred,d,origin_week)
        o=d.widx[origin_week];regions=d.regions
        self.margins[o]=margin(diagnostics,regions)
        # Each historical forecast is rebuilt with training labels no later than its own origin.
        for j in range(max(11,o-14),o-1):
            if j not in self.margins:
                self.alert_model.predict_data(d,d.weeks[j])
                self.margins[j]=margin(self.alert_model.diagnostics,regions)
        labels={(r['date_week'],r['region']):int(float(r['target_alert_t2'])) for r in rows
                if r['target_alert_t2'] and r['target_week_t2']<=origin_week}
        train_margin=[];train_y=[];target_weeks=[]
        for j in range(max(0,o-14),o-1):
            if j not in self.margins:continue
            for g,reg in enumerate(regions):
                y=labels.get((d.weeks[j],reg));m=self.margins[j][g]
                if y is not None and np.isfinite(m):
                    train_margin.append(m);train_y.append(y);target_weeks.append(d.weeks[j+2])
        threshold=0.
        if len(train_y)>=30:
            m=np.array(train_margin);y=np.array(train_y);grid=np.linspace(-.16,.16,65)
            prob=1/(1+np.exp(-np.clip((m[:,None]-grid[None,:])/.01,-30,30)))
            yy=y[:,None]
            tp=(prob*yy).sum(0);fp=(prob*(1-yy)).sum(0)
            fn=((1-prob)*yy).sum(0);tn=((1-prob)*(1-yy)).sum(0)
            expected=.5*((2*tp+1)/(2*tp+fp+fn+2)+(2*tn+1)/(2*tn+fp+fn+2))
            objective=expected-np.abs(grid)
            indices=np.flatnonzero(objective>=objective.max()-1e-12)
            threshold=grid[indices[np.argmin(abs(grid[indices]))]]
        self.audit=dict(origin=origin_week,n_calibration=len(train_y),threshold=float(threshold),
                        max_label_week=max(target_weeks) if target_weeks else None)
        out={}
        for g,reg in enumerate(regions):
            m=self.margins[o][g]
            out[reg]=(pred[reg][0],int(m>=threshold) if np.isfinite(m) else pred[reg][1])
        return out

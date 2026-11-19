# [개요] 지역 상태, 추세, 처리장 편향을 분리하는 인과적 상태 공간 예측. 최종 예측에 약하게만 결합한다.
# 입력: 보정된 예측과 처리장 구성. 출력: 상태 공간 예측과 결합한 농도.
"""A weakly weighted causal state-space forecast with latent plant offsets."""
import numpy as np

class StateSpace:
    def correct(self,pred,rows,d,origin,plant_bias,core_pred):
        o=d.widx[origin];raw,_=plant_bias.memberships(rows,d,origin);C=raw.copy();L=d.L.T;comp=d.comp
        corelp=np.array([np.log10(max(core_pred[g][0],1e-25)) for g in d.regions])
        ok=np.isfinite(corelp)&np.isfinite(L[o])
        if ok.sum()<6 or not np.isfinite(comp[o]):return pred
        shared=float(comp[o]+np.median(corelp[ok]-L[o,ok]))
        for j in range(1,o+1):
            for g in range(len(d.regions)):
                if not C[j,g].any():C[j,g]=C[j-1,g]
        out={}
        for g,reg in enumerate(d.regions):
            ids=np.flatnonzero(raw[:o+1,g].sum(0)>0);size=2+len(ids)
            state=np.zeros(size);P=np.eye(size)*.03;P[:2,:2]=np.eye(2)
            A=np.eye(size);A[0,1]=.7;A[1,1]=.7;Q=np.diag([.01,.001]+[1e-5]*len(ids));initialized=False
            for j in range(o+1):
                state=A@state;P=A@P@A.T+Q
                if np.isfinite(L[j,g]) and np.isfinite(comp[j]) and raw[j,g].any():
                    y=L[j,g]-comp[j]
                    if not initialized:state[0]=y;initialized=True
                    h=np.r_[1.,0.,raw[j,g,ids]];error=y-h@state
                    variance=.1*(1+max(0,abs(error)-.4)/.2)
                    v=P@h;gain=v/(h@v+variance);state=state+gain*error;P=P-np.outer(gain,v)
            if initialized:
                future=A@A@state;membership=C[max(0,o-25):o+1,g][:,ids].mean(0)
                alternative=shared+future[0]+membership@future[2:]
                sites=d.NS[g,o] if np.isfinite(d.NS[g,o]) else 0.
                weight=.1*float(np.clip(4/max(sites,1),.25,2))
                lp=(1-weight)*np.log10(max(pred[reg][0],1e-25))+weight*alternative
                out[reg]=(float(10**np.clip(lp,-25,30)),pred[reg][1])
            else:out[reg]=pred[reg]
        return out

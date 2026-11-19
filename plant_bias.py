# [개요] 처리장 구성 보정. n_sites, pop_sampled로 채취 처리장 구성을 추론하고, 구성이 바뀐 과거 주차의 농도 변화로 편향을 추정해 뺀다.
# 입력: 처리장 정보(plant_meta.csv의 ID, 시.도, 처리인구)와 주간 자료. 출력: 구성 변화 편향이 보정된 농도.
"""Infer observed plant membership; estimate composition bias from past changes only."""
import os,csv,itertools
from pathlib import Path
import numpy as np

class PlantBias:
    def __init__(self):
        with (Path(os.environ['KOWAS_STATIC_DIR'])/'plant_meta.csv').open(encoding='utf-8',newline='') as f:
            # No registry/last-sample fields are used.
            self.meta=[{k:r[k] for k in ('plant_id','region','first_sample_week','treatment_population')} for r in csv.DictReader(f)]
        self.combinations={};self.membership_cache={};self.audit={}

    def memberships(self,rows,d,origin):
        active=[p for p in self.meta if p['first_sample_week']<=origin and p['treatment_population']]
        ids=[p['plant_id'] for p in active];lookup={p:i for i,p in enumerate(ids)}
        raw=np.zeros((len(d.weeks),len(d.regions),len(ids)))
        ri={r:i for i,r in enumerate(d.regions)}
        for row in rows:
            wk=row['date_week'];g=ri[row['region']];j=d.widx[wk]
            n=int(row['n_sites'] or 0);pop=float(row['pop_sampled'] or 0)
            plants=[p for p in active if p['region']==row['region'] and p['first_sample_week']<=wk]
            if n<1 or pop<=0 or len(plants)<n:continue
            pids=tuple(p['plant_id'] for p in plants);pops=tuple(float(p['treatment_population']) for p in plants)
            key=(pids,pops,n);fullkey=(key,pop)
            if fullkey not in self.membership_cache:
                if key not in self.combinations:
                    combos=list(itertools.combinations(range(len(plants)),n))
                    sums=np.array([sum(pops[h] for h in c) for c in combos])
                    self.combinations[key]=(combos,sums)
                combos,sums=self.combinations[key];error=abs(sums-pop)/max(pop,1)
                nearest=np.flatnonzero(error<=error.min()+1e-12);v=np.zeros(len(plants))
                if error.min()<=.01:
                    for k in nearest:
                        for h in combos[k]:v[h]+=1/(n*len(nearest))
                self.membership_cache[fullkey]=v
            raw[j,g,[lookup[p] for p in pids]]=self.membership_cache[fullkey]
        return raw,ids

    def correct(self,pred,rows,d,origin):
        raw,ids=self.memberships(rows,d,origin);C=raw.copy();o=d.widx[origin];L=d.L.T;comp=d.comp
        for j in range(1,len(d.weeks)):
            for g in range(len(d.regions)):
                if not C[j,g].any():C[j,g]=C[j-1,g]
        out={};detail=[]
        for g,reg in enumerate(d.regions):
            cols=np.flatnonzero(raw[:o+1,g].sum(0)>0);delta=0.;alert_delta=0.;ntrain=0
            if len(cols)>=2:
                xx=[];yy=[]
                for j in range(max(1,o-103),o+1):
                    for lag in (1,2,4):
                        k=j-lag
                        if k<0 or not (np.isfinite(L[j,g]) and np.isfinite(L[k,g]) and np.isfinite(comp[j]) and np.isfinite(comp[k])):continue
                        if not raw[j,g].any() or not raw[k,g].any():continue
                        diff=raw[j,g,cols]-raw[k,g,cols]
                        if np.abs(diff).sum()<1e-6:continue
                        xx.append(diff);yy.append(L[j,g]-L[k,g]-(comp[j]-comp[k]))
                ntrain=len(yy)
                if ntrain>=15:
                    xx=np.array(xx);yy=np.array(yy);pen=np.eye(len(cols))*3.;wt=np.ones(ntrain)
                    for _ in range(15):
                        beta=np.linalg.solve((xx.T*wt)@xx+pen,(xx.T*wt)@yy)
                        wt=np.minimum(1,.2/np.maximum(abs(yy-xx@beta),1e-8))
                    ix=np.arange(max(0,o-2),o+1);ok=np.isfinite(L[ix,g])&raw[ix,g].any(1);ix=ix[ok]
                    if len(ix):
                        weights=10**(L[ix,g]-np.max(L[ix,g]));anchor=np.average(raw[ix,g][:,cols],axis=0,weights=weights)
                    else:anchor=C[o,g,cols]
                    history=C[max(0,o-12):o+1,g][:,cols]
                    frequency=np.array([np.sum(np.all(np.isclose(history,row[None,:]),axis=1)) for row in history])
                    modal=history[int(np.argmax(frequency))]
                    future=.5*(modal+history.mean(0))
                    delta=float(np.clip((future-anchor)@beta,-.5,.5))
                    alert_future=C[max(0,o-12):o+1,g][:,cols].mean(0)
                    alert_delta=float(np.clip((alert_future-anchor)@beta,-.5,.5))
            out[reg]=(float(10**np.clip(np.log10(max(pred[reg][0],1e-25))+delta,-25,30)),pred[reg][1])
            detail.append(dict(region=reg,n_training_pairs=ntrain,correction=delta,alert_correction=alert_delta))
        self.audit=dict(origin=origin,allowed_plants=len(ids),max_observation_week=d.weeks[o],details=detail)
        return out

"""Predeclared descriptive association, separate from advance warning skill."""
from pathlib import Path
import numpy as np
import pandas as pd
from research_calibration_gate import BASE,ART,OUT as PARENT,SEEDS,FOLDS,read,sha,js
from state_gate_core import state_features
from residual_shape_core import BINS
OUT=ART/'state_gate_v1'
PLAN=BASE/'研究笔记/第51轮_相似风况与校准失效识别_2026-11-02.md'
def run():
 assert not OUT.exists(),'preserve previous attempt';OUT.mkdir()
 inv=read(PARENT/'inventory.json');raw=np.load(ART/'data.npz')['raw'];rows=[]
 for fold in FOLDS:
  d=np.load(ART/'seasonal_probability_v1'/fold/'data.npz');ids=np.load(PARENT/fold/'population.npz')['ids']
  for seed in SEEDS:
   q={c:np.load(next(v['q_source'] for v in inv if (v['fold'],v['seed'],v['case'])==(fold,seed,c)))['q'] for c in ['R','W28']}
   x=state_features(raw,ids,q['R'],q['W28'],float(d['center']),float(d['scale']))
   sw=np.load(PARENT/fold/f'hourly_scores_W28_{seed}.npz')['crps'];sm=np.load(PARENT/fold/f'hourly_scores_M50_{seed}.npz')['crps']
   for j,(a,b) in enumerate(BINS):
    count=np.isfinite(sw[:,a:b]).sum((1,2));delta=np.divide(np.nansum(sw[:,a:b]-sm[:,a:b],(1,2)),count,out=np.full(len(ids),np.nan),where=count>0)
    for i,issue in enumerate(ids):rows.append(dict(fold=fold,seed=seed,issue=int(issue),lead_start=a+1,lead_end=b,observed=int(count[i]),harm=delta[i],**{f'f{k}':x[i,j,k] for k in range(6)}))
 df=pd.DataFrame(rows);df.to_csv(OUT/'origin_diagnosis.csv',index=False,encoding='utf-8-sig')
 # Unit for descriptive summaries is an origin/lead-group, averaging seeds first.
 avg=df.groupby(['fold','issue','lead_start','lead_end'],as_index=False).mean(numeric_only=True);out=[]
 for scope in ['ALL',*FOLDS]:
  z=avg if scope=='ALL' else avg[avg.fold==scope]
  for name,mask in [('high_wind',z.f0>=1),('rising_wind',z.f1>=.5),('falling_wind',z.f1<=-.5),('downward_calibration',z.f4>=.5),('narrowing',z.f5<=np.log(.75))]:
   for state in [False,True]:
    part=z[mask==state];known=part[np.isfinite(part.harm)]
    out.append(dict(scope=scope,signal=name,active=state,n=len(part),observed=len(known),mean_harm=known.harm.mean(),harm_fraction=(known.harm>0).mean()))
 pd.DataFrame(out).to_csv(OUT/'diagnostic_strata.csv',index=False,encoding='utf-8-sig')
 js(OUT/'diagnosis_complete.json',dict(status='COMPLETE',rows=len(df),descriptive_only=True,predeclared_groups=True,plan_sha=sha(PLAN)))
 print(pd.DataFrame(out).query("scope=='ALL'").to_string(index=False),flush=True)
if __name__=='__main__':run()

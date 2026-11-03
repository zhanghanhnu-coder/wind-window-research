"""Post-result explanation ONLY: hindsight shifts never feed any forecast."""
from pathlib import Path
import numpy as np,pandas as pd
from research_window_error_attribution import OUT,ART,FOLDS,SEEDS,read,sha,js
BINS=[(0,6),(6,24),(24,48),(48,72)]

def run():
 assert read(OUT/'evaluation_complete.json')['status']=='COMPLETE';assert not (OUT/'shift_lag_diagnostic.csv').exists()
 plan=ART.parent.parent/'研究笔记/第48轮负结果后的校准滞后诊断补充_2026-11-02.md';identity={str(p):sha(p) for p in [Path(__file__),plan,OUT/'paired_bounds.csv',OUT/'marginal_paired.csv',OUT/'shift_audit.csv']};js(OUT/'lag_diagnostic_identity.json',identity);rows=[]
 for fold in FOLDS:
  pop=np.load(OUT/fold/'population.npz');y=np.load(ART/'seasonal_probability_v1'/fold/'test_truth.npz')['y'];scale=float(np.load(ART/'seasonal_probability_v1'/fold/'data.npz')['scale'])
  for seed in SEEDS:
   source=next(r['base']['source'] for r in read(OUT/'source_inventory.json') if r['fold']==fold and r['seed']==seed);q=np.load(source)['q']
   for block in np.unique(pop['blocks']):
    pos=np.flatnonzero(pop['blocks']==block);fit=np.load(OUT/fold/f'shift_{seed}_w{block}.npz');grid=fit['grid'];zero=int(np.flatnonzero(grid==0)[0])
    for g,(a,b) in enumerate(BINS):
     qq=q[pos,a:b].astype(float);yy=y[pos,a:b];valid=np.isfinite(yy);counts=valid.sum((1,2));use=counts>0;loss=[]
     for delta in grid:
      prob=(np.maximum(0,qq+delta)<=10).mean(-1);z=(prob-(yy<=10))**2;loss.append(float((np.where(valid,z,0).sum((1,2))[use]/counts[use]).mean()) if use.any() else np.nan)
     loss=np.array(loss);objective=loss+.002*(grid/scale)**2;deployed=int(np.flatnonzero(grid==fit['delta'][g])[0]);oracle=min(np.flatnonzero(objective<=np.nanmin(objective)+1e-12),key=lambda i:(abs(grid[i]),grid[i])) if use.any() else zero
     rows.append(dict(fold=fold,seed=seed,block=int(block),lead_start=a+1,lead_end=b,applied_origins=len(pos),observed_origins=int(use.sum()),observed_labels=int(valid.sum()),historical_delta=float(grid[deployed]),hindsight_delta=float(grid[oracle]),fallback=bool(np.isnan(fit['objectives'][g]).all()),historical_objective_change=float(fit['objectives'][g,deployed]-fit['objectives'][g,zero]),future_brier_base=float(loss[zero]),future_brier_deployed=float(loss[deployed]),future_brier_change=float(loss[deployed]-loss[zero]),future_objective_change=float(objective[deployed]-objective[zero]),hindsight_objective_advantage=float(objective[deployed]-objective[oracle]),same_choice=bool(deployed==oracle),hindsight_only=True))
 pd.DataFrame(rows).to_csv(OUT/'shift_lag_diagnostic.csv',index=False,encoding='utf-8-sig')
 for p,h in identity.items():assert sha(p)==h
 js(OUT/'lag_diagnostic_complete.json',dict(status='COMPLETE',rows=len(rows),hindsight_only=True,forecast_outputs_changed=False));print('LAG DIAGNOSTIC',len(rows),flush=True)
if __name__=='__main__':run()

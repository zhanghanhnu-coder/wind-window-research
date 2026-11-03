"""Verify every hindsight diagnostic row; outputs never enter predictions."""
import json
import numpy as np,pandas as pd
from research_window_error_attribution import OUT,ART,SEEDS,FOLDS,read,sha

def hindsight_index(objectives,grid):
 objectives=np.asarray(objectives,float);grid=np.asarray(grid,float)
 if not np.isfinite(objectives).any():return None
 best=np.flatnonzero(objectives<=np.nanmin(objectives)+1e-12)
 return int(min(best,key=lambda i:(abs(grid[i]),grid[i])))

def run():
 assert read(OUT/'lag_diagnostic_complete.json')['status']=='COMPLETE'
 for p,h in read(OUT/'lag_diagnostic_identity.json').items():assert sha(p)==h
 rows=pd.read_csv(OUT/'shift_lag_diagnostic.csv');count=0
 for fold in FOLDS:
  pop=np.load(OUT/fold/'population.npz');truth=np.load(ART/'seasonal_probability_v1'/fold/'test_truth.npz')['y'];scale=float(np.load(ART/'seasonal_probability_v1'/fold/'data.npz')['scale'])
  for seed in SEEDS:
   src=next(x['base']['source'] for x in read(OUT/'source_inventory.json') if x['fold']==fold and x['seed']==seed);q=np.load(src)['q'].astype(float);scores={c:np.load(OUT/fold/f'hourly_scores_{c}_{seed}.npz')['brier'] for c in ['B','M']}
   for row in rows[(rows.fold==fold)&(rows.seed==seed)].itertuples():
    pos=np.flatnonzero(pop['blocks']==row.block);a=row.lead_start-1;b=row.lead_end;fit=np.load(OUT/fold/f'shift_{seed}_w{row.block}.npz');g=[0,6,24,48].index(a);grid=fit['grid'];dep=int(np.flatnonzero(grid==row.historical_delta)[0]);zero=int(np.flatnonzero(grid==0)[0]);y=truth[pos,a:b];valid=np.isfinite(y);cnt=valid.sum((1,2));use=cnt>0
    def observed_mean(z):return float((np.nansum(z,(1,2))[use]/cnt[use]).mean()) if use.any() else np.nan
    base=observed_mean(scores['B'][pos,a:b]);used=observed_mean(scores['M'][pos,a:b]);np.testing.assert_allclose([base,used,used-base],[row.future_brier_base,row.future_brier_deployed,row.future_brier_change],rtol=0,atol=3e-12)
    objectives=[]
    for d in grid:
     pp=(q[pos,a:b]<=10-d).mean(-1);loss=np.where(valid,(pp-(y<=10))**2,np.nan);objectives.append(observed_mean(loss)+.002*(d/scale)**2)
    objectives=np.array(objectives);oracle=hindsight_index(objectives,grid)
    if oracle is None:
     assert row.observed_labels==0 and row.observed_origins==0 and row.hindsight_delta==0
     assert np.isnan(row.future_objective_change) and np.isnan(row.hindsight_objective_advantage)
    else:
     assert row.hindsight_delta==grid[oracle];np.testing.assert_allclose([objectives[dep]-objectives[zero],objectives[dep]-objectives[oracle]],[row.future_objective_change,row.hindsight_objective_advantage],rtol=0,atol=3e-12)
    np.testing.assert_allclose(fit['objectives'][g,dep]-fit['objectives'][g,zero],row.historical_objective_change,rtol=0,atol=3e-12,equal_nan=True);assert row.observed_origins==use.sum() and row.observed_labels==valid.sum() and row.hindsight_only;count+=1
 result=dict(status='PASS',rows=count,hindsight_grid_objectives_recomputed=count*25,independent_deployed_scores=True,forecast_files_modified=False);(OUT/'lag_diagnostic_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result,flush=True)
if __name__=='__main__':run()

import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from research_rolling import OUT,SEEDS,ART,js
from rolling_core import event_matrix,map_probability
from probability_core import score_scenarios

def main():
 conf=json.loads((OUT/'solver_sensitivity_frozen.json').read_text(encoding='utf-8'));assert hashlib.sha256(Path(__file__).with_name('rolling_solver_sensitivity.py').read_bytes()).hexdigest()==conf['source_sha256']
 d=np.load(OUT/'data.npz');ids=d['ids'];te=d['test_positions'];old_a=pd.read_csv(OUT/'calibration_audit.csv');a=pd.read_csv(OUT/'solver_sensitivity_audit.csv');member=json.loads((OUT/'membership.json').read_text(encoding='utf-8'));changed=0;rechecked=0
 for s in SEEDS:
  q=d[f'q_{s}'];r=d['truth']-q[...,1];h=np.load(OUT/f'calibration_history_{s}.npz');wpack=np.load(OUT/f'tilt_weights_2000_{s}.npz');oldw=np.load(OUT/f'tilt_weights_{s}.npz');scores=np.load(OUT/f'scores_tilted28_2000_{s}.npz');before=np.load(OUT/f'scores_tilted28_{s}.npz')
  for j,k in enumerate(te):
   issue=int(ids[k]);w=wpack[f'issue_{issue}'];n=len(w);assert abs(w.sum()-1)<1e-12 and np.min(w)>=.25/n-1e-12
   row=a[(a.seed==s)&(a.issue==issue)].iloc[0];old=old_a[(old_a.seed==s)&(old_a.issue==issue)].iloc[0]
   if old.tilt_success:np.testing.assert_equal(w,oldw[f'issue_{issue}'])
   if row.weight_max_change>0:
    assert not old.tilt_success;changed+=1;sc=np.maximum(0,q[k,...,1]+r[member[str(issue)]['28']]);E=event_matrix(sc);target=map_probability(E.mean(0),n,h['theta'][j]);objective=.5*np.mean((E.T@w-target)**2)+.01*np.sum(w*np.log(w*n));assert objective<=row.old_deployed_objective+1e-10
    if row.success:np.testing.assert_allclose(objective,row.objective,atol=1e-12)
    result=score_scenarios(sc,d['truth'][k],w)
    for metric,v in result.items():np.testing.assert_allclose(v,scores[metric][j],atol=1e-12,equal_nan=True)
    rechecked+=1
   else:
    for metric in scores.files:np.testing.assert_equal(scores[metric][j],before[metric][j])
 raw=np.load(ART/'data.npz');times=pd.DatetimeIndex(pd.to_datetime(raw['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');tt=times[ids[te]];table=pd.read_csv(OUT/'solver_sensitivity_metrics.csv')
 for row in table.itertuples():
  with np.load(OUT/f'scores_tilted28_2000_{row.seed}.npz') as pack:
   mask=np.ones(len(te),bool) if row.month==0 else tt.month==row.month;v=pack[row.metric][mask];np.testing.assert_allclose(np.nanmean(v),row.value,atol=1e-12);assert np.isfinite(v).sum()==row.n
 weeks=((tt-pd.Timestamp('2024-11-01',tz=tt.tz)).days//7).to_numpy();keys=np.unique(weeks);draw=np.random.default_rng(20261102).integers(0,len(keys),(2000,len(keys)));cp=pd.read_csv(OUT/'solver_sensitivity_comparisons.csv')
 for row in cp.itertuples():
  diff=[]
  for s in SEEDS:
   with np.load(OUT/f'scores_tilted28_2000_{s}.npz') as candidate,np.load(OUT/f'scores_{row.baseline}_{s}.npz') as base:diff.append(candidate[row.metric]-base[row.metric])
  dd=np.mean(diff,axis=0);ss=np.array([np.nansum(dd[weeks==k]) for k in keys]);nn=np.array([np.isfinite(dd[weeks==k]).sum() for k in keys]);boot=ss[draw].sum(1)/nn[draw].sum(1);np.testing.assert_allclose([np.nanmean(dd),*np.quantile(boot,[.025,.975])],[row.difference,row.low95,row.high95],atol=1e-12)
 result=dict(solves=len(a),successful=int(a.success.sum()),max_iterations=int(a.iterations.max()),changed_weights=changed,changed_score_cases_recomputed=rechecked,unchanged_solutions_preserved=int(old_a.tilt_success.sum()),metric_rows_recomputed=len(table),paired_rows_recomputed=len(cp),parameters_changed='maxiter only: 200 to 2000',primary_results_overwritten=False)
 js('solver_sensitivity_verification.json',result);print(result)

if __name__=='__main__':main()

"""Numerical-only extension, frozen before execution; preserves primary replay."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from probability_core import score_scenarios
from rolling_core import tilt_objective,event_matrix,map_probability
from research_rolling import OUT,SEEDS,METRICS,ART,csv,js

def main():
 rule={'maxiter':2000,'ftol':1e-12,'gtol':1e-9,'z_bounds':[-10,10],'target':'identical stored direct28 target from prequential calibration','objective':'identical tilt_objective','status':'post-hoc numerical sensitivity; original 200-step methods retained','all_solves':696,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 config=OUT/'solver_sensitivity_frozen.json'
 if config.exists():assert json.loads(config.read_text(encoding='utf-8'))==rule
 else:js(config.name,rule)
 data=np.load(OUT/'data.npz');raw=np.load(ART/'data.npz');times=pd.DatetimeIndex(pd.to_datetime(raw['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');ids=data['ids'];te=data['test_positions'];tt=times[ids[te]];membership=json.loads((OUT/'membership.json').read_text(encoding='utf-8'));audit=[];metrics=[]
 for s in SEEDS:
  history=np.load(OUT/f'calibration_history_{s}.npz');q=data[f'q_{s}'];res=data['truth']-q[...,1];oldw=np.load(OUT/f'tilt_weights_{s}.npz');newweights={}
  with np.load(OUT/f'scores_tilted28_{s}.npz') as original:arrays={key:original[key].copy() for key in original.files}
  for j,k in enumerate(te):
   issue=int(ids[k]);ix=membership[str(issue)]['28'];sc=np.maximum(0,q[k,...,1]+res[ix]);E=event_matrix(sc);target=map_probability(E.mean(0),len(sc),history['theta'][j]);n=len(sc)
   fit=minimize(tilt_objective,np.zeros(n),args=(E,target),jac=True,method='L-BFGS-B',bounds=[(-10,10)]*n,options={'maxiter':2000,'ftol':1e-12,'gtol':1e-9})
   success=bool(fit.success and np.isfinite(fit.fun));w=.25/n+.75*softmax(fit.x) if success else np.ones(n)/n;newweights[f'issue_{issue}']=w;difference=float(np.max(abs(w-oldw[f'issue_{issue}'])))
   # Same score routine, only recompute changed weights; unchanged outputs copied exactly.
   if difference>0:
    result=score_scenarios(sc,data['truth'][k],w)
    for key,value in result.items():arrays[key][j]=value
   oldprob=E.T@oldw[f'issue_{issue}'];prior=np.ones(n)/n;oldobjective=.5*np.mean((oldprob-target)**2)+.01*np.sum(oldw[f'issue_{issue}']*np.log(oldw[f'issue_{issue}']/prior))
   audit.append(dict(seed=s,issue=issue,success=success,iterations=int(fit.nit),message=str(fit.message),objective=float(fit.fun),old_deployed_objective=float(oldobjective),weight_max_change=difference,ess=float(1/(w@w)),gap=float(np.sqrt(np.mean((E.T@w-target)**2)))))
  np.savez_compressed(OUT/f'scores_tilted28_2000_{s}.npz',**arrays);np.savez_compressed(OUT/f'tilt_weights_2000_{s}.npz',**newweights)
  for month in [0,11,12]:
   mask=np.ones(len(te),bool) if month==0 else tt.month==month
   for key in METRICS:
    v=arrays[key][mask];metrics.append(dict(seed=s,method='tilted28_2000',month=month,metric=key,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
  print(s,'2000-step Brier',np.nanmean(arrays['brier6']),flush=True)
 csv('solver_sensitivity_audit.csv',audit);csv('solver_sensitivity_metrics.csv',metrics)
 week=((tt-pd.Timestamp('2024-11-01',tz=tt.tz)).days//7).to_numpy();keys=np.unique(week);draw=np.random.default_rng(20261102).integers(0,len(keys),(2000,len(keys)));rows=[]
 for baseline in ['frozen32','rolling28','tilted28']:
  for metric in ['brier6','crps','length_crps72','first_cdf_brier','any6_brier72']:
   differences=[]
   for s in SEEDS:
    with np.load(OUT/f'scores_tilted28_2000_{s}.npz') as a,np.load(OUT/f'scores_{baseline}_{s}.npz') as b:differences.append(a[metric]-b[metric])
   diff=np.mean(differences,axis=0);sums=np.array([np.nansum(diff[week==k]) for k in keys]);counts=np.array([np.isfinite(diff[week==k]).sum() for k in keys]);bootstrap=sums[draw].sum(1)/counts[draw].sum(1);lo,hi=np.quantile(bootstrap,[.025,.975]);rows.append(dict(candidate='tilted28_2000',baseline=baseline,metric=metric,difference=float(np.nanmean(diff)),low95=lo,high95=hi,blocks=len(keys),n=int(np.isfinite(diff).sum())))
 csv('solver_sensitivity_comparisons.csv',rows);print('numerical sensitivity complete')

if __name__=='__main__':main()

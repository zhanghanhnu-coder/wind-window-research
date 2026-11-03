"""Audit sequential information boundaries and saved experiment results."""
import json,subprocess,sys,re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from model import ART
from research_rolling import OUT,SEEDS,METHODS,JOINT,METRICS,sha,js,generate_scenarios
from rolling_core import mature_paths,mature_events,fit_map,event_matrix,OFFSETS

def main():
 old=json.loads((OUT/'previous_hashes.json').read_text(encoding='utf-8'));assert all(sha(ART/k)==v for k,v in old.items())
 identity=json.loads((OUT/'identity.json').read_text(encoding='utf-8'));assert all(sha(Path(k))==v for k,v in identity.items())
 proc=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ART.parent,capture_output=True,text=True,encoding='utf-8',errors='replace');(OUT/'tests.log').write_text(proc.stdout+proc.stderr,encoding='utf-8');assert proc.returncode==0,proc.stderr[-1800:];tests=int(re.search(r'Ran (\d+) tests',proc.stderr).group(1))
 data=np.load(OUT/'data.npz');raw=np.load(ART/'data.npz');ids=data['ids'];te=data['test_positions'];times=pd.DatetimeIndex(pd.to_datetime(raw['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');tt=times[ids[te]]
 assert np.array_equal(ids[te],raw['test_ids']);np.testing.assert_equal(data['truth'],np.stack([raw['raw'][i+1:i+73] for i in ids]));complete=np.isfinite(data['truth']).all((1,2));start=int(data['warmup_start'])
 members=json.loads((OUT/'membership.json').read_text(encoding='utf-8'));bank=pd.read_csv(OUT/'bank_audit.csv');audit=pd.read_csv(OUT/'calibration_audit.csv');metrics=pd.read_csv(OUT/'metrics.csv')
 for key,value in members.items():
  issue=int(key)
  for name,days in [('14',14),('28',28),('all',None)]:
   expected=np.flatnonzero(complete&(ids+72<=issue-1)&((ids+72>=issue-24*days) if days is not None else True))
   if len(expected)<8:expected=np.flatnonzero(complete&(ids+72<=issue-1))
   np.testing.assert_equal(value[name],expected)
 assert (bank.last_label<=bank.issue-1).all();assert (audit.latest_window_label<=audit.issue-1).all()
 # Optimizer fallback is a frozen part of the method, not a failed experiment.
 assert (audit.loc[~audit.map_success,'map_fallback']).all()
 for r in audit[~audit.map_success].itertuples():np.testing.assert_equal([r.map_a,r.map_b],[1,0])
 for r in audit[~audit.tilt_success].itertuples():
  with np.load(OUT/f'tilt_weights_{r.seed}.npz') as w:
   v=w[f'issue_{r.issue}'];np.testing.assert_allclose(v,np.full(len(v),1/len(v)),atol=0,rtol=0)
 max_base=0.;recomputed=0;formula_checks=0;history_checks=0
 oldloss=np.load(ART/'v2/paired_losses.npz')['graph24']
 for si,s in enumerate(SEEDS):
  hist=np.load(OUT/f'calibration_history_{s}.npz');weights=np.load(OUT/f'tilt_weights_{s}.npz');q=data[f'q_{s}'];res=data['truth']-q[...,1]
  for j,k in enumerate(te):
   issue=int(ids[k]);past=np.flatnonzero((ids>=start)&(ids<issue));x,y,norig,latest=mature_events(ids[past],hist['probabilities'][past],hist['labels'][past],hist['sizes'][past],issue,start)
   r=audit[(audit.seed==s)&(audit.issue==issue)].iloc[0];assert len(y)==r.map_n and norig==r.map_origins and latest==r.latest_window_label
   # Selected calibration labels must precede issue even after arbitrary poisoning of unavailable labels.
   if j in [0,116,231]:
    altered=hist['labels'][past].copy();end=ids[past,None,None]+OFFSETS[None,:,None]+6;future=np.broadcast_to(end>issue-1,altered.shape);altered[future]=1-np.nan_to_num(altered[future]);xx,yy,nn,ll=mature_events(ids[past],hist['probabilities'][past],altered,hist['sizes'][past],issue,start);np.testing.assert_equal(x,xx);np.testing.assert_equal(y,yy)
    theta,info=fit_map(x,y,norig);np.testing.assert_allclose(theta,hist['theta'][j],atol=1e-9);history_checks+=1
    paths={name:np.array(ix) for name,ix in members[str(issue)].items()};samples=generate_scenarios(data,q,k,paths,data['static_positions']);samples['tilted28']=(samples['rolling28'][0],weights[f'issue_{issue}'])
    target_truth=data['truth'][k]
    for m,(sc,w) in samples.items():
     w=np.ones(len(sc))/len(sc) if w is None else w;assert np.isfinite(sc).all() and (sc>=0).all();assert abs(w.sum()-1)<1e-12
     if m=='tilted28':assert (w>=.25/len(w)-1e-12).all()
     with np.load(OUT/f'scores_{m}_{s}.npz') as saved:
      probs=np.stack([w@((sc[:,a:a+6]<=10).all(1)) for a in OFFSETS]);truth=np.stack([np.where(np.isfinite(target_truth[a:a+6]).all(0),(target_truth[a:a+6]<=10).all(0),np.nan) for a in OFFSETS]);np.testing.assert_allclose((probs-truth)**2,saved['brier6'][j],atol=1e-12,equal_nan=True)
      small=sc[:,:3,:2].astype(float);y0=target_truth[:3,:2];brute=np.sum(w[:,None,None]*abs(small-y0),axis=0)-.5*np.sum(w[:,None,None,None]*w[None,:,None,None]*abs(small[:,None]-small[None,:]),axis=(0,1));np.testing.assert_allclose(brute,saved['crps'][j,:3,:2],atol=1e-10,equal_nan=True)
      # Independent sliding-window CDF via cumulative sums, no imported first_start.
      cs=np.concatenate([np.zeros((len(sc),1,17)),np.cumsum(sc<=10,axis=1)],axis=1);windows=cs[:,6:]-cs[:,:-6]==6;predcdf=np.maximum.accumulate(windows,axis=1);yp=np.r_[np.zeros((1,17)),np.cumsum(target_truth<=10,axis=0)];ycdf=np.maximum.accumulate(yp[6:]-yp[:-6]==6,axis=0);forecast=np.sum(w[:,None,None]*predcdf,axis=0);loss=np.mean((forecast-ycdf)**2,axis=0);loss[~np.isfinite(target_truth).all(0)]=np.nan;np.testing.assert_allclose(loss,saved['first_cdf_brier'][j],atol=1e-12,equal_nan=True)
      assert np.nanmin(saved['any6_prob48'][j]-saved['any6_prob24'][j])>=-1e-12;assert np.nanmin(saved['any6_prob72'][j]-saved['any6_prob48'][j])>=-1e-12;formula_checks+=1
   history_checks+=1
  for m in METHODS:
   with np.load(OUT/f'scores_{m}_{s}.npz') as pack:
    for row in metrics[(metrics.seed==s)&(metrics.method==m)].itertuples():
     mask=np.ones(len(te),bool) if row.month==0 else tt.month==row.month;values=pack[row.metric][mask];np.testing.assert_allclose(np.nanmean(values),row.value,atol=1e-12);assert np.isfinite(values).sum()==row.n;recomputed+=1
    if m=='frozen32':max_base=max(max_base,float(np.nanmax(abs(pack['brier6']-oldloss[si]))))
    if m=='frozen58':
     with np.load(ART/f'probability_v1/scores_joint_all_{s}.npz') as previous:
      for metric in METRICS:np.testing.assert_allclose(pack[metric],previous[metric],atol=1e-12,rtol=1e-12,equal_nan=True)
    assert np.isfinite(pack['brier6']).sum()==15459
 assert max_base==0
 comp=pd.read_csv(OUT/'paired_comparisons.csv');week=((tt-pd.Timestamp('2024-11-01',tz=tt.tz)).days//7).to_numpy();keys=np.unique(week);draw=np.random.default_rng(20261102).integers(0,len(keys),(2000,len(keys)))
 for r in comp.itertuples():
  diff=[]
  for s in SEEDS:
   with np.load(OUT/f'scores_{r.candidate}_{s}.npz') as a,np.load(OUT/f'scores_{r.baseline}_{s}.npz') as b:
    assert np.array_equal(np.isfinite(a[r.metric]),np.isfinite(b[r.metric]));diff.append(a[r.metric]-b[r.metric])
  delta=np.mean(diff,axis=0);ss=np.array([np.nansum(delta[week==k]) for k in keys]);nn=np.array([np.isfinite(delta[week==k]).sum() for k in keys]);boot=ss[draw].sum(1)/nn[draw].sum(1);np.testing.assert_allclose([np.nanmean(delta),*np.quantile(boot,[.025,.975])],[r.difference,r.low95,r.high95],atol=1e-12)
 result=dict(unit_tests_passed=tests,previous_artifacts_unchanged=len(old),identity_files=len(identity),metric_rows_recomputed=recomputed,paired_rows_recomputed=len(comp),bank_membership_entries_checked=len(members)*3,calibration_time_audit_rows=len(audit),calibration_refits=9,independent_case_score_checks=formula_checks,baseline_max_case_difference=max_base,map_optimizer_failures=int((~audit.map_success).sum()),tilt_optimizer_failures=int((~audit.tilt_success).sum()),scope='All memberships and fitting-label times, all metric means/counts and paired intervals checked; 3 fixed issues x 3 seeds x 8 joint methods independently checked for window Brier, selected-cell CRPS and first-window CDF. All optimizer failures verified to use predeclared fallback. No second full experiment or original-model training.')
 js('verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

"""Independent artifact and numerical audit of round14; does not rewrite predictions."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from model import ART
OUT=ART/'static_window_v1';SRC=ART/'direct_window_v1';OFF=[0,24,48,66]
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 old=read('previous_hashes.json');ident=read('identity.json')
 for p,h in old.items():assert sha(ART/p)==h,p
 for p,h in ident.items():assert sha(Path(p))==h,p
 dat=np.load(OUT/'evaluation_data.npz');src=np.load(SRC/'split_data.npz');base=np.load(ART/'data.npz');ids=dat['cal_ids'];testids=dat['test_ids'];sel=dat['selected_positions'];np.testing.assert_array_equal(ids[sel],base['cal_ids']);assert len(set(ids[sel]))==58
 assert np.max(ids+72)<np.min(testids);np.testing.assert_array_equal(testids,base['test_ids'])
 # Build labels independently from raw future slices with explicit loops.
 for part,pids in [('cal',ids),('test',testids)]:
  truth=[]
  for i in pids:
   labs=[]
   for off in OFF:
    raw=base['raw'][i+1+off:i+1+off+6];lab=np.where(np.isfinite(raw).all(0),(raw<=10).all(0).astype(float),np.nan);labs.append(lab)
   truth.append(labs)
  np.testing.assert_array_equal(np.array(truth),dat[part+'_labels'])
 y=dat['test_labels'];inventory=read('inventory.json');fits=pd.read_csv(OUT/'fits.csv');errors={};probs={};refit=0;maxdiff=0.
 for item in inventory:
  name=item['id'];m=item['model'];s=item['seed'];method=item['method'];saved=np.load(OUT/f'predictions_{name}.npz');p=saved['test'];np.testing.assert_array_equal(saved['offsets'],OFF);assert p.shape==y.shape
  if m=='oct_frequency58':
   obs=dat['cal_labels'][sel];expected=np.broadcast_to((np.nansum(obs,0)+1)/(np.isfinite(obs).sum(0)+2),y.shape);np.testing.assert_array_equal(p,expected)
  else:
   file=SRC/(f'predictions_{m}_{s}_budget25.npz' if m not in ['scenario32','logistic'] else f'predictions_{m}_{s}.npz' if m=='scenario32' else 'predictions_logistic_single.npz');source=np.load(file);pos=[list(source['offsets']).index(a) for a in OFF];tp=source['test'][:,pos]
   if method in ['raw','frozen']:np.testing.assert_array_equal(p,tp)
   else:
    theta=saved['theta'];qq=np.clip(tp.astype(float),1e-6,1-1e-6);expected=expit(theta[None,:,0,None]*np.log(qq/(1-qq))+theta[None,:,1,None]);np.testing.assert_allclose(p,expected,atol=1e-14,rtol=0)
    rows=np.arange(len(ids)) if method=='shared112' else sel;cp=source['cal'][:,pos][rows];cy=dat['cal_labels'][rows]
    for j,t in enumerate(theta):
     which=slice(j,j+1) if method=='byoffset58' else slice(None);v=cy[:,which];mask=np.isfinite(v);q=np.clip(cp[:,which][mask].astype(float),1e-6,1-1e-6);x=np.log(q/(1-q));v=v[mask]
     def fun(z):
      pred=expit(z[0]*x+z[1]);return np.mean(-v*np.log(pred)-(1-v)*np.log1p(-pred))+.01*((z[0]-1)**2+z[1]**2)
     # Different solver, numeric derivatives and tighter convergence for the two-parameter convex fit.
     result=minimize(fun,[1.,0.],method='SLSQP',bounds=[(.25,4),(-3,3)],options={'maxiter':500,'ftol':1e-13});assert result.success,result.message
     maxdiff=max(maxdiff,float(np.max(abs(result.x-t))));np.testing.assert_allclose(result.x,t,atol=2e-5,rtol=0);refit+=1
     row=fits[(fits.model==m)&(fits.seed==s)&(fits.method==method)&(fits.group==j)].iloc[0];assert int(row.n)==len(v) and row.success and not row.fallback
  err=(p-y)**2;np.testing.assert_array_equal(err,np.load(OUT/f'errors_{name}.npz')['errors']);errors[name]=err;probs[name]=p
  if method=='raw':
   olderr=np.load(SRC/(f'errors_{m}_{s}_budget25.npz' if s else 'errors_logistic_single.npz'))['test_primary_errors'];np.testing.assert_array_equal(err,olderr)
  if method=='frozen':np.testing.assert_array_equal(err,np.load(SRC/f'errors_scenario32_{s}.npz')['test_primary_errors'])
 times=pd.DatetimeIndex(pd.to_datetime(base['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai')[testids];metrics=pd.read_csv(OUT/'metrics.csv')
 for r in metrics.itertuples():
  err=errors[r.id];mask=np.ones(len(y),bool) if r.month==0 else times.month==r.month;v=err if r.metric=='brier_primary' else err[:,OFF.index(int(r.metric.removeprefix('brier_offset')))];assert np.isclose(np.nanmean(v[mask]),r.value,atol=1e-13,rtol=0) and np.isfinite(v[mask]).sum()==r.n
 rel=pd.read_csv(OUT/'reliability.csv');high=pd.read_csv(OUT/'high_confidence.csv')
 for table,isbins in [(rel,True),(high,False)]:
  for r in table.itertuples():
   mask=np.ones(len(y),bool) if r.month==0 else times.month==r.month;pp=probs[r.id][mask].ravel();yy=y[mask].ravel();v=np.isfinite(yy);pp=pp[v];yy=yy[v];mm=((pp>=r.bin/10)&(pp<(r.bin+1)/10 if r.bin<9 else pp<=1)) if isbins else pp>=.8
   assert mm.sum()==r.n
   if mm.any():assert np.isclose(pp[mm].mean(),r.mean_probability,atol=1e-13,rtol=0) and np.isclose(yy[mm].mean(),r.observed_rate,atol=1e-13,rtol=0)
   else:assert pd.isna(r.mean_probability) and pd.isna(r.observed_rate)
   if not isbins:assert np.isclose(mm.mean(),r.fraction,atol=1e-13,rtol=0)
 groups={}
 for it in inventory:groups.setdefault((it['model'],it['method']),[]).append(errors[it['id']])
 week=((times-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();keys=np.unique(week);rng=np.random.default_rng(20261104);draw=rng.integers(0,len(keys),(2000,len(keys)));pairs=pd.read_csv(OUT/'paired_comparisons.csv')
 for r in pairs.itertuples():
  delta=np.mean(groups[(r.candidate,r.method)],0)-np.mean(groups[(r.baseline,r.baseline_method)],0);sums=np.array([np.nansum(delta[week==k]) for k in keys]);counts=np.array([np.isfinite(delta[week==k]).sum() for k in keys]);boot=(sums[draw].sum(1)/counts[draw].sum(1));lo,hi=np.quantile(boot,[.025,.975]);np.testing.assert_allclose([np.nanmean(delta),lo,hi],[r.difference,r.low95,r.high95],atol=1e-13,rtol=0)
 summary=dict(status='PASS',previous_unchanged=len(old),frozen_identities=len(ident),prediction_sets=len(inventory),independent_parameter_refits=refit,maximum_parameter_difference=maxdiff,metric_rows=len(metrics),reliability_rows=len(rel),high_confidence_rows=len(high),paired_rows=len(pairs),valid_primary=int(np.isfinite(y).sum()),same58=True,october_all_labels_mature=True,raw_and_scenario_case_errors_exact=True)
 (OUT/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':run()

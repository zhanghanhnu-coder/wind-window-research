"""Independent round16 accounting, parameter, marginal and path checks."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
from research_lead_calibration import CONFIGS,SEEDS,NEW,paths
from crps_distribution_core import affine_objective
OUT=ART/'crps_lead_calibration_v1';SRC=ART/'crps_distribution_v1';BINS=[(0,6),(6,24),(24,48),(48,72)]
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def empirical(q,y):
 q=np.sort(np.asarray(q,float),axis=-1);k=q.shape[-1];return abs(q-y[...,None]).mean(-1)-np.sum(q*(2*np.arange(k)-k+1),-1)/(k*k)
def transform(q,theta,scale,smooth):
 pp=np.stack([np.interp(np.arange(1,73),[3.5,15.5,36.5,60.5],theta[:,j]) for j in range(2)],-1) if smooth else np.vstack([np.tile(t,(b-a,1)) for t,(a,b) in zip(theta,BINS)])
 med=q[...,9,None];return np.maximum(0,med+pp[None,:,0,None,None]*scale+pp[None,:,1,None,None]*(q-med))
def run():
 old=read('previous_hashes.json');identity=read('identity.json')
 for p,h in old.items():assert sha(ART/p)==h,p
 for p,h in identity.items():assert sha(Path(p))==h,p
 ev=np.load(OUT/'evaluation_data.npz');assert sha(OUT/'evaluation_data.npz')==sha(SRC/'evaluation_data.npz');meta=json.loads((ART/'metadata.json').read_text(encoding='utf-8'));scale=meta['scale'];dat=np.load(ART/'data.npz');complete=ev['complete_positions'];selected=ev['selected57'];truth=ev['test_truth'];ct=ev['cal_truth'][complete]
 np.testing.assert_array_equal(ev['cal_ids'][complete],dat['cal_ids']);assert max(ev['cal_ids'][complete]+72)<min(ev['test_ids']);np.testing.assert_array_equal(truth,np.stack([dat['raw'][i+1:i+73] for i in ev['test_ids']]))
 fitrows=pd.read_csv(OUT/'calibration.csv');objective_checks=0;gradient_checks=0;distribution_checks=0
 for model in CONFIGS:
  for seed in SEEDS:
   pred=np.load(SRC/f'predictions_{model}_{seed}_budget40.npz');cp=pred['cal'][complete].astype(float);params=np.load(OUT/f'parameters_{model}_{seed}.npz');info=read(f'calibration_{model}_{seed}.json')
   for kind in ['affine','shift']:
    theta=params[kind];np.testing.assert_array_equal(theta,info[kind]);assert theta.shape==(4,2) and np.all(abs(theta[:,0])<=2) and np.all((theta[:,1]>=.25)&(theta[:,1]<=4))
    if kind=='shift':np.testing.assert_array_equal(theta[:,1],np.ones(4))
    for j,(a,b) in enumerate(BINS):
     row=info[kind+'_info'][j];q=cp[:,a:b]/scale;y=ct[:,a:b].astype(float)/scale;med=q[...,9,None];qq=np.maximum(0,med+theta[j,0]+theta[j,1]*(q-med));objective=np.mean(empirical(qq,y))+.001*(theta[j,0]**2+(theta[j,1]-1)**2);assert abs(objective-row['objective_used'])<1e-10 and row['valid_targets']==58*(b-a)*17;objective_checks+=1
     if row['fallback']:np.testing.assert_array_equal(theta[j],[0,1]);assert not row['success']
     f=fitrows[(fitrows.model==model)&(fitrows.seed==seed)&(fitrows.kind==kind)&(fitrows.group==j)].iloc[0];assert np.isclose(f.delta,theta[j,0],atol=1e-15) and np.isclose(f.gamma,theta[j,1],atol=1e-15)
     if kind=='affine':
      q=q.reshape(-1,19);y=y.ravel();t=np.array([.137,1.173]);_,g=affine_objective(t,q,y);eps=1e-7;num=np.array([(affine_objective(t+eps*np.eye(2)[k],q,y)[0]-affine_objective(t-eps*np.eye(2)[k],q,y)[0])/(2*eps) for k in range(2)]);np.testing.assert_allclose(g,num,atol=3e-6,rtol=0)
      # Independent pinball identity gradient; finite differences can cross absolute-value kinks.
      spread=q-q[:,9,None];z=q[:,9,None]+t[0]+t[1]*spread;v=np.maximum(z,0);tau=(np.arange(19)+.5)/19;indicator=(v>y[:,None]).astype(float)+.5*(v==y[:,None]);dq=2/19*(indicator-tau)*(z>0);pin_grad=np.array([dq.sum(1).mean(),(dq*spread).sum(1).mean()])+.002*np.array([t[0],t[1]-1]);np.testing.assert_allclose(g,pin_grad,atol=1e-12,rtol=0);gradient_checks+=1
   for method in NEW:
    q=transform(pred['test'].astype(float),params['shift' if method=='bin_shift' else 'affine'],scale,method=='smooth_affine');assert np.isfinite(q).all() and (q>=0).all() and (np.diff(q,axis=-1)>=0).all();scores=np.load(OUT/f'scores_{model}__{method}__{seed}.npz');np.testing.assert_allclose(empirical(q,truth),scores['crps'],atol=2e-13,rtol=0,equal_nan=True);distribution_checks+=1
 inventory=read('inventory.json');cases=0;copied=0;selectedcases=[0,len(truth)//2,len(truth)-1]
 for item in inventory:
  scores=np.load(OUT/f'scores_{item["id"]}.npz');jj=np.load(OUT/f'jumps_{item["id"]}.npz')
  if item['method'] not in NEW:
   method='affine_ecc57' if item['method']=='global' else item['method'];assert sha(OUT/f'scores_{item["id"]}.npz')==sha(SRC/f'scores_{item["model"]}__{method}__{item["seed"]}.npz');copied+=1;continue
  for i,sc in enumerate(paths(item,meta,ev)):
   if i not in selectedcases:continue
   y=truth[i];sc=sc.astype(float);k=len(sc);brute=abs(sc-y).mean(0)-.5*abs(sc[:,None]-sc[None,:]).mean((0,1));np.testing.assert_allclose(brute,scores['crps'][i],atol=2e-12,rtol=0,equal_nan=True)
   for duration in [3,6,12]:
    values=[]
    for off in [0,24,48,72-duration]:
     obs=y[off:off+duration];p=(sc[:,off:off+duration]<=10).all(1).mean(0);values.append(np.where(np.isfinite(obs).all(0),(p-(obs<=10).all(0))**2,np.nan))
    np.testing.assert_allclose(np.array(values),scores[f'brier{duration}'][i],atol=2e-14,rtol=0,equal_nan=True)
   def longest(row):return np.diff(np.r_[-1,np.flatnonzero(~row),len(row)]).max()-1
   length=np.array([[longest(sc[si,:,n]<=10) for si in range(k)] for n in range(17)]);observed=np.array([longest(y[:,n]<=10) for n in range(17)]);valid=np.isfinite(y).all(0);expected=np.where(valid,empirical(length,observed),np.nan);np.testing.assert_allclose(expected,scores['length_crps72'][i],atol=2e-13,rtol=0,equal_nan=True)
   seen=np.zeros((k,17),bool);actual=np.zeros(17,bool);errors=[]
   for off in range(67):
    seen|=(sc[:,off:off+6]<=10).all(1);actual|=(y[off:off+6]<=10).all(0);errors.append((seen.mean(0)-actual)**2)
   np.testing.assert_allclose(np.where(valid,np.mean(errors,0),np.nan),scores['first_cdf_brier'][i],atol=2e-14,rtol=0,equal_nan=True)
   dif=abs(sc[:,1:]-sc[:,:-1]);hourly=dif.mean((0,2));np.testing.assert_allclose(hourly,jj['hourly'][i],atol=1e-13,rtol=0);np.testing.assert_allclose(hourly[[5,23,47]],jj['boundaries'][i],atol=1e-13,rtol=0);assert abs(dif[:,np.setdiff1d(np.arange(71),[5,23,47])].mean()-jj['other_mean'][i])<1e-13;cases+=1
 times=pd.DatetimeIndex(pd.to_datetime(dat['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai')[ev['test_ids']];cached={it['id']:dict(np.load(OUT/f'scores_{it["id"]}.npz')) for it in inventory};metrics=pd.read_csv(OUT/'metrics.csv')
 for r in metrics.itertuples():
  mask=np.ones(len(times),bool) if r.month==0 else times.month==r.month;v=cached[r.id][r.metric][mask];assert abs(np.nanmean(v)-r.value)<1e-11 and np.isfinite(v).sum()==r.n
 lead=pd.read_csv(OUT/'lead_metrics.csv')
 for r in lead.itertuples():
  v=cached[r.id][r.metric][:,r.lead_start-1:r.lead_end];assert abs(np.nanmean(v)-r.value)<1e-11 and np.isfinite(v).sum()==r.n
 windows=pd.read_csv(OUT/'windows_by_lead.csv')
 for r in windows.itertuples():
  j=[0,24,48,66].index(r.offset);z=cached[r.id];np.testing.assert_allclose([np.nanmean(z[k][:,j]) for k in ['brier6','prob6','label6']],[r.brier6,r.mean_probability,r.observed_rate],atol=1e-13,rtol=0)
 jumps=pd.read_csv(OUT/'jumps.csv')
 for r in jumps.itertuples():
  j=np.load(OUT/f'jumps_{r.id}.npz');np.testing.assert_allclose([*j['boundaries'].mean(0),j['other_mean'].mean()],[r.boundary6_7,r.boundary24_25,r.boundary48_49,r.other_mean],atol=1e-13,rtol=0)
 rel=pd.read_csv(OUT/'reliability.csv');high=pd.read_csv(OUT/'high_confidence.csv')
 for table,isbins in [(rel,True),(high,False)]:
  for r in table.itertuples():
   mask=np.ones(len(times),bool) if r.month==0 else times.month==r.month;p=cached[r.id]['prob6'][mask].ravel();y=cached[r.id]['label6'][mask].ravel();good=np.isfinite(y);p=p[good];y=y[good];mm=((p>=r.bin/10)&(p<(r.bin+1)/10 if r.bin<9 else p<=1)) if isbins else p>=.8;assert mm.sum()==r.n
   if mm.any():np.testing.assert_allclose([p[mm].mean(),y[mm].mean()],[r.mean_probability,r.observed_rate],atol=1e-13,rtol=0)
   else:assert pd.isna(r.mean_probability) and pd.isna(r.observed_rate)
   if not isbins:assert abs(mm.mean()-r.fraction)<1e-13
 week=((times-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();keys=np.unique(week);draw=np.random.default_rng(20261107).integers(0,len(keys),(2000,len(keys)));pairs=pd.read_csv(OUT/'paired_comparisons.csv')
 for r in pairs.itertuples():
  delta=np.mean([cached[f'{r.candidate}__{r.method}__{s}'][r.metric]-cached[f'{r.baseline}__{r.baseline_method}__{s}'][r.metric] for s in SEEDS],0);sums=np.array([np.nansum(delta[week==k]) for k in keys]);counts=np.array([np.isfinite(delta[week==k]).sum() for k in keys]);boot=sums[draw].sum(1)/counts[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975]);np.testing.assert_allclose([np.nanmean(delta),lo,hi],[r.difference,r.low95,r.high95],atol=1e-12,rtol=0)
 summary=dict(status='PASS',previous_unchanged=len(old),frozen_identities=len(identity),parameter_objectives=objective_checks,gradient_checks=gradient_checks,all_test_marginal_groups=distribution_checks,independent_path_cases=cases,copied_old_score_groups=copied,metric_rows=len(metrics),lead_rows=len(lead),window_rows=len(windows),jump_rows=len(jumps),reliability_rows=len(rel),high_confidence_rows=len(high),paired_rows=len(pairs),only_october_fit=True)
 (OUT/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':run()

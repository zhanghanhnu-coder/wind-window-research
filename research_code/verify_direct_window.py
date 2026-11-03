"""Verify completed direct-window experiment without refitting research choices."""
import json,re,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from model import ART
from direct_window_core import WindowNet,predict_windows,linear_features,PRIMARY
from research_direct_window import OUT,SEEDS,CONFIGS,sha,js

def main():
 old=json.loads((OUT/'previous_hashes.json').read_text(encoding='utf-8'));assert all(sha(ART/k)==v for k,v in old.items());identity=json.loads((OUT/'identity.json').read_text(encoding='utf-8'));assert all(sha(Path(k))==v for k,v in identity.items())
 p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ART.parent,capture_output=True,text=True,encoding='utf-8',errors='replace');(OUT/'tests.log').write_text(p.stdout+p.stderr,encoding='utf-8');assert p.returncode==0,p.stderr[-2000:];nt=int(re.search(r'Ran (\d+) tests',p.stderr).group(1))
 data=np.load(ART/'data.npz');split=np.load(OUT/'split_data.npz');times=pd.DatetimeIndex(pd.to_datetime(data['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');rows=pd.read_csv(OUT/'metrics.csv');training=pd.read_csv(OUT/'training_summary.csv')
 assert np.array_equal(split['test_ids'],data['test_ids']);counts={}
 for name,start,end in [('train','2024-01-12','2024-09-01'),('val','2024-09-01','2024-10-01'),('cal','2024-10-01','2024-11-01'),('test','2024-11-01','2025-01-01')]:
  ids=split[name+'_ids'];assert (times[ids]>=pd.Timestamp(start,tz=times.tz)).all();assert (times[ids+72]<pd.Timestamp(end,tz=times.tz)).all();truth=np.stack([data['raw'][i+1:i+73] for i in ids]);expected=[]
  # Independent rolling-sum label construction.
  safe=np.where(np.isfinite(truth),truth<=10,False);sumgood=np.concatenate([np.zeros((len(ids),1,17)),np.cumsum(safe,axis=1)],axis=1);sumvalid=np.concatenate([np.zeros((len(ids),1,17)),np.cumsum(np.isfinite(truth),axis=1)],axis=1);expected=np.where(sumvalid[:,6:]-sumvalid[:,:-6]==6,(sumgood[:,6:]-sumgood[:,:-6]==6).astype(float),np.nan);np.testing.assert_equal(split[name+'_labels'],expected);counts[name]=int(np.isfinite(expected[:,PRIMARY]).sum())
 model_checks=0;best12_matches=0
 for row in training.itertuples():
  hist=json.loads((OUT/f'history_{row.model}_{row.seed}.json').read_text(encoding='utf-8'));values=np.array([r['validation_brier'] for r in hist]);assert len(hist)==25 and np.isfinite(values).all()
  assert row.best_epoch25==int(np.argmin(values))+1;assert row.best_epoch12==int(np.argmin(values[:12]))+1
  for budget in [12,25]:
   ck=torch.load(OUT/f'{row.model}_{row.seed}_budget{budget}.pt',weights_only=False,map_location='cpu');cfg=ck['config'];net=WindowNet(ck['adj'],hidden=cfg['hidden'],gated=cfg['gated']);net.load_state_dict(ck['state']);assert sum(p.numel() for p in net.parameters())==row.parameters
   pred=np.load(OUT/f'predictions_{row.model}_{row.seed}_budget{budget}.npz')
   for part in ['val','cal','test']:
    ids=split[part+'_ids'];got=predict_windows(net,data['x'],ids[[0,-1]]);np.testing.assert_allclose(got,pred[part][[0,-1]],atol=2e-6,rtol=0)
   np.testing.assert_allclose(np.nanmean((pred['val'][:,PRIMARY]-split['val_labels'][:,PRIMARY])**2),ck['validation_brier'],atol=1e-8);model_checks+=1
  if row.best_epoch12==row.best_epoch25:
   with np.load(OUT/f'predictions_{row.model}_{row.seed}_budget12.npz') as a,np.load(OUT/f'predictions_{row.model}_{row.seed}_budget25.npz') as b:
    for part in ['val','cal','test']:np.testing.assert_equal(a[part],b[part])
   best12_matches+=1
 obj=joblib.load(OUT/'logistic.joblib');ft=linear_features(data['x'],split['train_ids']);np.testing.assert_allclose(obj['scaler'].mean_,ft.astype(float).mean(0),atol=1e-10)
 pred=np.load(OUT/'predictions_logistic_single.npz')
 for part in ['val','cal','test']:
  ids=split[part+'_ids'];features=obj['scaler'].transform(linear_features(data['x'],ids[[0,-1]]));result=np.stack([obj['models'][int(a)].predict_proba(features)[:,1].reshape(2,17) for a in PRIMARY],axis=1);np.testing.assert_allclose(result,pred[part][[0,-1]],atol=1e-9)
 all_checked=0;baseline_diff=0.
 for file in OUT.glob('predictions_*.npz'):
  ident=file.stem[len('predictions_'):];parts=ident.split('_');seed=next((int(v) for v in parts if v.isdigit()),0);budget=int(parts[-1][6:]) if parts[-1].startswith('budget') else 0;model=ident.rsplit('_',2)[0] if budget else ident.rsplit('_',1)[0];pred=np.load(file);off=pred['offsets'];pi=np.array([int(np.flatnonzero(off==a)[0]) for a in PRIMARY]);errors=np.load(OUT/f'errors_{ident}.npz')
  for part in ['val','cal','test']:
   if part not in pred:continue
   y=split[part+'_labels'][:,off];assert np.isfinite(pred[part]).all() and np.min(pred[part])>=0 and np.max(pred[part])<=1;loss=(pred[part]-y)**2;np.testing.assert_equal(loss,errors[part+'_all_errors']);np.testing.assert_equal(loss[:,pi],errors[part+'_primary_errors'])
   rr=rows[(rows.model==model)&(rows.seed==seed)&(rows.budget==budget)&(rows.period==part)]
   for r in rr.itertuples():
    mask=np.ones(len(y),bool) if r.month==0 else times[split[part+'_ids']].month==r.month
    v=loss if r.metric=='brier_available_starts' else loss[:,pi] if r.metric=='brier_primary' else loss[:,int(np.flatnonzero(off==int(r.metric[len('brier_offset'):]))[0])]
    np.testing.assert_allclose(np.nanmean(v[mask]),r.value,atol=1e-12);assert np.isfinite(v[mask]).sum()==r.n;all_checked+=1
  if model=='scenario32':
   orig=np.load(ART/'v2/paired_losses.npz')['graph24'][SEEDS.index(seed)];baseline_diff=max(baseline_diff,float(np.nanmax(abs(errors['test_primary_errors']-orig))));assert baseline_diff==0
 def losses(name,seed):
  suffix=f'{seed}_budget25' if name in CONFIGS else 'single' if name in ['logistic','climatology','persistence'] else str(seed)
  with np.load(OUT/f'errors_{name}_{suffix}.npz') as f:return f['test_primary_errors']
 tt=times[split['test_ids']];week=((tt-pd.Timestamp('2024-11-01',tz=tt.tz)).days//7).to_numpy();keys=np.unique(week);draw=np.random.default_rng(20261103).integers(0,len(keys),(2000,len(keys)));paired=pd.read_csv(OUT/'paired_comparisons.csv')
 for r in paired.itertuples():
  delta=np.mean([losses(r.candidate,s)-losses(r.baseline,s) for s in SEEDS],axis=0);ss=np.array([np.nansum(delta[week==k]) for k in keys]);nn=np.array([np.isfinite(delta[week==k]).sum() for k in keys]);boot=ss[draw].sum(1)/nn[draw].sum(1);np.testing.assert_allclose([np.nanmean(delta),*np.quantile(boot,[.025,.975])],[r.difference,r.low95,r.high95],atol=1e-12)
 result=dict(tests=nt,previous_artifacts_unchanged=len(old),identity_files=len(identity),neural_trainings=len(training),checkpoints_reloaded=model_checks,budget12_25_identical=best12_matches,metric_rows_recomputed=all_checked,paired_rows_recomputed=len(paired),baseline_max_case_difference=baseline_diff,valid_primary_by_split=counts,scope='All target labels independently reconstructed; 30 checkpoints first/last inference and September-selection audit; saved per-case errors, means/counts and intervals recomputed; logistic scaler and predictions checked. Not second training.')
 js('verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

"""Independent chronological, prediction, scoring and stratified-block audit for 25."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import torch
from model import ART
from crps_distribution_core import AtomNet,predict_atoms
from research_crps_distribution import guard

OUT=ART/'seasonal_probability_v1';SEEDS=[20241101,20241102,20241103];MODELS=['gated_graph_atoms','gated_temporal_atoms','gated_mean_atoms'];METHODS=['raw','smooth_affine'];BINS=[(0,6),(6,24),(24,48),(48,72)];METRICS=['crps','brier_hourly','mae','coverage','width','low_miss','high_miss','interval_score']
DATES={'M05':['2024-03-01','2024-04-01','2024-05-01','2024-07-01'],'M07':['2024-05-01','2024-06-01','2024-07-01','2024-09-01'],'M09':['2024-07-01','2024-08-01','2024-09-01','2024-11-01'],'M11':['2024-09-01','2024-10-01','2024-11-01','2025-01-01']}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=2e-11):np.testing.assert_allclose(a,b,atol=tol,rtol=0,equal_nan=True)
def crps(q,y):
 q=np.asarray(q,np.float64);y=np.asarray(y,np.float64);e=y[...,None]-q;tau=(np.arange(19)+.5)/19
 return 2*np.maximum(tau*e,(tau-1)*e).mean(-1)
def reference_scores(q,y):
 q=np.asarray(q,np.float64);y=np.asarray(y,np.float64);valid=np.isfinite(y);p=np.count_nonzero(q<=10,axis=-1)/19;label=(y<=10).astype(float);lo=q[...,1];hi=q[...,17]
 r=dict(crps=crps(q,y),probability=p,label=label,brier_hourly=(p-label)**2,mae=abs(q[...,9]-y),coverage=(lo<=y)&(y<=hi),width=hi-lo,low_miss=y<lo,high_miss=y>hi,interval_score=hi-lo+np.where(y<lo,10*(lo-y),0)+np.where(y>hi,10*(y-hi),0))
 return {k:v if k=='probability' else np.where(valid,v,np.nan) for k,v in r.items()}
def compare_table(name,rows,keys):
 actual=pd.read_csv(OUT/name);expected=pd.DataFrame(rows);assert len(actual)==len(expected),(name,len(actual),len(expected));a=actual.sort_values(keys).reset_index(drop=True);b=expected.sort_values(keys).reset_index(drop=True)
 for k in b:
  if b[k].dtype.kind in 'biufc':close(a[k],b[k])
  else:assert a[k].fillna('').tolist()==b[k].fillna('').tolist(),(name,k)
 return len(actual)

def check_data(raw,times):
 allfold={};support=[];node_support=[];nodes=read(ART/'metadata.json')['nodes']
 for name,dates in DATES.items():
  f=dict(np.load(OUT/name/'data.npz'));cut,cal,test,end=[pd.Timestamp(x,tz=times.tz) for x in dates];fit=times<cut;values=np.stack([raw['mean'][fit],raw['raw'][fit]]);center=float(np.nanmean(values));scale=float(np.nanstd(values));assert center==float(f['center']) and scale==float(f['scale'])
  features=np.stack([raw['mean'],raw['raw']],axis=-1).astype(np.float32);valid=np.isfinite(features).all(-1,keepdims=True);x=np.concatenate([np.nan_to_num((features-center)/scale,nan=0),valid.astype(np.float32)],axis=-1);np.testing.assert_array_equal(f['x'],x)
  corr=pd.DataFrame(raw['mean'][fit]).corr(min_periods=100).fillna(0).to_numpy();np.fill_diagonal(corr,0);A=np.zeros(corr.shape)
  for j,row in enumerate(corr):
   chosen=np.argsort(row)[-4:];A[j,chosen]=np.maximum(row[chosen],0)
  # pandas correlations are column-contiguous. NumPy reductions can differ by
  # one/two float64 ULPs across layouts; prove both the exact source reduction
  # and the exact float32 matrix actually consumed by the network.
  column_order=np.array(A,order='F');column_order/=np.maximum(column_order.sum(1,keepdims=True),1e-6);np.testing.assert_array_equal(column_order,f['adj'])
  A/=np.maximum(A.sum(axis=1)[:,None],1e-6);np.testing.assert_array_max_ulp(A,f['adj'],2);np.testing.assert_array_equal(A.astype(np.float32),f['adj'].astype(np.float32));ids={}
  for split,start,stop,stride in [('train',pd.Timestamp('2024-01-12',tz=times.tz),cut,3),('val',cut,cal,6),('cal',cal,test,6),('test',test,end,6)]:
   candidates=np.arange(23,len(times)-72,stride);index=np.array([i for i in candidates if times[i]>=start and times[i+72]<stop and valid[i-23:i+1].mean()>=.65]);np.testing.assert_array_equal(index,f[split+'_ids']);ids[split]=index
   # Keep the original slice layout when reproducing its descriptive float32
   # mean; advanced indexing changes reduction order despite identical values.
   y=np.stack([raw['raw'][i+1:i+73] for i in index]);np.testing.assert_array_equal(y,np.load(OUT/name/f'{split}_truth.npz')['y']);v=np.isfinite(y)
   support.append(dict(fold=name,split=split,origins=len(index),valid_origins=int(v.any((1,2)).sum()),full_72h_allnodes=int(v.all((1,2)).sum()),valid_labels=int(v.sum()),total_labels=int(y.size),valid_fraction=float(v.mean()),mean_wind=float(np.nanmean(y)),wind_le10_rate=float(np.count_nonzero(y[v]<=10)/v.sum()),first_issue=str(times[index[0]]),last_outcome=str(times[index[-1]+72]),center=center,scale=scale))
   for j,node in enumerate(nodes):
    z=y[:,:,j];vv=np.isfinite(z);node_support.append(dict(fold=name,split=split,node=node,valid_labels=int(vv.sum()),total_labels=int(z.size),valid_fraction=float(vv.mean()),mean_wind=float(np.nanmean(z)),wind_le10_rate=float((z[vv]<=10).mean()) if vv.any() else np.nan))
   if split=='cal':assert v.any((1,2)).sum()>=20 and all(np.isfinite(y[:,a:b]).sum()>=100 for a,b in BINS)
  assert times[ids['train'][-1]+72]<cut and times[ids['val'][-1]+72]<cal and times[ids['cal'][-1]+72]<test
  if name=='M11':
   meta=read(ART/'metadata.json');assert center==meta['center'] and scale==meta['scale'];np.testing.assert_array_equal(x,raw['x']);np.testing.assert_array_equal(column_order,raw['adj']);old=np.load(ART/'crps_distribution_v1/evaluation_data.npz')
   for split in ids:np.testing.assert_array_equal(ids[split],old[split+'_ids'])
  allfold[name]=f
 compare_table('data_support.csv',support,['fold','split'])
 for row in node_support:row['node']=int(row['node'])
 compare_table('node_data_support.csv',node_support,['fold','split','node']);return allfold

def models_and_scores(folds,times):
 totals=dict(checkpoints=0,full_prediction_reloads=0,reused_prediction_groups=0,calibration_objectives=0,score_arrays=0,scalar_pairwise_crps=0);metrics=[];months=[];nodes=[];leads=[];reliability=[];calrows=[];cache={};training=pd.read_csv(OUT/'training_summary.csv');node_names=read(ART/'metadata.json')['nodes'];assert len(training)==36 and training.reused.sum()==9
 for name,f in folds.items():
  path=OUT/name;testy=np.load(path/'test_truth.npz')['y'];caly=np.load(path/'cal_truth.npz')['y'];valy=np.load(path/'val_truth.npz')['y'];n=len(f['adj']);seed_hashes={};ts=times[f['test_ids']]
  for model in MODELS:
   expected=f['adj'] if model=='gated_graph_atoms' else np.eye(n) if model=='gated_temporal_atoms' else (np.ones((n,n))-np.eye(n))/(n-1)
   for seed in SEEDS:
    guard();ck=torch.load(path/f'{model}_{seed}_budget40.pt',map_location='cpu',weights_only=False);np.testing.assert_array_equal(ck['adj'],expected);assert ck['scale']==float(f['scale']) and ck['seed']==seed and ck['budget']==40;torch.manual_seed(seed);net=AtomNet(expected,float(f['scale']),hidden=24,gated=True);ph=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in net.parameters())).hexdigest();assert seed_hashes.setdefault(seed,ph)==ph;assert sum(p.numel() for p in net.parameters())==45984;net.load_state_dict(ck['state']);pred=np.load(path/f'predictions_{model}_{seed}_budget40.npz');hist=read(path/f'history_{model}_{seed}.json');assert [h['epoch'] for h in hist]==list(range(1,41));best=min(hist,key=lambda r:r['validation_crps']);assert ck['epoch']==best['epoch'];close(ck['validation_crps'],best['validation_crps'],1e-12)
    row=training[(training.fold==name)&(training.model==model)&(training.seed==seed)].iloc[0];assert row.initial_parameter_sha256==ph and row.best_epoch==ck['epoch'] and row.parameters==45984 and row.reused==(name=='M11');close(row.validation_crps,ck['validation_crps'])
    for split in ['val','cal','test']:
     fresh=predict_atoms(net,f['x'],f[split+'_ids']);np.testing.assert_array_equal(fresh,pred[split]);totals['full_prediction_reloads']+=1
     if name=='M11':
      src=ART/('spatial_outage_controls_v1' if model=='gated_mean_atoms' else 'crps_distribution_v1');old=np.load(src/f'predictions_{model}_{seed}_budget40.npz');np.testing.assert_array_equal(fresh,old[split]);assert sha(path/f'{model}_{seed}_budget40.pt')==sha(src/f'{model}_{seed}_budget40.pt');totals['reused_prediction_groups']+=1
    close(np.nanmean(crps(pred['val'],valy)),ck['validation_crps'],1e-6);totals['checkpoints']+=1
    theta=np.load(path/f'calibration_{model}_{seed}.npz')['theta'];details=read(path/f'calibration_{model}_{seed}.json');assert theta.shape==(4,2) and np.isfinite(theta).all() and (abs(theta[:,0])<=2).all() and (theta[:,1]>=.25).all() and (theta[:,1]<=4).all()
    for j,(a,b) in enumerate(BINS):
     q=pred['cal'][:,a:b].astype(float)/float(f['scale']);y=caly[:,a:b].astype(float)/float(f['scale']);v=np.maximum(0,q[...,9,None]+theta[j,0]+theta[j,1]*(q-q[...,9,None]));obj=np.nanmean(crps(v,y))+.001*(theta[j,0]**2+(theta[j,1]-1)**2);info=details[j];close(info['objective_used'],obj,1e-10);assert info['valid_targets']==np.isfinite(y).sum() and info['lead_start']==a+1 and info['lead_end']==b;close([info['delta'],info['gamma']],theta[j],1e-14)
     if info['fallback']:np.testing.assert_array_equal(theta[j],[0,1]);assert not info['success']
     calrows.append(dict(fold=name,model=model,seed=seed,**info));totals['calibration_objectives']+=1
    anchors=np.array([3.5,15.5,36.5,60.5]);delta=np.interp(np.arange(1,73),anchors,theta[:,0])[None,:,None,None];gamma=np.interp(np.arange(1,73),anchors,theta[:,1])[None,:,None,None];rawq=pred['test'].astype(float);smooth=np.maximum(0,rawq[...,9,None]+delta*float(f['scale'])+gamma*(rawq-rawq[...,9,None]));np.testing.assert_array_equal(smooth,np.load(path/f'calibrated_{model}_{seed}.npz')['q'])
    for method,q in [('raw',rawq),('smooth_affine',smooth)]:
     assert np.isfinite(q).all() and (np.diff(q,axis=-1)>=0).all();ref=reference_scores(q,testy);actual=np.load(path/f'scores_{model}_{seed}_{method}.npz');assert set(actual.files)==set(ref)
     for k,v in ref.items():close(v,actual[k],3e-12);totals['score_arrays']+=1
     for i in [0,len(q)//2,len(q)-1]:
      for h in [0,35,71]:
       for node in [0,8,16]:
        atoms=q[i,h,node];y=float(testy[i,h,node]);value=np.mean(abs(atoms-y))-.5*np.mean(abs(atoms[:,None]-atoms[None,:]));close(value,actual['crps'][i,h,node],3e-12);totals['scalar_pairwise_crps']+=1
     cache[(name,model,seed,method)]={k:ref[k] for k in ['crps','brier_hourly']};common=dict(fold=name,model=model,seed=seed,method=method)
     for metric in METRICS:
      v=ref[metric];metrics.append(dict(**common,metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
      for m in sorted(set(ts.month)):
       z=v[ts.month==m];months.append(dict(**common,month=int(m),metric=metric,value=float(np.nanmean(z)),n=int(np.isfinite(z).sum())))
      for j,node in enumerate(node_names):
       z=v[:,:,j];nodes.append(dict(**common,node=int(node),metric=metric,value=float(np.nanmean(z)),n=int(np.isfinite(z).sum())))
      for a,b in BINS:
       z=v[:,a:b];leads.append(dict(**common,lead_start=a+1,lead_end=b,metric=metric,value=float(np.nanmean(z)),n=int(np.isfinite(z).sum())))
     valid=np.isfinite(ref['label']);p=ref['probability'][valid];y=ref['label'][valid]
     for b in range(10):
      ix=(p>=b/10)&(p<(b+1)/10 if b<9 else p<=1);reliability.append(dict(**common,bin=b,n=int(ix.sum()),mean_probability=float(p[ix].mean()) if ix.any() else np.nan,observed_rate=float(y[ix].mean()) if ix.any() else np.nan))
    print('AUDITED',name,model,seed,flush=True)
 df=pd.DataFrame(metrics)
 for group,g in df.groupby(['model','seed','method','metric']):metrics.append(dict(fold='ALL',**dict(zip(['model','seed','method','metric'],group)),value=float(np.dot(g.value,g.n)/g.n.sum()),n=int(g.n.sum())))
 keys=['fold','model','seed','method','metric'];tables={}
 for filename,rows,k in [('metrics.csv',metrics,keys),('month_metrics.csv',months,keys+['month']),('node_metrics.csv',nodes,keys+['node']),('lead_metrics.csv',leads,keys+['lead_start']),('reliability.csv',reliability,keys[:-1]+['bin']),('calibration.csv',calrows,['fold','model','seed','lead_start'])]:tables[filename]=compare_table(filename,rows,k)
 totals['table_rows']=tables;return totals,cache

def comparisons(folds,times,cache):
 rng=np.random.default_rng(20261113);weights={};weeks={}
 for name,f in folds.items():
  w=((times[f['test_ids']]-pd.Timestamp(DATES[name][2],tz=times.tz)).days//7).to_numpy();weeks[name]=w;k=int(w.max())+1;draw=rng.integers(0,k,(2000,k));weights[name]=np.stack([np.bincount(row,minlength=k)[w] for row in draw])
 df=pd.read_csv(OUT/'paired_comparisons.csv');assert len(df)==70 and df.primary.sum()==4
 for r in df.itertuples():
  selected=list(folds) if r.scope=='ALL' else [r.scope];num=np.zeros(2000);den=np.zeros(2000);ss=0.;nn=0;blocks=[];origins=0
  for name in selected:
   a=np.stack([cache[(name,r.candidate,s,r.method)][r.metric] for s in SEEDS]);b=np.stack([cache[(name,r.baseline,s,r.baseline_method)][r.metric] for s in SEEDS]);np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b));diff=(a-b).mean(0);sums=np.nansum(diff,(1,2));counts=np.isfinite(diff).sum((1,2));num+=weights[name]@sums;den+=weights[name]@counts;ss+=sums.sum();nn+=counts.sum();blocks.append(len(np.unique(weeks[name][counts>0])));origins+=len(sums)
  ok=den>0;eligible=min(blocks)>=4 and ok.sum()>=1900;lo,hi=np.quantile(num[ok]/den[ok],[.025,.975]) if eligible else [np.nan,np.nan];close(r.difference,ss/nn);close(r.low95,lo);close(r.high95,hi);assert r.n==nn and r.origins==origins and r.valid_draws==ok.sum() and r.nonempty_blocks==sum(blocks) and r.minimum_blocks_in_fold==min(blocks) and r.interval_supported==eligible;assert r.primary==(r.scope=='ALL' and r.method==r.baseline_method=='smooth_affine')
 return len(df)

def run():
 guard();previous=read(OUT/'previous_hashes.json');ident=read(OUT/'identity.json')
 for p,h in previous.items():assert sha(ART/p)==h,p
 for p,h in ident.items():assert sha(Path(p))==h,p
 phase=read(OUT/'prediction_complete.json');assert phase['models']==36 and phase['methods']==2 and phase['calibration_groups']==144
 d=dict(np.load(ART/'data.npz'));times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');folds=check_data(d,times);totals,cache=models_and_scores(folds,times);totals['paired_comparisons']=comparisons(folds,times,cache);totals.update(status='PASS',previous_unchanged=len(previous),frozen_identities=len(ident),chronological_folds=4,training_new=27,training_reused=9,evaluation_origins=sum(len(f['test_ids']) for f in folds.values()));(OUT/'verification.json').write_text(json.dumps(totals,indent=2),encoding='utf-8');print(json.dumps(totals,indent=2),flush=True)
if __name__=='__main__':run()

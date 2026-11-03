"""Frozen sequential replay; all new outputs isolated in rolling_v1."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import torch
from model import ART,Net,origins,predict
from probability_core import score_scenarios
from rolling_core import mature_paths,event_matrix,mature_events,fit_map,map_probability,tilt_weights,OFFSETS
from research_probability import SEEDS,METRICS

OUT=ART/'rolling_v1';PLAN=ART.parent.parent/'研究笔记/滚动校准研究方案_2026-11-01.md'
JOINT=['frozen32','frozen58','rolling28_32','rolling28','rolling14','expanding','blend28','tilted28'];METHODS=JOINT+['direct28']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')

def prepare():
 OUT.mkdir(exist_ok=True);frozen=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0];p=OUT/'protocol_frozen.md'
 if p.exists():assert p.read_text(encoding='utf-8')==frozen
 else:p.write_text(frozen,encoding='utf-8')
 old=OUT/'previous_hashes.json'
 if not old.exists():js('previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.pt','.npz','.csv','.joblib']})
 files=[Path(__file__),Path(__file__).with_name('rolling_core.py'),Path(__file__).with_name('probability_core.py'),ART/'data.npz',ART/'metadata.json',ART/'probability_v1/predictions.npz',p]+[ART/'v2'/('graph24.pt' if s==SEEDS[0] else f'graph24_{s}.pt') for s in SEEDS]
 identity={str(p):sha(p) for p in files};ip=OUT/'identity.json'
 if ip.exists():assert json.loads(ip.read_text(encoding='utf-8'))==identity,'new version required for changed code/data'
 else:js('identity.json',identity)

def generate_data():
 d=np.load(ART/'data.npz');old=np.load(ART/'probability_v1/predictions.npz');meta=json.loads((ART/'metadata.json').read_text(encoding='utf-8'));times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai')
 ids=origins(times,'2024-10-01','2025-01-01',6);ids=np.array([i for i in ids if d['x'][i-23:i+1,:,-1].mean()>=.65]);truth=np.stack([d['raw'][i+1:i+73] for i in ids]);lookup={i:j for j,i in enumerate(ids)}
 te=np.array([lookup[i] for i in d['test_ids']]);cal=np.array([lookup[i] for i in d['cal_ids']]);start=int(np.flatnonzero(times==pd.Timestamp('2024-10-16',tz=times.tz))[0])
 data=dict(ids=ids,truth=truth,test_positions=te,static_positions=cal,warmup_start=np.array(start))
 reused=[]
 for s in SEEDS:
  ck=torch.load(ART/'v2'/('graph24.pt' if s==SEEDS[0] else f'graph24_{s}.pt'),weights_only=False,map_location='cpu');net=Net(ck['adj'],dilation=ck['dilation']);net.load_state_dict(ck['state']);q=predict(net,d['x'],ids)*ck['scale']+ck['center']
  # Keep exactly the archived forecast values on historical comparisons.
  for part in ['cal','test']:
   positions=np.array([lookup[i] for i in old[part+'_ids']]);diff=float(np.max(abs(q[positions]-old[f'{part}_q_{s}'])));assert diff<2e-5
   q[positions]=old[f'{part}_q_{s}'];reused.append(dict(seed=s,part=part,n=len(positions),max_inference_difference=diff))
  data[f'q_{s}']=q
 np.savez_compressed(OUT/'data.npz',**data);csv('prediction_reuse.csv',reused)
 js('counts.json',dict(archive_origins=len(ids),test_origins=len(te),warmup_origins=int(((ids>=start)&(ids<ids[te[0]])).sum()),complete_archive_paths=int(np.isfinite(truth).all((1,2)).sum()),static_paths=len(cal),first_test=str(times[ids[te[0]]]),last_test=str(times[ids[te[-1]]])))
 return data,times,meta

def window_truth(y):
 return np.stack([np.where(np.isfinite(y[a:a+6]).all(0),(y[a:a+6]<=10).all(0),np.nan) for a in OFFSETS])

def generate_scenarios(data,q,k,paths,static):
 med=q[k,...,1];y=data['truth'];res=y-q[...,1]
 r28=np.maximum(0,med+res[paths['28']]);r14=np.maximum(0,med+res[paths['14']]);exp=np.maximum(0,med+res[paths['all']]);fix=np.maximum(0,med+res[static]);rng=np.random.default_rng(20241101);sel=rng.choice(len(fix),32,replace=len(fix)<32);sel28=np.random.default_rng(20241101).choice(len(r28),32,replace=len(r28)<32)
 return {'frozen32':(fix[sel],None),'frozen58':(fix,None),'rolling28_32':(r28[sel28],None),'rolling28':(r28,None),'rolling14':(r14,None),'expanding':(exp,None),'blend28':(np.concatenate([fix,r28]),np.r_[np.full(len(fix),.5/len(fix)),np.full(len(r28),.5/len(r28))])}

def run(data,times,meta):
 ids=data['ids'];truth=data['truth'];start=int(data['warmup_start']);testset=set(data['test_positions']);complete=np.isfinite(truth).all((1,2));static=data['static_positions'];membership={}
 labels=np.stack([window_truth(y) for y in truth]);all_metrics=[];reliability=[];high=[];offsetrows=[];audit=[];all_bankrows=[]
 for s in SEEDS:
  q=data[f'q_{s}'];res=truth-q[...,1];history_p=np.full((len(ids),4,17),np.nan);sizes=np.zeros(len(ids),int);store={m:[] for m in METHODS};theta_store=[];weight_store={};case_store={}
  for k,issue in enumerate(ids):
   if issue<start:continue
   paths={name:mature_paths(ids,complete,issue,days) for name,days in [('14',14),('28',28),('all',None)]}
   if min(len(v) for v in paths.values())<8:raise ValueError('insufficient mature complete paths')
   membership[str(int(issue))]={name:idx.tolist() for name,idx in paths.items()}
   sc=np.maximum(0,q[k,...,1]+res[paths['28']]);E=event_matrix(sc);p=E.mean(0).reshape(4,17);sizes[k]=len(sc)
   # Calibration sees previously issued probabilities only; current is appended afterwards.
   if k in testset:
    hist=np.flatnonzero((ids>=start)&(ids<issue));x,y,norig,latest=mature_events(ids[hist],history_p[hist],labels[hist],sizes[hist],int(issue),start)
    assert np.isfinite(x).all() and (latest is None or latest<=issue-1)
    theta,fitinfo=fit_map(x,y,norig);target=map_probability(p,len(sc),theta);w,ti=tilt_weights(E,target)
    samples=generate_scenarios(data,q,k,paths,static);samples['tilted28']=(sc,w);theta_store.append(theta);weight_store[f'issue_{issue}']=w
    for m,(sample,weights) in samples.items():store[m].append(score_scenarios(sample,truth[k],weights,meta['scale']))
    label=labels[k];valid=np.isfinite(label);store['direct28'].append(dict(brier6=np.where(valid,(target-label)**2,np.nan),prob6=np.where(valid,target,np.nan),label6=label))
    row=dict(seed=s,issue=int(issue),time=str(times[issue]),map_a=float(theta[0]),map_b=float(theta[1]),map_n=fitinfo['n'],map_origins=norig,map_success=fitinfo['success'],map_fallback=fitinfo['fallback'],latest_window_label=latest,map_reason=fitinfo['reason'],**{'tilt_'+key:value for key,value in ti.items()})
    row.update(target_mean=float(target.mean()),raw_probability_mean=float(p.mean()),tilted_probability_mean=float((E.T@w).mean()));audit.append(row)
    if len(store['frozen32']) in [1,117,232] and s==SEEDS[0]:
     number=len(store['frozen32'])-1
     for m,(sample,ww) in samples.items():case_store[f'case{number}_{m}']=sample;case_store[f'case{number}_{m}_weights']=np.ones(len(sample))/len(sample) if ww is None else ww
     case_store[f'case{number}_target']=target;case_store[f'case{number}_truth']=truth[k]
    if s==SEEDS[0]:
     for name,idx in paths.items():
      requested=None if name=='all' else int(name);count_recent=len(idx) if requested is None else int(((ids+72<=issue-1)&complete&(ids+72>=issue-requested*24)).sum())
      all_bankrows.append(dict(issue=int(issue),time=str(times[issue]),bank=name,n=len(idx),recent_n=count_recent,fallback=count_recent<8,last_label=int(np.max(ids[idx]+72)),first_label=int(np.min(ids[idx]+72)),resample32=len(idx)<32 if name=='28' else False))
    if len(store['frozen32'])%60==0:print(s,'evaluated',len(store['frozen32']),flush=True)
   history_p[k]=p
  for m in METHODS:
   arrays={name:np.stack([r[name] for r in store[m]]) for name in store[m][0]};np.savez_compressed(OUT/f'scores_{m}_{s}.npz',**arrays)
   tt=times[ids[data['test_positions']]]
   for month in [0,11,12]:
    mask=np.ones(len(tt),bool) if month==0 else tt.month==month
    for metric in METRICS:
     if metric not in arrays:continue
     a=arrays[metric][mask];all_metrics.append(dict(seed=s,method=m,month=month,metric=metric,value=float(np.nanmean(a)),n=int(np.isfinite(a).sum())))
    pp=arrays['prob6'][mask].ravel();yy=arrays['label6'][mask].ravel();valid=np.isfinite(yy);pp=pp[valid];yy=yy[valid]
    for b in range(10):
     bm=(pp>=b/10)&((pp<(b+1)/10) if b<9 else pp<=1)
     reliability.append(dict(seed=s,method=m,month=month,bin=b,n=int(bm.sum()),mean_probability=float(pp[bm].mean()) if bm.any() else np.nan,observed_rate=float(yy[bm].mean()) if bm.any() else np.nan))
    bm=pp>=.8;high.append(dict(seed=s,method=m,month=month,n=int(bm.sum()),fraction=float(bm.mean()),mean_probability=float(pp[bm].mean()) if bm.any() else np.nan,observed_rate=float(yy[bm].mean()) if bm.any() else np.nan))
    for j,a in enumerate(OFFSETS):
     v=arrays['brier6'][mask,j];offsetrows.append(dict(seed=s,method=m,month=month,offset=int(a),brier=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
   print(s,m,'Brier',np.nanmean(arrays['brier6']),flush=True)
  np.savez_compressed(OUT/f'calibration_history_{s}.npz',probabilities=history_p,sizes=sizes,labels=labels,theta=np.stack(theta_store))
  np.savez_compressed(OUT/f'tilt_weights_{s}.npz',**weight_store)
  if s==SEEDS[0]:np.savez_compressed(OUT/'fixed_case_scenarios.npz',**case_store)
 csv('metrics.csv',all_metrics);csv('reliability.csv',reliability);csv('high_confidence.csv',high);csv('offset_metrics.csv',offsetrows);csv('calibration_audit.csv',audit);csv('bank_audit.csv',all_bankrows);js('membership.json',membership)

def compare(times):
 week=((times-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();keys=np.unique(week);draw=np.random.default_rng(20261102).integers(0,len(keys),(2000,len(keys)));rows=[]
 pairs=[(m,'frozen32') for m in METHODS[1:]]+[('rolling28','frozen58'),('tilted28','rolling28'),('direct28','rolling28')]
 metrics=['brier6','crps','interval_score','length_crps24','length_crps48','length_crps72','any6_brier24','any6_brier48','any6_brier72','first_cdf_brier','group_brier','spatial_variogram','temporal_variogram']
 for method,base in pairs:
  for metric in (['brier6'] if method=='direct28' else metrics):
   dd=[]
   for s in SEEDS:
    with np.load(OUT/f'scores_{method}_{s}.npz') as a,np.load(OUT/f'scores_{base}_{s}.npz') as b:
     assert np.array_equal(np.isfinite(a[metric]),np.isfinite(b[metric]));dd.append(a[metric]-b[metric])
   diff=np.mean(dd,axis=0);ss=np.array([np.nansum(diff[week==k]) for k in keys]);nn=np.array([np.isfinite(diff[week==k]).sum() for k in keys]);boot=ss[draw].sum(1)/nn[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975])
   rows.append(dict(candidate=method,baseline=base,metric=metric,difference=float(np.nanmean(diff)),low95=lo,high95=hi,blocks=len(keys),n=int(np.isfinite(diff).sum()),primary_contrast=(method,base) in [('rolling28_32','frozen32'),('tilted28','rolling28')] and metric=='brier6'))
 csv('paired_comparisons.csv',rows)

if __name__=='__main__':
 prepare();data,times,meta=generate_data();run(data,times,meta);compare(times[data['ids'][data['test_positions']]]);print('COMPLETE',flush=True)

"""Post-processing experiment. Does not train or overwrite any historical model."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
import torch
from model import ART,Net,origins,predict
from probability_core import *

OUT=ART/'probability_v1'
PLAN=ART.parent.parent/'研究笔记/概率联合情景研究方案_2026-11-01.md'
SEEDS=[20241101,20241102,20241103]
METHODS=['joint32','joint_all','scaled_joint','pooled_ecc','tuned_ecc','analog_joint','time_broken','space_broken']
GAMMAS=[.75,1.,1.25,1.5];KS=[8,16,32]
METRICS=['brier6','crps','coverage','width','interval_score','low_miss','high_miss','brier3','brier12','length_crps24','length_crps48','length_crps72','any6_brier24','any6_brier48','any6_brier72','first_cdf_brier','group_brier','spatial_variogram','temporal_variogram','ess']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(frame,name):pd.DataFrame(frame).to_csv(OUT/name,index=False,encoding='utf-8-sig')

def prepare():
 OUT.mkdir(exist_ok=True)
 frozen=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0]
 pf=OUT/'protocol_frozen.md'
 if pf.exists():assert pf.read_text(encoding='utf-8')==frozen,'frozen protocol changed'
 else:pf.write_text(frozen,encoding='utf-8')
 old=OUT/'previous_hashes.json'
 if not old.exists():write_json(old,{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 files=[ART/'data.npz',ART/'metadata.json',Path(__file__),Path(__file__).with_name('probability_core.py'),pf]
 files += [ART/'v2'/('graph24.pt' if s==SEEDS[0] else f'graph24_{s}.pt') for s in SEEDS]
 identity={str(p):sha(p) for p in files}
 ip=OUT/'identity.json'
 if ip.exists():assert json.loads(ip.read_text(encoding='utf-8'))==identity,'source identity changed; inspect before rerun'
 else:write_json(ip,identity)

def prediction_data():
 d=np.load(ART/'data.npz');meta=json.loads((ART/'metadata.json').read_text(encoding='utf-8'))
 t=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');x=d['x'];raw=d['raw']
 cal=origins(t,'2024-10-01','2024-11-01',6);cal=np.array([i for i in cal if x[i-23:i+1,:,-1].mean()>=.65]);te=d['test_ids']
 truth=np.stack([raw[i+1:i+73] for i in cal]);testtruth=np.stack([raw[i+1:i+73] for i in te]);ctx=context(raw,cal);testctx=context(raw,te)
 boundary=np.flatnonzero(t==pd.Timestamp('2024-10-16',tz=t.tz))[0];fit,val=calibration_split(cal,72,boundary)
 assert np.max(cal[fit]+72)<np.min(cal[val]);assert np.max(cal+72)<np.min(te)
 assert np.array_equal(cal[np.isfinite(truth).all(axis=(1,2))],d['cal_ids'])
 data=dict(cal_ids=cal,test_ids=te,cal_truth=truth,test_truth=testtruth,cal_context=ctx,test_context=testctx,inner_fit=fit,inner_val=val)
 for s in SEEDS:
  ck=torch.load(ART/'v2'/('graph24.pt' if s==SEEDS[0] else f'graph24_{s}.pt'),weights_only=False,map_location='cpu')
  net=Net(ck['adj'],dilation=ck['dilation']);net.load_state_dict(ck['state'])
  data[f'cal_q_{s}']=predict(net,x,cal)*ck['scale']+ck['center'];data[f'test_q_{s}']=predict(net,x,te)*ck['scale']+ck['center']
 np.savez_compressed(OUT/'predictions.npz',**data)
 write_json(OUT/'sample_counts.json',dict(cal_origins=len(cal),cal_complete=int(np.isfinite(truth).all((1,2)).sum()),inner_fit_origins=int(fit.sum()),inner_fit_complete=int(np.isfinite(truth[fit]).all((1,2)).sum()),inner_val_origins=int(val.sum()),inner_val_complete=int(np.isfinite(truth[val]).all((1,2)).sum()),purged_origins=int((~(fit|val)).sum()),test_origins=len(te),test_complete_all_nodes=int(np.isfinite(testtruth).all((1,2)).sum()),calibration_first=str(t[cal[0]]),calibration_last_label=str(t[cal[-1]+72]),test_first=str(t[te[0]]),test_last=str(t[te[-1]])))
 return data,t,meta

def tune(data):
 rows=[];fit=data['inner_fit'];val=data['inner_val'];yy=data['cal_truth'][val]
 for s in SEEDS:
  bank=make_bank(data[f'cal_q_{s}'][fit],data['cal_truth'][fit],data['cal_context'][fit]);q=data[f'cal_q_{s}'][val]
  for gamma in GAMMAS:
   templ=ecc_template(bank['standard'],bank['pools'],gamma);total=0.;count=0
   for qq,y in zip(q,yy):
    loss=crps(np.maximum(0,qq[...,1]+spread(qq)*templ),y);total+=np.nansum(loss);count+=np.isfinite(loss).sum()
   rows.append(dict(seed=s,parameter='gamma',value=gamma,metric='crps',score=total/count,n=count))
  for k in KS:
   total=0.;count=0
   for qq,y,c in zip(q,yy,data['cal_context'][val]):
    sc=np.maximum(0,qq[...,1]+bank['residual']);w=analog_weights(bank['contexts'],c,k)
    for a in [0,24,48,66]:
     obs=y[a:a+6];valid=np.isfinite(obs).all(0);label=(obs<=10).all(0);p=window_prob(sc,6,a,w);loss=(p-label)**2
     total+=loss[valid].sum();count+=valid.sum()
   rows.append(dict(seed=s,parameter='k',value=k,metric='brier6',score=total/count,n=count))
 df=pd.DataFrame(rows);csv(df,'inner_selection_trials.csv');selected={}
 for key in ['gamma','k']:
  scores=df[df.parameter==key].groupby('value',sort=True).score.mean();selected[key]=float(scores.idxmin())
 selected['k']=int(selected['k']);write_json(OUT/'selected_parameters.json',selected);return selected

def evaluate(data,t,meta,selected):
 rows=[];reliability=[];confidence=[];subgroups=[];times=t[data['test_ids']]
 for s in SEEDS:
  bank=make_bank(data[f'cal_q_{s}'],data['cal_truth'],data['cal_context']);store={m:[] for m in METHODS}
  for i,(q,y,ctx) in enumerate(zip(data[f'test_q_{s}'],data['test_truth'],data['test_context'])):
   samples=scenarios(bank,q,ctx,**selected)
   for m,(sc,w) in samples.items():store[m].append(score_scenarios(sc,y,w,meta['scale']))
   if i%60==0:print(f'seed {s}: evaluated {i}/{len(times)} origins',flush=True)
  for m in METHODS:
   arrays={k:np.stack([r[k] for r in store[m]]) for k in store[m][0]};np.savez_compressed(OUT/f'scores_{m}_{s}.npz',**arrays)
   for month in [0,11,12]:
    select=np.ones(len(times),bool) if month==0 else times.month==month
    for metric in METRICS:
     a=arrays[metric][select];rows.append(dict(seed=s,method=m,month=month,metric=metric,value=float(np.nanmean(a)),n=int(np.isfinite(a).sum())))
    p=arrays['prob6'][select].ravel();y=arrays['label6'][select].ravel();valid=np.isfinite(y);p=p[valid];y=y[valid]
    for b in range(10):
     mask=(p>=b/10)&((p<(b+1)/10) if b<9 else (p<=1))
     reliability.append(dict(seed=s,method=m,month=month,bin=b,n=int(mask.sum()),mean_probability=float(p[mask].mean()) if mask.any() else None,observed_rate=float(y[mask].mean()) if mask.any() else None))
    mask=p>=.8;confidence.append(dict(seed=s,method=m,month=month,n=int(mask.sum()),total=len(p),fraction=float(mask.mean()),mean_probability=float(p[mask].mean()) if mask.any() else None,observed_rate=float(y[mask].mean()) if mask.any() else None))
   for a,b in BINS:
    for metric in ['crps','coverage','width','interval_score','low_miss','high_miss']:
     v=arrays[metric][:,a:b];subgroups.append(dict(seed=s,method=m,lead_start=a+1,lead_end=b,metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
   print(s,m,'Brier',np.nanmean(arrays['brier6']),'CRPS',np.nanmean(arrays['crps']),flush=True)
 csv(rows,'metrics.csv');csv(reliability,'reliability_bins.csv');csv(confidence,'high_confidence.csv');csv(subgroups,'lead_metrics.csv')

def comparisons(times):
 weeks=((times-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();keys=np.unique(weeks);draw=np.random.default_rng(20261101).integers(0,len(keys),(2000,len(keys)));rows=[]
 chosen=['brier6','crps','interval_score','length_crps24','length_crps48','length_crps72','any6_brier24','any6_brier48','any6_brier72','first_cdf_brier','group_brier','spatial_variogram','temporal_variogram']
 for baseline,candidates in [('joint32',METHODS[1:]),('tuned_ecc',['time_broken','space_broken'])]:
  for m in candidates:
   for metric in chosen:
    delta=[]
    for s in SEEDS:
     with np.load(OUT/f'scores_{m}_{s}.npz') as aa,np.load(OUT/f'scores_{baseline}_{s}.npz') as bb:
      assert np.array_equal(np.isfinite(aa[metric]),np.isfinite(bb[metric]));delta.append(aa[metric]-bb[metric])
    diff=np.mean(delta,axis=0);sums=np.array([np.nansum(diff[weeks==k]) for k in keys]);counts=np.array([np.isfinite(diff[weeks==k]).sum() for k in keys]);boot=sums[draw].sum(1)/counts[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975])
    rows.append(dict(candidate=m,baseline=baseline,metric=metric,difference=float(np.nanmean(diff)),low95=lo,high95=hi,blocks=len(keys),bootstrap=2000,n=int(np.isfinite(diff).sum())))
 csv(rows,'paired_comparisons.csv')

def main():
 prepare();data,t,meta=prediction_data();selected=tune(data);print('SELECTED',selected,flush=True)
 evaluate(data,t,meta,selected);comparisons(t[data['test_ids']]);print('COMPLETE',flush=True)

if __name__=='__main__':main()

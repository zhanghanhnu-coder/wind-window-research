"""Round16, fixed lead calibration on frozen round15 distributions."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import pandas as pd
from model import ART
from lead_calibration_core import fit_groups,apply_groups,jump_summary,BINS
from crps_distribution_core import couple_atoms
from probability_core import score_scenarios
from research_crps_distribution import CONFIGS,SEEDS,METRICS,COMPARE_METRICS,guard,scenario_generator as old_scenarios
SRC=ART/'crps_distribution_v1';OUT=ART/'crps_lead_calibration_v1';PLAN=ART.parent.parent/'研究笔记/分提前量分布校准研究方案_2026-11-01.md'
NEW=['bin_affine','smooth_affine','bin_shift']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')

def prepare():
 guard();OUT.mkdir(exist_ok=True);f=OUT/'protocol_frozen.md';protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0]
 if f.exists():assert f.read_text(encoding='utf-8')==protocol
 else:f.write_text(protocol,encoding='utf-8')
 if not (OUT/'previous_hashes.json').exists():js('previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 files=[f,Path(__file__)]+[Path(__file__).with_name(n) for n in ['lead_calibration_core.py','crps_distribution_core.py','research_crps_distribution.py','probability_core.py']]+[SRC/'identity.json',SRC/'evaluation_data.npz',ART/'metadata.json',ART/'data.npz']+sorted(SRC.glob('predictions_*budget40.npz'))+sorted(SRC.glob('bank_*.npz'))
 identity={str(p):sha(p) for p in files};ip=OUT/'identity.json'
 if ip.exists():assert read(ip.name)==identity
 else:js(ip.name,identity)
 shutil.copy2(SRC/'evaluation_data.npz',OUT/'evaluation_data.npz')

def fit(meta,ev):
 records=[];inventory=[];complete=ev['complete_positions'];truth=ev['cal_truth'][complete]
 for seed in SEEDS:
  for model in CONFIGS:
   guard();done=OUT/f'calibration_{model}_{seed}.json'
   if done.exists():saved=read(done.name)
   else:
    q=np.load(SRC/f'predictions_{model}_{seed}_budget40.npz')['cal'][complete];theta,info=fit_groups(q,truth,meta['scale']);shift,sinfo=fit_groups(q,truth,meta['scale'],True);saved=dict(affine=theta.tolist(),shift=shift.tolist(),affine_info=info,shift_info=sinfo);js(done.name,saved)
   np.savez_compressed(OUT/f'parameters_{model}_{seed}.npz',affine=np.array(saved['affine']),shift=np.array(saved['shift']))
   for kind in ['affine','shift']:
    for j,row in enumerate(saved[kind+'_info']):records.append(dict(model=model,seed=seed,kind=kind,group=j,**row))
   for method in [*NEW,'global','raw_ecc57','residual57']:inventory.append(dict(id=f'{model}__{method}__{seed}',model=model,method=method,seed=seed))
   print('fitted',model,seed,flush=True)
  for method in ['old57','old32']:inventory.append(dict(id=f'old_graph__{method}__{seed}',model='old_graph',method=method,seed=seed))
 csv('calibration.csv',records);js('inventory.json',inventory)

def paths(item,meta,ev):
 if item['method'] not in NEW:
  old=dict(item);old['method']='affine_ecc57' if item['method']=='global' else item['method'];yield from old_scenarios(old,meta,ev['complete_positions'],ev['selected57']);return
 pred=np.load(SRC/f'predictions_{item["model"]}_{item["seed"]}_budget40.npz');bank=np.load(SRC/f'bank_{item["model"]}_{item["seed"]}.npz');params=np.load(OUT/f'parameters_{item["model"]}_{item["seed"]}.npz');theta=params['shift' if item['method']=='bin_shift' else 'affine']
 for q in pred['test']:yield couple_atoms(apply_groups(q,theta,meta['scale'],item['method']=='smooth_affine'),bank['ranks'])

def evaluate(meta,ev):
 raw=np.load(ART/'data.npz');times=pd.DatetimeIndex(pd.to_datetime(raw['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai')[ev['test_ids']];truth=ev['test_truth'];rows=[];reliability=[];high=[];leads=[];window=[];jumps=[]
 actual=[jump_summary(y[None]) for y in truth];np.savez_compressed(OUT/'actual_jumps.npz',**{k:np.stack([r[k] for r in actual]) for k in actual[0]})
 for item in read('inventory.json'):
  guard();dest=OUT/f'scores_{item["id"]}.npz';jfile=OUT/f'jumps_{item["id"]}.npz'
  if item['method'] not in NEW:
   method='affine_ecc57' if item['method']=='global' else item['method'];shutil.copy2(SRC/f'scores_{item["model"]}__{method}__{item["seed"]}.npz',dest)
  existing=dest.exists();jj=[];ss=[]
  if not existing or not jfile.exists():
   for i,(sc,y) in enumerate(zip(paths(item,meta,ev),truth)):
    jj.append(jump_summary(sc))
    if not existing:ss.append(score_scenarios(sc,y,train_scale=meta['scale']))
    if i%100==0:guard();print(item['id'],'origin',i,flush=True)
   assert len(jj)==len(times);np.savez_compressed(jfile,**{k:np.stack([r[k] for r in jj]) for k in jj[0]})
   if not existing:np.savez_compressed(dest,**{k:np.stack([r[k] for r in ss]) for k in ss[0]})
  scores=dict(np.load(dest));jump=dict(np.load(jfile))
  for month in [0,11,12]:
   mask=np.ones(len(times),bool) if month==0 else times.month==month
   for metric in METRICS:
    a=scores[metric][mask];rows.append(dict(**item,month=month,metric=metric,value=float(np.nanmean(a)),n=int(np.isfinite(a).sum())))
   p=scores['prob6'][mask].ravel();y=scores['label6'][mask].ravel();valid=np.isfinite(y);p=p[valid];y=y[valid]
   for b in range(10):
    mm=(p>=b/10)&(p<(b+1)/10 if b<9 else p<=1);reliability.append(dict(**item,month=month,bin=b,n=int(mm.sum()),mean_probability=float(p[mm].mean()) if mm.any() else np.nan,observed_rate=float(y[mm].mean()) if mm.any() else np.nan))
   mm=p>=.8;high.append(dict(**item,month=month,n=int(mm.sum()),fraction=float(mm.mean()),mean_probability=float(p[mm].mean()) if mm.any() else np.nan,observed_rate=float(y[mm].mean()) if mm.any() else np.nan))
  for a,b in BINS:
   for metric in ['crps','coverage','width','interval_score','low_miss','high_miss']:
    values=scores[metric][:,a:b];leads.append(dict(**item,lead_start=a+1,lead_end=b,metric=metric,value=float(np.nanmean(values)),n=int(np.isfinite(values).sum())))
  for j,off in enumerate([0,24,48,66]):window.append(dict(**item,offset=off,brier6=float(np.nanmean(scores['brier6'][:,j])),mean_probability=float(np.nanmean(scores['prob6'][:,j])),observed_rate=float(np.nanmean(scores['label6'][:,j]))))
  jumps.append(dict(**item,boundary6_7=float(jump['boundaries'][:,0].mean()),boundary24_25=float(jump['boundaries'][:,1].mean()),boundary48_49=float(jump['boundaries'][:,2].mean()),other_mean=float(jump['other_mean'].mean())))
  print('RESULT',item['id'],'Brier',np.nanmean(scores['brier6']),'CRPS',np.nanmean(scores['crps']),flush=True)
 csv('metrics.csv',rows);csv('reliability.csv',reliability);csv('high_confidence.csv',high);csv('lead_metrics.csv',leads);csv('windows_by_lead.csv',window);csv('jumps.csv',jumps);return times

def compare(times):
 pairs=[('gated_graph_atoms','bin_affine','gated_graph_atoms','global',True),('gated_graph_atoms','bin_affine','old_graph','old57',True),('gated_graph_atoms','bin_affine','gated_temporal_atoms','bin_affine',True)]
 for model in CONFIGS:
  if model!='gated_graph_atoms':pairs.append((model,'bin_affine',model,'global',False))
  pairs.extend([(model,'smooth_affine',model,'bin_affine',False),(model,'bin_shift',model,'bin_affine',False),(model,'bin_affine',model,'residual57',False)])
 pairs.extend([('gated_graph_atoms','bin_affine',m,'bin_affine',False) for m in ['plain_graph_atoms','wide_plain_graph_atoms']]);pairs.append(('old_graph','old57','old_graph','old32',False));week=((times-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();keys=np.unique(week);draw=np.random.default_rng(20261107).integers(0,len(keys),(2000,len(keys)));rows=[]
 for model,method,base,bmethod,primary in pairs:
  for metric in COMPARE_METRICS:
   delta=[]
   for seed in SEEDS:
    with np.load(OUT/f'scores_{model}__{method}__{seed}.npz') as aa,np.load(OUT/f'scores_{base}__{bmethod}__{seed}.npz') as bb:
     assert np.array_equal(np.isfinite(aa[metric]),np.isfinite(bb[metric]));delta.append(aa[metric]-bb[metric])
   diff=np.mean(delta,0);sums=np.array([np.nansum(diff[week==k]) for k in keys]);counts=np.array([np.isfinite(diff[week==k]).sum() for k in keys]);boot=sums[draw].sum(1)/counts[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975]);rows.append(dict(candidate=model,method=method,baseline=base,baseline_method=bmethod,metric=metric,difference=float(np.nanmean(diff)),low95=lo,high95=hi,blocks=len(keys),primary=primary and metric in ['crps','brier6']))
 csv('paired_comparisons.csv',rows)

if __name__=='__main__':
 prepare();meta=json.loads((ART/'metadata.json').read_text(encoding='utf-8'));ev=np.load(OUT/'evaluation_data.npz');fit(meta,ev);compare(evaluate(meta,ev));print('COMPLETE',flush=True)

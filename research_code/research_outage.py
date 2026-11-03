"""Round21: frozen models under controlled single-target historical outages."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import pandas as pd
import torch
from model import ART,Net,predict
from crps_distribution_core import AtomNet
from outage_core import eligible_cases,target_predict,target_paths,node_scores
from research_crps_distribution import guard,SEEDS
OUT=ART/'sensor_outage_v1';PLAN=ART.parent.parent/'研究笔记/传感器历史缺测与图信息研究方案_2026-11-02.md';MODELS=['graph24','temporal24','gated_graph_atoms','gated_temporal_atoms'];LENGTHS=[0,6,12,24];METRICS=['crps','brier6','length_crps72','first_cdf_brier','coverage','width','low_miss','high_miss']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def checkpoint(model,seed):return ART/'v2'/(model+('' if seed==SEEDS[0] else f'_{seed}')+'.pt') if model in MODELS[:2] else ART/'crps_distribution_v1'/f'{model}_{seed}_budget40.pt'
def network(model,seed):
 ck=torch.load(checkpoint(model,seed),map_location='cpu',weights_only=False);net=Net(ck['adj'],dilation=ck['dilation']) if model in MODELS[:2] else AtomNet(ck['adj'],ck['scale'],hidden=ck['config']['hidden'],gated=ck['config']['gated']);net.load_state_dict(ck['state']);net.eval();return net,ck

def prepare():
 guard();OUT.mkdir(exist_ok=True);f=OUT/'protocol_frozen.md';protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0]
 if f.exists():assert f.read_text(encoding='utf-8')==protocol
 else:f.write_text(protocol,encoding='utf-8')
 if not (OUT/'previous_hashes.json').exists():js('previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 files=[f,Path(__file__),Path(__file__).with_name('outage_core.py'),ART/'data.npz',ART/'metadata.json',ART/'probability_v1/predictions.npz',ART/'crps_distribution_v1/evaluation_data.npz']+[Path(__file__).with_name(n) for n in ['model.py','direct_window_core.py','crps_distribution_core.py','lead_calibration_core.py','probability_core.py']]
 for model in MODELS:
  for seed in SEEDS:
   files.append(checkpoint(model,seed))
   if model in MODELS[2:]:files.extend([ART/'crps_lead_calibration_v1'/f'parameters_{model}_{seed}.npz',ART/'crps_distribution_v1'/f'bank_{model}_{seed}.npz',ART/'crps_distribution_v1'/f'predictions_{model}_{seed}_budget40.npz'])
 identity={str(p):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==identity
 else:js('identity.json',identity)
 if not (OUT/'recipe.json').exists():js('recipe.json',dict(models=MODELS,seeds=SEEDS,outage_hours=LENGTHS,scenario_count=57,masking='target last hours, all three features zero',qualification='target history24 complete and remaining farm fraction>=.65',fit='none; frozen clean October calibration',start_utc=datetime.now(timezone.utc).isoformat()))

def inputs():
 d=np.load(ART/'data.npz');ids=d['test_ids'];histories=np.stack([d['x'][i-23:i+1] for i in ids]);eligible=eligible_cases(histories);positions,targets=np.where(eligible);ev=np.load(ART/'crps_distribution_v1/evaluation_data.npz');assert max(ev['selected57_ids']+72)<min(ids);info=dict(histories=histories,eligible=eligible,positions=positions,targets=targets,issue_ids=ids[positions],timestamps=d['timestamps'][ids[positions]],source_cal_ids=ev['selected57_ids'],selected57=ev['selected57']);path=OUT/'input_data.npz'
 if not path.exists():np.savez_compressed(path,**info)
 else:
  saved=np.load(path)
  for k,v in info.items():np.testing.assert_array_equal(v,saved[k])
 js('input_identity.json',{str(path):sha(path)});full=(histories[...,2]==1).all(1);js('eligibility.json',dict(potential_cases=int(eligible.size),eligible_cases=int(eligible.sum()),original_incomplete_target_history=int((~full).sum()),excluded_by_farm_quality=int((full&~eligible).sum()),origins=len(ids),nodes=histories.shape[2]));return info,d

def forecast_all(info,d):
 histories=info['histories'][info['positions']];targets=info['targets'];meta=read(ART/'metadata.json');old=np.load(ART/'probability_v1/predictions.npz');ev=np.load(ART/'crps_distribution_v1/evaluation_data.npz');checks=[]
 for model in MODELS:
  for seed in SEEDS:
   guard();net,ck=network(model,seed);ordinary=model in MODELS[:2];bankpath=OUT/f'bank_{model}_{seed}.npz'
   if not bankpath.exists():
    if ordinary:
     if model=='graph24':residual=(old['cal_truth']-old[f'cal_q_{seed}'][...,1])[ev['complete_positions']][info['selected57']]
     else:
      cal_ids=d['cal_ids'];cal_q=predict(net,d['x'],cal_ids)*ck['scale']+ck['center'];truth=np.stack([d['raw'][i+1:i+73] for i in cal_ids]);residual=(truth-cal_q[...,1])[info['selected57']]
     assert np.isfinite(residual).all();np.savez_compressed(bankpath,residual=residual)
    else:
     bank=np.load(ART/'crps_distribution_v1'/f'bank_{model}_{seed}.npz');params=np.load(ART/'crps_lead_calibration_v1'/f'parameters_{model}_{seed}.npz');np.savez_compressed(bankpath,ranks=bank['ranks'],theta=params['affine'],scale=meta['scale'])
   for length in LENGTHS:
    guard();path=OUT/f'predictions_{model}_{seed}_h{length}.npz'
    if not path.exists():
     q=target_predict(net,histories,targets,length)
     if ordinary:q=q*ck['scale']+ck['center']
     np.savez_compressed(path,q=q);print('PREDICTED',model,seed,length,len(q),flush=True)
   q=np.load(OUT/f'predictions_{model}_{seed}_h0.npz')['q']
   if model=='graph24':ref=old[f'test_q_{seed}'][info['positions'],:,targets,:]
   elif ordinary:ref=predict(net,d['x'],d['test_ids'])*ck['scale']+ck['center'];ref=ref[info['positions'],:,targets,:]
   else:ref=np.load(ART/'crps_distribution_v1'/f'predictions_{model}_{seed}_budget40.npz')['test'][info['positions'],:,targets,:]
   diff=float(np.max(abs(q-ref)));np.testing.assert_allclose(q,ref,atol=1e-5,rtol=0);checks.append(dict(model=model,seed=seed,maximum_absolute_difference=diff,values=q.size))
 js('clean_prediction_checks.json',checks);js('prediction_complete.json',dict(flows=48,utc=datetime.now(timezone.utc).isoformat()))

def score_all(info,d):
 assert (OUT/'prediction_complete.json').exists();truth=np.stack([d['raw'][i+1:i+73,j] for i,j in zip(info['issue_ids'],info['targets'])]);np.savez_compressed(OUT/'evaluation_truth.npz',truth=truth);rows=[];reliability=[];high=[];times=pd.DatetimeIndex(pd.to_datetime(info['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai')
 for model in MODELS:
  for seed in SEEDS:
   bank=dict(np.load(OUT/f'bank_{model}_{seed}.npz'))
   for length in LENGTHS:
    guard();path=OUT/f'scores_{model}_{seed}_h{length}.npz'
    if not path.exists():
     q=np.load(OUT/f'predictions_{model}_{seed}_h{length}.npz')['q'];scores=[]
     for i,(qi,yi,node) in enumerate(zip(q,truth,info['targets'])):
      scores.append(node_scores(target_paths(qi,bank,int(node),model in MODELS[:2]),yi))
      if i%500==0:guard()
     np.savez_compressed(path,**{k:np.stack([s[k] for s in scores]) for k in scores[0]})
    data=np.load(path)
    for month in [0,11,12]:
     mask=np.ones(len(times),bool) if month==0 else times.month==month;common=dict(model=model,seed=seed,outage_hours=length,month=month)
     for metric in METRICS:a=data[metric][mask];rows.append(dict(**common,metric=metric,value=float(np.nanmean(a)),n=int(np.isfinite(a).sum())))
     prob=data['prob6'][mask].ravel();label=data['label6'][mask].ravel();valid=np.isfinite(label);prob=prob[valid];label=label[valid];m=prob>=.8;high.append(dict(**common,n=int(m.sum()),fraction=float(m.mean()),mean_probability=float(prob[m].mean()) if m.any() else np.nan,observed_rate=float(label[m].mean()) if m.any() else np.nan))
     for b in range(10):
      m=(prob>=b/10)&(prob<(b+1)/10 if b<9 else prob<=1);reliability.append(dict(**common,bin=b,n=int(m.sum()),mean_probability=float(prob[m].mean()) if m.any() else np.nan,observed_rate=float(label[m].mean()) if m.any() else np.nan))
    print('SCORED',model,seed,length,'Brier',np.nanmean(data['brier6']),'CRPS',np.nanmean(data['crps']),flush=True)
 csv('metrics.csv',rows);csv('high_confidence.csv',high);csv('reliability.csv',reliability)

def compare(info):
 times=pd.DatetimeIndex(pd.to_datetime(info['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');weeks=((times-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();keys=np.arange(9);draw=np.random.default_rng(20261110).integers(0,len(keys),(2000,len(keys)));rows=[]
 def interval(diff):
  sums=np.array([np.nansum(diff[weeks==k]) for k in keys]);counts=np.array([np.isfinite(diff[weeks==k]).sum() for k in keys]);den=counts[draw].sum(1);valid=den>0;boot=sums[draw][valid].sum(1)/den[valid];lo,hi=np.quantile(boot,[.025,.975]);return dict(difference=float(np.nanmean(diff)),low95=lo,high95=hi,blocks=int(sum(counts>0)),valid_draws=int(sum(valid)),n=int(np.isfinite(diff).sum()))
 for model in MODELS:
  for length in LENGTHS[1:]:
   for metric in METRICS:
    delta=[]
    for seed in SEEDS:
     a=np.load(OUT/f'scores_{model}_{seed}_h{length}.npz')[metric];b=np.load(OUT/f'scores_{model}_{seed}_h0.npz')[metric];np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b));delta.append(a-b)
    rows.append(dict(kind='own_degradation',candidate=model,baseline=model,outage_hours=length,metric=metric,primary=False,**interval(np.mean(delta,0))))
 for graph,own in [('graph24','temporal24'),('gated_graph_atoms','gated_temporal_atoms')]:
  for length in LENGTHS:
   for metric in METRICS:
    raw=[];did=[]
    for seed in SEEDS:
     a=np.load(OUT/f'scores_{graph}_{seed}_h{length}.npz')[metric];b=np.load(OUT/f'scores_{own}_{seed}_h{length}.npz')[metric];a0=np.load(OUT/f'scores_{graph}_{seed}_h0.npz')[metric];b0=np.load(OUT/f'scores_{own}_{seed}_h0.npz')[metric];np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b));raw.append(a-b);did.append((a-a0)-(b-b0))
    rows.append(dict(kind='absolute_graph_difference',candidate=graph,baseline=own,outage_hours=length,metric=metric,primary=False,**interval(np.mean(raw,0))))
    if length:rows.append(dict(kind='difference_in_degradation',candidate=graph,baseline=own,outage_hours=length,metric=metric,primary=length==24 and metric in ['crps','brier6'],**interval(np.mean(did,0))))
 csv('paired_comparisons.csv',rows)

if __name__=='__main__':
 prepare();info,d=inputs();forecast_all(info,d);score_all(info,d);compare(info);print('COMPLETE',flush=True)

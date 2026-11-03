"""Round15: proper finite-distribution training and full-path dependence controls."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,time
import numpy as np
import pandas as pd
import torch
from model import ART
from crps_distribution_core import AtomNet,masked_crps,predict_atoms,fit_affine,transform_atoms,rank_template,couple_atoms
from probability_core import score_scenarios,break_dependence,BINS
OUT=ART/'crps_distribution_v1';SRC=ART/'direct_window_v1';PLAN=ART.parent.parent/'研究笔记/CRPS分布训练研究方案_2026-11-01.md'
SEEDS=[20241101,20241102,20241103]
CONFIGS={'plain_graph_atoms':dict(hidden=24,gated=False,graph=True),'wide_plain_graph_atoms':dict(hidden=27,gated=False,graph=True),'gated_temporal_atoms':dict(hidden=24,gated=True,graph=False),'gated_graph_atoms':dict(hidden=24,gated=True,graph=True)}
DEADLINE=datetime(2026,11,2,5,23,15,tzinfo=timezone.utc)
METRICS=['brier6','crps','coverage','width','interval_score','low_miss','high_miss','brier3','brier12','length_crps24','length_crps48','length_crps72','any6_brier24','any6_brier48','any6_brier72','first_cdf_brier','group_brier','spatial_variogram','temporal_variogram','ess']
COMPARE_METRICS=['brier6','crps','length_crps72','any6_brier72','first_cdf_brier','group_brier','spatial_variogram','temporal_variogram']
def guard():
 if datetime.now(timezone.utc)>=DEADLINE:raise RuntimeError('15-hour authorized experiment deadline reached')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))

def prepare():
 guard();OUT.mkdir(exist_ok=True);f=OUT/'protocol_frozen.md';protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0]
 if f.exists():assert f.read_text(encoding='utf-8')==protocol
 else:f.write_text(protocol,encoding='utf-8')
 if not (OUT/'previous_hashes.json').exists():js('previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 files=[f,Path(__file__)]+[Path(__file__).with_name(n) for n in ['crps_distribution_core.py','direct_window_core.py','probability_core.py','model.py']]+[ART/'data.npz',ART/'metadata.json',SRC/'split_data.npz',ART/'probability_v1/predictions.npz']
 ident={str(p):sha(p) for p in files};ip=OUT/'identity.json'
 if ip.exists():assert read(ip.name)==ident,'frozen source changed; create a new version'
 else:js(ip.name,ident)
 if not (OUT/'recipe.json').exists():js('recipe.json',dict(configs=CONFIGS,seeds=SEEDS,atoms=19,scenarios=57,epochs=40,secondary_budget=20,batch=32,lr=.001,gradient_clip=1.,selection='September hourly CRPS',deadline_utc=DEADLINE.isoformat(),start_utc=datetime.now(timezone.utc).isoformat()))

def data_load():
 d=np.load(ART/'data.npz');meta=json.loads((ART/'metadata.json').read_text(encoding='utf-8'));sp=np.load(SRC/'split_data.npz');ids={k:sp[k+'_ids'] for k in ['train','val','cal','test']};truth={k:np.stack([d['raw'][i+1:i+73] for i in v]) for k,v in ids.items()};complete=np.isfinite(truth['cal']).all((1,2));positions=np.flatnonzero(complete);np.testing.assert_array_equal(ids['cal'][positions],d['cal_ids']);assert len(positions)==58 and max(ids['cal']+72)<min(ids['test'])
 selected=np.random.default_rng(20261105).choice(58,57,replace=False);np.savez_compressed(OUT/'evaluation_data.npz',**{k+'_ids':v for k,v in ids.items()},cal_truth=truth['cal'],test_truth=truth['test'],complete_positions=positions,selected57=selected,selected57_ids=d['cal_ids'][selected])
 return d,meta,ids,truth,positions,selected

def train(d,meta,ids,truth):
 xx=torch.from_numpy(np.stack([d['x'][i-23:i+1] for i in ids['train']]));yy=torch.from_numpy(truth['train']);yv=torch.from_numpy(truth['val']);scale=meta['scale'];rows=[]
 for seed in SEEDS:
  for name,cfg in CONFIGS.items():
   guard();done=OUT/f'completed_{name}_{seed}.json'
   if done.exists():rows.append(read(done.name));print('resume trained',name,seed,flush=True);continue
   torch.manual_seed(seed);A=d['adj'] if cfg['graph'] else np.eye(len(d['adj']));net=AtomNet(A,scale,hidden=cfg['hidden'],gated=cfg['gated']);optimizer=torch.optim.Adam(net.parameters(),lr=.001);best=float('inf');best20=float('inf');history=[];start=time.monotonic()
   for epoch in range(1,41):
    guard();net.train();order=np.random.default_rng(epoch).permutation(len(xx));total=0.;count=0
    for a in range(0,len(xx),32):
     ix=order[a:a+32];optimizer.zero_grad();loss=masked_crps(net(xx[ix]),yy[ix])/scale;assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(grad);optimizer.step();total+=loss.item()*len(ix);count+=len(ix)
    pred=predict_atoms(net,d['x'],ids['val']);val=float(masked_crps(torch.from_numpy(pred),yv));record=dict(epoch=epoch,train_normalized_crps=total/count,validation_crps=val,elapsed_seconds=time.monotonic()-start);history.append(record)
    if val<best:best=val;state=copy.deepcopy(net.state_dict());bestepoch=epoch
    if epoch<=20 and val<best20:best20=val;state20=copy.deepcopy(net.state_dict());bestepoch20=epoch
    js(f'history_{name}_{seed}.json',history);print(seed,name,'epoch',epoch,'val',round(val,6),'seconds',round(time.monotonic()-start,1),flush=True)
   for budget,st,epoch,vc in [(40,state,bestepoch,best),(20,state20,bestepoch20,best20)]:
    net.load_state_dict(st);torch.save(dict(state=st,adj=A,scale=scale,config=cfg,seed=seed,epoch=epoch,validation_crps=vc,budget=budget),OUT/f'{name}_{seed}_budget{budget}.pt');values={k:predict_atoms(net,d['x'],ids[k]) for k in ['val','cal','test']};np.savez_compressed(OUT/f'predictions_{name}_{seed}_budget{budget}.npz',**values)
   row=dict(model=name,seed=seed,parameters=sum(p.numel() for p in net.parameters()),best_epoch40=bestepoch,best_epoch20=bestepoch20,validation_crps40=best,validation_crps20=best20,seconds=time.monotonic()-start);js(done.name,row);rows.append(row)
 csv('training_summary.csv',rows)

def calibrate(meta,truth,complete,selected):
 rows=[];inventory=[]
 for seed in SEEDS:
  for name in CONFIGS:
   guard();params=OUT/f'calibration_{name}_{seed}.json';pp=np.load(OUT/f'predictions_{name}_{seed}_budget40.npz');cal=pp['cal'][complete]
   if params.exists():info=read(params.name);theta=np.array([info['delta'],info['gamma']])
   else:theta,info=fit_affine(cal,truth['cal'][complete],meta['scale']);js(params.name,info)
   residual=truth['cal'][complete]-cal[...,9];r=residual[selected];ranks=rank_template(r);np.savez_compressed(OUT/f'bank_{name}_{seed}.npz',residual=r,ranks=ranks,theta=theta);rows.append(dict(model=name,seed=seed,**info,tied_adjacent=int((np.diff(np.sort(r,axis=0),axis=0)==0).sum())))
   methods=['raw_ecc57','affine_ecc57','residual57']+(['time_broken','space_broken'] if name=='gated_graph_atoms' else [])
   for method in methods:inventory.append(dict(id=f'{name}__{method}__{seed}',model=name,method=method,seed=seed))
   print('calibrated',name,seed,'theta',theta,'success',info['success'],flush=True)
  for method in ['old57','old32']:inventory.append(dict(id=f'old_graph__{method}__{seed}',model='old_graph',method=method,seed=seed))
 csv('calibration.csv',rows);js('inventory.json',inventory)

def scenario_generator(item,meta,complete,selected):
 name=item['model'];seed=item['seed'];method=item['method']
 if name=='old_graph':
  old=np.load(ART/'probability_v1/predictions.npz');r=(old['cal_truth']-old[f'cal_q_{seed}'][...,1])[complete];idx=selected if method=='old57' else np.random.default_rng(20241101).choice(58,32,replace=False)
  for med in old[f'test_q_{seed}'][...,1]:yield np.maximum(0,med+r[idx])
 else:
  pack=np.load(OUT/f'predictions_{name}_{seed}_budget40.npz');bank=np.load(OUT/f'bank_{name}_{seed}.npz')
  for q in pack['test']:
   if method=='residual57':sc=np.maximum(0,q[...,9]+bank['residual'])
   else:
    atoms=q if method=='raw_ecc57' else transform_atoms(q,bank['theta'],meta['scale']);sc=couple_atoms(atoms,bank['ranks'])
    if method in ['time_broken','space_broken']:sc=break_dependence(sc,method.split('_')[0])
   yield sc

def evaluate(d,meta,ids,truth,complete,selected):
 times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai')[ids['test']];rows=[];reliability=[];high=[];lead=[]
 for item in read('inventory.json'):
  guard();path=OUT/f'scores_{item["id"]}.npz'
  if path.exists():arrays=dict(np.load(path));print('resume scores',item['id'],flush=True)
  else:
   records=[]
   for i,(sc,y) in enumerate(zip(scenario_generator(item,meta,complete,selected),truth['test'])):
    records.append(score_scenarios(sc,y,train_scale=meta['scale']))
    if i%80==0:guard();print(item['id'],'scored',i,flush=True)
   assert len(records)==len(times);arrays={k:np.stack([r[k] for r in records]) for k in records[0]};np.savez_compressed(path,**arrays)
  for month in [0,11,12]:
   mask=np.ones(len(times),bool) if month==0 else times.month==month
   for metric in METRICS:
    v=arrays[metric][mask];rows.append(dict(**item,month=month,metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
   p=arrays['prob6'][mask].ravel();y=arrays['label6'][mask].ravel();valid=np.isfinite(y);p=p[valid];y=y[valid]
   for b in range(10):
    mm=(p>=b/10)&(p<(b+1)/10 if b<9 else p<=1);reliability.append(dict(**item,month=month,bin=b,n=int(mm.sum()),mean_probability=float(p[mm].mean()) if mm.any() else np.nan,observed_rate=float(y[mm].mean()) if mm.any() else np.nan))
   mm=p>=.8;high.append(dict(**item,month=month,n=int(mm.sum()),fraction=float(mm.mean()),mean_probability=float(p[mm].mean()) if mm.any() else np.nan,observed_rate=float(y[mm].mean()) if mm.any() else np.nan))
  for a,b in BINS:
   for metric in ['crps','coverage','width','interval_score','low_miss','high_miss']:
    v=arrays[metric][:,a:b];lead.append(dict(**item,lead_start=a+1,lead_end=b,metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
  print('RESULT',item['id'],'CRPS',np.nanmean(arrays['crps']),'Brier6',np.nanmean(arrays['brier6']),flush=True)
 csv('metrics.csv',rows);csv('reliability.csv',reliability);csv('high_confidence.csv',high);csv('lead_metrics.csv',lead);return times

def compare(times):
 pairs=[('gated_graph_atoms','affine_ecc57','old_graph','old57',True),('gated_graph_atoms','affine_ecc57','gated_temporal_atoms','affine_ecc57',True)]
 for name in CONFIGS:
  pairs.extend([(name,'affine_ecc57',name,'raw_ecc57',False),(name,'affine_ecc57',name,'residual57',False),(name,'residual57','old_graph','old57',False)])
 pairs.extend([('gated_graph_atoms','affine_ecc57',n,'affine_ecc57',False) for n in ['plain_graph_atoms','wide_plain_graph_atoms']]);pairs.append(('old_graph','old57','old_graph','old32',False));pairs.extend([('gated_graph_atoms',m,'gated_graph_atoms','affine_ecc57',False) for m in ['time_broken','space_broken']])
 weeks=((times-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();keys=np.unique(weeks);draw=np.random.default_rng(20261105).integers(0,len(keys),(2000,len(keys)));rows=[]
 for model,method,base,bmethod,primary in pairs:
  for metric in COMPARE_METRICS:
   delta=[]
   for seed in SEEDS:
    with np.load(OUT/f'scores_{model}__{method}__{seed}.npz') as aa,np.load(OUT/f'scores_{base}__{bmethod}__{seed}.npz') as bb:
     assert np.array_equal(np.isfinite(aa[metric]),np.isfinite(bb[metric]));delta.append(aa[metric]-bb[metric])
   diff=np.mean(delta,0);ss=np.array([np.nansum(diff[weeks==k]) for k in keys]);nn=np.array([np.isfinite(diff[weeks==k]).sum() for k in keys]);boot=ss[draw].sum(1)/nn[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975]);rows.append(dict(candidate=model,method=method,baseline=base,baseline_method=bmethod,metric=metric,difference=float(np.nanmean(diff)),low95=lo,high95=hi,blocks=len(keys),primary=primary and metric in ['crps','brier6']))
 csv('paired_comparisons.csv',rows)

if __name__=='__main__':
 prepare();d,meta,ids,truth,complete,selected=data_load();train(d,meta,ids,truth);calibrate(meta,truth,complete,selected);compare(evaluate(d,meta,ids,truth,complete,selected));print('COMPLETE',flush=True)

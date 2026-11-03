"""Round22: specific graph structure versus common-farm information under outage."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,time,shutil
import numpy as np
import pandas as pd
import torch
from model import ART,Net,predict
from crps_distribution_core import AtomNet,masked_crps,predict_atoms,rank_template
from lead_calibration_core import fit_groups
from spatial_controls_core import control_graphs,make_control_network
from outage_core import target_predict,target_paths,node_scores
from research_crps_distribution import guard,SEEDS
OUT=ART/'spatial_outage_controls_v1';SOURCE=ART/'sensor_outage_v1';PLAN=ART.parent.parent/'研究笔记/缺测条件下空间信息来源对照方案_2026-11-02.md';MODELS=['mean24','random24','gated_mean_atoms','gated_random_atoms'];LENGTHS=[0,6,12,24];METRICS=['crps','brier6','length_crps72','first_cdf_brier','coverage','width','low_miss','high_miss']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def old_checkpoint(model,seed):return ART/'v2'/(model+('' if seed==SEEDS[0] else f'_{seed}')+'.pt')

def prepare():
 guard();assert (SOURCE/'final_manifest.json').exists();OUT.mkdir(exist_ok=True);f=OUT/'protocol_frozen.md';protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0]
 if f.exists():assert f.read_text(encoding='utf-8')==protocol
 else:f.write_text(protocol,encoding='utf-8')
 if not (OUT/'previous_hashes.json').exists():js('previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 files=[f,Path(__file__)]+[Path(__file__).with_name(n) for n in ['spatial_controls_core.py','outage_core.py','model.py','direct_window_core.py','crps_distribution_core.py','lead_calibration_core.py','probability_core.py','research_crps_distribution.py']]+[ART/'data.npz',ART/'metadata.json',ART/'crps_distribution_v1/evaluation_data.npz',ART/'direct_window_v1/split_data.npz',SOURCE/'input_data.npz',SOURCE/'final_manifest.json']+[old_checkpoint(m,s) for m in MODELS[:2] for s in SEEDS]
 identity={str(p):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==identity
 else:js('identity.json',identity)
 shutil.copy2(SOURCE/'input_data.npz',OUT/'input_data.npz');d=np.load(ART/'data.npz');graphs,perm=control_graphs(d['adj'])
 for kind in graphs:
  for seed in SEEDS:np.testing.assert_allclose(graphs[kind],torch.load(old_checkpoint(kind+'24',seed),weights_only=False,map_location='cpu')['adj'],atol=1e-7,rtol=0)
 if not (OUT/'recipe.json').exists():js('recipe.json',dict(models=MODELS,seeds=SEEDS,epochs=40,batch=32,lr=.001,gradient_clip=1.,hidden=24,atoms=19,scenarios=57,selection='September hourly CRPS only',permutation=perm.tolist(),outage_hours=LENGTHS,graphs={k:v.tolist() for k,v in graphs.items()},start_utc=datetime.now(timezone.utc).isoformat()))
 return d,graphs

def train(d,graphs):
 meta=read(ART/'metadata.json');ev=np.load(ART/'crps_distribution_v1/evaluation_data.npz');ids={k:ev[k+'_ids'] for k in ['train','val','cal','test']};x=d['x'];raw=d['raw'];xx=torch.from_numpy(np.stack([x[i-23:i+1] for i in ids['train']]));yy=torch.from_numpy(np.stack([raw[i+1:i+73] for i in ids['train']]));vy=torch.from_numpy(np.stack([raw[i+1:i+73] for i in ids['val']]));rows=[]
 for model in MODELS[2:]:
  kind='mean' if 'mean' in model else 'random';A=graphs[kind]
  for seed in SEEDS:
   done=OUT/f'trained_{model}_{seed}.json'
   if done.exists():rows.append(read(done));continue
   guard();net=make_control_network(A,meta['scale'],seed);opt=torch.optim.Adam(net.parameters(),lr=.001);best=float('inf');history=[];start=time.monotonic()
   for epoch in range(1,41):
    guard();net.train();order=np.random.default_rng(epoch).permutation(len(xx));total=0.;count=0
    for a in range(0,len(xx),32):
     ix=order[a:a+32];opt.zero_grad();loss=masked_crps(net(xx[ix]),yy[ix])/meta['scale'];assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(grad);opt.step();total+=loss.item()*len(ix);count+=len(ix)
    pred=predict_atoms(net,x,ids['val']);value=float(masked_crps(torch.from_numpy(pred),vy));history.append(dict(epoch=epoch,train_normalized_crps=total/count,validation_crps=value,elapsed_seconds=time.monotonic()-start))
    if value<best:best=value;state=copy.deepcopy(net.state_dict());bestepoch=epoch
    js(f'history_{model}_{seed}.json',history);print('TRAIN',model,seed,epoch,'val',round(value,6),flush=True)
   net.load_state_dict(state);torch.save(dict(state=state,adj=A,scale=meta['scale'],config=dict(hidden=24,gated=True,graph_kind=kind),seed=seed,epoch=bestepoch,validation_crps=best,budget=40),OUT/f'{model}_{seed}_budget40.pt');np.savez_compressed(OUT/f'predictions_{model}_{seed}_budget40.npz',**{k:predict_atoms(net,x,ids[k]) for k in ['val','cal','test']});row=dict(model=model,seed=seed,parameters=sum(p.numel() for p in net.parameters()),best_epoch=bestepoch,validation_crps=best,seconds=time.monotonic()-start);js(done.name,row);rows.append(row)
 csv('training_summary.csv',rows)

def forecasts(d):
 info=np.load(OUT/'input_data.npz');meta=read(ART/'metadata.json');ev=np.load(ART/'crps_distribution_v1/evaluation_data.npz');hist=info['histories'][info['positions']];targets=info['targets'];complete=ev['complete_positions'];selected=info['selected57'];calibration=[];checks=[]
 for model in MODELS:
  ordinary=model in MODELS[:2]
  for seed in SEEDS:
   guard();ck=torch.load(old_checkpoint(model,seed) if ordinary else OUT/f'{model}_{seed}_budget40.pt',map_location='cpu',weights_only=False);net=Net(ck['adj'],dilation=ck['dilation']) if ordinary else AtomNet(ck['adj'],ck['scale'],hidden=24,gated=True);net.load_state_dict(ck['state']);net.eval();bankpath=OUT/f'bank_{model}_{seed}.npz'
   if not bankpath.exists():
    if ordinary:
     cm=predict(net,d['x'],d['cal_ids'])[...,1]*ck['scale']+ck['center'];truth=np.stack([d['raw'][i+1:i+73] for i in d['cal_ids']]);np.savez_compressed(bankpath,residual=(truth-cm)[selected])
    else:
     cp=np.load(OUT/f'predictions_{model}_{seed}_budget40.npz')['cal'];theta,details=fit_groups(cp[complete],ev['cal_truth'][complete],meta['scale']);residual=ev['cal_truth'][complete][selected]-cp[complete][selected,...,9];np.savez_compressed(bankpath,ranks=rank_template(residual),residual=residual,theta=theta,scale=meta['scale']);js(f'calibration_{model}_{seed}.json',details)
   if not ordinary:
    for row in read(OUT/f'calibration_{model}_{seed}.json'):calibration.append(dict(model=model,seed=seed,**row))
   for length in LENGTHS:
    path=OUT/f'predictions_{model}_{seed}_h{length}.npz'
    if not path.exists():
     guard();q=target_predict(net,hist,targets,length)
     if ordinary:q=q*ck['scale']+ck['center']
     np.savez_compressed(path,q=q);print('PREDICTED',model,seed,length,flush=True)
   q=np.load(OUT/f'predictions_{model}_{seed}_h0.npz')['q'];ref=predict(net,d['x'],d['test_ids'])*ck['scale']+ck['center'] if ordinary else np.load(OUT/f'predictions_{model}_{seed}_budget40.npz')['test'];ref=ref[info['positions'],:,targets,:];np.testing.assert_allclose(q,ref,atol=1e-5,rtol=0);checks.append(dict(model=model,seed=seed,max_difference=float(np.max(abs(q-ref)))))
 csv('calibration.csv',calibration);js('clean_prediction_checks.json',checks);js('prediction_complete.json',dict(flows=48,utc=datetime.now(timezone.utc).isoformat()))

def score():
 assert (OUT/'prediction_complete.json').exists();info=np.load(OUT/'input_data.npz');truth=np.load(SOURCE/'evaluation_truth.npz')['truth'];times=pd.DatetimeIndex(pd.to_datetime(info['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');rows=[];high=[];rel=[]
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
      m=(prob>=b/10)&(prob<(b+1)/10 if b<9 else prob<=1);rel.append(dict(**common,bin=b,n=int(m.sum()),mean_probability=float(prob[m].mean()) if m.any() else np.nan,observed_rate=float(label[m].mean()) if m.any() else np.nan))
    print('SCORED',model,seed,length,'Brier',np.nanmean(data['brier6']),'CRPS',np.nanmean(data['crps']),flush=True)
 csv('metrics.csv',rows);csv('high_confidence.csv',high);csv('reliability.csv',rel)

def compare():
 info=np.load(OUT/'input_data.npz');times=pd.DatetimeIndex(pd.to_datetime(info['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');weeks=((times-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();draw=np.random.default_rng(20261110).integers(0,9,(2000,9));rows=[]
 for graph,controls in [('graph24',MODELS[:2]),('gated_graph_atoms',MODELS[2:])]:
  for control in controls:
   for length in LENGTHS:
    for metric in METRICS:
     raw=[];did=[]
     for seed in SEEDS:
      a=np.load(SOURCE/f'scores_{graph}_{seed}_h{length}.npz')[metric];b=np.load(OUT/f'scores_{control}_{seed}_h{length}.npz')[metric];np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b));raw.append(a-b)
      if length:did.append((a-np.load(SOURCE/f'scores_{graph}_{seed}_h0.npz')[metric])-(b-np.load(OUT/f'scores_{control}_{seed}_h0.npz')[metric]))
     for kind,values in [('absolute_graph_difference',raw)]+([('difference_in_degradation',did)] if length else []):
      diff=np.mean(values,0);sums=np.array([np.nansum(diff[weeks==k]) for k in range(9)]);counts=np.array([np.isfinite(diff[weeks==k]).sum() for k in range(9)]);den=counts[draw].sum(1);valid=den>0;boot=sums[draw][valid].sum(1)/den[valid];lo,hi=np.quantile(boot,[.025,.975]);rows.append(dict(kind=kind,candidate=graph,baseline=control,outage_hours=length,metric=metric,primary=kind=='absolute_graph_difference' and length==24 and metric in ['crps','brier6'],difference=float(np.nanmean(diff)),low95=lo,high95=hi,blocks=int(sum(counts>0)),valid_draws=int(sum(valid)),n=int(np.isfinite(diff).sum())))
 csv('paired_comparisons.csv',rows)

if __name__=='__main__':
 prepare_data,graphs=prepare();train(prepare_data,graphs);forecasts(prepare_data);score();compare();print('COMPLETE',flush=True)

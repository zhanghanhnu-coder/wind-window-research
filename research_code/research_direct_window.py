"""Pre-fixed direct-window study, isolated from all historical experiments."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,copy,time,warnings
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from model import ART,origins
from direct_window_core import WindowNet,window_labels,masked_loss,predict_windows,linear_features,persistence_prob,PRIMARY

OUT=ART/'direct_window_v1';PLAN=ART.parent.parent/'研究笔记/直接窗口预测研究方案_2026-11-01.md'
SEEDS=[20241101,20241102,20241103]
CONFIGS={'plain_graph_bce':dict(hidden=24,gated=False,graph=True,loss='bce'),'wide_plain_graph_bce':dict(hidden=32,gated=False,graph=True,loss='bce'),'gated_temporal_bce':dict(hidden=24,gated=True,graph=False,loss='bce'),'gated_graph_bce':dict(hidden=24,gated=True,graph=True,loss='bce'),'gated_graph_brier':dict(hidden=24,gated=True,graph=True,loss='brier')}
DEADLINE=datetime(2026,11,2,5,23,15,tzinfo=timezone.utc)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,value):pd.DataFrame(value).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def time_guard():
 if datetime.now(timezone.utc)>=DEADLINE:raise RuntimeError('15-hour deadline reached; preserve outputs and stop new work')

def prepare():
 OUT.mkdir(exist_ok=True);protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0];f=OUT/'protocol_frozen.md'
 if f.exists():assert f.read_text(encoding='utf-8')==protocol
 else:f.write_text(protocol,encoding='utf-8')
 old=OUT/'previous_hashes.json'
 if not old.exists():js(old.name,{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 files=[f,Path(__file__),Path(__file__).with_name('direct_window_core.py'),ART/'data.npz',ART/'metadata.json',ART/'probability_v1/predictions.npz']
 identity={str(p):sha(p) for p in files};i=OUT/'identity.json'
 if i.exists():assert json.loads(i.read_text(encoding='utf-8'))==identity,'source changed: inspect and create new version'
 else:js(i.name,identity)
 js('recipe.json',dict(configs=CONFIGS,seeds=SEEDS,epochs=25,secondary_epoch_cap=12,batch=32,lr=.001,loss_selection='September Brier at offsets 0/24/48/66',deadline_utc=DEADLINE.isoformat(),started_utc=datetime.now(timezone.utc).isoformat()))

def load_splits():
 d=np.load(ART/'data.npz');x=d['x'];raw=d['raw'];times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');old=np.load(ART/'probability_v1/predictions.npz')
 splits={k:origins(times,a,b,stride) for k,a,b,stride in [('train','2024-01-12','2024-09-01',3),('val','2024-09-01','2024-10-01',6)]}
 splits={k:np.array([i for i in ids if x[i-23:i+1,:,-1].mean()>=.65]) for k,ids in splits.items()};splits['cal']=old['cal_ids'];splits['test']=d['test_ids'];labels={k:window_labels(np.stack([raw[i+1:i+73] for i in ids])) for k,ids in splits.items()}
 np.savez_compressed(OUT/'split_data.npz',**{k+'_ids':ids for k,ids in splits.items()},**{k+'_labels':lab for k,lab in labels.items()})
 js('split_summary.json',{k:dict(origins=len(ids),first=str(times[ids[0]]),last=str(times[ids[-1]]),last_target=str(times[ids[-1]+72]),valid_primary=int(np.isfinite(labels[k][:,PRIMARY]).sum()),positive_primary=float(np.nanmean(labels[k][:,PRIMARY]))) for k,ids in splits.items()})
 return d,times,splits,labels

def baselines(d,splits,labels):
 prior=np.nanmean(labels['train'],axis=(0,1));np.savez_compressed(OUT/'prior.npz',prior=prior)
 for name in ['climatology','persistence']:
  values={}
  for part in ['val','cal','test']:
   values[part]=np.broadcast_to(prior,(len(splits[part]),67,len(prior))).copy() if name=='climatology' else persistence_prob(d['raw'],splits[part],prior)
  np.savez_compressed(OUT/f'predictions_{name}_single.npz',offsets=np.arange(67),**values)
 features={part:linear_features(d['x'],ids) for part,ids in splits.items()};scaler=StandardScaler().fit(features['train']);features={part:scaler.transform(v) for part,v in features.items()};models={};preds={part:[] for part in ['val','cal','test']};status=[]
 for a in PRIMARY:
  y=labels['train'][:,a].ravel();valid=np.isfinite(y);clf=LogisticRegression(C=1,max_iter=2000,solver='lbfgs')
  with warnings.catch_warnings(record=True) as ww:
   warnings.simplefilter('always',ConvergenceWarning);clf.fit(features['train'][valid],y[valid])
  converged=not any(isinstance(w.message,ConvergenceWarning) for w in ww);assert converged,'logistic failed fixed 2000-iteration budget'
  models[int(a)]=clf;status.append(dict(offset=int(a),training_rows=int(valid.sum()),iterations=int(clf.n_iter_[0]),converged=converged))
  for part in preds:preds[part].append(clf.predict_proba(features[part])[:,1].reshape(len(splits[part]),len(prior)))
 joblib.dump(dict(scaler=scaler,models=models,feature_definition='local24x3; mask-weighted farm24x3; node onehot'),OUT/'logistic.joblib');csv('logistic_fit.csv',status)
 np.savez_compressed(OUT/'predictions_logistic_single.npz',offsets=PRIMARY,**{part:np.stack(v,axis=1) for part,v in preds.items()})
 old=np.load(ART/'probability_v1/predictions.npz');selected=np.random.default_rng(20241101).choice(len(d['cal_ids']),32,replace=False);complete=np.isfinite(old['cal_truth']).all((1,2))
 for s in SEEDS:
  residual=(old['cal_truth']-old[f'cal_q_{s}'][...,1])[complete];med=old[f'test_q_{s}'][...,1];pred=[]
  for q in med:
   sc=np.maximum(0,q+residual[selected]);pred.append(np.stack([(sc[:,a:a+6]<=10).all(1).mean(0) for a in range(67)]))
  np.savez_compressed(OUT/f'predictions_scenario32_{s}.npz',offsets=np.arange(67),test=np.stack(pred))
 print('simple, logistic, scenario baselines saved',flush=True)

def train(d,splits,labels):
 x=d['x'];tr=splits['train'];xx=torch.from_numpy(np.stack([x[i-23:i+1] for i in tr]));yy=torch.from_numpy(labels['train']);records=[]
 for seed in SEEDS:
  time_guard()
  for name,cfg in CONFIGS.items():
   done=OUT/f'completed_{name}_{seed}.json'
   if done.exists():
    records.append(json.loads(done.read_text(encoding='utf-8')));print('resume complete',name,seed,flush=True);continue
   time_guard();torch.manual_seed(seed);A=d['adj'] if cfg['graph'] else np.eye(d['adj'].shape[0]);net=WindowNet(A,hidden=cfg['hidden'],gated=cfg['gated']);optimizer=torch.optim.Adam(net.parameters(),lr=.001);best=float('inf');best12=float('inf');history=[];started=time.monotonic()
   for epoch in range(1,26):
    time_guard();net.train();order=np.random.default_rng(epoch).permutation(len(tr));loss_sum=0.;count=0
    for start in range(0,len(tr),32):
     idx=order[start:start+32];optimizer.zero_grad();out=net(xx[idx]);loss=masked_loss(out,yy[idx],cfg['loss']);assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1);assert torch.isfinite(grad);optimizer.step();loss_sum+=loss.item()*len(idx);count+=len(idx)
    pv=predict_windows(net,x,splits['val']);vb=float(np.nanmean((pv[:,PRIMARY]-labels['val'][:,PRIMARY])**2));record=dict(epoch=epoch,train_loss=loss_sum/count,validation_brier=vb,elapsed_seconds=time.monotonic()-started);history.append(record)
    if vb<best:
     best=vb;state=copy.deepcopy(net.state_dict());be=epoch
    if epoch<=12 and vb<best12:
     best12=vb;state12=copy.deepcopy(net.state_dict());be12=epoch
    js(f'history_{name}_{seed}.json',history)
    print(seed,name,'epoch',epoch,'val',round(vb,6),'seconds',round(time.monotonic()-started,1),flush=True)
   for budget,st,bepoch,bval in [('25',state,be,best),('12',state12,be12,best12)]:
    net.load_state_dict(st);torch.save(dict(state=st,adj=A,config=cfg,seed=seed,epoch=bepoch,validation_brier=bval,epoch_budget=int(budget)),OUT/f'{name}_{seed}_budget{budget}.pt')
    values={part:predict_windows(net,x,splits[part]) for part in ['val','cal','test']};np.savez_compressed(OUT/f'predictions_{name}_{seed}_budget{budget}.npz',offsets=np.arange(67),**values)
   row=dict(model=name,seed=seed,parameters=sum(p.numel() for p in net.parameters()),best_epoch25=be,best_epoch12=be12,validation_brier25=best,validation_brier12=best12,training_seconds=time.monotonic()-started);js(done.name,row);records.append(row)
 csv('training_summary.csv',records)

def evaluate(times,splits,labels):
 rows=[];rel=[];high=[];nodes=[]
 for path in sorted(OUT.glob('predictions_*.npz')):
  ident=path.stem[len('predictions_'):];parts=ident.split('_');seed=next((int(p) for p in parts if p.isdigit()),0);budget=int(parts[-1][6:]) if parts[-1].startswith('budget') else 0
  name=ident.rsplit('_',2)[0] if budget else ident.rsplit('_',1)[0];pred=np.load(path);off=pred['offsets'];primary_positions=np.array([int(np.flatnonzero(off==a)[0]) for a in PRIMARY]);pack={}
  for period in ['val','cal','test']:
   if period not in pred:continue
   p=pred[period];y=labels[period][:,off];assert p.shape==y.shape and np.isfinite(p).all() and np.min(p)>=0 and np.max(p)<=1;loss=(p-y)**2;primary=loss[:,primary_positions];pp=p[:,primary_positions];yy=y[:,primary_positions]
   pack[period+'_primary_errors']=primary;pack[period+'_all_errors']=loss
   for month in ([0,11,12] if period=='test' else [0]):
    mask=np.ones(len(p),bool) if month==0 else times[splits[period]].month==month
    for metric,v in [('brier_primary',primary),('brier_available_starts',loss)]:rows.append(dict(model=name,seed=seed,budget=budget,period=period,month=month,metric=metric,value=float(np.nanmean(v[mask])),n=int(np.isfinite(v[mask]).sum()),starts=len(PRIMARY) if metric=='brier_primary' else len(off)))
    for j,a in enumerate(PRIMARY):rows.append(dict(model=name,seed=seed,budget=budget,period=period,month=month,metric=f'brier_offset{a}',value=float(np.nanmean(primary[mask,j])),n=int(np.isfinite(primary[mask,j]).sum()),starts=1))
    if period=='test':
     pr=pp[mask].ravel();obs=yy[mask].ravel();valid=np.isfinite(obs);pr=pr[valid];obs=obs[valid]
     for b in range(10):
      mm=(pr>=b/10)&((pr<(b+1)/10) if b<9 else pr<=1);rel.append(dict(model=name,seed=seed,budget=budget,month=month,bin=b,n=int(mm.sum()),mean_probability=float(pr[mm].mean()) if mm.any() else np.nan,observed_rate=float(obs[mm].mean()) if mm.any() else np.nan))
     mm=pr>=.8;high.append(dict(model=name,seed=seed,budget=budget,month=month,n=int(mm.sum()),fraction=float(mm.mean()),mean_probability=float(pr[mm].mean()) if mm.any() else np.nan,observed_rate=float(obs[mm].mean()) if mm.any() else np.nan))
   if period=='test':
    for j in range(y.shape[2]):nodes.append(dict(model=name,seed=seed,budget=budget,node_index=j,brier=float(np.nanmean(primary[:,:,j])),n=int(np.isfinite(primary[:,:,j]).sum())))
  np.savez_compressed(OUT/f'errors_{ident}.npz',**pack)
 csv('metrics.csv',rows);csv('reliability.csv',rel);csv('high_confidence.csv',high);csv('node_metrics.csv',nodes)

def compare(times,splits):
 def loss(name,seed,budget=25):
  suffix=f'{seed}_budget{budget}' if name in CONFIGS else 'single' if name in ['logistic','climatology','persistence'] else str(seed)
  with np.load(OUT/f'errors_{name}_{suffix}.npz') as p:return p['test_primary_errors']
 tt=times[splits['test']];week=((tt-pd.Timestamp('2024-11-01',tz=tt.tz)).days//7).to_numpy();keys=np.unique(week);draw=np.random.default_rng(20261103).integers(0,len(keys),(2000,len(keys)));pairs=[(n,'scenario32') for n in [*CONFIGS,'logistic','climatology','persistence']]+[('gated_graph_bce',n) for n in ['gated_temporal_bce','plain_graph_bce','wide_plain_graph_bce','gated_graph_brier','logistic']];rows=[]
 for candidate,base in pairs:
  d=np.mean([loss(candidate,s)-loss(base,s) for s in SEEDS],axis=0);ss=np.array([np.nansum(d[week==k]) for k in keys]);nn=np.array([np.isfinite(d[week==k]).sum() for k in keys]);boot=ss[draw].sum(1)/nn[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975]);rows.append(dict(candidate=candidate,baseline=base,difference=float(np.nanmean(d)),low95=lo,high95=hi,blocks=len(keys),primary=(candidate,base) in [('gated_graph_bce','scenario32'),('gated_graph_bce','gated_temporal_bce')]))
 csv('paired_comparisons.csv',rows)

if __name__=='__main__':
 prepare();time_guard();d,times,splits,labels=load_splits();baselines(d,splits,labels);train(d,splits,labels);evaluate(times,splits,labels);compare(times,splits);print('COMPLETE',flush=True)

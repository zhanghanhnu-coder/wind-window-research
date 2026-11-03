"""Exploratory diagnostics; no tuning of the served model on evaluation outcomes."""
import json,copy,pickle,joblib
import numpy as np
import pandas as pd
import torch
from model import *
from forecast import Forecaster
from replay import replay

def seed_audit():
 f=Forecaster();d=np.load(ART/'data.npz');x=d['x'];raw=d['raw'];idx=f.times;center=f.meta['center'];scale=f.meta['scale'];y=((raw-center)/scale).astype('float32')
 tr=origins(idx,'2024-01-12','2024-09-01',3);va=origins(idx,'2024-09-01','2024-10-01',6)
 tr=np.array([i for i in tr if x[i-23:i+1,:,-1].mean()>=.65]);va=np.array([i for i in va if x[i-23:i+1,:,-1].mean()>=.65]);test=d['test_ids'];cal=d['cal_ids'];truth=np.stack([raw[i+1:i+73] for i in test]);caltruth=np.stack([raw[i+1:i+73] for i in cal]);rows=[]
 for seed in [20241101,20241102,20241103]:
  for name,A in [('temporal',np.eye(len(f.nodes))),('graph',d['adj'])]:
   if seed==20241101:net=f.models[name];bestepoch=f.meta[name]['best_epoch']
   else:
    torch.manual_seed(seed);net=Net(A);opt=torch.optim.Adam(net.parameters(),lr=.001);best=float('inf')
    for epoch in range(1,13):
     net.train()
     for xx,yy in batches(x,y,tr,32,True,epoch):
      opt.zero_grad();loss=pinball(net(xx),yy);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
     q=predict(net,x,va);vy=np.stack([y[i+1:i+73] for i in va]);vl=float(pinball(torch.from_numpy(q),torch.from_numpy(vy)))
     if vl<best:best=vl;state=copy.deepcopy(net.state_dict());bestepoch=epoch
    net.load_state_dict(state);torch.save({'state':state,'adj':A,'center':center,'scale':scale,'best_epoch':bestepoch},ART/f'{name}_seed{seed}.pt')
   median=predict(net,x,test)[...,1]*scale+center;cm=predict(net,x,cal)[...,1]*scale+center
   bank=caltruth-cm;sel=np.random.default_rng(20241101).choice(len(bank),32,replace=len(bank)<32);sc=np.maximum(0,median[:,None]+bank[sel][None]);p=(sc[:,:,:6]<=10).all(2).mean(1);valid=np.isfinite(truth[:,:6]).all(1);label=(truth[:,:6]<=10).all(1)
   rows.append({'seed':seed,'model':name,'validation_selected_epoch':bestepoch,'mae_all_72h':float(np.nanmean(abs(median-truth))),'brier_first6_threshold10':float(((p-label)**2)[valid].mean())})
   print('SEED',rows[-1],flush=True)
 pd.DataFrame(rows).to_csv(ART/'graph_seed_audit.csv',index=False,encoding='utf-8-sig')

def probability_audit():
 f=Forecaster();d=np.load(ART/'data.npz');ids=d['test_ids'];raw=f.raw;truth=np.stack([raw[i+1:i+73] for i in ids]);records=[];errors={};probs={};labels={}
 for name in ['graph','temporal','persistence']:
  sc=np.stack([f.scenarios(str(f.times[i]),name)[0] for i in ids]);ps=[];ys=[];valids=[]
  for offset in [0,24,48,66]:
   t=truth[:,offset:offset+6];valid=np.isfinite(t).all(1);y=(t<=10).all(1);p=(sc[:,:,offset:offset+6]<=10).all(2).mean(1)
   independent=(sc[:,:,offset:offset+6]<=10).mean(1).prod(1)
   ps.append(p);ys.append(y);valids.append(valid)
   for variant,pv in [('joint_paths',p),('independent_hours',independent)]:
    for month in [11,12]:
     mask=valid & (f.times[ids].month.to_numpy()==month)[:,None]
     records.append({'model':name,'variant':variant,'month':month,'offset':offset,'brier':float(((pv-y)**2)[mask].mean()),'n':int(mask.sum())})
  e=(np.stack(ps,1)-np.stack(ys,1))**2;errors[name]=np.where(np.stack(valids,1),e,np.nan)
  probs[name]=ps[0];labels[name]=(ys[0],valids[0])
 pd.DataFrame(records).to_csv(ART/'probability_audit.csv',index=False,encoding='utf-8-sig')
 # Paired weekly blocks keep all turbines/horizons in a week together.
 diff=errors['graph']-errors['temporal'];weeks=((f.times[ids]-pd.Timestamp('2024-11-01',tz='Asia/Shanghai')).days//7).to_numpy();keys=np.unique(weeks)
 sums=np.array([np.nansum(diff[weeks==k]) for k in keys]);counts=np.array([np.isfinite(diff[weeks==k]).sum() for k in keys]);rng=np.random.default_rng(20241101);draw=rng.integers(0,len(keys),(2000,len(keys)));boot=sums[draw].sum(1)/counts[draw].sum(1)
 result={'contrast':'graph minus temporal; threshold10; six-hour windows offsets0,24,48,66','difference':float(np.nanmean(diff)),'block_bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),'weekly_blocks':len(keys),'replicates':2000,'interpretation':'negative favours graph; overlapping windows and previously inspected single year make this exploratory, not a confirmatory test'}
 (ART/'paired_graph_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 # Earlier six-hour binary classifier: same origin/device/valid labels intersection.
 samples=pickle.load(open(ROOT.parent/'研究笔记/基线实验_2026-11-01/完整实验样本.pkl','rb'));samples=samples[samples.origin.isin(f.times[ids]) & samples.future6max.notna()].copy();parts=[]
 pos={str(t):i for i,t in enumerate(f.times[ids])}
 for month,group in samples.groupby(samples.origin.dt.month):
  if month not in [11,12]:continue
  pack=joblib.load(ROOT.parent/f'研究笔记/第二轮B实验_2026-11-01/w_{month}_10_multi_max_expand.joblib');p=pack['model'].predict_proba(group[pack['features']])[:,1];logit=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).reshape(-1,1);cal=pack['calibrator'].predict_proba(logit)[:,1]
  group['prior_classifier']=cal
  for name in probs:group[name]=[probs[name][pos[str(t)],f.nodes.index(str(n))] for t,n in zip(group.origin,group.device)]
  parts.append(group)
 aligned=pd.concat(parts);y=(aligned.future6max<=10).to_numpy();comparison=[]
 for name in ['prior_classifier','graph','temporal','persistence']:comparison.append({'model':name,'brier':float(np.mean((aligned[name].to_numpy()-y)**2)),'n':len(aligned),'target':'immediate6h threshold10 same complete-feature rows','training_note':'prior classifier expands monthly; demo models fixed Jan-Aug. Different training policies, comparison of deployed candidates not pure architecture ablation.'})
 pd.DataFrame(comparison).to_csv(ART/'prior_baseline_comparison.csv',index=False,encoding='utf-8-sig');print('PAIRED',result,'PRIOR',comparison,flush=True)

def sensitivity():
 f=Forecaster();rows=[]
 for start,days in [('2024-11-01',30),('2024-12-01',28)]:
  for th in [10,12]:
   for p in [.5,.8]:
    for policy,weight in [('fifo',1),('expected',1),('cvar',.25),('cvar',1),('robust',1)]:
     r=replay(f,start=start,days=days,threshold=th,minimum_probability=p,policy=policy,risk_weight=weight)
     rows.append(r['settings']|r['summary']);print('SENS',start,th,p,policy,weight,r['summary']['completed'],r['summary']['failed_attempts'],flush=True)
 pd.DataFrame(rows).drop(columns=['solver_counts']).to_csv(ART/'sensitivity.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':seed_audit();probability_audit();sensitivity()

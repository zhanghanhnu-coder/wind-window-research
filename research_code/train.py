"""Fixed exploratory 2024 split. Never select models using Nov-Dec results."""
import json,copy,time,hashlib
import numpy as np
import pandas as pd
import torch
from model import *

def run():
 torch.manual_seed(20241101);np.random.seed(20241101)
 mean,mx=load_data();idx=mean.index;raw=mx.to_numpy().astype('float32')
 fit=idx<pd.Timestamp('2024-09-01',tz=idx.tz)
 center=float(np.nanmean(np.stack([mean.to_numpy()[fit],raw[fit]])))
 scale=float(np.nanstd(np.stack([mean.to_numpy()[fit],raw[fit]])))
 x=inputs(mean,mx,center,scale);y=(raw-center)/scale
 corr=mean.loc[fit].corr(min_periods=100).fillna(0).to_numpy();np.fill_diagonal(corr,0)
 adj=np.zeros_like(corr)
 for i in range(len(adj)):
  js=np.argsort(corr[i])[-4:];adj[i,js]=np.maximum(corr[i,js],0)
 adj=adj/np.maximum(adj.sum(1,keepdims=True),1e-6)
 tr=origins(idx,'2024-01-12','2024-09-01',3);va=origins(idx,'2024-09-01','2024-10-01',6)
 # Availability gate uses input only, not future completeness.
 tr=np.array([i for i in tr if x[i-23:i+1,:,-1].mean()>=.65]);va=np.array([i for i in va if x[i-23:i+1,:,-1].mean()>=.65])
 cal=origins(idx,'2024-10-01','2024-11-01',6)
 cal=np.array([i for i in cal if np.isfinite(raw[i+1:i+73]).all()])
 test=origins(idx,'2024-11-01','2025-01-01',6)
 test=np.array([i for i in test if x[i-23:i+1,:,-1].mean()>=.65])
 truth=np.stack([raw[i+1:i+73] for i in test]);rows=[];window=[];coverage=[];histories={}
 metadata={'seed':20241101,'train':'2024-01-12 to 2024-08-31','validation':'2024-09','residual_bank':'2024-10','evaluation':'2024-11 to 2024-12; exploratory, previously inspected year','center':center,'scale':scale,'nodes':list(mean.columns),'n_train_origins':len(tr),'n_validation_origins':len(va),'n_residual_trajectories':len(cal),'n_test_origins':len(test),'data_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'graph':'training-only top-4 positive Pearson correlation; not physical coordinates','target':'hour-ending maximum of 12 exported 5-minute wind values','versions':{'torch':torch.__version__,'numpy':np.__version__},'scenario_count':32}
 np.savez_compressed(ART/'data.npz',x=x,raw=raw,mean=mean.to_numpy(),timestamps=np.array([str(t) for t in idx]),adj=adj,test_ids=test,cal_ids=cal)
 for name,A in [('temporal',np.eye(len(adj))),('graph',adj),('persistence',None)]:
  t0=time.time()
  if A is not None:
   torch.manual_seed(20241101);net=Net(A);opt=torch.optim.Adam(net.parameters(),lr=.001);best=float('inf');hist=[]
   for epoch in range(1,13):
    net.train();losses=[]
    for xx,yy in batches(x,y,tr,32,True,epoch):
     opt.zero_grad();loss=pinball(net(xx),yy);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();losses.append(loss.item())
    q=predict(net,x,va);vy=np.stack([y[i+1:i+73] for i in va]);vl=float(pinball(torch.from_numpy(q),torch.from_numpy(vy)))
    hist.append({'epoch':epoch,'train_pinball':float(np.mean(losses)),'validation_pinball':vl})
    if vl<best:best=vl;state=copy.deepcopy(net.state_dict());bestepoch=epoch
    print(name,epoch,round(float(np.mean(losses)),5),round(vl,5),flush=True)
   net.load_state_dict(state);torch.save({'state':state,'adj':A,'center':center,'scale':scale,'best_epoch':bestepoch},ART/f'{name}.pt')
   med=predict(net,x,test)[...,1]*scale+center;cm=predict(net,x,cal)[...,1]*scale+center
   histories[name]=hist;metadata[name]={'best_epoch':bestepoch,'validation_pinball':best,'seconds':time.time()-t0}
  else:
   def persist(ids):
    vals=[]
    for i in ids:
     past=raw[i-23:i+1];last=np.array([z[np.isfinite(z)][-1] if np.isfinite(z).any() else center for z in past.T])
     vals.append(np.broadcast_to(last,(72,len(adj))))
    return np.array(vals)
   med=persist(test);cm=persist(cal)
  residual=np.stack([raw[i+1:i+73] for i in cal])-cm
  np.savez_compressed(ART/f'{name}_residuals.npz',residual=residual,origins=np.array([str(idx[i]) for i in cal]))
  rng=np.random.default_rng(20241101);selected=rng.choice(len(residual),32,replace=len(residual)<32)
  scenarios=np.maximum(0,med[:,None]+residual[selected][None])
  for horizon in [1,6,24,72]:
   e=med[:,horizon-1]-truth[:,horizon-1];rows.append({'model':name,'horizon_hours':horizon,'mae':float(np.nanmean(abs(e))),'n':int(np.isfinite(e).sum())})
  lo,hi=np.quantile(scenarios,[.1,.9],axis=1);mask=np.isfinite(truth)
  coverage.append({'model':name,'interval':'10%-90% empirical scenario range','coverage':float(((truth>=lo)&(truth<=hi))[mask].mean()),'mean_width':float((hi-lo)[mask].mean()),'n':int(mask.sum())})
  for th in [8,10,12]:
   for start in [0,24,48,66]:
    observed=truth[:,start:start+6];valid=np.isfinite(observed).all(1);label=(observed<=th).all(1)
    p=(scenarios[:,:,start:start+6]<=th).all(2).mean(1)
    window.append({'model':name,'threshold':th,'start_offset':start,'duration':6,'brier':float(((p-label)**2)[valid].mean()),'n':int(valid.sum()),'observed_success_rate':float(label[valid].mean())})
  print('FINISHED',name,'seconds',round(time.time()-t0,1),flush=True)
 pd.DataFrame(rows).to_csv(ART/'forecast_metrics.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(window).to_csv(ART/'window_metrics.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(coverage).to_csv(ART/'interval_metrics.csv',index=False,encoding='utf-8-sig')
 metadata['graph_definition']='training-only top-4 positive Pearson correlation; not physical coordinates'
 (ART/'training_history.json').write_text(json.dumps(histories,ensure_ascii=False,indent=2),encoding='utf-8')
 (ART/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(metadata,ensure_ascii=False),flush=True)

if __name__=='__main__':run()

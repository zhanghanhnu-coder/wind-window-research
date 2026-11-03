"""Pre-fixed 24h receptive-field and graph-ablation study. Preserve v1 artifacts."""
import json,copy,hashlib
import numpy as np
import pandas as pd
import torch
from model import Net,ART,origins,batches,pinball,predict

OUT=ART/'v2';OUT.mkdir(exist_ok=True)

def run():
 d=np.load(ART/'data.npz');meta=json.loads((ART/'metadata.json').read_text(encoding='utf-8'));x=d['x'];raw=d['raw'];center=meta['center'];scale=meta['scale'];y=((raw-center)/scale).astype('float32');times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai')
 tr=origins(times,'2024-01-12','2024-09-01',3);va=origins(times,'2024-09-01','2024-10-01',6)
 tr=np.array([i for i in tr if x[i-23:i+1,:,-1].mean()>=.65]);va=np.array([i for i in va if x[i-23:i+1,:,-1].mean()>=.65]);te=d['test_ids'];cal=d['cal_ids'];truth=np.stack([raw[i+1:i+73] for i in te]);ct=np.stack([raw[i+1:i+73] for i in cal]);vy=np.stack([y[i+1:i+73] for i in va]);n=raw.shape[1]
 perm=np.random.default_rng(20241101).permutation(n);graphs={'temporal24':np.eye(n),'graph24':d['adj'],'mean24':(np.ones((n,n))-np.eye(n))/(n-1),'random24':d['adj'][perm][:,perm]}
 recipe={'version':'2.0','seeds':[20241101,20241102,20241103],'epochs':12,'dilation':5,'input_hours':24,'receptive_field':25,'permutation':perm.tolist(),'primary':'Brier threshold10 duration6 offsets0/24/48/66 pooled over valid cells','splits':'Jan-Aug fit; Sep epoch selection; Oct scenario bank; Nov-Dec exploratory evaluation','source_data_sha256':hashlib.sha256((ART/'data.npz').read_bytes()).hexdigest(),'not_confirmatory':'2024 has already been inspected; no claims of unseen-year generalization'}
 (OUT/'recipe.json').write_text(json.dumps(recipe,ensure_ascii=False,indent=2),encoding='utf-8');records=[];windows=[];saved_errors={};history={}
 selected=np.random.default_rng(20241101).choice(len(cal),32,replace=len(cal)<32)
 for seed in recipe['seeds']:
  for name,A in graphs.items():
   torch.manual_seed(seed);net=Net(A,dilation=5);opt=torch.optim.Adam(net.parameters(),lr=.001);best=float('inf');hist=[]
   for epoch in range(1,13):
    net.train()
    for xx,yy in batches(x,y,tr,32,True,epoch):
     opt.zero_grad();loss=pinball(net(xx),yy);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
    q=predict(net,x,va);vl=float(pinball(torch.from_numpy(q),torch.from_numpy(vy)));hist.append(vl)
    if vl<best:best=vl;state=copy.deepcopy(net.state_dict());be=epoch
   net.load_state_dict(state);med=predict(net,x,te)[...,1]*scale+center;cm=predict(net,x,cal)[...,1]*scale+center;residual=ct-cm;sc=np.maximum(0,med[:,None]+residual[selected][None]);errors=[]
   for offset in [0,24,48,66]:
    obs=truth[:,offset:offset+6];valid=np.isfinite(obs).all(1);label=(obs<=10).all(1);p=(sc[:,:,offset:offset+6]<=10).all(2).mean(1);e=np.where(valid,(p-label)**2,np.nan);errors.append(e)
    for month in [0,11,12]:
     m=np.ones(len(te),bool) if month==0 else times[te].month==month
     windows.append({'model':name,'seed':seed,'month':month,'offset':offset,'brier':float(np.nanmean(e[m])),'n':int(np.isfinite(e[m]).sum())})
   errors=np.stack(errors,axis=1);saved_errors.setdefault(name,[]).append(errors)
   record={'model':name,'seed':seed,'selected_epoch':be,'validation_pinball':best,'mae_all':float(np.nanmean(abs(med-truth))),'brier_primary':float(np.nanmean(errors)),'brier_first6':float(np.nanmean(errors[:,0]))}
   for h in [1,6,24,72]:record[f'mae_{h}h']=float(np.nanmean(abs(med[:,h-1]-truth[:,h-1])))
   records.append(record);history[f'{name}_{seed}']=hist
   filename=name if seed==20241101 else f'{name}_{seed}'
   torch.save({'state':state,'adj':A,'center':center,'scale':scale,'best_epoch':be,'dilation':5,'seed':seed},OUT/f'{filename}.pt')
   if seed==20241101:np.savez_compressed(OUT/f'{name}_residuals.npz',residual=residual,origins=np.array([str(times[i]) for i in cal]))
   print(json.dumps(record),flush=True)
 pd.DataFrame(records).to_csv(OUT/'ablation_metrics.csv',index=False,encoding='utf-8-sig');pd.DataFrame(windows).to_csv(OUT/'ablation_windows.csv',index=False,encoding='utf-8-sig')
 (OUT/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8');np.savez_compressed(OUT/'paired_losses.npz',**{k:np.stack(v) for k,v in saved_errors.items()},test_ids=te)
 weeks=((times[te]-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();keys=np.unique(weeks);draw=np.random.default_rng(20241101).integers(0,len(keys),(2000,len(keys)));comparisons=[]
 for baseline in ['temporal24','mean24','random24']:
  delta=np.mean(np.stack(saved_errors['graph24'])-np.stack(saved_errors[baseline]),axis=0);sums=np.array([np.nansum(delta[weeks==k]) for k in keys]);counts=np.array([np.isfinite(delta[weeks==k]).sum() for k in keys]);boot=sums[draw].sum(1)/counts[draw].sum(1);ci=np.quantile(boot,[.025,.975])
  comparisons.append({'candidate':'graph24','baseline':baseline,'mean_brier_difference':float(np.nanmean(delta)),'low95':float(ci[0]),'high95':float(ci[1]),'blocks':len(keys),'replicates':2000,'negative_favours_graph':True})
 pd.DataFrame(comparisons).to_csv(OUT/'paired_comparisons.csv',index=False,encoding='utf-8-sig');print('PAIRED',comparisons,flush=True)

if __name__=='__main__':run()

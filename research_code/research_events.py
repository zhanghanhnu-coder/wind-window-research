"""Pre-specified event propagation study, isolated from earlier experiments."""
from pathlib import Path
import argparse,copy,hashlib,json,itertools,shutil
import numpy as np
import pandas as pd
import torch
from model import ART,origins,pinball
from propagation_graph import lag_profile,lag_adjacency,directed_pair_edges
from event_graph import event_scores,crossings,state_at,stable_rule,shifted_hour_features,EventNet,require_identity,KINDS

ROOT=Path(__file__).resolve().parent.parent
OUT=ART/'event_propagation_v1';OUT.mkdir(exist_ok=True)
RESULT=ROOT/'核心结果/事件传播研究';RESULT.mkdir(parents=True,exist_ok=True)
MODELS=['reference','event_candidate','event_zero','event_shuffle','event_stable'];SEEDS=[20241101,20241102,20241103]

def dump(obj,name): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(rows,name): pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def load():
 d=np.load(ART/'data.npz');meta=json.loads((ART/'metadata.json').read_text(encoding='utf-8'));f=np.load(ART/'propagation_v1/fine_data.npz')
 times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');ft=pd.DatetimeIndex(f['timestamps'])
 return d,meta,f,times,ft

def prepare():
 d,meta,f,times,ft=load();nodes=meta['nodes'];n=len(nodes)
 # Preserve previous artifacts once. Never overwrite that record on a repeated run.
 if not (OUT/'previous_hashes.json').exists():
  prior={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ART.rglob('*') if p.is_file() and 'event_propagation_v1' not in p.parts and p.suffix in ['.pt','.npz','.csv']};dump(prior,'previous_hashes.json')
 plan=(ROOT/'研究笔记/事件传播研究方案_2026-11-01.md').read_text(encoding='utf-8').split('## 执行记录')[0]
 if not (OUT/'protocol_frozen.md').exists():(OUT/'protocol_frozen.md').write_text(plan,encoding='utf-8')
 elif (OUT/'protocol_frozen.md').read_text(encoding='utf-8')!=plan:raise ValueError('Frozen protocol changed')
 scores=event_scores(f['wind'],f['direction']);fit=ft<pd.Timestamp('2024-09-01',tz=ft.tz)
 thresholds={'wind':float(np.nanquantile(abs(scores['wind_change'][fit]),.95)),'turn':max(30.,float(np.nanquantile(abs(scores['turn'][fit]),.95)))}
 config={'thresholds':thresholds,'lag_steps':12,'step_minutes':5,'event_history_steps':24,'event_future_steps':48,'state_lookback_steps':72,'seeds':SEEDS,'epochs':12,'models':MODELS,'data_sha256':hashlib.sha256((ART/'data.npz').read_bytes()).hexdigest(),'fine_data_sha256':hashlib.sha256((ART/'propagation_v1/fine_data.npz').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((OUT/'protocol_frozen.md').read_bytes()).hexdigest(),'scope':'2024 exploratory; causal online event detection, offline lag diagnostics; not physical propagation proof'}
 dump(config,'recipe.json');events=[]
 for kind,signal,threshold in [('up',scores['wind_change'],thresholds['wind']),('down',-scores['wind_change'],thresholds['wind']),('turn',abs(scores['turn']),thresholds['turn'])]:
  for index in crossings(signal,threshold,72):
   eligible=index>=24 and index+48<len(ft) and ft[index-24].month==ft[index].month==ft[index+48].month
   events.append({'index':index,'kind':kind,'time':str(ft[index]),'month':int(ft[index].month),'strength':float(signal[index]),'window_eligible':bool(eligible)})
 events.sort(key=lambda e:(e['index'],KINDS.index(e['kind'])))
 for k,e in enumerate(events):e['event_id']=k
 dump(events,'events.json');csv(events,'event_catalog.csv');np.savez_compressed(OUT/'event_signals.npz',**scores)
 observations=[];allprofiles=[]
 for event in events:
  if not event['window_eligible']:continue
  signal=scores['turn_signal'] if event['kind']=='turn' else scores['wind_signal'];ix=event['index'];v=signal[ix-24:ix+49]
  for i,j in itertools.combinations(range(n),2):
   ls,ss,count=lag_profile(v[:,i],v[:,j],12,'corr',36)
   if count<36:continue
   best=max(0,float(ss.max()));ties=np.flatnonzero(ss>=best-1e-12);k=min(ties,key=lambda k:(abs(ls[k]),ls[k])) if len(ties) else 12
   observations.append({'event_id':event['event_id'],'kind':event['kind'],'time':event['time'],'month':event['month'],'node_i':nodes[i],'node_j':nodes[j],'lag_steps':int(ls[k]),'score':best,'zero_score':float(ss[12]),'n_common':count})
   allprofiles.append(ss)
 obs=pd.DataFrame(observations);profiles=np.asarray(allprofiles,dtype=np.float32);csv(obs,'event_pair_observations.csv');np.savez_compressed(OUT/'event_profiles.npz',scores=profiles)
 summaries=[];graphs={};follow=[];train=obs[obs.month<=8]
 for kind in KINDS[1:]:
  candidate=[];stable=[]
  for (ni,nj),g in train[train.kind==kind].groupby(['node_i','node_j']):
   lagcounts=g.lag_steps.value_counts();lag=int(min(lagcounts.index,key=lambda l:(-lagcounts[l],abs(l),l)));p=profiles[g.index];gains=p[:,lag+12]-p[:,12];good=stable_rule(g.lag_steps,g.month,gains,lag)
   weight=float(g.score.clip(lower=0).mean());i=nodes.index(ni);j=nodes.index(nj)
   row={'kind':kind,'node_i':ni,'node_j':nj,'events':len(g),'months':int(g.month.nunique()),'lag_minutes':lag*5,'score':weight,'sign_agreement':float((np.sign(g.lag_steps)==np.sign(lag)).mean()),'within5min_agreement':float((abs(g.lag_steps-lag)<=1).mean()),'fixed_gain_over_zero':float(np.mean(gains)),'stable_pass':bool(good),'candidate_pass':len(g)>=6 and weight>0};summaries.append(row)
   if len(g)>=6:candidate.extend(directed_pair_edges(i,j,lag,weight))
   if good:stable.extend(directed_pair_edges(i,j,lag,weight))
   for period,months in [('train',range(1,9)),('sep_oct',[9,10]),('nov_dec',[11,12])]:
    z=obs[(obs.kind==kind)&(obs.node_i==ni)&(obs.node_j==nj)&obs.month.isin(months)];pp=profiles[z.index]
    follow.append({'kind':kind,'node_i':ni,'node_j':nj,'period':period,'events':len(z),'fixed_lag_minutes':lag*5,'stable_train_pass':bool(good),'candidate_train_pass':bool(row['candidate_pass']),'sign_agreement':float((np.sign(z.lag_steps)==np.sign(lag)).mean()) if len(z) else None,'within5min_agreement':float((abs(z.lag_steps-lag)<=1).mean()) if len(z) else None,'fixed_gain_over_zero':float(np.mean(pp[:,lag+12]-pp[:,12])) if len(z) else None})
  graphs[kind+'_candidate']=lag_adjacency(n,candidate,12);graphs[kind+'_stable']=lag_adjacency(n,stable,12)
 csv(summaries,'pair_event_summary.csv');csv(follow,'forward_stability.csv');np.savez_compressed(OUT/'event_graphs.npz',**graphs)
 edges=[]
 for name,a in graphs.items():
  for lag,receiver,source in zip(*np.where(a>0)):edges.append({'graph':name,'source':nodes[source],'receiver':nodes[receiver],'lag_minutes':int(lag)*5,'weight':float(a[lag,receiver,source])})
 csv(edges,'event_graph_edges.csv')
 # Coarse hour origin i corresponds to exclusive fine index 12*i.
 cutoffs=np.arange(len(times))*12;state=np.array([KINDS.index(state_at(events,int(t))) for t in cutoffs],np.int8)
 shifted=np.stack([shifted_hour_features(f['wind'],cutoffs,lag,meta['center'],meta['scale']) for lag in range(13)])
 assert np.allclose(shifted[0],d['x'],rtol=1e-5,atol=1e-5), 'Canonical hourly feature mismatch'
 # All references use the same arithmetic implementation, avoiding a preprocessing confound.
 shifted[0]=d['x'];base=np.einsum('ij,tjf->tif',d['adj'],d['x']);perm=np.random.default_rng(20241101).permutation(n);neighbors={}
 for model in MODELS:
  bykind=[]
  for kind in KINDS:
   if model=='reference' or kind=='none':bykind.append(base.copy());continue
   a=graphs[kind+('_stable' if model=='event_stable' else '_candidate')].copy()
   if model=='event_zero':a=np.pad(a.sum(0,keepdims=True),((0,12),(0,0),(0,0)))
   elif model=='event_shuffle':a=a[:,perm][:,:,perm]
   aggregate=np.einsum('lij,ltjf->tif',a,shifted);has=a.sum((0,2))>0
   bykind.append(np.where(has[None,:,None],.5*base+.5*aggregate,base))
  neighbors[model]=np.stack(bykind).astype(np.float32)
 np.savez_compressed(OUT/'model_inputs.npz',state=state,**neighbors)
 stats={'thresholds':thresholds,'events_total':len(events),'events_by_kind':pd.DataFrame(events).groupby('kind').size().to_dict(),'events_by_month_kind':pd.DataFrame(events).groupby(['month','kind']).size().reset_index(name='count').to_dict('records'),'eligible_events':sum(e['window_eligible'] for e in events),'pair_observations':len(obs),'graphs':{name:{'edges':int((a>0).sum()),'nonzero_edges':int((a[1:]>0).sum())} for name,a in graphs.items()},'test_origins_by_state':{kind:int((state[d['test_ids']]==k).sum()) for k,kind in enumerate(KINDS)}}
 dump(stats,'event_summary.json');print(json.dumps(stats,ensure_ascii=False),flush=True)

def identity():
 paths=[OUT/'recipe.json',OUT/'event_graphs.npz',OUT/'model_inputs.npz',ART/'data.npz',ART/'metadata.json',Path(__file__),ROOT/'风电运维Demo/event_graph.py',ROOT/'风电运维Demo/model.py']
 return hashlib.sha256(''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in paths).encode()).hexdigest()

def event_batches(x,neighbor,state,y,ids,batch=32,shuffle=False,seed=0):
 ids=ids.copy()
 if shuffle:np.random.default_rng(seed).shuffle(ids)
 for a in range(0,len(ids),batch):
  ii=ids[a:a+batch];xx=np.stack([np.concatenate([x[i-23:i+1],neighbor[state[i],i-23:i+1]],-1) for i in ii]);yy=np.stack([y[i+1:i+73] for i in ii])
  yield torch.from_numpy(xx),torch.from_numpy(yy)

def prediction(net,x,neighbor,state,y,ids):
 net.eval();out=[]
 with torch.no_grad():
  for xx,_ in event_batches(x,neighbor,state,y,ids,64):out.append(net(xx).numpy())
 return np.concatenate(out)

def train():
 d,meta,f,times,ft=load();inputs=np.load(OUT/'model_inputs.npz');state=inputs['state'];token=identity();x=d['x'];raw=d['raw'];center=meta['center'];scale=meta['scale'];y=((raw-center)/scale).astype('float32')
 tr=origins(times,'2024-01-12','2024-09-01',3);va=origins(times,'2024-09-01','2024-10-01',6);tr=np.array([i for i in tr if x[i-23:i+1,:,-1].mean()>=.65]);va=np.array([i for i in va if x[i-23:i+1,:,-1].mean()>=.65]);te=d['test_ids'];cal=d['cal_ids']
 truth=np.stack([raw[i+1:i+73] for i in te]);ct=np.stack([raw[i+1:i+73] for i in cal]);vy=np.stack([y[i+1:i+73] for i in va]);selected=np.random.default_rng(20241101).choice(len(cal),32,replace=False)
 records=[];windows=[];losses={}
 for seed in SEEDS:
  for model in MODELS:
   p=OUT/f'{model}_{seed}';ckpath=p.with_suffix('.pt');scorepath=OUT/f'{model}_{seed}_scores.npz';rp=OUT/f'{model}_{seed}_record.json';neighbor=inputs[model]
   if ckpath.exists() and scorepath.exists() and rp.exists():
    row=json.loads(rp.read_text());require_identity(row,token);saved=np.load(scorepath);errors=saved['errors'];med=saved['median'];print('REUSE',model,seed,flush=True)
   else:
    torch.manual_seed(seed);net=EventNet();opt=torch.optim.Adam(net.parameters(),lr=.001);best=float('inf');history=[]
    for epoch in range(1,13):
     net.train()
     for xx,yy in event_batches(x,neighbor,state,y,tr,32,True,epoch):
      opt.zero_grad();loss=pinball(net(xx),yy);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
     q=prediction(net,x,neighbor,state,y,va);vl=float(pinball(torch.from_numpy(q),torch.from_numpy(vy)));history.append(vl)
     if vl<best:best=vl;weights=copy.deepcopy(net.state_dict());be=epoch
    net.load_state_dict(weights);med=prediction(net,x,neighbor,state,y,te)[...,1]*scale+center;cm=prediction(net,x,neighbor,state,y,cal)[...,1]*scale+center;residual=ct-cm;sc=np.maximum(0,med[:,None]+residual[selected][None]);errors=[]
    for offset in [0,24,48,66]:
     obs=truth[:,offset:offset+6];valid=np.isfinite(obs).all(1);label=(obs<=10).all(1);prob=(sc[:,:,offset:offset+6]<=10).all(2).mean(1);errors.append(np.where(valid,(prob-label)**2,np.nan))
    errors=np.stack(errors,axis=1);row={'model':model,'seed':seed,'selected_epoch':be,'validation_pinball':best,'brier_primary':float(np.nanmean(errors)),'mae_all':float(np.nanmean(abs(med-truth))),'identity':token}
    for h in [1,6,24,72]:row[f'mae_{h}h']=float(np.nanmean(abs(med[:,h-1]-truth[:,h-1])))
    torch.save({'state':weights,'identity':token,'selected_epoch':be,'seed':seed,'model':model},ckpath);np.savez_compressed(scorepath,errors=errors,median=med,residual=residual,test_ids=te,cal_ids=cal,identity=np.array(token));dump(row,rp.name);dump(history,f'{model}_{seed}_history.json')
   records.append(row);losses.setdefault(model,[]).append(errors)
   scopes={'all':np.ones(len(te),bool),'recent_event':state[te]>0,'no_recent_event':state[te]==0,'nov':times[te].month==11,'dec':times[te].month==12}
   for kind,k in zip(KINDS[1:],range(1,4)):scopes[kind]=state[te]==k
   for scope,mask in scopes.items():
    e=errors[mask];entry={'model':model,'seed':seed,'scope':scope,'origins':int(mask.sum()),'valid_windows':int(np.isfinite(e).sum()),'brier':float(np.nanmean(e)) if np.isfinite(e).any() else None}
    for h in [1,6,24,72]:entry[f'mae_{h}h']=float(np.nanmean(abs(med[mask,h-1]-truth[mask,h-1]))) if mask.any() else None
    windows.append(entry)
   csv(records,'training_metrics.csv');csv(windows,'scope_metrics.csv');print('TRAINED',json.dumps(row),flush=True)
 np.savez_compressed(OUT/'paired_losses.npz',**{k:np.stack(v) for k,v in losses.items()},test_ids=te)
 weeks=((times[te]-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();comparisons=[]
 pairs=[(model,'reference') for model in MODELS[1:]]+[('event_candidate','event_zero'),('event_candidate','event_shuffle')]
 for model,base in pairs:
  a=np.asarray(losses[model]);b=np.asarray(losses[base]);assert np.array_equal(np.isfinite(a),np.isfinite(b));delta=np.mean(a-b,axis=0)
  for scope,mask in [('all',np.ones(len(te),bool)),('recent_event',state[te]>0)]:
   keys=np.unique(weeks[mask]);sums=np.array([np.nansum(delta[mask&(weeks==k)]) for k in keys]);counts=np.array([np.isfinite(delta[mask&(weeks==k)]).sum() for k in keys]);use=counts>0;sums=sums[use];counts=counts[use]
   if not len(counts):continue
   draw=np.random.default_rng(20241101).integers(0,len(counts),(2000,len(counts)));boot=sums[draw].sum(1)/counts[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975])
   comparisons.append({'candidate':model,'baseline':base,'scope':scope,'difference':float(np.nanmean(delta[mask])),'low95':lo,'high95':hi,'blocks':len(counts),'origins':int(mask.sum()),'valid_windows':int(np.isfinite(delta[mask]).sum()),'unadjusted_exploratory':True})
 csv(comparisons,'paired_comparisons.csv');print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('step',choices=['prepare','train']);args=parser.parse_args();globals()[args.step]()

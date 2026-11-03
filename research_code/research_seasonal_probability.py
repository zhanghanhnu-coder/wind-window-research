"""Round25: fixed chronological folds; complete forecast phase precedes scoring."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,shutil,time
import numpy as np
import pandas as pd
import torch
from model import ART
from crps_distribution_core import AtomNet,masked_crps,predict_atoms
from lead_calibration_core import fit_groups,apply_groups,BINS
from seasonal_probability_core import FOLDS,build_fold,score_atoms,check_calibration
from research_crps_distribution import guard,SEEDS

OUT=ART/'seasonal_probability_v1';PLAN=ART.parent.parent/'研究笔记/全年滚动分段概率预测研究方案_2026-11-02.md'
MODELS=['gated_graph_atoms','gated_temporal_atoms','gated_mean_atoms'];METHODS=['raw','smooth_affine']
METRICS=['crps','brier_hourly','mae','coverage','width','low_miss','high_miss','interval_score']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def source_dir(model):return ART/('spatial_outage_controls_v1' if model=='gated_mean_atoms' else 'crps_distribution_v1')
def adjacency(f,model):
 n=len(f['adj'])
 return f['adj'] if model=='gated_graph_atoms' else np.eye(n) if model=='gated_temporal_atoms' else (np.ones((n,n))-np.eye(n))/(n-1)

def prepare():
 guard();assert (ART/'lookahead_execution_v1/final_manifest.json').exists();OUT.mkdir(exist_ok=True);protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0];frozen=OUT/'protocol_frozen.md'
 if frozen.exists():assert frozen.read_text(encoding='utf-8')==protocol
 else:frozen.write_text(protocol,encoding='utf-8')
 files=[frozen,Path(__file__)]+[Path(__file__).with_name(n) for n in ['seasonal_probability_core.py','model.py','crps_distribution_core.py','direct_window_core.py','lead_calibration_core.py','research_crps_distribution.py']]+[ART/'data.npz',ART/'metadata.json',ART/'crps_distribution_v1/evaluation_data.npz']
 for model in MODELS:
  for seed in SEEDS:files.extend([source_dir(model)/f'{model}_{seed}_budget40.pt',source_dir(model)/f'predictions_{model}_{seed}_budget40.npz',source_dir(model)/f'history_{model}_{seed}.json'])
 ident={str(p):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==ident
 else:js(OUT/'identity.json',ident)
 if not (OUT/'previous_hashes.json').exists():js(OUT/'previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 if not (OUT/'recipe.json').exists():js(OUT/'recipe.json',dict(folds=FOLDS,models=MODELS,seeds=SEEDS,epochs=40,batch=32,lr=.001,clip=1.,atoms=19,methods=METHODS,bootstrap_seed=20261113,bootstrap_replicates=2000,selection='preceding validation month raw masked CRPS',calibration='all valid labels in preceding calibration month',new_trainings=27,reused_models=9,start_utc=datetime.now(timezone.utc).isoformat()))
 d=dict(np.load(ART/'data.npz'));folds={};support=[];nodes=read(ART/'metadata.json')['nodes'];node_support=[];times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai')
 for name,cfg in FOLDS.items():
  fold=build_fold(d,name);path=OUT/name;path.mkdir(exist_ok=True);folds[name]=fold;np.savez_compressed(path/'data.npz',**fold)
  for split in ['train','val','cal','test']:
   ids=fold[split+'_ids'];y=np.stack([d['raw'][i+1:i+73] for i in ids]);valid=np.isfinite(y);r=dict(fold=name,split=split,origins=len(ids),valid_origins=int(valid.any((1,2)).sum()),full_72h_allnodes=int(valid.all((1,2)).sum()),valid_labels=int(valid.sum()),total_labels=int(y.size),valid_fraction=float(valid.mean()),mean_wind=float(np.nanmean(y)),wind_le10_rate=float((y[valid]<=10).mean()),first_issue=str(times[ids[0]]),last_outcome=str(times[ids[-1]+72]),center=fold['center'],scale=fold['scale']);support.append(r)
   for j,node in enumerate(nodes):
    a=y[:,:,j];v=np.isfinite(a);node_support.append(dict(fold=name,split=split,node=node,valid_labels=int(v.sum()),total_labels=int(a.size),valid_fraction=float(v.mean()),mean_wind=float(np.nanmean(a)),wind_le10_rate=float((a[v]<=10).mean()) if v.any() else np.nan))
   if split=='cal':check_calibration(y)
   np.savez_compressed(path/f'{split}_truth.npz',y=y)
  if name=='M11':
   meta=read(ART/'metadata.json');old=np.load(ART/'crps_distribution_v1/evaluation_data.npz');assert fold['center']==meta['center'] and fold['scale']==meta['scale'];np.testing.assert_array_equal(fold['x'],d['x']);np.testing.assert_array_equal(fold['adj'],d['adj'])
   for split in ['train','val','cal','test']:np.testing.assert_array_equal(fold[split+'_ids'],old[split+'_ids'])
   js(path/'preprocessing_reuse_equivalence.json',dict(passed=True,fields=['x','center','scale','adj','train_ids','val_ids','cal_ids','test_ids'],comparison='exact array equality before reuse'))
 csv('data_support.csv',support);csv('node_data_support.csv',node_support);return d,folds

def train(folds):
 rows=[];reuse=[]
 for name,f in folds.items():
  path=OUT/name;xx=torch.from_numpy(np.stack([f['x'][i-23:i+1] for i in f['train_ids']]));yy=torch.from_numpy(np.load(path/'train_truth.npz')['y']);vy=torch.from_numpy(np.load(path/'val_truth.npz')['y']);scale=f['scale']
  for seed in SEEDS:
   initial=None
   for model in MODELS:
    guard();A=adjacency(f,model);torch.manual_seed(seed);net=AtomNet(A,scale,hidden=24,gated=True);paramhash=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in net.parameters())).hexdigest()
    if initial is None:initial=paramhash
    else:assert initial==paramhash
    done=path/f'trained_{model}_{seed}.json'
    if done.exists():rows.append(read(done));continue
    start=time.monotonic();ckpath=path/f'{model}_{seed}_budget40.pt';predpath=path/f'predictions_{model}_{seed}_budget40.npz';hp=path/f'history_{model}_{seed}.json'
    if name=='M11':
     src=source_dir(model);ck=torch.load(src/ckpath.name,weights_only=False,map_location='cpu');np.testing.assert_array_equal(A,ck['adj']);assert scale==ck['scale'];net.load_state_dict(ck['state']);old=np.load(src/predpath.name);values={}
     for split in ['val','cal','test']:
      values[split]=predict_atoms(net,f['x'],f[split+'_ids']);np.testing.assert_array_equal(values[split],old[split]);reuse.append(dict(model=model,seed=seed,split=split,max_difference=0.,elements=int(values[split].size)))
     shutil.copy2(src/ckpath.name,ckpath);shutil.copy2(src/hp.name,hp);np.savez_compressed(predpath,**values);bestepoch=ck['epoch'];best=ck['validation_crps']
    else:
     opt=torch.optim.Adam(net.parameters(),lr=.001);best=float('inf');history=[]
     for epoch in range(1,41):
      guard();net.train();order=np.random.default_rng(epoch).permutation(len(xx));total=0.;count=0
      for a in range(0,len(xx),32):
       ix=order[a:a+32];opt.zero_grad();loss=masked_crps(net(xx[ix]),yy[ix])/scale;assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(grad);opt.step();total+=loss.item()*len(ix);count+=len(ix)
      pred=predict_atoms(net,f['x'],f['val_ids']);value=float(masked_crps(torch.from_numpy(pred),vy));history.append(dict(epoch=epoch,train_normalized_crps=total/count,validation_crps=value,elapsed_seconds=time.monotonic()-start))
      if value<best:best=value;state=copy.deepcopy(net.state_dict());bestepoch=epoch
      js(hp,history)
      if epoch%5==0:print('TRAIN',name,model,seed,epoch,round(value,6),flush=True)
     net.load_state_dict(state);torch.save(dict(state=state,adj=A,center=f['center'],scale=scale,config=dict(hidden=24,gated=True,graph_kind=model),seed=seed,epoch=bestepoch,validation_crps=best,budget=40),ckpath);np.savez_compressed(predpath,**{split:predict_atoms(net,f['x'],f[split+'_ids']) for split in ['val','cal','test']})
    row=dict(fold=name,model=model,seed=seed,parameters=sum(p.numel() for p in net.parameters()),initial_parameter_sha256=paramhash,best_epoch=bestepoch,validation_crps=best,reused=name=='M11',source_checkpoint=str(source_dir(model)/ckpath.name) if name=='M11' else '',seconds=time.monotonic()-start);js(done,row);rows.append(row);print('FINISHED_MODEL',name,model,seed,'best',bestepoch,flush=True)
 csv('training_summary.csv',rows)
 if reuse:js(OUT/'reuse_predictions_equivalence.json',reuse)
 assert len(rows)==36

def calibrate(folds):
 rows=[]
 for name,f in folds.items():
  path=OUT/name;y=np.load(path/'cal_truth.npz')['y'];counts=check_calibration(y)
  for model in MODELS:
   for seed in SEEDS:
    guard();dest=path/f'calibration_{model}_{seed}.json';pack=np.load(path/f'predictions_{model}_{seed}_budget40.npz');pfile=path/f'calibration_{model}_{seed}.npz'
    if dest.exists():details=read(dest);theta=np.load(pfile)['theta']
    else:theta,details=fit_groups(pack['cal'],y,f['scale']);np.savez_compressed(pfile,theta=theta);js(dest,details)
    assert [r['valid_targets'] for r in details]==counts
    q=apply_groups(pack['test'],theta,f['scale'],smooth=True);assert np.isfinite(q).all() and (np.diff(q,axis=-1)>=0).all();np.savez_compressed(path/f'calibrated_{model}_{seed}.npz',q=q)
    rows.extend([dict(fold=name,model=model,seed=seed,**r) for r in details]);print('CALIBRATED',name,model,seed,flush=True)
 csv('calibration.csv',rows);assert len(rows)==144;js(OUT/'prediction_complete.json',dict(models=36,methods=2,calibration_groups=144,new_trainings=27,reused=9,utc=datetime.now(timezone.utc).isoformat()))

def evaluate(d,folds):
 assert (OUT/'prediction_complete.json').exists();alltimes=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');nodes=read(ART/'metadata.json')['nodes'];rows=[];months=[];by_node=[];lead=[];reliability=[]
 for name,f in folds.items():
  path=OUT/name;y=np.load(path/'test_truth.npz')['y'];times=alltimes[f['test_ids']]
  for model in MODELS:
   for seed in SEEDS:
    for method in METHODS:
     guard();dest=path/f'scores_{model}_{seed}_{method}.npz';q=np.load(path/f'predictions_{model}_{seed}_budget40.npz')['test'] if method=='raw' else np.load(path/f'calibrated_{model}_{seed}.npz')['q'];s=score_atoms(q,y);np.savez_compressed(dest,**s);common=dict(fold=name,model=model,seed=seed,method=method)
     for metric in METRICS:
      v=s[metric];rows.append(dict(**common,metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
      for month in np.unique(times.month):
       a=v[times.month==month];months.append(dict(**common,month=int(month),metric=metric,value=float(np.nanmean(a)),n=int(np.isfinite(a).sum())))
      for j,node in enumerate(nodes):
       a=v[:,:,j];by_node.append(dict(**common,node=node,metric=metric,value=float(np.nanmean(a)),n=int(np.isfinite(a).sum())))
      for a,b in BINS:
       z=v[:,a:b];lead.append(dict(**common,lead_start=a+1,lead_end=b,metric=metric,value=float(np.nanmean(z)),n=int(np.isfinite(z).sum())))
     valid=np.isfinite(s['label']);p=s['probability'][valid];label=s['label'][valid]
     for b in range(10):
      m=(p>=b/10)&(p<(b+1)/10 if b<9 else p<=1);reliability.append(dict(**common,bin=b,n=int(m.sum()),mean_probability=float(p[m].mean()) if m.any() else np.nan,observed_rate=float(label[m].mean()) if m.any() else np.nan))
     print('SCORED',name,model,seed,method,'CRPS',np.nanmean(s['crps']),'hourlyBrier',np.nanmean(s['brier_hourly']),flush=True)
 df=pd.DataFrame(rows)
 for keys,g in df.groupby(['model','seed','method','metric'],sort=False):rows.append(dict(fold='ALL',**dict(zip(['model','seed','method','metric'],keys)),value=float(np.sum(g.value*g.n)/g.n.sum()),n=int(g.n.sum())))
 csv('metrics.csv',rows);csv('month_metrics.csv',months);csv('node_metrics.csv',by_node);csv('lead_metrics.csv',lead);csv('reliability.csv',reliability)

def compare(d,folds):
 times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');rng=np.random.default_rng(20261113);weeks={};draws={}
 for name,f in folds.items():
  weeks[name]=((times[f['test_ids']]-pd.Timestamp(FOLDS[name]['test_start'],tz=times.tz)).days//7).to_numpy();k=int(weeks[name].max())+1;draws[name]=rng.integers(0,k,(2000,k))
 pairs=[('gated_graph_atoms',method,base,method) for method in METHODS for base in MODELS[1:]]+[(model,'smooth_affine',model,'raw') for model in MODELS];rows=[]
 for model,method,base,bmethod in pairs:
  for metric in ['crps','brier_hourly']:
   sums={};counts={};sizes={}
   for name in folds:
    path=OUT/name;diff=[]
    for seed in SEEDS:
     a=np.load(path/f'scores_{model}_{seed}_{method}.npz')[metric];b=np.load(path/f'scores_{base}_{seed}_{bmethod}.npz')[metric];np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b));diff.append(a-b)
    z=np.mean(diff,0);k=draws[name].shape[1];sums[name]=np.array([np.nansum(z[weeks[name]==j]) for j in range(k)]);counts[name]=np.array([np.isfinite(z[weeks[name]==j]).sum() for j in range(k)]);sizes[name]=len(z)
   for scope in ['ALL']+list(folds):
    selected=list(folds) if scope=='ALL' else [scope];num=np.zeros(2000);den=np.zeros(2000);total=0.;n=0;blocks=[]
    for name in selected:
     s=sums[name];c=counts[name];draw=draws[name];num+=s[draw].sum(1);den+=c[draw].sum(1);total+=s.sum();n+=c.sum();blocks.append(int((c>0).sum()))
    valid=den>0;ok=min(blocks)>=4 and valid.sum()>=1900;lo,hi=np.quantile(num[valid]/den[valid],[.025,.975]) if ok else [np.nan,np.nan]
    rows.append(dict(scope=scope,candidate=model,method=method,baseline=base,baseline_method=bmethod,metric=metric,primary=scope=='ALL' and method==bmethod=='smooth_affine',difference=float(total/n),low95=float(lo),high95=float(hi),n=int(n),origins=sum(sizes[x] for x in selected),nonempty_blocks=sum(blocks),minimum_blocks_in_fold=min(blocks),valid_draws=int(valid.sum()),interval_supported=bool(ok)))
 csv('paired_comparisons.csv',rows)

if __name__=='__main__':
 data,fold_data=prepare();train(fold_data);calibrate(fold_data);evaluate(data,fold_data);compare(data,fold_data);print('COMPLETE',flush=True)

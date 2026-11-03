"""Round32 prediction phase: same-parameter multiscale networks and empirical controls."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,time
import numpy as np
import pandas as pd
import torch
from model import ART
from crps_distribution_core import masked_crps
from multiscale_context_core import MultiScaleAtomNet,multiscale_inputs,predict_multiscale,context_features,climate_forecast,analog_forecast,LEVELS
from lead_calibration_core import fit_groups,apply_groups,BINS
from seasonal_probability_core import FOLDS
from seasonal_rolling_core import calibrate_at
from research_crps_distribution import guard,SEEDS
OUT=ART/'multiscale_context_v1';SRC=ART/'seasonal_probability_v1';COP=ART/'partial_copula_v1';ROLLING=ART/'seasonal_rolling_v1';PLAN=ART.parent.parent/'研究笔记/多时间尺度历史与简单概率基线方案_2026-11-02.md'
OLD={'context1_graph':'gated_graph_atoms','context1_own':'gated_temporal_atoms','context1_mean':'gated_mean_atoms'};NEW={'context3_graph':(3,'graph'),'context7_graph':(7,'graph'),'context7_own':(7,'own'),'context7_mean':(7,'mean')};STATS=['climate28','analog112'];FAMILIES=list(OLD)+list(NEW)+STATS
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def js(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def verify_parents():
 counts={}
 for parent in [SRC,COP,ROLLING]:
  manifest=read(parent/'final_manifest.json')
  for p,h in manifest.items():assert sha(Path(p))==h,p
  counts[parent.name]=len(manifest)
 return counts
def adjacency(f,kind):
 n=len(f['adj']);return f['adj'] if kind=='graph' else np.eye(n) if kind=='own' else (np.ones((n,n))-np.eye(n))/(n-1)
def prepare():
 guard();verify_parents();OUT.mkdir(exist_ok=True);frozen=OUT/'protocol_frozen.md';protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0]
 if frozen.exists():assert frozen.read_text(encoding='utf-8')==protocol
 else:frozen.write_text(protocol,encoding='utf-8')
 files=[frozen,Path(__file__)]+[Path(__file__).parent/n for n in ['multiscale_context_core.py','crps_distribution_core.py','direct_window_core.py','model.py','lead_calibration_core.py','seasonal_probability_core.py','seasonal_rolling_core.py','research_crps_distribution.py','tests/test_multiscale_context.py']]+[p/'final_manifest.json' for p in [SRC,COP,ROLLING]]+[ART/'data.npz',ART/'metadata.json'];identity={str(p):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==identity
 else:js(OUT/'identity.json',identity)
 active=[OUT,ART/'window_pooling_v1',ART/'missing_window_bounds_v1']
 if not (OUT/'previous_hashes.json').exists():js(OUT/'previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and not any(root in p.parents for root in active) and p.suffix in ['.npz','.pt','.csv','.joblib']})
 if not (OUT/'recipe.json').exists():js(OUT/'recipe.json',dict(folds=FOLDS,new_models=NEW,old_models=OLD,statistics=STATS,seeds=SEEDS,epochs=40,batch=32,lr=.001,clip=1.,parameters=45984,encoder_fusion='shared weights; fine/coarse final states averaged equally',methods=['raw','frozen','rolling28'],common_copula='round26 gated_graph_atoms partial05 by fold and seed',bootstrap_seed=20261119,bootstrap_replicates=2000,new_trainings=48,unique_statistical_producers=8,concurrently_active_excluded_from_old_snapshot=[str(x) for x in active],start_utc=datetime.now(timezone.utc).isoformat()))
 data=dict(np.load(ART/'data.npz'));folds={name:dict(np.load(SRC/name/'data.npz')) for name in FOLDS};times=pd.DatetimeIndex(pd.to_datetime(data['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');rows=[]
 for name,f in folds.items():
  path=OUT/name;path.mkdir(exist_ok=True);assert min(f['train_ids'])>=167;fit=times<pd.Timestamp(FOLDS[name]['train_end'],tz=times.tz);counts=np.isfinite(data['raw'][fit]).sum(0);assert counts.min()>=24;prior=np.stack([np.quantile(data['raw'][fit,j][np.isfinite(data['raw'][fit,j])],LEVELS,method='inverted_cdf') for j in range(17)]);np.savez_compressed(path/'baseline_prior.npz',q=prior,counts=counts,fit_mask=fit)
  rows.append(dict(fold=name,minimum_train_issue=int(min(f['train_ids'])),long_history_start=int(min(f['train_ids'])-167),minimum_prior_values=int(counts.min()),train_origins=len(f['train_ids']),val_origins=len(f['val_ids']),cal_origins=len(f['cal_ids']),test_origins=len(f['test_ids'])))
 csv('data_preflight.csv',rows);return data,folds
def train_and_archive(folds):
 rows=[];original=pd.read_csv(SRC/'training_summary.csv')
 for name,f in folds.items():
  path=OUT/name;archive=np.load(ROLLING/name/'archive_data.npz');ids=archive['ids'];position={int(i):j for j,i in enumerate(ids)};known={}
  for split in ['cal','test']:
   for j,i in enumerate(f[split+'_ids']):known[int(i)]=(split,j)
  newpos=np.array([j for j,i in enumerate(ids) if int(i) not in known],int);yy=torch.from_numpy(np.load(SRC/name/'train_truth.npz')['y']);vy=torch.from_numpy(np.load(SRC/name/'val_truth.npz')['y']);scale=float(f['scale']);inputs={span:tuple(torch.from_numpy(x) for x in multiscale_inputs(f['x'],f['train_ids'],span)) for span in [3,7]}
  for seed in SEEDS:
   for family,(span,kind) in NEW.items():
    guard();done=path/f'trained_{family}_{seed}.json'
    if done.exists():
     record=read(done)
     for file,h in record['output_hashes'].items():assert sha(Path(file))==h,file
     rows.append(record);continue
    A=adjacency(f,kind);torch.manual_seed(seed);net=MultiScaleAtomNet(A,scale,span);paramhash=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in net.parameters())).hexdigest();expected=original[(original.fold==name)&(original.seed==seed)].initial_parameter_sha256.unique();assert len(expected)==1 and paramhash==expected[0];assert sum(p.numel() for p in net.parameters())==45984;optimizer=torch.optim.Adam(net.parameters(),lr=.001);fine,coarse=inputs[span];best=float('inf');history=[];started=time.monotonic();hp=path/f'history_{family}_{seed}.json'
    for epoch in range(1,41):
     guard();net.train();order=np.random.default_rng(epoch).permutation(len(fine));total=0.;count=0
     for start in range(0,len(fine),32):
      ix=order[start:start+32];optimizer.zero_grad();loss=masked_crps(net(fine[ix],coarse[ix]),yy[ix])/scale;assert torch.isfinite(loss);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(gradient);optimizer.step();total+=loss.item()*len(ix);count+=len(ix)
     prediction=predict_multiscale(net,f['x'],f['val_ids']);value=float(masked_crps(torch.from_numpy(prediction),vy));assert np.isfinite(value);history.append(dict(epoch=epoch,train_normalized_crps=total/count,validation_crps=value,elapsed_seconds=time.monotonic()-started))
     if value<best:best=value;state=copy.deepcopy(net.state_dict());bestepoch=epoch
     js(hp,history)
     if epoch%5==0:print('TRAIN',name,family,seed,epoch,round(value,6),flush=True)
    net.load_state_dict(state);ckpath=path/f'{family}_{seed}_budget40.pt';predpath=path/f'predictions_{family}_{seed}.npz';apath=path/f'archive_predictions_{family}_{seed}.npz';torch.save(dict(state=state,adj=A,center=f['center'],scale=scale,span=span,family=family,seed=seed,epoch=bestepoch,validation_crps=best,budget=40),ckpath);pred={split:predict_multiscale(net,f['x'],f[split+'_ids']) for split in ['val','cal','test']};np.savez_compressed(predpath,**pred);q=np.empty((len(ids),72,17,19),np.float32)
    for i,(split,j) in known.items():q[position[i]]=pred[split][j]
    if len(newpos):q[newpos]=predict_multiscale(net,f['x'],ids[newpos])
    np.savez_compressed(apath,q=q);record=dict(fold=name,model=family,seed=seed,span=span,graph_kind=kind,parameters=45984,initial_parameter_sha256=paramhash,best_epoch=bestepoch,validation_crps=best,training_and_export_seconds=time.monotonic()-started,archive_origins=len(ids),new_archive_origins=len(newpos),output_hashes={str(p):sha(p) for p in [ckpath,predpath,apath,hp]});js(done,record);rows.append(record);print('FINISHED_MODEL',name,family,seed,'best',bestepoch,'seconds',round(record['training_and_export_seconds'],1),flush=True)
 csv('training_summary.csv',rows);assert len(rows)==48;js(OUT/'training_complete.json',dict(new_models=48,utc=datetime.now(timezone.utc).isoformat()))
def statistical_archives(data,folds):
 raw=data['raw'];features=context_features(raw);summary=[]
 for name,f in folds.items():
  path=OUT/name;a=np.load(ROLLING/name/'archive_data.npz');ids=a['ids'];position={int(i):j for j,i in enumerate(ids)};prior=np.load(path/'baseline_prior.npz')['q']
  for family in STATS:
   guard();done=path/f'statistics_{family}.json'
   if done.exists():
    record=read(done)
    for file,h in record['output_hashes'].items():assert sha(Path(file))==h,file
    summary.append(record);continue
   predictions=[];climate_counts=[];climate_fallback=[];source_ids=np.full((len(ids),17,32),-1,int);valid_targets=np.zeros((len(ids),72,17),np.int16);qualified=np.zeros((len(ids),17),bool)
   for j,issue in enumerate(ids):
    guard()
    if family=='climate28':q,details=climate_forecast(raw,int(issue),prior);climate=details
    else:
     q,details=analog_forecast(raw,int(issue),prior,float(f['scale']),features);climate=details['climate'];valid_targets[j]=np.asarray(details['valid_targets']).T;qualified[j]=details['current_qualified']
     for node,selected in enumerate(details['source_ids']):source_ids[j,node,:len(selected)]=selected
    predictions.append(q);climate_counts.append(climate['counts']);mask=np.zeros(17,bool);mask[climate['fallback_nodes']]=True;climate_fallback.append(mask)
    if j%100==0:print('STATISTICS',name,family,j,flush=True)
   q=np.stack(predictions);apath=path/f'archive_predictions_{family}_0.npz';ppath=path/f'predictions_{family}_0.npz';auditpath=path/f'statistical_sources_{family}.npz';np.savez_compressed(apath,q=q);np.savez_compressed(ppath,**{split:q[[position[int(i)] for i in f[split+'_ids']]] for split in ['cal','test']});np.savez_compressed(auditpath,ids=ids,climate_counts=climate_counts,climate_fallback=climate_fallback,source_ids=source_ids,valid_targets=valid_targets,current_qualified=qualified);record=dict(fold=name,model=family,producer_seed=0,archive_origins=len(ids),cal_origins=len(f['cal_ids']),test_origins=len(f['test_ids']),climate_fallback_node_origins=int(np.sum(climate_fallback)),analog_fallback_hour_node_targets=int((valid_targets<8).sum()) if family=='analog112' else None,output_hashes={str(p):sha(p) for p in [apath,ppath,auditpath]});js(done,record);summary.append(record)
 csv('statistical_summary.csv',summary);assert len(summary)==8
def producers():
 rows=[]
 for name in FOLDS:
  for family in FAMILIES:
   for seed in ([0] if family in STATS else SEEDS):
    old=family in OLD;apath=ROLLING/name/f'archive_predictions_{OLD[family]}_{seed}.npz' if old else OUT/name/f'archive_predictions_{family}_{seed}.npz';ppath=SRC/name/f'predictions_{OLD[family]}_{seed}_budget40.npz' if old else OUT/name/f'predictions_{family}_{seed}.npz';rows.append(dict(fold=name,model=family,producer_seed=seed,reused=old,archive_source=str(apath),archive_sha256=sha(apath),predictions_source=str(ppath),predictions_sha256=sha(ppath)))
 assert len(rows)==92;js(OUT/'producers.json',rows);return rows
def calibrate(folds):
 producer_rows=producers();fixedrows=[];updates=[];detailrows=[];predsources={};support_fallbacks=0;optimizer_failures=0;fitted=0
 for producer in producer_rows:
  guard();name=producer['fold'];family=producer['model'];seed=producer['producer_seed'];path=OUT/name;f=folds[name];scale=float(f['scale'])
  if producer['reused']:
   source_model=OLD[family];predsources[name,family,seed]={'raw':(Path(producer['predictions_source']),'test'),'frozen':(SRC/name/f'calibrated_{source_model}_{seed}.npz','q'),'rolling28':(ROLLING/name/f'predictions_{source_model}_{seed}_rolling28.npz','q')};continue
  pack=np.load(producer['predictions_source']);q=pack['test'];cy=np.load(SRC/name/'cal_truth.npz')['y'];cp=path/f'calibration_{family}_{seed}.npz';cj=path/f'calibration_{family}_{seed}.json'
  if cp.exists() and cj.exists():theta=np.load(cp)['theta'];details=read(cj)
  else:theta,details=fit_groups(pack['cal'],cy,scale);np.savez_compressed(cp,theta=theta);js(cj,details)
  fixedrows.extend([dict(fold=name,model=family,producer_seed=seed,**r) for r in details]);frozen=apply_groups(q,theta,scale,True);frozenpath=path/f'predictions_{family}_{seed}_frozen.npz';np.savez_compressed(frozenpath,q=frozen);a=np.load(ROLLING/name/'archive_data.npz');archive=np.load(producer['archive_source'])['q'];schedule=read(ROLLING/name/'schedule.json');previous=theta.copy();rolling=np.empty(frozen.shape,float)
  for entry in schedule:
   guard();block=entry['block'];issue=entry['issue_id'];positions=np.asarray(entry['positions']);base=f'update_{family}_{seed}_rolling28_w{block}';up=path/(base+'.npz');uj=path/(base+'.json')
   if block==0:
    current=theta.copy();record=dict(audit=dict(issue_id=issue,initial=True,reused_previous=False,fitted_groups=0,optimizer_failures=0,valid_origins=int(np.isfinite(cy).any((1,2)).sum()),valid_labels=int(np.isfinite(cy).sum()),earliest_label_id=int(min(f['cal_ids'])+1),latest_label_id=int(max(f['cal_ids']+72))),details=details);np.savez_compressed(up,theta=current);js(uj,record);rolling[positions]=frozen[positions]
   elif up.exists() and uj.exists():current=np.load(up)['theta'];record=read(uj);rolling[positions]=apply_groups(q[positions],current,scale,True)
   else:
    r=calibrate_at(archive,a['truth'],a['ids'],issue,28,scale,previous);current=r['theta'];record=dict(audit=dict(initial=False,**r['audit']),details=r['details']);np.savez_compressed(up,theta=current,mask=r['mask']);js(uj,record);rolling[positions]=apply_groups(q[positions],current,scale,True)
   au=record['audit'];assert au['latest_label_id'] is None or au['latest_label_id']<issue;previous=current.copy();fitted+=au['fitted_groups'];support_fallbacks+=int(au['reused_previous']);optimizer_failures+=au['optimizer_failures'];updates.append(dict(fold=name,model=family,producer_seed=seed,block=block,update_id=issue,applied_origins=len(positions),**au))
   if block>0:detailrows.extend([dict(fold=name,model=family,producer_seed=seed,block=block,**r) for r in record['details']])
  rollingpath=path/f'predictions_{family}_{seed}_rolling28.npz';np.savez_compressed(rollingpath,q=rolling);predsources[name,family,seed]={'raw':(Path(producer['predictions_source']),'test'),'frozen':(frozenpath,'q'),'rolling28':(rollingpath,'q')};print('CALIBRATED',name,family,seed,flush=True)
 inventory=[]
 for name in FOLDS:
  for family in FAMILIES:
   for seed in SEEDS:
    producer_seed=0 if family in STATS else seed;ranks=COP/name/f'ranks_gated_graph_atoms_{seed}_partial05.npz'
    for method,(source,key) in predsources[name,family,producer_seed].items():inventory.append(dict(fold=name,model=family,seed=seed,producer_seed=producer_seed,method=method,source=str(source),source_key=key,source_sha256=sha(source),ranks_source=str(ranks),ranks_sha256=sha(ranks),origins=len(folds[name]['test_ids']),score_source=str(OUT/name/f'scores_{family}_{seed}_{method}.npz')))
 assert len(fixedrows)==224 and len(updates)==504 and len(detailrows)==1792 and len(inventory)==324;csv('fixed_calibration.csv',fixedrows);csv('update_audit.csv',updates);csv('rolling_calibration.csv',detailrows);js(OUT/'inventory.json',inventory);verify_parents();js(OUT/'prediction_complete.json',dict(status='COMPLETE',new_neural_trainings=48,unique_producers=92,new_producers=56,unique_statistical_producers=8,flows=324,forecast_origins=sum(r['origins'] for r in inventory),fixed_groups=len(fixedrows),weekly_updates=448,rolling_groups=fitted,support_fallbacks=support_fallbacks,optimizer_failures=optimizer_failures,fixed_optimizer_failures=sum(r['fallback'] for r in fixedrows),utc=datetime.now(timezone.utc).isoformat()));print('PREDICTIONS_COMPLETE',flush=True)
if __name__=='__main__':
 data,folds=prepare();train_and_archive(folds);statistical_archives(data,folds);calibrate(folds)

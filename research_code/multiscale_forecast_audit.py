"""Independent input blocks, empirical order statistics and mature calibration audit."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from model import ART
from multiscale_context_core import MultiScaleAtomNet,predict_multiscale
from crps_distribution_core import masked_crps
from verify_seasonal_rolling import objective,transform
from research_crps_distribution import guard
OUT=ART/'multiscale_context_v1';SRC=ART/'seasonal_probability_v1';ROLLING=ART/'seasonal_rolling_v1';COP=ART/'partial_copula_v1'
FOLDS={'M05':'2024-03-01','M07':'2024-05-01','M09':'2024-07-01','M11':'2024-09-01'};SEEDS=[20241101,20241102,20241103];NEW={'context3_graph':(3,'graph'),'context7_graph':(7,'graph'),'context7_own':(7,'own'),'context7_mean':(7,'mean')};OLD={'context1_graph':'gated_graph_atoms','context1_own':'gated_temporal_atoms','context1_mean':'gated_mean_atoms'};STATS=['climate28','analog112'];BINS=[(0,6),(6,24),(24,48),(48,72)]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=3e-12):np.testing.assert_allclose(a,b,rtol=0,atol=tol,equal_nan=True)
def independent_inputs(x,ids,span):
 fine=[];coarse=[]
 for issue in ids:
  fine.append(x[issue-23:issue+1]);blocks=[]
  for end in range(int(issue)-23*span,int(issue)+1,span):
   block=x[end-span+1:end+1].astype(float);valid=block[:, :,2]>0;count=valid.sum(0);value=np.zeros((17,3),float)
   for node in range(17):
    if count[node]:value[node,:2]=np.sum(block[:,node,:2]*valid[:,node,None],axis=0)/count[node]
    value[node,2]=count[node]/span
   blocks.append(value)
  coarse.append(blocks)
 return np.asarray(fine,np.float32),np.asarray(coarse,np.float32)
def independent_predict(net,x,ids):
 net.eval();out=[]
 with torch.no_grad():
  for start in range(0,len(ids),32):
   fine,coarse=independent_inputs(x,ids[start:start+32],net.span);out.append(net(torch.from_numpy(fine),torch.from_numpy(coarse)).numpy())
 return np.concatenate(out)
def empirical(values):
 values=np.sort(values[np.isfinite(values)]);assert len(values)>0;index=np.ceil(((np.arange(19)+.5)/19)*len(values)).astype(int)-1;return values[index]
def feature_reference(raw):
 x=np.asarray(raw,float);features=np.zeros((len(x),17,4));qualified=np.zeros((len(x),17),bool)
 for i in range(len(x)):
  window=x[max(0,i-23):i+1];valid=np.isfinite(window);count=valid.sum(0);mean=np.where(valid,window,0).sum(0)/np.maximum(count,1);recent=x[max(0,i-5):i+1];before=x[max(0,i-11):max(0,i-5)];rv=np.isfinite(recent);bv=np.isfinite(before);rc=rv.sum(0);bc=bv.sum(0);rm=np.where(rv,recent,0).sum(0)/np.maximum(rc,1);bm=np.where(bv,before,0).sum(0)/np.maximum(bc,1);std=np.sqrt(np.where(valid,(window-mean)**2,0).sum(0)/np.maximum(count,1));features[i]=np.column_stack([mean,rm,rm-bm,std]);qualified[i]=(count>=18)&(rc>=4)&(bc>=4)
 return features,qualified
def statistical_reference(raw,issue,prior,scale,features,qualified,family):
 q=np.broadcast_to(prior,(72,17,19)).copy();counts=[];fallback=[];sources=np.full((17,32),-1,int);valid_targets=np.zeros((72,17),int);current=np.zeros(17,bool)
 for node in range(17):
  bank=raw[max(0,issue-672):issue,node];good=bank[np.isfinite(bank)];counts.append(len(good));fallback.append(len(good)<24)
  if len(good)>=24:q[:,node]=empirical(good)
 if family=='analog112':
  candidates=np.array([j for j in range(23,issue,6) if j+72<issue and j>=issue-2688],int);current=qualified[issue]
  for node in range(17):
   if not current[node]:continue
   eligible=candidates[qualified[candidates,node]];distance=np.sum(((features[eligible,node]-features[issue,node])/scale)**2,axis=-1);chosen=eligible[np.argsort(distance,kind='stable')[:32]];sources[node,:len(chosen)]=chosen
   for h in range(72):
    v=raw[chosen+h+1,node];v=v[np.isfinite(v)];valid_targets[h,node]=len(v)
    if len(v)>=8:q[h,node]=empirical(v)
 return q,np.array(counts),np.array(fallback),sources,valid_targets,current
def audit_forecasts():
 data=np.load(ART/'data.npz');raw=data['raw'];times=pd.DatetimeIndex(pd.to_datetime(data['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');features,qualified=feature_reference(raw);training=pd.read_csv(OUT/'training_summary.csv');oldtraining=pd.read_csv(SRC/'training_summary.csv');counts=dict(new_checkpoints=0,full_neural_reload_splits=0,full_archive_reload_groups=0,future_input_perturbations=0,statistical_archive_forecasts=0,statistical_sources=0,fixed_objectives=0,weekly_maturity_masks=0,rolling_objectives=0,independent_refits=0,future_label_perturbations=0,applied_calibration_flows=0,reused_producers=0);refits=[];producers=read(OUT/'producers.json');assert len(producers)==92 and len(training)==48
 for name,train_end in FOLDS.items():
  f=dict(np.load(SRC/name/'data.npz'));scale=float(f['scale']);a=np.load(ROLLING/name/'archive_data.npz');ids=a['ids'];position={int(i):j for j,i in enumerate(ids)};path=OUT/name;fit=times<pd.Timestamp(train_end,tz=times.tz);prior=np.stack([empirical(raw[fit,node]) for node in range(17)]);savedprior=np.load(path/'baseline_prior.npz');np.testing.assert_array_equal(savedprior['q'],prior);np.testing.assert_array_equal(savedprior['fit_mask'],fit);np.testing.assert_array_equal(savedprior['counts'],np.isfinite(raw[fit]).sum(0))
  for family,(span,kind) in NEW.items():
   adj=f['adj'] if kind=='graph' else np.eye(17) if kind=='own' else (np.ones((17,17))-np.eye(17))/16
   for seed in SEEDS:
    guard();ck=torch.load(path/f'{family}_{seed}_budget40.pt',map_location='cpu',weights_only=False);np.testing.assert_array_equal(ck['adj'],adj);assert ck['span']==span and ck['family']==family and ck['seed']==seed and ck['budget']==40 and ck['scale']==scale;torch.manual_seed(seed);net=MultiScaleAtomNet(adj,scale,span);ph=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in net.parameters())).hexdigest();expected=oldtraining[(oldtraining.fold==name)&(oldtraining.seed==seed)].initial_parameter_sha256.unique();assert len(expected)==1 and ph==expected[0];assert sum(p.numel() for p in net.parameters())==45984;net.load_state_dict(ck['state']);history=read(path/f'history_{family}_{seed}.json');assert [r['epoch'] for r in history]==list(range(1,41));best=min(history,key=lambda r:r['validation_crps']);assert ck['epoch']==best['epoch'];close(ck['validation_crps'],best['validation_crps']);row=training[(training.fold==name)&(training.model==family)&(training.seed==seed)].iloc[0];assert row.initial_parameter_sha256==ph and row.best_epoch==ck['epoch'] and row.parameters==45984;pred=np.load(path/f'predictions_{family}_{seed}.npz');archive=np.load(path/f'archive_predictions_{family}_{seed}.npz')['q'];known=set()
    for split in ['val','cal','test']:
     reload=independent_predict(net,f['x'],f[split+'_ids']);np.testing.assert_array_equal(reload,pred[split]);counts['full_neural_reload_splits']+=1
     if split=='val':close(float(masked_crps(torch.from_numpy(reload),torch.from_numpy(np.load(SRC/name/'val_truth.npz')['y']))),ck['validation_crps'])
     else:
      np.testing.assert_array_equal(archive[[position[int(i)] for i in f[split+'_ids']]],pred[split]);known.update(map(int,f[split+'_ids']))
    new=np.array([j for j,i in enumerate(ids) if int(i) not in known],int)
    if len(new):np.testing.assert_array_equal(archive[new],independent_predict(net,f['x'],ids[new]))
    at=int(ids[0]);changed=f['x'].copy();changed[at+1:]=12345.;np.testing.assert_array_equal(predict_multiscale(net,changed,np.array([at])),predict_multiscale(net,f['x'],np.array([at])));counts['future_input_perturbations']+=1;counts['full_archive_reload_groups']+=1;counts['new_checkpoints']+=1;done=read(path/f'trained_{family}_{seed}.json')
    for file,h in done['output_hashes'].items():assert sha(Path(file))==h,file
    print('MODEL_AUDIT',name,family,seed,flush=True)
  for family in STATS:
   guard();archive=np.load(path/f'archive_predictions_{family}_0.npz')['q'];source=np.load(path/f'statistical_sources_{family}.npz');np.testing.assert_array_equal(source['ids'],ids)
   for j,issue in enumerate(ids):
    q,c,fb,s,v,current=statistical_reference(raw,int(issue),prior,scale,features,qualified,family);np.testing.assert_array_equal(q,archive[j]);np.testing.assert_array_equal(c,source['climate_counts'][j]);np.testing.assert_array_equal(fb,source['climate_fallback'][j]);np.testing.assert_array_equal(s,source['source_ids'][j]);np.testing.assert_array_equal(v,source['valid_targets'][j]);np.testing.assert_array_equal(current,source['current_qualified'][j]);counts['statistical_archive_forecasts']+=1
   pred=np.load(path/f'predictions_{family}_0.npz')
   for split in ['cal','test']:np.testing.assert_array_equal(pred[split],archive[[position[int(i)] for i in f[split+'_ids']]])
   done=read(path/f'statistics_{family}.json')
   for file,h in done['output_hashes'].items():assert sha(Path(file))==h,file
   counts['statistical_sources']+=1;print('STATISTICS_AUDIT',name,family,flush=True)
  schedule=read(ROLLING/name/'schedule.json');cy=np.load(SRC/name/'cal_truth.npz')['y'];target=ids[:,None]+np.arange(1,73)
  for producer in [p for p in producers if p['fold']==name]:
   guard();family=producer['model'];seed=producer['producer_seed'];assert sha(Path(producer['archive_source']))==producer['archive_sha256'] and sha(Path(producer['predictions_source']))==producer['predictions_sha256']
   if producer['reused']:
    assert family in OLD and producer['archive_source']==str(ROLLING/name/f'archive_predictions_{OLD[family]}_{seed}.npz') and producer['predictions_source']==str(SRC/name/f'predictions_{OLD[family]}_{seed}_budget40.npz');counts['reused_producers']+=1;continue
   archive=np.load(producer['archive_source'])['q'];pred=np.load(producer['predictions_source']);theta0=np.load(path/f'calibration_{family}_{seed}.npz')['theta'];details0=read(path/f'calibration_{family}_{seed}.json');fixed=np.load(path/f'predictions_{family}_{seed}_frozen.npz')['q'];rolling=np.load(path/f'predictions_{family}_{seed}_rolling28.npz')['q'];np.testing.assert_array_equal(fixed,transform(pred['test'],theta0,scale));previous=theta0.copy()
   for g,(lo,hi) in enumerate(BINS):
    mask=np.isfinite(cy[:,lo:hi]);qq=pred['cal'][:,lo:hi][mask].astype(float)/scale;yy=cy[:,lo:hi][mask].astype(float)/scale;value=objective(theta0[g],qq,yy)[0];close(value,details0[g]['objective_used'],1e-10);assert details0[g]['valid_targets']==mask.sum();counts['fixed_objectives']+=1
    if details0[g]['fallback']:np.testing.assert_array_equal(theta0[g],[0.,1.])
   for entry in schedule:
    block=entry['block'];issue=entry['issue_id'];pos=np.asarray(entry['positions']);z=np.load(path/f'update_{family}_{seed}_rolling28_w{block}.npz');record=read(path/f'update_{family}_{seed}_rolling28_w{block}.json');theta=z['theta'];au=record['audit'];assert au['issue_id']==issue and au['initial']==(block==0)
    if block==0:np.testing.assert_array_equal(theta,theta0);np.testing.assert_array_equal(rolling[pos],fixed[pos]);assert record['details']==details0 and max(f['cal_ids']+72)<issue
    else:
     mask=np.isfinite(a['truth'])&((ids[:,None]<issue)&(target>=issue-672)&(target<issue))[:,:,None];np.testing.assert_array_equal(mask,z['mask']);selected=np.flatnonzero(mask.any((1,2)));gc=[int(mask[:,lo:hi].sum()) for lo,hi in BINS];support=len(selected)>=20 and min(gc)>=100;known=target[mask.any(-1)];assert au['valid_origins']==len(selected) and au['valid_labels']==mask.sum() and au['source_positions']==selected.tolist() and au['source_ids']==ids[selected].tolist() and au['group_counts']==gc and au['reused_previous']==(not support) and au['fitted_groups']==(4 if support else 0);assert au['earliest_label_id']==(int(known.min()) if len(known) else None) and au['latest_label_id']==(int(known.max()) if len(known) else None);counts['weekly_maturity_masks']+=1
     if not support:np.testing.assert_array_equal(theta,previous)
     for g,(lo,hi) in enumerate(BINS):
      info=record['details'][g];assert info['valid_targets']==gc[g];close([info['delta'],info['gamma']],theta[g],1e-14)
      if support:
       qq=archive[:,lo:hi][mask[:,lo:hi]].astype(float)/scale;yy=a['truth'][:,lo:hi][mask[:,lo:hi]].astype(float)/scale;value=objective(theta[g],qq,yy)[0];close(value,info['objective_used'],1e-10);counts['rolling_objectives']+=1;assert -2<=theta[g,0]<=2 and .25<=theta[g,1]<=4
       if info['fallback']:np.testing.assert_array_equal(theta[g],[0.,1.]);assert not info['success']
       else:assert info['success'];close(value,info['objective_candidate'],1e-10)
       if seed in [0,SEEDS[0]] and block==4:
        fit=minimize(objective,[0.,1.],args=(qq,yy),jac=True,method='L-BFGS-B',bounds=[(-2,2),(.25,4)],options=dict(maxiter=200,ftol=1e-10,gtol=1e-7));used=fit.x if fit.success and np.isfinite(fit.fun) else np.array([0.,1.]);ref=objective(used,qq,yy)[0];assert abs(ref-value)<2e-6;refits.append(dict(fold=name,model=family,producer_seed=seed,block=block,group=g,independent_success=bool(fit.success),independent_objective=ref,saved_objective=value,max_parameter_difference=float(max(abs(used-theta[g])))));counts['independent_refits']+=1
      else:assert info['objective_used'] is None and info['fallback']
     assert au['optimizer_failures']==(sum(r['fallback'] for r in record['details']) if support else 0)
     if ((family=='context7_graph' and seed==SEEDS[0]) or family in STATS) and block==4:
      from seasonal_rolling_core import calibrate_at
      polluted=a['truth'].copy();polluted[np.broadcast_to((target>=issue)[:,:,None],polluted.shape)]=98765.;changed=calibrate_at(archive,polluted,ids,issue,28,scale,previous);np.testing.assert_array_equal(changed['theta'],theta);np.testing.assert_array_equal(changed['mask'],mask);assert changed['audit']=={k:v for k,v in au.items() if k!='initial'};counts['future_label_perturbations']+=1
    np.testing.assert_array_equal(rolling[pos],transform(pred['test'][pos],theta,scale));previous=theta.copy()
   counts['applied_calibration_flows']+=2;print('CALIBRATION_AUDIT',name,family,seed,flush=True)
 pd.DataFrame(refits).to_csv(OUT/'independent_refits.csv',index=False,encoding='utf-8-sig');assert counts['new_checkpoints']==48 and counts['statistical_sources']==8 and counts['fixed_objectives']==224 and counts['weekly_maturity_masks']==448 and counts['reused_producers']==36;counts.update(audit_calibration_tables(producers));return counts

def audit_calibration_tables(producers):
 fixed=[];updates=[];details=[]
 for producer in producers:
  if producer['reused']:continue
  name=producer['fold'];family=producer['model'];seed=producer['producer_seed'];path=OUT/name
  fixed.extend([dict(fold=name,model=family,producer_seed=seed,**r) for r in read(path/f'calibration_{family}_{seed}.json')])
  for entry in read(ROLLING/name/'schedule.json'):
   block=entry['block'];record=read(path/f'update_{family}_{seed}_rolling28_w{block}.json');updates.append(dict(fold=name,model=family,producer_seed=seed,block=block,update_id=entry['issue_id'],applied_origins=len(entry['positions']),**record['audit']))
   if block:details.extend([dict(fold=name,model=family,producer_seed=seed,block=block,**r) for r in record['details']])
 counts={}
 for file,rows,keys in [('fixed_calibration.csv',fixed,['fold','model','producer_seed','group']),('update_audit.csv',updates,['fold','model','producer_seed','block']),('rolling_calibration.csv',details,['fold','model','producer_seed','block','group'])]:
  expected=pd.DataFrame(rows);keys=[k for k in keys if k in expected];expected=expected.sort_values(keys).reset_index(drop=True);saved=pd.read_csv(OUT/file).sort_values(keys).reset_index(drop=True);assert list(saved.columns)==list(expected.columns) and len(saved)==len(expected)
  for k in expected:
   if expected[k].dtype.kind in 'biufc':close(saved[k],expected[k],3e-11)
   else:
    norm=lambda x: '' if x is None or (isinstance(x,float) and np.isnan(x)) else str(x)
    assert saved[k].map(norm).tolist()==expected[k].map(norm).tolist(),(file,k)
  counts[file.removesuffix('.csv')+'_rows']=len(rows)
 assert [counts[k] for k in ['fixed_calibration_rows','update_audit_rows','rolling_calibration_rows']]==[224,504,1792]
 return counts

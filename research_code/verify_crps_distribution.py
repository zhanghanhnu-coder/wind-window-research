"""Audit frozen trained distributions, coupling, scores, and temporal boundaries."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import torch
from model import ART
from crps_distribution_core import AtomNet,predict_atoms,affine_objective
from research_crps_distribution import CONFIGS,SEEDS,scenario_generator
OUT=ART/'crps_distribution_v1'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def empirical(q,y):
 q=np.sort(np.asarray(q,float),axis=-1);k=q.shape[-1];return np.mean(abs(q-y[...,None]),-1)-np.sum(q*(2*np.arange(k)-k+1),-1)/(k*k)
def run():
 old=read('previous_hashes.json');identity=read('identity.json')
 for p,h in old.items():assert sha(ART/p)==h,p
 for p,h in identity.items():assert sha(Path(p))==h,p
 d=np.load(ART/'data.npz');meta=json.loads((ART/'metadata.json').read_text(encoding='utf-8'));ev=np.load(OUT/'evaluation_data.npz');src=np.load(ART/'direct_window_v1/split_data.npz');times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');ids={k:ev[k+'_ids'] for k in ['train','val','cal','test']}
 bounds={'train':('2024-01-12','2024-09-01'),'val':('2024-09-01','2024-10-01'),'cal':('2024-10-01','2024-11-01'),'test':('2024-11-01','2025-01-01')}
 for k,ii in ids.items():
  np.testing.assert_array_equal(ii,src[k+'_ids']);a,b=bounds[k];assert times[ii].min()>=pd.Timestamp(a,tz=times.tz) and times[ii+72].max()<pd.Timestamp(b,tz=times.tz);assert all(d['x'][i-23:i+1,:,-1].mean()>=.65 for i in ii)
 caltruth=np.stack([d['raw'][i+1:i+73] for i in ids['cal']]);truth=np.stack([d['raw'][i+1:i+73] for i in ids['test']]);np.testing.assert_array_equal(caltruth,ev['cal_truth']);np.testing.assert_array_equal(truth,ev['test_truth']);complete=ev['complete_positions'];selected=ev['selected57'];assert np.isfinite(caltruth[complete]).all();np.testing.assert_array_equal(ids['cal'][complete],d['cal_ids']);np.testing.assert_array_equal(selected,np.random.default_rng(20261105).choice(58,57,replace=False));assert len(set(selected))==57;assert max(ids['cal'][complete]+72)<min(ids['test'])
 checkpoints=0;maxpred=0.;budgetrows=[];calibration=read('recipe.json');frame=pd.read_csv(OUT/'calibration.csv');atomchecks=0
 for name,cfg in CONFIGS.items():
  for seed in SEEDS:
   history=read(f'history_{name}_{seed}.json');assert len(history)==40
   for budget in [20,40]:
    ck=torch.load(OUT/f'{name}_{seed}_budget{budget}.pt',weights_only=False,map_location='cpu');expected=min(history[:budget],key=lambda r:r['validation_crps']);assert ck['epoch']==expected['epoch'] and ck['validation_crps']==expected['validation_crps'];net=AtomNet(ck['adj'],ck['scale'],hidden=cfg['hidden'],gated=cfg['gated']);net.load_state_dict(ck['state']);pred=np.load(OUT/f'predictions_{name}_{seed}_budget{budget}.npz')
    for part in ['val','cal','test']:
     q=pred[part];assert q.shape==(len(ids[part]),72,17,19) and np.isfinite(q).all() and (q>=0).all() and (np.diff(q,axis=-1)>=0).all();fresh=predict_atoms(net,d['x'],ids[part][[0,-1]]);delta=float(np.max(abs(fresh-q[[0,-1]])));maxpred=max(maxpred,delta);np.testing.assert_allclose(fresh,q[[0,-1]],atol=1e-5,rtol=1e-6)
    vy=np.stack([d['raw'][i+1:i+73] for i in ids['val']]);vc=np.nanmean(empirical(pred['val'],vy));assert abs(vc-ck['validation_crps'])<2e-6
    budgetrows.append(dict(model=name,seed=seed,budget=budget,selected_epoch=ck['epoch'],test_raw_atom_crps=float(np.nanmean(empirical(pred['test'],truth)))));checkpoints+=1
   pred=np.load(OUT/f'predictions_{name}_{seed}_budget40.npz');bank=np.load(OUT/f'bank_{name}_{seed}.npz');theta=bank['theta'];info=read(f'calibration_{name}_{seed}.json');r=caltruth[complete][selected]-pred['cal'][complete][selected,...,9];np.testing.assert_array_equal(r,bank['residual']);expectedranks=np.argsort(np.argsort(r,axis=0,kind='stable'),axis=0,kind='stable');np.testing.assert_array_equal(expectedranks,bank['ranks']);assert -2<=theta[0]<=2 and .25<=theta[1]<=4
   if not info['success']:np.testing.assert_array_equal(theta,[0,1]);assert info['fallback']
   q=pred['cal'][complete].astype(float)/meta['scale'];y=caltruth[complete].astype(float)/meta['scale'];med=q[...,9,None];v=np.maximum(0,med+theta[0]+theta[1]*(q-med));objective=float(np.mean(empirical(v,y))+.001*(theta[0]**2+(theta[1]-1)**2));assert abs(objective-info['objective_used'])<1e-10
   q=q.reshape(-1,19);y=y.ravel();t=np.array([.137,1.173]);f,g=affine_objective(t,q,y);eps=1e-6;num=np.array([(affine_objective(t+eps*np.eye(2)[j],q,y)[0]-affine_objective(t-eps*np.eye(2)[j],q,y)[0])/(2*eps) for j in range(2)]);np.testing.assert_allclose(g,num,atol=3e-6,rtol=0)
   for method in ['raw_ecc57','affine_ecc57']:
    q=pred['test'].astype(float)
    if method=='affine_ecc57':med=q[...,9,None];q=np.maximum(0,med+theta[0]*meta['scale']+theta[1]*(q-med))
    direct=empirical(q,truth);scored=np.load(OUT/f'scores_{name}__{method}__{seed}.npz')['crps'];np.testing.assert_allclose(direct,scored,atol=2e-13,rtol=0,equal_nan=True);atomchecks+=1
 pd.DataFrame(budgetrows).to_csv(OUT/'budget_sensitivity.csv',index=False,encoding='utf-8-sig')
 # Independent full-path checks on three fixed origin positions for every scenario flow.
 samplepositions=[0,len(truth)//2,len(truth)-1];casechecks=0;inventory=read('inventory.json')
 for item in inventory:
  saved=np.load(OUT/f'scores_{item["id"]}.npz')
  for idx,sc in enumerate(scenario_generator(item,meta,complete,selected)):
   if idx not in samplepositions:continue
   y=truth[idx];sc=sc.astype(float);k=len(sc);brute=np.mean(abs(sc-y),axis=0)-.5*np.mean(abs(sc[:,None]-sc[None,:]),axis=(0,1));np.testing.assert_allclose(brute,saved['crps'][idx],atol=2e-12,rtol=0,equal_nan=True)
   for duration in [3,6,12]:
    errors=[]
    for off in [0,24,48,72-duration]:
     obs=y[off:off+duration];valid=np.isfinite(obs).all(0);p=(sc[:,off:off+duration]<=10).all(1).mean(0);e=np.where(valid,(p-(obs<=10).all(0))**2,np.nan);errors.append(e)
    np.testing.assert_allclose(np.array(errors),saved[f'brier{duration}'][idx],atol=2e-14,rtol=0,equal_nan=True)
   # Gaps between failed hours independently define longest passing runs.
   for horizon in [24,48,72]:
    def runlength(row):return np.diff(np.r_[-1,np.flatnonzero(~row),len(row)]).max()-1
    sgood=sc[:,:horizon]<=10;ygood=y[:horizon]<=10;smax=np.array([[runlength(sgood[si,:,ni]) for ni in range(17)] for si in range(k)]);ymax=np.array([runlength(ygood[:,ni]) for ni in range(17)])
    valid=np.isfinite(y[:horizon]).all(0);loss=np.where(valid,empirical(smax.T,ymax),np.nan);np.testing.assert_allclose(loss,saved[f'length_crps{horizon}'][idx],atol=2e-13,rtol=0,equal_nan=True)
   anybefore=np.zeros((k,17),bool);obsbefore=np.zeros(17,bool);cdf=[]
   for start in range(67):
    anybefore|=(sc[:,start:start+6]<=10).all(1);obsbefore|=(y[start:start+6]<=10).all(0);cdf.append((anybefore.mean(0)-obsbefore)**2)
   expected=np.where(np.isfinite(y).all(0),np.mean(cdf,axis=0),np.nan);np.testing.assert_allclose(expected,saved['first_cdf_brier'][idx],atol=2e-14,rtol=0,equal_nan=True);casechecks+=1
  if item['method']=='old32':
   prior=np.load(ART/f'probability_v1/scores_joint32_{item["seed"]}.npz')
   for key in saved.files:np.testing.assert_allclose(saved[key],prior[key],atol=2e-13,rtol=0,equal_nan=True)
 for seed in SEEDS:
  base=np.load(OUT/f'scores_gated_graph_atoms__affine_ecc57__{seed}.npz')
  for method in ['time_broken','space_broken']:
   broken=np.load(OUT/f'scores_gated_graph_atoms__{method}__{seed}.npz')
   for metric in ['crps','coverage','width','interval_score','low_miss','high_miss']+(['brier3','brier6','brier12','length_crps24','length_crps48','length_crps72','first_cdf_brier'] if method=='space_broken' else []):np.testing.assert_allclose(base[metric],broken[metric],atol=2e-13,rtol=0,equal_nan=True)
 metricframe=pd.read_csv(OUT/'metrics.csv');te=times[ids['test']];cached={}
 for item in inventory:cached[item['id']]=dict(np.load(OUT/f'scores_{item["id"]}.npz'))
 for r in metricframe.itertuples():
  mask=np.ones(len(te),bool) if r.month==0 else te.month==r.month;v=cached[r.id][r.metric][mask];assert abs(np.nanmean(v)-r.value)<1e-11 and np.isfinite(v).sum()==r.n
 leads=pd.read_csv(OUT/'lead_metrics.csv')
 for r in leads.itertuples():
  v=cached[r.id][r.metric][:,r.lead_start-1:r.lead_end];assert abs(np.nanmean(v)-r.value)<1e-11 and np.isfinite(v).sum()==r.n
 pairs=pd.read_csv(OUT/'paired_comparisons.csv');week=((te-pd.Timestamp('2024-11-01',tz=te.tz)).days//7).to_numpy();keys=np.unique(week);draw=np.random.default_rng(20261105).integers(0,len(keys),(2000,len(keys)))
 for r in pairs.itertuples():
  delta=np.mean([cached[f'{r.candidate}__{r.method}__{s}'][r.metric]-cached[f'{r.baseline}__{r.baseline_method}__{s}'][r.metric] for s in SEEDS],0);sums=np.array([np.nansum(delta[week==k]) for k in keys]);counts=np.array([np.isfinite(delta[week==k]).sum() for k in keys]);boot=sums[draw].sum(1)/counts[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975]);np.testing.assert_allclose([np.nanmean(delta),lo,hi],[r.difference,r.low95,r.high95],atol=1e-12,rtol=0)
 summary=dict(status='PASS',previous_unchanged=len(old),frozen_identities=len(identity),checkpoints_reloaded=checkpoints,max_checkpoint_prediction_difference=maxpred,exact_atom_vs_scenario_crps_groups=atomchecks,independent_full_path_cases=casechecks,metric_rows=len(metricframe),lead_metric_rows=len(leads),paired_rows=len(pairs),old32_all_scores_reproduced=True,calibration_only_october=True)
 (OUT/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':run()

"""Independent round22 graph, selection, calibration, inference and score audit."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import torch
from model import ART,Net,predict
from crps_distribution_core import AtomNet,predict_atoms
from verify_outage import reconstruct,empirical,close
from research_crps_distribution import guard
OUT=ART/'spatial_outage_controls_v1';SRC=ART/'sensor_outage_v1'
MODELS=['mean24','random24','gated_mean_atoms','gated_random_atoms'];SEEDS=[20241101,20241102,20241103];LENGTHS=[0,6,12,24]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def checkpoint(m,s):return ART/'v2'/(m+('' if s==SEEDS[0] else f'_{s}')+'.pt') if m in MODELS[:2] else OUT/f'{m}_{s}_budget40.pt'

def graphs(d):
 a=d['adj'];n=len(a);perm=np.random.default_rng(20241101).permutation(n);expected={'mean':(np.ones((n,n))-np.eye(n))/(n-1),'random':a[perm][:,perm]};recipe=read(OUT/'recipe.json');np.testing.assert_array_equal(recipe['permutation'],perm)
 for kind,adj in expected.items():np.testing.assert_array_equal(recipe['graphs'][kind],adj)
 return expected

def selected_training(net,ck,model,seed,d,ev):
 history=read(OUT/f'history_{model}_{seed}.json');assert len(history)==40 and [h['epoch'] for h in history]==list(range(1,41));best=min(history,key=lambda h:h['validation_crps']);assert best['epoch']==ck['epoch'];close(best['validation_crps'],ck['validation_crps'],1e-12);assert sum(p.numel() for p in net.parameters())==45984
 pred=dict(np.load(OUT/f'predictions_{model}_{seed}_budget40.npz'));checks=0
 for split in ['val','cal','test']:
  fresh=predict_atoms(net,d['x'],ev[split+'_ids']);np.testing.assert_array_equal(fresh,pred[split]);checks+=1
 y=np.stack([d['raw'][i+1:i+73] for i in ev['val_ids']]);q=pred['val'].astype(float);k=19;score=np.mean(abs(q-y[...,None]),-1)-np.sum(q*(2*np.arange(k)-k+1),-1)/k**2
 close(float(np.nanmean(score)),ck['validation_crps'],1e-6)
 return pred,checks

def bank_check(net,ck,model,seed,d,ev,info,pred):
 bank=dict(np.load(OUT/f'bank_{model}_{seed}.npz'));selected=info['selected57'];ordinary=model in MODELS[:2]
 if ordinary:
  q=predict(net,d['x'],d['cal_ids']);y=np.stack([d['raw'][i+1:i+73] for i in d['cal_ids']]);np.testing.assert_array_equal(bank['residual'],(y-(q[...,1]*ck['scale']+ck['center']))[selected]);return bank,0
 complete=ev['complete_positions'];q=pred['cal'][complete];y=ev['cal_truth'][complete];res=(y-q[...,9])[selected];np.testing.assert_array_equal(bank['residual'],res)
 order=np.argsort(res,axis=0,kind='stable');ranks=np.empty_like(order)
 np.put_along_axis(ranks,order,np.broadcast_to(np.arange(57)[:,None,None],order.shape),axis=0);np.testing.assert_array_equal(bank['ranks'],ranks)
 theta=bank['theta'];assert theta.shape==(4,2) and (abs(theta[:,0])<=2).all() and (theta[:,1]>=.25).all() and (theta[:,1]<=4).all();scale=float(bank['scale']);close(scale,read(ART/'metadata.json')['scale']);details=read(OUT/f'calibration_{model}_{seed}.json');rows=pd.read_csv(OUT/'calibration.csv')
 for j,(a,b) in enumerate([(0,6),(6,24),(24,48),(48,72)]):
  z=q[:,a:b].astype(float)/scale;obs=y[:,a:b].astype(float)/scale;med=z[...,9,None];v=np.maximum(0,med+theta[j,0]+theta[j,1]*(z-med));score=abs(v-obs[...,None]).mean(-1)-np.sum(v*(2*np.arange(19)-18),-1)/361;objective=score.mean()+.001*(theta[j,0]**2+(theta[j,1]-1)**2);row=details[j];close(row['objective_used'],objective,1e-10);assert row['valid_targets']==58*(b-a)*17 and row['lead_start']==a+1 and row['lead_end']==b
  if row['fallback']:np.testing.assert_array_equal(theta[j],[0,1]);assert not row['success']
  f=rows[(rows.model==model)&(rows.seed==seed)&(rows.lead_start==a+1)].iloc[0];close(f.delta,theta[j,0],1e-14);close(f.gamma,theta[j,1],1e-14);close(f.objective_used,objective,1e-10)
 return bank,4

def score_check(q,bank,nodes,ordinary,truth,scores,select):
 for a in range(0,len(q),128):
  stop=min(a+128,len(q));sc=reconstruct(q[a:stop],bank,nodes[a:stop],ordinary);y=truth[a:stop];valid=np.isfinite(y);np.testing.assert_allclose(empirical(sc,y),scores['crps'][a:stop],atol=3e-12,rtol=0,equal_nan=True);ordered=np.sort(sc,axis=1);lo=ordered[:,5];hi=ordered[:,51]
  for key,value in [('coverage',(y>=lo)&(y<=hi)),('width',hi-lo),('low_miss',y<lo),('high_miss',y>hi)]:np.testing.assert_allclose(np.where(valid,value,np.nan),scores[key][a:stop],atol=3e-12,rtol=0,equal_nan=True)
  for j,h in enumerate([0,24,48,66]):
   ok=np.isfinite(y[:,h:h+6]).all(1);p=(sc[:,:,h:h+6]<=10).all(2).mean(1);label=(y[:,h:h+6]<=10).all(1)
   for key,value in [('prob6',p),('label6',label),('brier6',(p-label)**2)]:np.testing.assert_allclose(np.where(ok,value,np.nan),scores[key][a:stop,j],atol=1e-12,rtol=0,equal_nan=True)
 def longest(row):return np.diff(np.r_[-1,np.flatnonzero(~row),len(row)]).max()-1
 for i in select:
  sc=reconstruct(q[i:i+1],bank,nodes[i:i+1],ordinary)[0].astype(float);y=truth[i].astype(float);brute=np.mean(abs(sc-y),0)-.5*np.mean(abs(sc[:,None]-sc[None,:]),axis=(0,1));np.testing.assert_allclose(brute,scores['crps'][i],atol=3e-12,rtol=0,equal_nan=True);length=np.array([longest(row<=10) for row in sc]);actual=longest(y<=10);value=np.mean(abs(length-actual))-.5*np.mean(abs(length[:,None]-length[None,:]));close(scores['length_crps72'][i],value if np.isfinite(y).all() else np.nan,1e-12)
  def first(row):
   found=[h for h in range(67) if (row[h:h+6]<=10).all()];return min(found) if found else np.inf
  starts=np.array([first(row) for row in sc]);actual=first(y);cdf=(starts[:,None]<=np.arange(67)).mean(0);value=np.mean((cdf-(actual<=np.arange(67)))**2);close(scores['first_cdf_brier'][i],value if np.isfinite(y).all() else np.nan,1e-12)

def tables(cache,info):
 times=pd.DatetimeIndex(pd.to_datetime(info['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');weeks=((times-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();draw=np.random.default_rng(20261110).integers(0,9,(2000,9));metrics=pd.read_csv(OUT/'metrics.csv');high=pd.read_csv(OUT/'high_confidence.csv');rel=pd.read_csv(OUT/'reliability.csv');pairs=pd.read_csv(OUT/'paired_comparisons.csv');assert len(metrics)==1152 and len(high)==144 and len(rel)==1440 and len(pairs)==224 and pairs.primary.sum()==8
 for r in metrics.itertuples():
  s=cache[(r.model,r.seed,r.outage_hours)][r.metric];a=s if r.month==0 else s[times.month==r.month];close(r.value,np.nanmean(a));assert r.n==np.isfinite(a).sum()
 for frame,ishigh in [(high,True),(rel,False)]:
  for r in frame.itertuples():
   s=cache[(r.model,r.seed,r.outage_hours)];mask=np.ones(len(times),bool) if r.month==0 else times.month==r.month;p=s['prob6'][mask].ravel();y=s['label6'][mask].ravel();valid=np.isfinite(y);p=p[valid];y=y[valid];chosen=p>=.8 if ishigh else (p>=r.bin/10)&(p<(r.bin+1)/10 if r.bin<9 else p<=1);assert r.n==chosen.sum();close(r.mean_probability,p[chosen].mean() if chosen.any() else np.nan);close(r.observed_rate,y[chosen].mean() if chosen.any() else np.nan)
   if ishigh:close(r.fraction,chosen.mean())
 for m in ['graph24','gated_graph_atoms']:
  for s in SEEDS:
   for h in LENGTHS:cache[(m,s,h)]=dict(np.load(SRC/f'scores_{m}_{s}_h{h}.npz'))
 for r in pairs.itertuples():
  values=[]
  for s in SEEDS:
   a=cache[(r.candidate,s,r.outage_hours)][r.metric];b=cache[(r.baseline,s,r.outage_hours)][r.metric];np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b));v=a-b
   if r.kind=='difference_in_degradation':v-=cache[(r.candidate,s,0)][r.metric]-cache[(r.baseline,s,0)][r.metric]
   else:assert r.kind=='absolute_graph_difference'
   values.append(v)
  diff=np.mean(values,0);ss=np.array([np.nansum(diff[weeks==k]) for k in range(9)]);nn=np.array([np.isfinite(diff[weeks==k]).sum() for k in range(9)]);den=nn[draw].sum(1);valid=den>0;boot=ss[draw][valid].sum(1)/den[valid];lo,hi=np.quantile(boot,[.025,.975]);close(r.difference,np.nanmean(diff));close(r.low95,lo);close(r.high95,hi);assert r.n==np.isfinite(diff).sum() and r.blocks==sum(nn>0) and r.valid_draws==sum(valid);assert r.primary==(r.kind=='absolute_graph_difference' and r.outage_hours==24 and r.metric in ['crps','brier6'])
 return dict(metrics=len(metrics),high_confidence=len(high),reliability=len(rel),paired=len(pairs))

def run():
 guard();previous=read(OUT/'previous_hashes.json');identity=read(OUT/'identity.json')
 for n,h in previous.items():assert sha(ART/n)==h,n
 for n,h in identity.items():assert sha(Path(n))==h,n
 assert sha(OUT/'input_data.npz')==sha(SRC/'input_data.npz');assert read(OUT/'prediction_complete.json')['flows']==48
 d=np.load(ART/'data.npz');ev=np.load(ART/'crps_distribution_v1/evaluation_data.npz');info=np.load(OUT/'input_data.npz');hist=np.stack([d['x'][i-23:i+1] for i in d['test_ids']]);np.testing.assert_array_equal(hist,info['histories']);pos=info['positions'];nodes=info['targets'];truth=np.stack([d['raw'][i+1:i+73,n] for i,n in zip(d['test_ids'][pos],nodes)]);np.testing.assert_array_equal(truth,np.load(SRC/'evaluation_truth.npz')['truth']);np.testing.assert_array_equal(ev['selected57'],info['selected57']);np.testing.assert_array_equal(ev['selected57_ids'],info['source_cal_ids']);assert max(info['source_cal_ids']+72)<min(d['test_ids']);expected=graphs(d);select=np.flatnonzero(np.isin(pos,[0,116,231])&np.isin(nodes,[0,8,16]));cache={};scalar=0;reloads=0;objectives=0;maxdiff=0.;cleanchecks=[]
 for m in MODELS:
  ordinary=m in MODELS[:2];kind='mean' if 'mean' in m else 'random'
  for seed in SEEDS:
   guard();ck=torch.load(checkpoint(m,seed),map_location='cpu',weights_only=False);np.testing.assert_allclose(ck['adj'],expected[kind],atol=1e-7,rtol=0);net=Net(ck['adj'],dilation=ck['dilation']) if ordinary else AtomNet(ck['adj'],ck['scale'],hidden=24,gated=True);net.load_state_dict(ck['state']);net.eval();pred=None
   if not ordinary:pred,n=selected_training(net,ck,m,seed,d,ev);reloads+=n
   bank,n=bank_check(net,ck,m,seed,d,ev,info,pred);objectives+=n;reference=(predict(net,d['x'],d['test_ids'])*ck['scale']+ck['center']) if ordinary else pred['test'];reference=reference[pos,:,nodes,:]
   for h in LENGTHS:
    guard();q=np.load(OUT/f'predictions_{m}_{seed}_h{h}.npz')['q'];assert q.shape==(len(pos),72,3 if ordinary else 19) and np.isfinite(q).all() and (np.diff(q,axis=-1)>=0).all()
    if h==0:np.testing.assert_allclose(q,reference,atol=1e-5,rtol=0);cleanchecks.append(dict(model=m,seed=seed,max_difference=float(abs(q-reference).max())))
    for i in select:
     xx=hist[pos[i]].copy()
     if h:xx[-h:,nodes[i],:]=0
     with torch.no_grad():fresh=net(torch.from_numpy(xx[None])).numpy()[0,:,nodes[i],:]
     if ordinary:fresh=fresh*ck['scale']+ck['center']
     maxdiff=max(maxdiff,float(abs(fresh-q[i]).max()));np.testing.assert_allclose(fresh,q[i],atol=1e-5,rtol=0);scalar+=1
    scores=dict(np.load(OUT/f'scores_{m}_{seed}_h{h}.npz'));score_check(q,bank,nodes,ordinary,truth,scores,select);cache[(m,seed,h)]=scores;print('AUDITED',m,seed,h,flush=True)
 assert read(OUT/'clean_prediction_checks.json')==cleanchecks
 counts=tables(cache,info);result=dict(status='PASS',previous_unchanged=len(previous),frozen_identities=len(identity),new_checkpoints=6,full_prediction_reloads=reloads,independent_calibration_objectives=objectives,clean_groups=12,scalar_forecast_checks=scalar,maximum_scalar_difference=maxdiff,full_primary_score_cases=len(pos)*48,independent_long_window_cases=len(select)*48,table_rows=counts)
 (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':run()

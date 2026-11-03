"""Independent target/history indexing, inference, scoring and interval audit."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import torch
from model import ART,Net,predict
from crps_distribution_core import AtomNet
from research_crps_distribution import guard
OUT=ART/'sensor_outage_v1';MODELS=['graph24','temporal24','gated_graph_atoms','gated_temporal_atoms'];SEEDS=[20241101,20241102,20241103];LENGTHS=[0,6,12,24]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-9):
 if not np.isfinite(b):assert not np.isfinite(a)
 else:assert np.isclose(a,b,atol=tol,rtol=0),(a,b)
def empirical(x,y):
 x=np.sort(np.asarray(x,float),axis=1);k=x.shape[1];return np.mean(abs(x-y[:,None]),axis=1)-np.sum(x*(2*np.arange(k)-k+1)[None,:,None],axis=1)/k**2

def reconstruct(q,bank,nodes,ordinary):
 if ordinary:return np.maximum(0,q[:,None,:,1]+bank['residual'][:,:,nodes].transpose(2,0,1))
 theta=bank['theta'];pp=np.stack([np.interp(np.arange(1,73),[3.5,15.5,36.5,60.5],theta[:,j]) for j in range(2)],-1);q=q.astype(float);med=q[...,9,None];atoms=np.maximum(0,med+pp[None,:,0,None]*float(bank['scale'])+pp[None,:,1,None]*(q-med));values=np.repeat(atoms,3,axis=-1);ranks=bank['ranks'][:,:,nodes].transpose(2,1,0);return np.take_along_axis(values,ranks,axis=-1).transpose(0,2,1)

def run():
 guard();previous=read(OUT/'previous_hashes.json');identity=read(OUT/'identity.json');inputids=read(OUT/'input_identity.json')
 for n,h in previous.items():assert sha(ART/n)==h,n
 for n,h in {**identity,**inputids}.items():assert sha(Path(n))==h,n
 d=np.load(ART/'data.npz');info=np.load(OUT/'input_data.npz');ids=d['test_ids'];hist=np.stack([d['x'][i-23:i+1] for i in ids]);np.testing.assert_array_equal(hist,info['histories']);eligible=np.zeros((len(ids),17),bool)
 for i in range(len(ids)):
  for n in range(17):
   masked=hist[i].copy();masked[:,n,:]=0;eligible[i,n]=bool((hist[i,:,n,2]==1).all() and masked[...,2].mean()>=.65)
 pos,nodes=np.where(eligible);np.testing.assert_array_equal(eligible,info['eligible']);np.testing.assert_array_equal(pos,info['positions']);np.testing.assert_array_equal(nodes,info['targets']);np.testing.assert_array_equal(ids[pos],info['issue_ids']);truth=np.stack([d['raw'][i+1:i+73,j] for i,j in zip(ids[pos],nodes)]);np.testing.assert_array_equal(truth,np.load(OUT/'evaluation_truth.npz')['truth']);assert max(info['source_cal_ids']+72)<min(ids[pos]);ev=np.load(ART/'crps_distribution_v1/evaluation_data.npz');np.testing.assert_array_equal(info['selected57'],ev['selected57']);np.testing.assert_array_equal(info['source_cal_ids'],ev['selected57_ids']);assert read(OUT/'prediction_complete.json')['flows']==48
 counts=read(OUT/'eligibility.json');full=(hist[...,2]==1).all(1);assert counts['eligible_cases']==len(pos) and counts['potential_cases']==eligible.size and counts['original_incomplete_target_history']==(~full).sum() and counts['excluded_by_farm_quality']==(full&~eligible).sum();select=np.flatnonzero(np.isin(pos,[0,116,231])&np.isin(nodes,[0,8,16]));scalarchecks=0;cleanchecks=0;scorechecks=0;maxscalar=0.;cache={};old=np.load(ART/'probability_v1/predictions.npz')
 for model in MODELS:
  ordinary=model in MODELS[:2]
  for seed in SEEDS:
   guard();path=ART/'v2'/(model+('' if seed==SEEDS[0] else f'_{seed}')+'.pt') if ordinary else ART/'crps_distribution_v1'/f'{model}_{seed}_budget40.pt';ck=torch.load(path,weights_only=False,map_location='cpu');net=Net(ck['adj'],dilation=ck['dilation']) if ordinary else AtomNet(ck['adj'],ck['scale'],hidden=ck['config']['hidden'],gated=ck['config']['gated']);net.load_state_dict(ck['state']);net.eval();bank=dict(np.load(OUT/f'bank_{model}_{seed}.npz'))
   if ordinary:
    if model=='graph24':res=(old['cal_truth']-old[f'cal_q_{seed}'][...,1])[ev['complete_positions']][info['selected57']]
    else:
     calq=predict(net,d['x'],d['cal_ids'])*ck['scale']+ck['center'];cy=np.stack([d['raw'][i+1:i+73] for i in d['cal_ids']]);res=(cy-calq[...,1])[info['selected57']]
    np.testing.assert_array_equal(bank['residual'],res)
   else:
    np.testing.assert_array_equal(bank['ranks'],np.load(ART/'crps_distribution_v1'/f'bank_{model}_{seed}.npz')['ranks']);np.testing.assert_array_equal(bank['theta'],np.load(ART/'crps_lead_calibration_v1'/f'parameters_{model}_{seed}.npz')['affine']);close(float(bank['scale']),read(ART/'metadata.json')['scale'])
   if model=='graph24':reference=old[f'test_q_{seed}'][pos,:,nodes,:]
   elif ordinary:reference=(predict(net,d['x'],ids)*ck['scale']+ck['center'])[pos,:,nodes,:]
   else:reference=np.load(ART/'crps_distribution_v1'/f'predictions_{model}_{seed}_budget40.npz')['test'][pos,:,nodes,:]
   for length in LENGTHS:
    guard();q=np.load(OUT/f'predictions_{model}_{seed}_h{length}.npz')['q'];assert q.shape==(len(pos),72,3 if ordinary else 19) and np.isfinite(q).all() and (np.diff(q,axis=-1)>=0).all()
    if length==0:np.testing.assert_allclose(q,reference,atol=1e-5,rtol=0);cleanchecks+=1
    for ix in select:
     xx=hist[pos[ix]].copy()
     if length:xx[-length:,nodes[ix],:]=0
     with torch.no_grad():fresh=net(torch.from_numpy(xx[None])).numpy()[0,:,nodes[ix],:]
     if ordinary:fresh=fresh*ck['scale']+ck['center']
     maxscalar=max(maxscalar,float(np.max(abs(fresh-q[ix]))));np.testing.assert_allclose(fresh,q[ix],atol=1e-5,rtol=0);scalarchecks+=1
    scores=dict(np.load(OUT/f'scores_{model}_{seed}_h{length}.npz'));cache[(model,seed,length)]=scores
    for a in range(0,len(q),128):
     stop=min(a+128,len(q));sc=reconstruct(q[a:stop],bank,nodes[a:stop],ordinary);yy=truth[a:stop];valid=np.isfinite(yy);np.testing.assert_allclose(empirical(sc,yy),scores['crps'][a:stop],atol=3e-12,rtol=0,equal_nan=True);ordered=np.sort(sc,axis=1);lo=ordered[:,5];hi=ordered[:,51]
     for key,value in [('coverage',(yy>=lo)&(yy<=hi)),('width',hi-lo),('low_miss',yy<lo),('high_miss',yy>hi)]:np.testing.assert_allclose(np.where(valid,value,np.nan),scores[key][a:stop],atol=3e-12,rtol=0,equal_nan=True)
     for j,off in enumerate([0,24,48,66]):
      ok=np.isfinite(yy[:,off:off+6]).all(1);pr=(sc[:,:,off:off+6]<=10).all(2).mean(1);label=(yy[:,off:off+6]<=10).all(1)
      for key,value in [('prob6',pr),('label6',label),('brier6',(pr-label)**2)]:np.testing.assert_allclose(np.where(ok,value,np.nan),scores[key][a:stop,j],atol=1e-12,rtol=0,equal_nan=True)
    for ix in select:
     sc=reconstruct(q[ix:ix+1],bank,nodes[ix:ix+1],ordinary)[0].astype(np.float64);yy=truth[ix].astype(np.float64);brute=np.mean(abs(sc-yy),0)-.5*np.mean(abs(sc[:,None]-sc[None,:]),axis=(0,1));np.testing.assert_allclose(brute,scores['crps'][ix],atol=3e-12,rtol=0,equal_nan=True)
     def longest(row):return np.diff(np.r_[-1,np.flatnonzero(~row),len(row)]).max()-1
     lengths=np.array([longest(row<=10) for row in sc]);actual=longest(yy<=10);value=np.mean(abs(lengths-actual))-.5*np.mean(abs(lengths[:,None]-lengths[None,:]));close(scores['length_crps72'][ix],value if np.isfinite(yy).all() else np.nan,1e-12)
     starts=[]
     for row in sc:
      found=[h for h in range(67) if (row[h:h+6]<=10).all()];starts.append(min(found) if found else np.inf)
     found=[h for h in range(67) if (yy[h:h+6]<=10).all()];actual=min(found) if found else np.inf;cdf=(np.asarray(starts)[:,None]<=np.arange(67)).mean(0);value=np.mean((cdf-(actual<=np.arange(67)))**2);close(scores['first_cdf_brier'][ix],value if np.isfinite(yy).all() else np.nan,1e-12);scorechecks+=1
    print('AUDITED',model,seed,length,flush=True)
 tables=verify_tables(cache,info);summary=dict(status='PASS',previous_unchanged=len(previous),frozen_identities=len(identity),eligible_cases=len(pos),clean_groups_checked=cleanchecks,independent_scalar_forecasts=scalarchecks,maximum_scalar_difference=maxscalar,full_primary_score_cases=len(pos)*48,selected_independent_window_cases=scorechecks,table_rows=tables,no_test_recalibration=True,no_counterfactual_nodes_joined=True)
 (OUT/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

def verify_tables(cache,info):
 metrics=pd.read_csv(OUT/'metrics.csv');rel=pd.read_csv(OUT/'reliability.csv');high=pd.read_csv(OUT/'high_confidence.csv');pairs=pd.read_csv(OUT/'paired_comparisons.csv');times=pd.DatetimeIndex(pd.to_datetime(info['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');weeks=((times-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();draw=np.random.default_rng(20261110).integers(0,9,(2000,9))
 assert len(metrics)==48*3*8 and len(rel)==48*3*10 and len(high)==48*3 and len(pairs)==208 and pairs.primary.sum()==4
 for row in metrics.itertuples():
  s=cache[(row.model,row.seed,row.outage_hours)][row.metric];a=s if row.month==0 else s[times.month==row.month];close(row.value,np.nanmean(a));assert row.n==np.isfinite(a).sum()
 for frame,ishigh in [(rel,False),(high,True)]:
  for row in frame.itertuples():
   s=cache[(row.model,row.seed,row.outage_hours)];mask=np.ones(len(times),bool) if row.month==0 else times.month==row.month;p=s['prob6'][mask].ravel();y=s['label6'][mask].ravel();valid=np.isfinite(y);p=p[valid];y=y[valid];chosen=p>=.8 if ishigh else (p>=row.bin/10)&(p<(row.bin+1)/10 if row.bin<9 else p<=1);assert row.n==chosen.sum();close(row.mean_probability,p[chosen].mean() if chosen.any() else np.nan);close(row.observed_rate,y[chosen].mean() if chosen.any() else np.nan)
   if ishigh:close(row.fraction,chosen.mean())
 for row in pairs.itertuples():
  delta=[]
  for seed in SEEDS:
   a=cache[(row.candidate,seed,row.outage_hours)][row.metric];b=cache[(row.baseline,seed,0 if row.kind=='own_degradation' else row.outage_hours)][row.metric];v=a-b
   if row.kind=='difference_in_degradation':v=v-(cache[(row.candidate,seed,0)][row.metric]-cache[(row.baseline,seed,0)][row.metric])
   delta.append(v)
  diff=np.mean(delta,0);ss=np.array([np.nansum(diff[weeks==k]) for k in range(9)]);nn=np.array([np.isfinite(diff[weeks==k]).sum() for k in range(9)]);den=nn[draw].sum(1);valid=den>0;boot=ss[draw][valid].sum(1)/den[valid];lo,hi=np.quantile(boot,[.025,.975]);close(row.difference,np.nanmean(diff));close(row.low95,lo);close(row.high95,hi);assert row.blocks==sum(nn>0) and row.valid_draws==sum(valid) and row.n==np.isfinite(diff).sum();assert row.primary==(row.kind=='difference_in_degradation' and row.outage_hours==24 and row.metric in ['crps','brier6'])
 return dict(metrics=len(metrics),reliability=len(rel),high_confidence=len(high),paired=len(pairs))

if __name__=='__main__':run()

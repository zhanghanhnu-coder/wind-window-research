"""Independent decision counts, shared-label bounds and block sensitivity."""
from pathlib import Path
import numpy as np
import pandas as pd
from model import ART
from research_residual_shape import read,sha,js
from verify_window_error_attribution import reference_bound,close
FOLDS=['M05','M07','M09','M11'];SEEDS=[20241101,20241102,20241103];CASES=['B','R','W28','M50','G28','K4','K6'];OUT=ART/'state_gate_v1'
def population(scope,kind):
 fs=FOLDS if scope=='ALL' else [scope];pop={f:dict(np.load(OUT/f/'population.npz')) for f in fs};y=np.concatenate([pop[f][kind+'_labels'] for f in fs]);keys=np.concatenate([pop[f][kind+'_keys'][pop[f][kind+'_inverse']].reshape(-1,2) for f in fs]);_,ix=np.unique(keys,axis=0,return_inverse=True);pp={c:np.stack([np.concatenate([np.load(OUT/f/f'diagnostic_{c}_{s}.npz')[kind] for f in fs]) for s in SEEDS]) for c in CASES};blocks=np.concatenate([pop[f]['blocks']+9*i for i,f in enumerate(fs)]);draws=np.concatenate([np.load(OUT/f/'draws.npz')['draws']+9*i for i,f in enumerate(fs)],axis=1)
 return y,ix.reshape(y.shape),pp,blocks,draws
def coeff(pp,a,b,metric):
 if metric=='brier':return np.mean(pp[a]**2-pp[b]**2,0),2*np.mean(pp[b]-pp[a],0)
 aa=(pp[a].mean(0)>=.8-1e-12).astype(float);bb=(pp[b].mean(0)>=.8-1e-12).astype(float)
 return (aa-bb,bb-aa) if metric=='FP' else (np.zeros_like(aa),bb-aa)
def run():
 assert read(OUT/'decisions_complete.json')['status']=='COMPLETE';assert read(OUT/'sensitivity_complete.json')['status']=='COMPLETE'
 identity={}
 for name in ['decision_identity.json','sensitivity_identity.json']:
  for p,h in read(OUT/name).items():assert sha(p)==h,p;identity[p]=h
 refs={(scope,kind):population(scope,kind) for scope in ['ALL',*FOLDS] for kind in ['single','group']};ncount=0
 for r in pd.read_csv(OUT/'decision_counts.csv').itertuples():
  y,ix,pp,_,_=refs[r.scope,r.kind];a=pp[r.case].mean(0)>=r.threshold-1e-12;tp=int((a&(y==1)).sum());fp=int((a&(y==0)).sum());fn=int((~a&(y==1)).sum());tn=int((~a&(y==0)).sum());ua=int((a&~np.isfinite(y)).sum());ur=int((~a&~np.isfinite(y)).sum());close([r.tp,r.fp,r.fn,r.tn,r.unknown_accepted,r.unknown_rejected,r.accepted,r.n],[tp,fp,fn,tn,ua,ur,a.sum(),a.size],0)
  divide=lambda a,b:a/b if b else np.nan
  close([r.precision_lower,r.precision_upper,r.missed_success_lower,r.missed_success_upper,r.fp_fraction_lower,r.fp_fraction_upper,r.fn_fraction_lower,r.fn_fraction_upper,r.coverage],[divide(tp,a.sum()),divide(tp+ua,a.sum()),divide(fn,tp+fn+ua),divide(fn+ur,tp+fn+ur),fp/a.size,(fp+ua)/a.size,fn/a.size,(fn+ur)/a.size,a.mean()]);ncount+=1
 nb=0
 for r in pd.read_csv(OUT/'decision_paired.csv').itertuples():
  y,ix,pp,blocks,draws=refs[r.scope,r.kind];c,d=coeff(pp,r.candidate,r.baseline,r.metric);lo,hi,bl,bu,bn,unique=reference_bound(c,d,y,ix,np.ones_like(y,bool),blocks,draws);close([lo,hi],[r.lower,r.upper]);s=np.load(OUT/f'decision_{r.scope}_{r.kind}_{r.candidate}_{r.baseline}_{r.metric}.npz');close(s['lower'],bl);close(s['upper'],bu);close(s['n'],bn,0);valid=np.isfinite(bl)&np.isfinite(bu);assert r.valid_draws==valid.sum()
  if r.supported:close([np.quantile(bl[valid],.025),np.quantile(bu[valid],.975)],[r.outer_low,r.outer_high])
  for side,value in [('lower',lo),('upper',hi)]:
   w=s['witness_'+side];close(w[np.isfinite(unique)],unique[np.isfinite(unique)],0);close(np.mean(c+d*w[ix]),value)
  nb+=1
  if nb%20==0:print('VERIFY DECISIONS',nb,flush=True)
 # Rebuild block IDs using integer epoch hours; do not trust producer's saved design.
 times=pd.DatetimeIndex(pd.to_datetime(np.load(ART/'data.npz')['timestamps'].tolist(),utc=True));hours=times.asi8//3600000000000;rng=np.random.default_rng(20261211);design={};testmonths=[5,7,9,11]
 for days in [3,7,14]:
  for shift in [0,.5]:
   blocks=[];draws=[];per=[];offset=0
   for fold,month in zip(FOLDS,testmonths):
    ids=np.load(OUT/fold/'population.npz')['ids'];anchor=pd.Timestamp(year=2024,month=month,day=1,tz='Asia/Shanghai').value//3600000000000-int(days*24*shift);groups=(hours[ids]-anchor)//(days*24);_,bb=np.unique(groups,return_inverse=True);k=len(np.unique(bb));dd=rng.integers(0,k,(2000,k));blocks.append(bb+offset);draws.append(dd+offset);per.append((bb,dd));offset+=k
   name=f'd{days}_s{int(shift*2)}';blocks=np.concatenate(blocks);draws=np.concatenate(draws,1);saved=np.load(OUT/f'sensitivity_design_{name}.npz');close(saved['blocks'],blocks,0);close(saved['draws'],draws,0);design[days,shift]=(blocks,draws,per)
 hour={f:{c:np.mean([np.load(OUT/f/f'hourly_scores_{c}_{s}.npz')['crps'] for s in SEEDS],0) for c in ['K6','W28','G28']} for f in FOLDS};ns=0
 for r in pd.read_csv(OUT/'block_sensitivity.csv').itertuples():
  blocks,draws,per=design[r.days,r.shift];name=f'd{r.days}_s{int(r.shift*2)}';saved=np.load(OUT/f'sensitivity_{r.candidate}_{r.baseline}_{name}_{r.metric}.npz')
  if r.metric=='crps':
   num=np.zeros(2000);den=np.zeros(2000);tot=0.;cnt=0;nonempty=[]
   for fold,(bb,dd) in zip(FOLDS,per):
    a=hour[fold][r.candidate];b=hour[fold][r.baseline];bn=[];bd=[]
    for k in np.unique(bb):bn.append(np.nansum(a[bb==k]-b[bb==k]));bd.append(np.isfinite(b[bb==k]).sum())
    bn=np.array(bn);bd=np.array(bd);num+=bn[dd].sum(1);den+=bd[dd].sum(1);tot+=bn.sum();cnt+=bd.sum();nonempty.append(int((bd>0).sum()))
   diff=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0);close(saved['difference'],diff);lo=hi=tot/cnt;low,high=np.nanquantile(diff,[.025,.975]);support=min(nonempty)>=4 and np.isfinite(diff).sum()>=1900
  else:
   y,ix,pp,_,_=refs['ALL','single'];c,d=coeff(pp,r.candidate,r.baseline,r.metric);lo,hi,bl,bu,bn,_=reference_bound(c,d,y,ix,np.ones_like(y,bool),blocks,draws);close(saved['lower'],bl);close(saved['upper'],bu);close(saved['n'],bn,0);valid=np.isfinite(bl)&np.isfinite(bu);nonempty=[len(np.unique(bb)) for bb,_ in per];support=min(nonempty)>=4 and valid.sum()>=1900;low=np.quantile(bl[valid],.025);high=np.quantile(bu[valid],.975)
  close([r.lower,r.upper],[lo,hi]);assert bool(r.supported)==support and r.minimum_nonempty_blocks==min(nonempty)
  if support:close([r.outer_low,r.outer_high],[low,high])
  ns+=1
  if ns%8==0:print('VERIFY STATE SENSITIVITY',ns,flush=True)
 # Independently aggregate detection outcomes; unavailable targets stay explicitly unevaluable.
 dd=pd.read_csv(OUT/'detection_with_control.csv');avg=dd.groupby(['fold','case','issue','lead_start','lead_end'],as_index=False).mean(numeric_only=True)
 for r in pd.read_csv(OUT/'advance_detection_summary.csv').itertuples():
  a=avg[(avg['case']==r.case)&(True if r.scope=='ALL' else avg.fold==r.scope)];z=a[np.isfinite(a.estimated_harm)&np.isfinite(a.actual_harm)];p=z.estimated_harm>0;y=z.actual_harm>0;close([r.tp,r.fp,r.fn,r.tn,r.evaluable,r.total],[(p&y).sum(),(p&~y).sum(),(~p&y).sum(),(~p&~y).sum(),len(z),len(a)],0)
  close(r.precision,(p&y).sum()/p.sum() if p.sum() else np.nan);close(r.recall,(p&y).sum()/y.sum() if y.sum() else np.nan)
 old=pd.read_csv(ART/'calibration_gate_v1/gate_audit.csv');control=dd[dd['case']=='G28'].merge(old[old['case']=='G28'],on=['fold','seed','issue','lead_start'],validate='one_to_one');close(control.estimated_harm,control.loss_4-control.loss_2)
 descriptive=pd.read_csv(OUT/'origin_diagnosis.csv').groupby(['fold','issue','lead_start','lead_end'],as_index=False).mean(numeric_only=True)
 for r in pd.read_csv(OUT/'diagnostic_strata.csv').itertuples():
  z=descriptive if r.scope=='ALL' else descriptive[descriptive.fold==r.scope]
  flags={'high_wind':z.f0>=1,'rising_wind':z.f1>=.5,'falling_wind':z.f1<=-.5,'downward_calibration':z.f4>=.5,'narrowing':z.f5<=np.log(.75)};part=z[flags[r.signal]==r.active];known=part[np.isfinite(part.harm)];assert r.n==len(part) and r.observed==len(known);close([r.mean_harm,r.harm_fraction],[known.harm.mean(),(known.harm>0).mean()])
 for p,h in identity.items():assert sha(p)==h,p
 result=dict(status='PASS',decision_count_rows=ncount,decision_bounds=nb,sensitivity_rows=ns,draws_recomputed=nb*2000+ns*2000,independent_block_rng=True);js(OUT/'decision_sensitivity_verification.json',result);print(result,flush=True)
if __name__=='__main__':run()

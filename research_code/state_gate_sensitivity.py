"""Predeclared 48 block sensitivity comparisons, with no model selection."""
from pathlib import Path
import numpy as np
import pandas as pd
from research_state_gate import OUT,ART,PLAN,FOLDS,SEEDS,read,sha,js
from state_gate_tables import population
from state_gate_decisions import decision_coefficients
from missing_window_bounds_core import coefficients
from window_error_attribution_core import selected_bounds,selected_bootstrap
PAIRS=[('K6','W28'),('K6','G28')]
def run():
 assert read(OUT/'evaluation_complete.json')['status']=='COMPLETE';assert not (OUT/'sensitivity_complete.json').exists()
 files=[Path(__file__),PLAN]+list(OUT.glob('*/population.npz'))+list(OUT.glob('*/diagnostic_*.npz'))+list(OUT.glob('*/hourly_scores_*.npz'));identity={str(p):sha(p) for p in files};js(OUT/'sensitivity_identity.json',identity)
 ts=pd.DatetimeIndex(pd.to_datetime(np.load(ART/'data.npz')['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');rng=np.random.default_rng(20261211);pop=population('ALL','single');rows=[]
 hour={f:{c:np.mean([np.load(OUT/f/f'hourly_scores_{c}_{s}.npz')['crps'] for s in SEEDS],0) for c in ['K6','W28','G28']} for f in FOLDS}
 for days in [3,7,14]:
  for shift in [0,.5]:
   blocks=[];draws=[];per=[];offset=0
   for fold,cfg in FOLDS.items():
    ids=np.load(OUT/fold/'population.npz')['ids'];anchor=pd.Timestamp(cfg['test_start'],tz='Asia/Shanghai')-pd.Timedelta(days=days*shift);bb=np.floor((ts[ids]-anchor).total_seconds()/(days*86400)).astype(int);_,bb=np.unique(bb,return_inverse=True);k=int(bb.max())+1;dd=rng.integers(0,k,(2000,k));blocks.append(bb+offset);draws.append(dd+offset);per.append((bb,dd));offset+=k
   blocks=np.concatenate(blocks);draws=np.concatenate(draws,1);name=f'd{days}_s{int(shift*2)}';np.savez_compressed(OUT/f'sensitivity_design_{name}.npz',blocks=blocks,draws=draws)
   for candidate,baseline in PAIRS:
    num=np.zeros(2000);den=np.zeros(2000);point_num=0.;point_den=0;nonempty=[]
    for fold,(bb,dd) in zip(FOLDS,per):
     diff=np.nansum(hour[fold][candidate]-hour[fold][baseline],(1,2));count=np.isfinite(hour[fold][baseline]).sum((1,2));sums=np.bincount(bb,weights=diff);counts=np.bincount(bb,weights=count);num+=sums[dd].sum(1);den+=counts[dd].sum(1);point_num+=diff.sum();point_den+=count.sum();nonempty.append(int((counts>0).sum()))
    good=den>0;support=min(nonempty)>=4 and good.sum()>=1900;arr=num[good]/den[good]
    rows.append(dict(candidate=candidate,baseline=baseline,days=days,shift=shift,metric='crps',lower=point_num/point_den,upper=point_num/point_den,outer_low=float(np.quantile(arr,.025)) if support else np.nan,outer_high=float(np.quantile(arr,.975)) if support else np.nan,minimum_nonempty_blocks=min(nonempty),supported=bool(support)))
    np.savez_compressed(OUT/f'sensitivity_{candidate}_{baseline}_{name}_crps.npz',difference=np.divide(num,den,out=np.full_like(num,np.nan),where=good))
    for metric in ['brier','FP','FN']:
     pa=pop['probabilities'][candidate];pb=pop['probabilities'][baseline];c,d=coefficients(pa,pb) if metric=='brier' else decision_coefficients(pa.mean(0),pb.mean(0),metric);mask=np.ones_like(pop['labels'],bool);bound=selected_bounds(c,d,pop['labels'],pop['inverse'],mask);sample=selected_bootstrap(c,d,pop['labels'],pop['inverse'],mask,blocks,draws);good=np.isfinite(sample['lower'])&np.isfinite(sample['upper']);support=min(len(np.unique(bb)) for bb,_ in per)>=4 and good.sum()>=1900
     rows.append(dict(candidate=candidate,baseline=baseline,days=days,shift=shift,metric=metric,lower=bound['lower'],upper=bound['upper'],outer_low=float(np.quantile(sample['lower'][good],.025)) if support else np.nan,outer_high=float(np.quantile(sample['upper'][good],.975)) if support else np.nan,minimum_nonempty_blocks=min(len(np.unique(bb)) for bb,_ in per),supported=bool(support)))
     np.savez_compressed(OUT/f'sensitivity_{candidate}_{baseline}_{name}_{metric}.npz',lower=sample['lower'],upper=sample['upper'],n=sample['n'])
   print('STATE SENSITIVITY',days,shift,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'block_sensitivity.csv',index=False,encoding='utf-8-sig');assert len(rows)==48
 for p,h in identity.items():assert sha(p)==h,p
 js(OUT/'sensitivity_complete.json',dict(status='COMPLETE',rows=48,seed=20261211))
if __name__=='__main__':run()

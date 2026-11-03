"""Predeclared block-length/anchor sensitivity; never chooses a forecast method."""
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from research_calibration_gate import OUT,ART,PLAN,SEEDS,FOLDS,read,sha,js
from calibration_gate_tables import population
from missing_window_bounds_core import coefficients
from window_error_attribution_core import selected_bounds,selected_bootstrap
PAIRS=[('G7','W28'),('M50','W28')]

def run():
 assert read(OUT/'evaluation_complete.json')['status']=='COMPLETE';assert not (OUT/'sensitivity_complete.json').exists()
 files=[Path(__file__),PLAN]+list(OUT.glob('*/population.npz'))+list(OUT.glob('*/fixed_cohorts.npz'))+list(OUT.glob('*/diagnostic_*.npz'))+list(OUT.glob('*/hourly_scores_*.npz'))
 identity={str(p):sha(p) for p in files};js(OUT/'sensitivity_identity.json',identity)
 ts=pd.DatetimeIndex(pd.to_datetime(np.load(ART/'data.npz')['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');rng=np.random.default_rng(20261210);rows=[];cache={kind:population('ALL',kind) for kind in ['single','group']}
 hour={f:{c:np.mean([np.load(OUT/f/f'hourly_scores_{c}_{s}.npz')['crps'] for s in SEEDS],0) for c in ['G7','M50','W28']} for f in FOLDS}
 for days in [3,7,14]:
  for shift in [0,.5]:
   blocks=[];draws=[];perfold=[];offset=0
   for fold,cfg in FOLDS.items():
    ids=np.load(OUT/fold/'population.npz')['ids'];anchor=pd.Timestamp(cfg['test_start'],tz='Asia/Shanghai')-pd.Timedelta(days=days*shift);bb=np.floor((ts[ids]-anchor).total_seconds()/(days*86400)).astype(int);_,bb=np.unique(bb,return_inverse=True);k=int(bb.max())+1;dd=rng.integers(0,k,(2000,k));blocks.append(bb+offset);draws.append(dd+offset);perfold.append((bb,dd));offset+=k
   blocks=np.concatenate(blocks);draws=np.concatenate(draws,1);name=f'd{days}_s{int(shift*2)}';np.savez_compressed(OUT/f'sensitivity_design_{name}.npz',blocks=blocks,draws=draws)
   for candidate,baseline in PAIRS:
    pair=f'{candidate}_{baseline}';num=np.zeros(2000);den=np.zeros(2000);total_num=0.;total_n=0;nonempty=[]
    for fold,(bb,dd) in zip(FOLDS,perfold):
     diff=np.nansum(hour[fold][candidate]-hour[fold][baseline],(1,2));count=np.isfinite(hour[fold][baseline]).sum((1,2));sums=np.bincount(bb,weights=diff);counts=np.bincount(bb,weights=count);num+=sums[dd].sum(1);den+=counts[dd].sum(1);total_num+=diff.sum();total_n+=count.sum();nonempty.append(int((counts>0).sum()))
    valid=den>0;support=bool(min(nonempty)>=4 and valid.sum()>=1900);loss=num[valid]/den[valid]
    rows.append(dict(candidate=candidate,baseline=baseline,days=days,anchor_shift_fraction=shift,metric='crps_all_observed',point_lower=total_num/total_n,point_upper=total_num/total_n,outer_low=float(np.quantile(loss,.025)) if support else np.nan,outer_high=float(np.quantile(loss,.975)) if support else np.nan,minimum_nonempty_blocks=min(nonempty),valid_draws=int(valid.sum()),supported=support))
    np.savez_compressed(OUT/f'sensitivity_{pair}_{name}_crps.npz',difference=np.divide(num,den,out=np.full_like(num,np.nan),where=valid))
    for kind,cohort in [('single','all'),('single','baseline_high'),('group','baseline_high')]:
     pop=cache[kind];c,d=coefficients(pop['probabilities'][candidate],pop['probabilities'][baseline]);mask=np.ones_like(pop['labels'],bool) if cohort=='all' else pop['high'];bound=selected_bounds(c,d,pop['labels'],pop['inverse'],mask);sample=selected_bootstrap(c,d,pop['labels'],pop['inverse'],mask,blocks,draws);valid=np.isfinite(sample['lower'])&np.isfinite(sample['upper']);nonempty=[];start=0
     for fold,(bb,_) in zip(FOLDS,perfold):
      mm=mask[start:start+len(bb)];nonempty.append(int(sum(mm[bb==k].any() for k in np.unique(bb))));start+=len(bb)
     support=bool(min(nonempty)>=4 and valid.sum()>=1900);metric=kind+'_'+cohort
     rows.append(dict(candidate=candidate,baseline=baseline,days=days,anchor_shift_fraction=shift,metric=metric,point_lower=bound['lower'],point_upper=bound['upper'],outer_low=float(np.quantile(sample['lower'][valid],.025)) if support else np.nan,outer_high=float(np.quantile(sample['upper'][valid],.975)) if support else np.nan,minimum_nonempty_blocks=min(nonempty),valid_draws=int(valid.sum()),supported=support))
     np.savez_compressed(OUT/f'sensitivity_{pair}_{name}_{metric}.npz',lower=sample['lower'],upper=sample['upper'],n=sample['n'])
   print('SENSITIVITY',days,shift,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'block_sensitivity.csv',index=False,encoding='utf-8-sig')
 for p,h in identity.items():assert sha(p)==h,p
 assert len(rows)==48
 js(OUT/'sensitivity_complete.json',dict(status='COMPLETE',rows=48,supported_rows=int(sum(r['supported'] for r in rows)),new_forecasts=False,utc=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__':run()

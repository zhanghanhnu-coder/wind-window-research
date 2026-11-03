"""Post-hoc time-block sensitivity of fixed W28-vs-B forecasts only."""
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from research_residual_warm import OUT,ART,BASE,SEEDS,FOLDS,read,sha,js
from residual_warm_tables import population
from missing_window_bounds_core import coefficients
from window_error_attribution_core import selected_bounds,selected_bootstrap

def completion_metadata(rows):
    return dict(status='COMPLETE',rows=len(rows),supported_rows=int(sum(r['supported'] for r in rows)),new_forecasts=False,utc=datetime.now(timezone.utc).isoformat())

def run():
    assert not (OUT/'block_sensitivity_complete.json').exists()
    plan=BASE/'研究笔记/第49轮_时间分块敏感性预设_2026-11-02.md'
    files=[Path(__file__),plan,OUT/'evaluation_complete.json']+list(OUT.glob('*/population.npz'))+list(OUT.glob('*/draws.npz'))+list(OUT.glob('*/diagnostic_*.npz'))+list(OUT.glob('*/hourly_scores_*.npz'));identity={str(p):sha(p) for p in files};js(OUT/'block_sensitivity_identity.json',identity)
    ts=pd.DatetimeIndex(pd.to_datetime(np.load(ART/'data.npz')['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');rng=np.random.default_rng(20261209);rows=[];cache={kind:population('ALL',kind) for kind in ['single','group']};hour={}
    for fold in FOLDS:
        z={c:np.mean([np.load(OUT/fold/f'hourly_scores_{c}_{s}.npz')['crps'] for s in SEEDS],0) for c in ['B','WT']};hour[fold]=(np.nansum(z['WT']-z['B'],(1,2)),np.isfinite(z['B']).sum((1,2)))
    for days in [3,7,14,28]:
        for shift in [0,.5]:
            blocks=[];draws=[];kcounts=[];perfold=[];offset=0
            for fold,cfg in FOLDS.items():
                ids=np.load(OUT/fold/'population.npz')['ids'];anchor=pd.Timestamp(cfg['test_start'],tz='Asia/Shanghai')-pd.Timedelta(days=days*shift);b=np.floor((ts[ids]-anchor).total_seconds()/(days*86400)).astype(int);_,b=np.unique(b,return_inverse=True);k=int(b.max())+1;d=rng.integers(0,k,(2000,k));blocks.append(b+offset);draws.append(d+offset);kcounts.append(k);perfold.append((b,d));offset+=k
            blocks=np.concatenate(blocks);draws=np.concatenate(draws,1);name=f'd{days}_shift{int(shift*2)}';np.savez_compressed(OUT/f'sensitivity_design_{name}.npz',blocks=blocks,draws=draws,fold_block_counts=kcounts)
            num=np.zeros(2000);den=np.zeros(2000);point_num=0.;point_n=0;nonempty=[]
            for fold,(b,d) in zip(FOLDS,perfold):
                diff,count=hour[fold];sums=np.bincount(b,weights=diff);counts=np.bincount(b,weights=count);num+=sums[d].sum(1);den+=counts[d].sum(1);point_num+=diff.sum();point_n+=count.sum();nonempty.append(int((counts>0).sum()))
            v=den>0;support=min(nonempty)>=4 and v.sum()>=1900;loss=num[v]/den[v];rows.append(dict(days=days,anchor_shift_fraction=shift,metric='crps_all_observed',point_lower=point_num/point_n,point_upper=point_num/point_n,outer_low=np.quantile(loss,.025) if support else np.nan,outer_high=np.quantile(loss,.975) if support else np.nan,minimum_nonempty_blocks=min(nonempty),valid_draws=int(v.sum()),supported=support))
            np.savez_compressed(OUT/f'sensitivity_{name}_crps.npz',difference=np.divide(num,den,out=np.full_like(num,np.nan),where=v))
            for kind in ['single','group']:
                pop=cache[kind];c,d=coefficients(pop['probabilities']['WT'],pop['probabilities']['B'])
                for cohort,mask in [('all',np.ones_like(pop['labels'],bool)),('baseline_high',pop['high'])]:
                    bound=selected_bounds(c,d,pop['labels'],pop['inverse'],mask);sample=selected_bootstrap(c,d,pop['labels'],pop['inverse'],mask,blocks,draws);v=np.isfinite(sample['lower'])&np.isfinite(sample['upper']);nonempty=[];start=0
                    for fold,(bb,_) in zip(FOLDS,perfold):
                        mm=mask[start:start+len(bb)];nonempty.append(int(sum(mm[bb==k].any() for k in np.unique(bb))));start+=len(bb)
                    support=min(nonempty)>=4 and v.sum()>=1900;rows.append(dict(days=days,anchor_shift_fraction=shift,metric=kind+'_'+cohort,point_lower=bound['lower'],point_upper=bound['upper'],outer_low=np.quantile(sample['lower'][v],.025) if support else np.nan,outer_high=np.quantile(sample['upper'][v],.975) if support else np.nan,minimum_nonempty_blocks=min(nonempty),valid_draws=int(v.sum()),supported=support));np.savez_compressed(OUT/f'sensitivity_{name}_{kind}_{cohort}.npz',lower=sample['lower'],upper=sample['upper'],n=sample['n'])
            print('SENSITIVITY',days,shift,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'block_sensitivity.csv',index=False,encoding='utf-8-sig')
    for p,h in identity.items():assert sha(p)==h,p
    js(OUT/'block_sensitivity_complete.json',completion_metadata(rows));print('SENSITIVITY COMPLETE',flush=True)

if __name__=='__main__':run()

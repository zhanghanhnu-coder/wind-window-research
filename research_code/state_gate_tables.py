"""Common-population comparisons, shared missing windows and marginal side effects."""
import numpy as np
import pandas as pd
from missing_window_bounds_core import coefficients
from window_error_attribution_core import selected_bounds,selected_bootstrap
from research_state_gate import OUT,ART,SEEDS,FOLDS,CASES,read,sha,js
from datetime import datetime,timezone
from pathlib import Path

PAIRS=[('K6','W28'),('K6','G28'),('K6','K4'),('K4','G28'),('K4','W28')]

def population(scope,kind):
    folds=list(FOLDS) if scope=='ALL' else [scope];labels=[];keys=[];blocks=[];draws=[];masks=[];blockoff=0;probs={c:[] for c in CASES}
    for fold in folds:
        path=OUT/fold;p=np.load(path/'population.npz');labels.append(p[kind+'_labels']);keys.append(p[kind+'_keys'][p[kind+'_inverse']].reshape(-1,2));blocks.append(p['blocks']+blockoff);draws.append(np.load(path/'draws.npz')['draws']+blockoff);blockoff+=int(p['blocks'].max())+1;masks.append(np.load(path/'fixed_cohorts.npz')[kind])
        for case in CASES:probs[case].append(np.stack([np.load(path/f'diagnostic_{case}_{s}.npz')[kind] for s in SEEDS]))
    y=np.concatenate(labels);uk,ix=np.unique(np.concatenate(keys),axis=0,return_inverse=True)
    return dict(labels=y,inverse=ix.reshape(y.shape),keys=uk,blocks=np.concatenate(blocks),draws=np.concatenate(draws,axis=1),high=np.concatenate(masks),probabilities={c:np.concatenate(v,axis=1) for c,v in probs.items()})

def evaluate():
    assert read(OUT/'experiment_complete.json')['status']=='COMPLETE'
    assert not (OUT/'evaluation_complete.json').exists()
    files=[Path(__file__),OUT.parent.parent.parent/'研究笔记/第51轮_相似风况与校准失效识别_2026-11-02.md',ART.parent/'window_error_attribution_core.py',ART.parent/'missing_window_bounds_core.py']+list(OUT.glob('*/diagnostic_*.npz'))+list(OUT.glob('*/fixed_cohorts.npz'))+list(OUT.glob('*/population.npz'))+list(OUT.glob('*/draws.npz'))+list(OUT.glob('*/hourly_scores_*.npz'))
    identity={str(p):sha(p) for p in files};js(OUT/'evaluation_identity.json',identity)
    paired=[];absolute=[];reliability=[];offsetrows=[];own=[]
    for scope in ['ALL',*FOLDS]:
        for kind in ['single','group']:
            pop=population(scope,kind);y=pop['labels'];inverse=pop['inverse'];ps=pop['probabilities'];events=17 if kind=='single' else 3
            for cohort,mask in [('all',np.ones_like(y,bool)),('baseline_high',pop['high'])]:
                for candidate,baseline in [(c,None) for c in CASES]+PAIRS:
                    c,d=coefficients(ps[candidate],ps[baseline] if baseline else None);bound=selected_bounds(c,d,y,inverse,mask);sample=selected_bootstrap(c,d,y,inverse,mask,pop['blocks'],pop['draws']);valid=np.isfinite(sample['lower'])&np.isfinite(sample['upper']);supported=int(valid.sum())>=1900
                    row=dict(scope=scope,kind=kind,cohort=cohort,candidate=candidate,baseline=baseline or 'absolute',primary=bool(scope=='ALL' and cohort=='baseline_high' and (candidate,baseline) in PAIRS[:2]),lower=bound['lower'],upper=bound['upper'],n=int(bound['n']),outer_low=float(np.quantile(sample['lower'][valid],.025)) if supported else np.nan,outer_high=float(np.quantile(sample['upper'][valid],.975)) if supported else np.nan,valid_draws=int(valid.sum()),interval_supported=supported,unknown_records=int((mask&np.isnan(y)).sum()),physical_events=int(np.unique(inverse[mask]).size),known_label_difference=float((c+d*np.nan_to_num(y))[mask&np.isfinite(y)].mean()) if (mask&np.isfinite(y)).any() else np.nan)
                    (paired if baseline else absolute).append(row)
                    np.savez_compressed(OUT/f'bounds_{scope}_{kind}_{cohort}_{candidate}_{baseline or "absolute"}.npz',witness_lower=bound.get('witness_lower',[]),witness_upper=bound.get('witness_upper',[]),bootstrap_lower=sample['lower'],bootstrap_upper=sample['upper'],bootstrap_n=sample['n'])
                    if row['primary']:print('PRIMARY',row,flush=True)
                for case in CASES:
                    p=ps[case].mean(0)
                    for off in [-1,0,24,48,66]:
                        m=mask.copy()
                        if off!=-1:m&=(np.repeat([0,24,48,66],events)==off)[None,:]
                        n=int(m.sum());success=int((m&(y==1)).sum());unknown=int((m&np.isnan(y)).sum());row=dict(scope=scope,kind=kind,cohort=cohort,case=case,offset=off,n=n,mean_probability=float(p[m].mean()) if n else np.nan,observed_lower=success/n if n else np.nan,observed_upper=(success+unknown)/n if n else np.nan,known_success=success,known_failure=int((m&(y==0)).sum()),unknown=unknown)
                        (reliability if off==-1 else offsetrows).append(row)
            for case in CASES:
                p=ps[case].mean(0);m=p>=.8-1e-12;n=int(m.sum());own.append(dict(scope=scope,kind=kind,case=case,n=n,total=p.size,mean_probability=float(p[m].mean()) if n else np.nan,observed_lower=float((y[m]==1).sum()/n) if n else np.nan,observed_upper=float(((y[m]==1).sum()+np.isnan(y[m]).sum())/n) if n else np.nan))
            print('TABLE',scope,kind,flush=True)
    for name,rows in [('paired_bounds.csv',paired),('absolute_bounds.csv',absolute),('fixed_cohort_reliability.csv',reliability),('offset_reliability.csv',offsetrows),('candidate_high_coverage.csv',own)]:pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
    marg=[]
    for scope in ['ALL',*FOLDS]:
        folds=list(FOLDS) if scope=='ALL' else [scope]
        loaded={fold:{case:{metric:np.mean([np.load(OUT/fold/f'hourly_scores_{case}_{s}.npz')[metric] for s in SEEDS],0) for metric in ['crps','brier','mae']} for case in CASES} for fold in folds}
        for start,end in [(0,72),(0,6),(6,24),(24,48),(48,72)]:
            for candidate,baseline in PAIRS:
                for metric in ['crps','brier','mae']:
                    sum_b=[];sum_m=[];count=[];block=[];draw=[];boff=0
                    for fold in folds:
                        z={c:loaded[fold][c][metric][:,start:end] for c in [baseline,candidate]};valid=np.isfinite(z[baseline]);assert np.array_equal(valid,np.isfinite(z[candidate]));sum_b.extend(np.nansum(z[baseline],axis=(1,2)));sum_m.extend(np.nansum(z[candidate],axis=(1,2)));count.extend(valid.sum((1,2)));bb=np.load(OUT/fold/'population.npz')['blocks'];block.extend(bb+boff);draw.append(np.load(OUT/fold/'draws.npz')['draws']+boff);boff+=int(bb.max())+1
                    count=np.array(count);sb=np.array(sum_b);sm=np.array(sum_m);block=np.array(block);draw=np.concatenate(draw,axis=1);bs=np.bincount(block,weights=sm-sb);bc=np.bincount(block,weights=count);samples=bs[draw].sum(1)/bc[draw].sum(1)
                    marg.append(dict(scope=scope,lead_start=start+1,lead_end=end,metric=metric,baseline=baseline,candidate=candidate,n=int(count.sum()),base=float(sb.sum()/count.sum()),value=float(sm.sum()/count.sum()),difference=float((sm-sb).sum()/count.sum()),low=float(np.quantile(samples,.025)),high=float(np.quantile(samples,.975)),known_observations_only=True))
    pd.DataFrame(marg).to_csv(OUT/'marginal_paired.csv',index=False,encoding='utf-8-sig')
    for p,h in identity.items():assert sha(p)==h,p
    assert len(paired)==100 and sum(r['primary'] for r in paired)==4 and len(absolute)==140
    js(OUT/'evaluation_complete.json',dict(status='COMPLETE',paired_rows=len(paired),primary_comparisons=4, exploratory_reused_2024=True,absolute_rows=len(absolute),fixed_reliability_rows=len(reliability),offset_rows=len(offsetrows),own_cohort_rows=len(own),marginal_rows=len(marg),utc=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__':evaluate()

"""Post-hoc explanation only: activation, historical mismatch, raw-vs-calibrated."""
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from research_residual_warm import OUT,ART,COLD,SEEDS,FOLDS,read,sha,js,BASE
from residual_warm_tables import population
from calibration_exposure_core import window_fallback
from residual_shape_core import weighted_quantile
from missing_window_bounds_core import coefficients
from window_error_attribution_core import selected_bounds,selected_bootstrap

def run():
    assert read(OUT/'evaluation_complete.json')['status']=='COMPLETE'
    assert not (OUT/'startup_diagnostic_complete.json').exists()
    files=[Path(__file__),ART.parent/'calibration_exposure_core.py',ART.parent/'tests/test_calibration_exposure.py',BASE/'研究笔记/第49轮_校准启用与分布变化诊断说明_2026-11-02.md',OUT/'fit_audit.csv']+list(OUT.glob('*/fit_*.npz'))
    identity={str(p):sha(p) for p in files};js(OUT/'startup_identity.json',identity)
    exposure=[];lag=[]
    for fold in FOLDS:
        pop=np.load(OUT/fold/'population.npz');ids=pop['ids'];blocks=pop['blocks'];truth=np.load(ART/'seasonal_probability_v1'/fold/'test_truth.npz')['y'];scale=float(np.load(ART/'seasonal_probability_v1'/fold/'data.npz')['scale'])
        for seed in SEEDS:
            raw=np.load(OUT/fold/f'predictions_R_{seed}.npz')['q'];s=np.maximum((raw[...,17]-raw[...,1])/2,.1*scale)
            for history in ['cold','warm']:
                fb=np.stack([np.load(OUT/fold/f'fit_{history}_{seed}_w{b}.npz')['fallback'] for b in range(9)])
                if history=='cold':
                    old=np.stack([np.load(COLD/fold/f'fit_{seed}_w{b}.npz')['fallback'] for b in range(9)]);np.testing.assert_array_equal(fb,old)
                for kind,count in [('single',17),('group',3)]:
                    f=window_fallback(blocks,fb,count);y=pop[kind+'_labels'];high=np.load(OUT/fold/'fixed_cohorts.npz')[kind]
                    for cohort,mask in [('all',np.ones_like(y,bool)),('baseline_high',high)]:
                        for off in [-1,0,24,48,66]:
                            m=mask.copy()
                            if off!=-1:m&=(np.repeat([0,24,48,66],count)==off)[None,:]
                            n=int(m.sum());nf=int((m&f).sum());exposure.append(dict(fold=fold,seed=seed,history=history,kind=kind,cohort=cohort,offset=off,n=n,fallback_records=nf,applied_records=n-nf,fallback_fraction=nf/n if n else np.nan,fallback_known_failures=int((m&f&(y==0)).sum())))
            for block in range(9):
                pos=np.flatnonzero(blocks==block);fit=np.load(OUT/fold/f'fit_warm_{seed}_w{block}.npz')
                for g,(a,b) in enumerate([(0,6),(6,24),(24,48),(48,72)]):
                    y=truth[pos,a:b];valid=np.isfinite(y);counts=valid.sum((1,2));use=counts>0;n=int(use.sum());z=(y-raw[pos,a:b,...,9])/s[pos,a:b];weights=np.zeros(len(pos));weights[use]=1/(n*counts[use]) if n else 0;w=np.broadcast_to(weights[:,None,None],y.shape)[valid]
                    future=weighted_quantile(z[valid],w,[.5])[0] if n else np.nan;past=float(fit['z_quantiles'][g,9]);lag.append(dict(fold=fold,seed=seed,block=block,issue=int(ids[pos[0]]),lead_start=a+1,lead_end=b,history_median_standardized_error=past,application_median_standardized_error=float(future),future_observed_origins=n,future_observed_labels=int(valid.sum()),sign_reversal=bool(past*future<0) if n else False,diagnostic_only_future_truth_used=True))
    pd.DataFrame(exposure).to_csv(OUT/'calibration_exposure.csv',index=False,encoding='utf-8-sig');pd.DataFrame(lag).to_csv(OUT/'historical_error_mismatch.csv',index=False,encoding='utf-8-sig')
    rows=[]
    for scope in ['ALL',*FOLDS]:
        for kind in ['single','group']:
            p=population(scope,kind);c,d=coefficients(p['probabilities']['R'],p['probabilities']['B'])
            for cohort,mask in [('all',np.ones_like(p['labels'],bool)),('baseline_high',p['high'])]:
                bound=selected_bounds(c,d,p['labels'],p['inverse'],mask);sample=selected_bootstrap(c,d,p['labels'],p['inverse'],mask,p['blocks'],p['draws']);valid=np.isfinite(sample['lower'])&np.isfinite(sample['upper'])
                rows.append(dict(scope=scope,kind=kind,cohort=cohort,candidate='R',baseline='B',posthoc=True,n=int(bound['n']),lower=bound['lower'],upper=bound['upper'],outer_low=float(np.quantile(sample['lower'][valid],.025)) if valid.sum()>=1900 else np.nan,outer_high=float(np.quantile(sample['upper'][valid],.975)) if valid.sum()>=1900 else np.nan,valid_draws=int(valid.sum())))
                np.savez_compressed(OUT/f'raw_calibrated_{scope}_{kind}_{cohort}.npz',bootstrap_lower=sample['lower'],bootstrap_upper=sample['upper'],bootstrap_n=sample['n'])
    pd.DataFrame(rows).to_csv(OUT/'raw_vs_calibrated_posthoc.csv',index=False,encoding='utf-8-sig')
    old=pd.read_csv(COLD/'fit_audit.csv');old=old.rename(columns={'z05':'z_first_tau_0p026316','z95':'z_last_tau_0p973684','z50':'z_median'});old.to_csv(OUT/'stage_a_fit_audit_clear_names.csv',index=False,encoding='utf-8-sig')
    (OUT/'schema_notes.md').write_text('A阶段原始fit_audit中的z05/z95实际对应19点网格首末，tau=0.5/19和18.5/19，命名不准确；保留原始记录，正确名称见stage_a_fit_audit_clear_names.csv。两阶段marginal_diagnosis中的z05/z95则确为实测标准化误差的5%和95%。historical_error_mismatch使用未来真值，只作解释，不能作为在线输入。\n',encoding='utf-8')
    for p,h in identity.items():assert sha(p)==h,p
    js(OUT/'startup_diagnostic_complete.json',dict(status='COMPLETE',exposure_rows=len(exposure),mismatch_rows=len(lag),raw_comparisons=len(rows),new_forecasts=False,utc=datetime.now(timezone.utc).isoformat()));print('STARTUP DIAGNOSTIC COMPLETE',flush=True)

if __name__=='__main__':run()

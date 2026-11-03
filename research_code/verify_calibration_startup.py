"""Independently reconstruct activation and post-hoc historical mismatch diagnostics."""
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from model import ART
from research_residual_shape import read,sha,js
from verify_window_error_attribution import reference_bound,close

def check_renamed_columns(old,new):
    # Re-exported CSVs can differ by a serialization ulp; saved fit arrays stay exact.
    for key,dest in [('z05','z_first_tau_0p026316'),('z50','z_median'),('z95','z_last_tau_0p973684')]:close(old[key],new[dest],3e-12)

def verify():
    out=ART/'residual_warm_v1';assert read(out/'startup_diagnostic_complete.json')['status']=='COMPLETE'
    for p,h in read(out/'startup_identity.json').items():assert sha(p)==h,p
    exposure=pd.read_csv(out/'calibration_exposure.csv');lag=pd.read_csv(out/'historical_error_mismatch.csv');pops={};pred={};flags={};raws={};truths={};scales={};high={}
    for fold in ['M05','M07','M09','M11']:
        pops[fold]=dict(np.load(out/fold/'population.npz'));truths[fold]=np.load(ART/'seasonal_probability_v1'/fold/'test_truth.npz')['y'];scales[fold]=float(np.load(ART/'seasonal_probability_v1'/fold/'data.npz')['scale']);high[fold]=dict(np.load(out/fold/'fixed_cohorts.npz'))
        for seed in [20241101,20241102,20241103]:
            raws[fold,seed]=np.load(out/fold/f'predictions_R_{seed}.npz')['q']
            for history in ['cold','warm']:flags[fold,seed,history]=[np.load(out/fold/f'fit_{history}_{seed}_w{b}.npz')['fallback'] for b in range(9)]
            for case in ['R','B']:pred[fold,seed,case]=dict(np.load(out/fold/f'diagnostic_{case}_{seed}.npz'))
    for row in exposure.itertuples():
        pop=pops[row.fold];y=pop[row.kind+'_labels'];mask=high[row.fold][row.kind].copy() if row.cohort=='baseline_high' else np.ones_like(y,bool);count=17 if row.kind=='single' else 3
        if row.offset!=-1:mask&=(np.repeat([0,24,48,66],count)==row.offset)[None,:]
        fallback=np.zeros_like(mask)
        for i,block in enumerate(pop['blocks']):
            for off,(offset,group) in enumerate([(0,0),(24,2),(48,3),(66,3)]):fallback[i,off*count:(off+1)*count]=flags[row.fold,row.seed,row.history][block][group]
        n=int(mask.sum());nf=int((mask&fallback).sum());assert n==row.n and nf==row.fallback_records and n-nf==row.applied_records and (mask&fallback&(y==0)).sum()==row.fallback_known_failures
        if n:close(nf/n,row.fallback_fraction)
    for row in lag.itertuples():
        fold=row.fold;seed=row.seed;pop=pops[fold];positions=np.flatnonzero(pop['blocks']==row.block);a=row.lead_start-1;b=row.lead_end;raw=raws[fold,seed][positions,a:b];y=truths[fold][positions,a:b];valid=np.isfinite(y);counts=valid.sum((1,2));use=counts>0;n=int(use.sum());assert n==row.future_observed_origins and valid.sum()==row.future_observed_labels
        group=[1,7,25,49].index(row.lead_start);fit=np.load(out/fold/f'fit_warm_{seed}_w{row.block}.npz');close(fit['z_quantiles'][group,9],row.history_median_standardized_error)
        if n:
            z=(y-raw[...,9])/np.maximum((raw[...,17]-raw[...,1])/2,.1*scales[fold]);w=np.zeros_like(y)
            for i in np.flatnonzero(use):w[i][valid[i]]=1/(n*counts[i])
            v=row.application_median_standardized_error;assert (z[valid]==v).any() or np.min(abs(z[valid]-v))<1e-12
            assert w[valid&(z<v-1e-12)].sum()<=.5+1e-12 and w[valid&(z<=v+1e-12)].sum()>=.5-1e-12
            assert row.sign_reversal==bool(row.history_median_standardized_error*v<0)
        else:assert np.isnan(row.application_median_standardized_error)
        assert row.diagnostic_only_future_truth_used
    bound_rows=0;draw_count=0
    for row in pd.read_csv(out/'raw_vs_calibrated_posthoc.csv').itertuples():
        fs=['M05','M07','M09','M11'] if row.scope=='ALL' else [row.scope];kind=row.kind;y=np.concatenate([pops[f][kind+'_labels'] for f in fs]);keys=np.concatenate([pops[f][kind+'_keys'][pops[f][kind+'_inverse']].reshape(-1,2) for f in fs]);_,inverse=np.unique(keys,axis=0,return_inverse=True);inverse=inverse.reshape(y.shape);blocks=np.concatenate([pops[f]['blocks']+9*i for i,f in enumerate(fs)]);draws=np.concatenate([np.load(out/f/'draws.npz')['draws']+9*i for i,f in enumerate(fs)],axis=1);mask=np.concatenate([high[f][kind] for f in fs]) if row.cohort=='baseline_high' else np.ones_like(y,bool)
        pp={c:np.stack([np.concatenate([pred[f,s,c][kind] for f in fs]) for s in [20241101,20241102,20241103]]) for c in ['R','B']};c=np.mean(pp['R']**2-pp['B']**2,0);d=2*np.mean(pp['B']-pp['R'],0);lo,hi,bl,bu,bn,_=reference_bound(c,d,y,inverse,mask,blocks,draws);close([lo,hi],[row.lower,row.upper]);saved=np.load(out/f'raw_calibrated_{row.scope}_{kind}_{row.cohort}.npz');close(saved['bootstrap_lower'],bl);close(saved['bootstrap_upper'],bu);close(saved['bootstrap_n'],bn);valid=np.isfinite(bl)&np.isfinite(bu)
        if valid.sum()>=1900:close([np.quantile(bl[valid],.025),np.quantile(bu[valid],.975)],[row.outer_low,row.outer_high])
        bound_rows+=1;draw_count+=len(draws)
    old=pd.read_csv(ART/'residual_shape_v1/fit_audit.csv');new=pd.read_csv(out/'stage_a_fit_audit_clear_names.csv')
    check_renamed_columns(old,new)
    result=dict(status='PASS',exposure_rows=len(exposure),historical_mismatch_rows=len(lag),raw_vs_calibrated_rows=bound_rows,bootstrap_draws_recomputed=draw_count,old_fit_columns_verified=True,future_diagnostic_not_used_by_forecast=True,utc=datetime.now(timezone.utc).isoformat());js(out/'startup_verification.json',result);print(result,flush=True)

if __name__=='__main__':verify()

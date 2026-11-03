"""Independent audit: empirical-CDF inequalities, integer-rank paths, pinball CRPS."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json
import numpy as np
import pandas as pd
from model import ART
from verify_window_error_attribution import reference_bound,close
from research_residual_shape import read,sha,js

FOLDS=['M05','M07','M09','M11'];SEEDS=[20241101,20241102,20241103]
BINS=[(0,6),(6,24),(24,48),(48,72)];OFFSETS=[0,24,48,66];GROUPS=[[0,1,2],[8,9,10],[13,15,16]]

def verify(study):
    warm=study=='warm';out=ART/('residual_warm_v1' if warm else 'residual_shape_v1');cases=['B','R','C','WL','WS','WT'] if warm else ['B','L','S','T']
    assert read(out/'evaluation_complete.json')['status']=='COMPLETE'
    ident={}
    for name in ['source_identity.json','evaluation_identity.json']:
        for p,h in read(out/name).items():assert sha(p)==h,p;ident[p]=h
    inv=read(out/'inventory.json');aud=pd.read_csv(out/'fit_audit.csv',float_precision='round_trip');diag=pd.read_csv(out/'marginal_diagnosis.csv',float_precision='round_trip');raw=np.load(ART/'data.npz')['raw']
    pops={};preds={};means={};parameter_groups=0;cdf_checks=0;forecasts=0;observed_cells=0;diagnostic_rows=0
    for fold in FOLDS:
        folder=out/fold;pop=dict(np.load(folder/'population.npz'));pops[fold]=pop;ids=pop['ids'];truth=np.stack([raw[i+1:i+73] for i in ids]);close(truth,np.load(ART/'seasonal_probability_v1'/fold/'test_truth.npz')['y'],0);scale=float(np.load(ART/'seasonal_probability_v1'/fold/'data.npz')['scale']);high=np.load(folder/'fixed_cohorts.npz')['single'];highhour=np.zeros_like(truth,bool)
        sy=[];gy=[]
        for j,off in enumerate(OFFSETS):
            highhour[:,off:off+6,:]=np.repeat(high[:,j*17:(j+1)*17,None],6,axis=2).transpose(0,2,1)
            z=truth[:,off:off+6];bad=(np.isfinite(z)&(z>10)).any(1);complete=np.isfinite(z).all(1);sy.append(np.where(bad,0.,np.where(complete,1.,np.nan)))
            for nodes in GROUPS:
                x=z[:,:,nodes];gy.append(np.where((np.isfinite(x)&(x>10)).any((1,2)),0.,np.where(np.isfinite(x).all((1,2)),1.,np.nan)))
        close(pop['single_labels'],np.concatenate(sy,1),0);close(pop['group_labels'],np.stack(gy,1),0)
        score_cache={}
        for seed in SEEDS:
            qcase={c:np.load(next(x['q_source'] for x in inv if (x['fold'],x['seed'],x['case'])==(fold,seed,c)))['q'].astype(float) for c in cases}
            base=qcase['R'] if warm else qcase['B'];ranks=np.load(next(x['ranks_source'] for x in inv if (x['fold'],x['seed'],x['case'])==(fold,seed,'B')))['ranks'];np.testing.assert_array_equal(np.sort(ranks,axis=0),np.broadcast_to(np.arange(57)[:,None,None],ranks.shape))
            if warm:
                arch=np.load(ART/'seasonal_rolling_v1'/fold/'archive_data.npz');aq=np.load(ART/'seasonal_rolling_v1'/fold/f'archive_predictions_gated_graph_atoms_{seed}.npz')['q'];before=arch['ids']<ids[0]
                original=np.load(ART/'seasonal_probability_v1'/fold/f'predictions_gated_graph_atoms_{seed}_budget40.npz')['test'];close(base,original,0);positions=np.searchsorted(arch['ids'],ids);close(arch['ids'][positions],ids,0);close(aq[positions],base,0);close(arch['truth'][positions],truth,0)
                histories=[('cold',base,truth,ids),('warm',np.concatenate([aq[before],base]),np.concatenate([arch['truth'][before],truth]),np.concatenate([arch['ids'][before],ids]))]
            else:histories=[('cold',base,truth,ids)]
            for history,hq,hy,hi in histories:
                for block in np.unique(pop['blocks']):
                    pos=np.flatnonzero(pop['blocks']==block);issue=int(ids[pos[0]]);filename=f'fit_{history}_{seed}_w{block}.npz' if warm else f'fit_{seed}_w{block}.npz';saved=np.load(folder/filename)
                    for g,(a,b) in enumerate(BINS):
                        t=hi[:,None,None]+np.arange(a+1,b+1)[None,:,None];valid=np.isfinite(hy[:,a:b])&(t<issue)&(t>=issue-672);counts=valid.sum((1,2));use=counts>0;n=int(use.sum());nlabel=int(valid.sum());fallback=n<20 or nlabel<100
                        row=aud[(aud.fold==fold)&(aud.seed==seed)&(aud.block==block)&(aud.lead_start==a+1)]
                        if warm:row=row[row.history==history]
                        assert len(row)==1;row=row.iloc[0];assert int(row.origins)==n and int(row.labels)==nlabel and bool(row.fallback)==fallback and bool(saved['fallback'][g])==fallback
                        if nlabel:
                            times=np.broadcast_to(t,valid.shape)[valid];assert row.latest_label==times.max()<issue and row.earliest_label==times.min()>=issue-672
                        zq=saved['z_quantiles'][g]
                        if fallback:assert np.isnan(zq).all()
                        else:
                            values=[];weights=[]
                            for i in np.flatnonzero(use):
                                ok=valid[i];qq=hq[i,a:b][ok];ss=np.maximum((qq[:,17]-qq[:,1])/2,.1*scale);values.extend((hy[i,a:b][ok]-qq[:,9])/ss);weights.extend(np.full(int(counts[i]),1/(n*counts[i])))
                            values=np.array(values);weights=np.array(weights);close(weights.sum(),1)
                            for v,p in zip(zq,(np.arange(19)+.5)/19):
                                assert (values==v).any(),'empirical quantile must be an observed residual'
                                assert weights[values<v].sum()<=p+2e-12 and weights[values<=v].sum()>=p-2e-12,'CDF quantile inequality failed'
                                cdf_checks+=1
                        targets=[('C','T')] if warm and history=='cold' else [('WL','L'),('WS','S'),('WT','T')] if warm else [('L','L'),('S','S'),('T','T')]
                        qq=base[pos,a:b];ss=np.maximum((qq[...,17]-qq[...,1])/2,.1*scale)[...,None];mm=qq[...,9,None]
                        for case,method in targets:
                            if fallback:expected=qq
                            elif method=='L':expected=np.clip(qq+ss*zq[9],0,None)
                            elif method=='S':expected=np.clip(mm+ss*(zq[9]+(zq-zq[::-1])/2),0,None)
                            else:expected=np.clip(mm+ss*zq,0,None)
                            close(qcase[case][pos,a:b],expected,0)
                        parameter_groups+=1
            for case,qq in qcase.items():
                assert np.isfinite(qq).all() and (qq>=0).all() and (np.diff(qq,axis=-1)>=0).all()
                stored=dict(np.load(folder/f'diagnostic_{case}_{seed}.npz'));preds[fold,seed,case]=stored;safe_count=(qq<=10).sum(-1);close(stored['hourly'],safe_count/19,0)
                for i in range(len(qq)):
                    safe=ranks//3<safe_count[i][None,:,:];single=[];group=[]
                    for off in OFFSETS:
                        single.extend(safe[:,off:off+6].all(1).mean(0))
                        for nodes in GROUPS:group.append(safe[:,off:off+6][:,:,nodes].all((1,2)).mean())
                    close(stored['single'][i],single,0);close(stored['group'][i],group,0);forecasts+=1
                score=np.load(folder/f'hourly_scores_{case}_{seed}.npz');error=truth[...,None]-qq;tau=(np.arange(19)+.5)/19;expected=2*np.mean((tau-(error<0))*error,axis=-1);close(score['crps'],expected);close(score['brier'],np.where(np.isfinite(truth),(safe_count/19-(truth<=10))**2,np.nan));close(score['mae'],abs(truth-qq[...,9]));observed_cells+=int(np.isfinite(truth).sum());score_cache[case,seed]={key:score[key] for key in ['crps','brier','mae']}
                rows=diag[(diag.fold==fold)&(diag.seed==seed)&(diag.case==case)]
                for row in rows.itertuples():
                    mask=highhour.copy() if row.cohort=='baseline_high_hours' else np.ones_like(truth,bool);mask[:,:row.lead_start-1]=False;mask[:,row.lead_end:]=False;valid=mask&np.isfinite(truth);sat=mask&(qq[...,-1]<=10);obs=int(valid.sum());fail=int((sat&np.isfinite(truth)&(truth>10)).sum());unknown=int((sat&~np.isfinite(truth)).sum());satn=int(sat.sum())
                    assert row.records==mask.sum() and row.observed==obs and row.unknown==(mask&~np.isfinite(truth)).sum() and row.saturated_records==satn and row.saturated_known_failures==fail and row.saturated_unknown==unknown
                    if obs:
                        e=(truth-qq[...,9])[valid];z=e/np.maximum((qq[...,17]-qq[...,1])/2,.1*scale)[valid]
                        close([np.mean(e),np.median(e),np.mean(abs(e)),np.mean(truth[valid]>qq[...,18][valid]),np.mean(truth[valid]>qq[...,17][valid]),np.mean(truth[valid]<qq[...,1][valid]),np.mean((truth[valid]>=qq[...,1][valid])&(truth[valid]<=qq[...,17][valid])),np.mean(truth[valid]<=10),*np.quantile(z,[.05,.5,.95])],[row.mean_error,row.median_error,row.mae,row.above_largest_fraction,row.above_eighteenth_fraction,row.below_second_fraction,row.central_interval_coverage,row.observed_safe,row.z05,row.z50,row.z95])
                    if mask.any():close([np.mean((qq[...,17]-qq[...,1])[mask]),np.mean((qq<=10).mean(-1)[mask]),np.mean((qq==0).mean(-1)[mask])],[row.mean_width,row.mean_predicted_safe,row.zero_atom_fraction])
                    if satn:close([fail/satn,(fail+unknown)/satn],[row.saturated_failure_lower,row.saturated_failure_upper])
                    diagnostic_rows+=1
            print('VERIFY',study,fold,seed,parameter_groups,forecasts,flush=True)
        for case in cases:means[fold,case]={metric:np.mean([score_cache[case,s][metric] for s in SEEDS],0) for metric in ['crps','brier','mae']}
        for kind in ['single','group']:np.testing.assert_array_equal(np.load(folder/'fixed_cohorts.npz')[kind],np.mean([preds[fold,s,'B'][kind] for s in SEEDS],0)>=.8-1e-12)
    refs={}
    for scope in ['ALL',*FOLDS]:
        fs=FOLDS if scope=='ALL' else [scope]
        for kind in ['single','group']:
            y=np.concatenate([pops[f][kind+'_labels'] for f in fs]);keys=np.concatenate([pops[f][kind+'_keys'][pops[f][kind+'_inverse']].reshape(-1,2) for f in fs]);_,ix=np.unique(keys,axis=0,return_inverse=True);ix=ix.reshape(y.shape);blocks=np.concatenate([pops[f]['blocks']+9*i for i,f in enumerate(fs)]);draws=np.concatenate([np.load(out/f/'draws.npz')['draws']+9*i for i,f in enumerate(fs)],axis=1);high=np.concatenate([np.load(out/f/'fixed_cohorts.npz')[kind] for f in fs]);pp={c:np.stack([np.concatenate([preds[f,s,c][kind] for f in fs]) for s in SEEDS]) for c in cases};refs[scope,kind]=(y,ix,blocks,draws,high,pp)
    nbound=0;nboot=0
    for row in pd.concat([pd.read_csv(out/'absolute_bounds.csv'),pd.read_csv(out/'paired_bounds.csv')]).itertuples():
        y,ix,blocks,draws,high,pp=refs[row.scope,row.kind];pa=pp[row.candidate];pb=None if row.baseline=='absolute' else pp[row.baseline];c=np.mean(pa**2,0) if pb is None else np.mean(pa**2-pb**2,0);d=1-2*np.mean(pa,0) if pb is None else 2*np.mean(pb-pa,0);mask=high if row.cohort=='baseline_high' else np.ones_like(y,bool)
        lo,hi,bl,bu,bn,unique=reference_bound(c,d,y,ix,mask,blocks,draws);close([lo,hi],[row.lower,row.upper]);assert row.n==mask.sum();saved=np.load(out/f'bounds_{row.scope}_{row.kind}_{row.cohort}_{row.candidate}_{row.baseline}.npz');close(saved['bootstrap_lower'],bl);close(saved['bootstrap_upper'],bu);close(saved['bootstrap_n'],bn,0);valid=np.isfinite(bl)&np.isfinite(bu);assert valid.sum()==row.valid_draws
        if row.interval_supported:close([np.quantile(bl[valid],.025),np.quantile(bu[valid],.975)],[row.outer_low,row.outer_high])
        for side,value in [('lower',lo),('upper',hi)]:
            w=saved['witness_'+side];close(w[np.isfinite(unique)],unique[np.isfinite(unique)],0);close(np.mean((c+d*w[ix])[mask]),value)
        nbound+=1;nboot+=len(bl)
        if nbound%40==0:print('BOUNDS',study,nbound,flush=True)
    nrel=0
    for file in ['fixed_cohort_reliability.csv','offset_reliability.csv','candidate_high_coverage.csv']:
        for row in pd.read_csv(out/file).itertuples():
            y,ix,blocks,draws,high,pp=refs[row.scope,row.kind];p=pp[row.case].mean(0);mask=p>=.8-1e-12 if file=='candidate_high_coverage.csv' else high.copy() if row.cohort=='baseline_high' else np.ones_like(y,bool)
            if file=='offset_reliability.csv':mask&=(np.repeat(OFFSETS,17 if row.kind=='single' else 3)==row.offset)[None,:]
            n=int(mask.sum());assert n==row.n
            if n:close([p[mask].mean(),(y[mask]==1).sum()/n,((y[mask]==1).sum()+np.isnan(y[mask]).sum())/n],[row.mean_probability,row.observed_lower,row.observed_upper])
            nrel+=1
    nmarg=0
    for row in pd.read_csv(out/'marginal_paired.csv').itertuples():
        fs=FOLDS if row.scope=='ALL' else [row.scope];bs=[];cs=[];ns=[];dd=[]
        for fold in fs:
            b=means[fold,row.baseline][row.metric][:,row.lead_start-1:row.lead_end];c=means[fold,row.candidate][row.metric][:,row.lead_start-1:row.lead_end];blocks=pops[fold]['blocks'];bs.append(np.array([np.nansum(b[blocks==k]) for k in range(9)]));cs.append(np.array([np.nansum(c[blocks==k]) for k in range(9)]));ns.append(np.array([np.isfinite(b[blocks==k]).sum() for k in range(9)]));dd.append(np.load(out/fold/'draws.npz')['draws'])
        n=np.sum(ns);bv=np.sum(bs)/n;cv=np.sum(cs)/n;num=sum((c-b)[d].sum(1) for b,c,d in zip(bs,cs,dd));den=sum(nc[d].sum(1) for nc,d in zip(ns,dd));draw=num/den;close([bv,cv,cv-bv,*np.quantile(draw,[.025,.975])],[row.base,row.value,row.difference,row.low,row.high]);assert n==row.n;nmarg+=1
    for p,h in ident.items():assert sha(p)==h,p
    result=dict(status='PASS',study=study,source_files=len(ident),parameter_groups=parameter_groups,weighted_cdf_checks=cdf_checks,scenario_forecasts=forecasts,observed_hour_losses=observed_cells,diagnostic_rows=diagnostic_rows,bound_rows=nbound,bootstrap_draws_recomputed=nboot,reliability_rows=nrel,marginal_comparison_rows=nmarg,method='independent CDF inequalities; explicit raw archive join; integer-rank weather events; midpoint pinball CRPS; independent shared-label add.at bounds; all saved bootstrap draws',utc=datetime.now(timezone.utc).isoformat())
    js(out/'verification.json',result);print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',choices=['cold','warm'],required=True);verify(p.parse_args().study)

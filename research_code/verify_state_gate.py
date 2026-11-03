"""Independent feature, neighbor, pinball-risk and forecast reconstruction."""
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from model import ART
from verify_window_error_attribution import reference_bound,close
from research_residual_shape import read,sha,js
FOLDS=['M05','M07','M09','M11'];SEEDS=[20241101,20241102,20241103]
BINS=[(0,6),(6,24),(24,48),(48,72)];OFFSETS=[0,24,48,66];GROUPS=[[0,1,2],[8,9,10],[13,15,16]]

def verify():
    study='state_gate';out=ART/'state_gate_v1';parent=ART/'calibration_gate_v1';cases=['B','R','W28','M50','G28','K4','K6']
    assert read(out/'evaluation_complete.json')['status']=='COMPLETE'
    ident={}
    for name in ['source_identity.json','evaluation_identity.json']:
        for p,h in read(out/name).items():assert sha(p)==h,p;ident[p]=h
    pm=read(parent/'final_manifest.json');copied=0
    for fold in FOLDS:
        names=['population.npz','fixed_cohorts.npz','draws.npz']+[f'{stem}_{c}_{s}.npz' for stem in ['diagnostic','hourly_scores'] for c in cases[:5] for s in SEEDS]
        for name in names:assert sha(parent/fold/name)==pm[str(parent/fold/name)]==sha(out/fold/name);copied+=1
    inv=read(out/'inventory.json');aud=pd.read_csv(out/'state_audit.csv');raw=np.load(ART/'data.npz')['raw'];det=pd.read_csv(out/'advance_detection.csv');origin_diag=pd.read_csv(out/'origin_diagnosis.csv')
    pops={};preds={};means={};forecasts=0;observed_cells=0;gate_groups=0;gate_risks=0;feature_values=0
    for fold in FOLDS:
        folder=out/fold;pop=dict(np.load(folder/'population.npz'));pops[fold]=pop;ids=pop['ids'];truth=np.stack([raw[i+1:i+73] for i in ids]);close(truth,np.load(ART/'seasonal_probability_v1'/fold/'test_truth.npz')['y'],0);data=np.load(ART/'seasonal_probability_v1'/fold/'data.npz');scale=float(data['scale']);center=float(data['center'])
        sy=[];gy=[]
        for off in OFFSETS:
            z=truth[:,off:off+6];sy.append(np.where((np.isfinite(z)&(z>10)).any(1),0.,np.where(np.isfinite(z).all(1),1.,np.nan)))
            for nodes in GROUPS:
                x=z[:,:,nodes];gy.append(np.where((np.isfinite(x)&(x>10)).any((1,2)),0.,np.where(np.isfinite(x).all((1,2)),1.,np.nan)))
        close(pop['single_labels'],np.concatenate(sy,1),0);close(pop['group_labels'],np.stack(gy,1),0);score_cache={}
        for seed in SEEDS:
            qcase={c:np.load(next(x['q_source'] for x in inv if (x['fold'],x['seed'],x['case'])==(fold,seed,c)))['q'].astype(float) for c in cases}
            base=qcase['R'];ranks=np.load(next(x['ranks_source'] for x in inv if (x['fold'],x['seed'],x['case'])==(fold,seed,'B')))['ranks'];np.testing.assert_array_equal(np.sort(ranks,axis=0),np.broadcast_to(np.arange(57)[:,None,None],ranks.shape))
            ar=np.load(ART/'seasonal_rolling_v1'/fold/'archive_data.npz');aq=np.load(ART/'seasonal_rolling_v1'/fold/f'archive_predictions_gated_graph_atoms_{seed}.npz')['q'];before=ar['ids']<ids[0];hi=np.concatenate([ar['ids'][before],ids]);hq=np.concatenate([aq[before],base]);hy=np.concatenate([ar['truth'][before],truth]);pre=int(before.sum());hist=parent/fold/f'honest_history_{seed}.npz';assert sha(hist)==pm[str(hist)];hw=np.load(hist)['q'];close(hw[pre:],qcase['W28'],0)
            stored=np.load(folder/f'features_{seed}.npz');close(stored['ids'],hi,0);assert stored['pretest_origins']==pre;features=np.empty_like(stored['features'])
            for i,t in enumerate(hi):
                a0=raw[max(0,t-5):t+1].astype(float);a1=raw[max(0,t-11):t-5].astype(float);a2=raw[max(0,t-23):t+1].astype(float);good0=a0[np.isfinite(a0)];good1=a1[np.isfinite(a1)];good2=a2[np.isfinite(a2)]
                m0=good0.sum()/len(good0) if len(good0) else center;m1=good1.sum()/len(good1) if len(good1) else center;vol=np.sqrt(np.mean((good2-good2.mean())**2)) if len(good2) else 0.
                for g,(a,b) in enumerate(BINS):
                    r=hq[i,a:b].astype(float);w=hw[i,a:b].astype(float);features[i,g]=[(m0-center)/scale,(m0-m1)/scale,vol/scale,len(good0)/a0.size,np.mean(r[...,9]-w[...,9])/scale,np.log((np.mean(w[...,17]-w[...,1])+.1*scale)/(np.mean(r[...,17]-r[...,1])+.1*scale))]
            close(features,stored['features']);feature_values+=features.size
            # Descriptive diagnostic rows are independently checked against frozen scores.
            sw=np.load(parent/fold/f'hourly_scores_W28_{seed}.npz')['crps'];sm=np.load(parent/fold/f'hourly_scores_M50_{seed}.npz')['crps']
            for row in origin_diag[(origin_diag.fold==fold)&(origin_diag.seed==seed)].itertuples():
                i=int(np.searchsorted(ids,row.issue));g=[a+1 for a,b in BINS].index(row.lead_start);a,b=BINS[g];close([getattr(row,f'f{k}') for k in range(6)],features[pre+i,g]);v=np.isfinite(sw[i,a:b]);assert row.observed==v.sum();actual=(sw[i,a:b]-sm[i,a:b])[v].mean() if v.any() else np.nan;close(row.harm,actual)
            daily=(ids-ids[0])//24
            for block in np.unique(daily):
                pos=np.flatnonzero(daily==block);issue=int(ids[pos[0]]);current=features[pre+pos[0]]
                for case,dim in [('K4',4),('K6',6)]:
                    saved=np.load(folder/f'state_{case}_{seed}_d{block}.npz');assert saved['issue']==issue
                    for g,(a,b) in enumerate(BINS):
                        times=hi[:,None,None]+np.arange(a+1,b+1)[None,:,None];mask=np.isfinite(hy[:,a:b])&(times<issue)&(times>=issue-672);counts=mask.sum((1,2));eligible=np.flatnonzero(counts>0);fallback=len(eligible)<20 or counts.sum()<100;selected=[]
                        if not fallback:
                            f=features[eligible,g,:dim];q1,q3=np.percentile(f,[25,75],axis=0);sc=np.maximum(q3-q1,.1);dist=np.sum(np.clip((f-current[g,:dim])/sc,-3,3)**2,axis=1);order=sorted(range(len(eligible)),key=lambda k:(dist[k],-hi[eligible[k]]));selected=eligible[order[:40]].tolist();fallback=counts[selected].sum()<100
                        close(saved['selected_positions'][g][saved['selected_positions'][g]>=0],selected,0)
                        row=aud[(aud.fold==fold)&(aud.seed==seed)&(aud['case']==case)&(aud.block==block)&(aud.lead_start==a+1)].iloc[0];assert row.eligible_origins==len(eligible) and row.selected_origins==len(selected) and row.selected_labels==counts[selected].sum() and bool(row.fallback)==fallback
                        if selected:
                            tt=np.broadcast_to(times,mask.shape)[selected][mask[selected]];assert row.latest_label==tt.max()<issue and row.earliest_label==tt.min()>=issue-672
                        if fallback:weight=.5;risks=np.full(5,np.nan)
                        else:
                            risks=[]
                            for ww in [0,.25,.5,.75,1]:
                                each=[]
                                for i in selected:
                                    qq=(1-ww)*hq[i,a:b][mask[i]]+ww*hw[i,a:b][mask[i]];e=hy[i,a:b][mask[i]][:,None]-qq;tau=(np.arange(19)+.5)/19;each.append(np.mean(2*np.mean((tau-(e<0))*e,axis=-1)))
                                risks.append(float(np.mean(each)));gate_risks+=1
                            opts=[i for i,v in enumerate(risks) if v<=min(risks)+1e-12];chosen=min(opts,key=lambda k:(abs(k/4-.5),k/4));weight=chosen/4
                        close(saved['losses'][g],risks);close([saved['weights'][g],row.weight],[weight,weight],0);close(qcase[case][pos,a:b],(1-weight)*base[pos,a:b]+weight*qcase['W28'][pos,a:b]);gate_groups+=1
                        dr=det[(det.fold==fold)&(det.seed==seed)&(det['case']==case)&(det.issue==issue)&(det.lead_start==a+1)].iloc[0];estimated=risks[4]-risks[2] if not fallback else np.nan;close(dr.estimated_harm,estimated);v=np.isfinite(sw[pos[0],a:b]);actual=(sw[pos[0],a:b]-sm[pos[0],a:b])[v].mean() if v.any() else np.nan;close(dr.actual_harm,actual);assert dr.observed==v.sum()
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
            print('VERIFY STATE',fold,seed,gate_groups,flush=True)
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
    result=dict(status='PASS',parent_copies=copied,feature_values=feature_values,gate_groups=gate_groups,gate_risks=gate_risks,scenario_forecasts=forecasts,observed_hour_losses=observed_cells,bound_rows=nbound,bootstrap_draws_recomputed=nboot,reliability_rows=nrel,marginal_rows=nmarg)
    js(out/'verification.json',result);print(result,flush=True)
if __name__=='__main__':verify()

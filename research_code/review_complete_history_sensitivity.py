"""Fixed complete-input-history subgroup; no new inference or tuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from precision_truth_core import load_truth
from review_baselines_core import loss_event_totals

BASE=Path(__file__).resolve().parent;OUT=BASE/'artifacts/review_followup_v1/baselines'
FOLDS=['M05','M07','M09','M11'];SEEDS=[20241101,20241102,20241103]

def run():
    times,_,_=load_truth();data={};support=[]
    for f,expected in zip(FOLDS,[61,49,109,173]):
        z=dict(np.load(OUT/f/'population.npz'));p=np.load(BASE/'artifacts/seasonal_probability_v1'/f/'data.npz');ids=z['ids']
        complete=np.array([(p['x'][i-23:i+1,:,2]==1).all() for i in ids]);assert complete.sum()==expected
        a=np.stack([np.load(OUT/f/f'probabilities_context1_graph_{s}_refit64_natural.npz')['single'] for s in SEEDS])[:,complete]
        b=np.stack([np.load(OUT/f/f'probabilities_context1_mean_{s}_refit64_natural.npz')['single'] for s in SEEDS])[:,complete]
        y=z['single_labels'][complete];ix=z['single_inverse'][complete]
        lo,hi,count=loss_event_totals(a,b,y,ix)
        # Independent reference: losses at both possible labels, then one shared outcome per event.
        zero=np.bincount(ix.ravel(),weights=np.mean(a*a-b*b,axis=0).ravel(),minlength=len(count))
        one=np.bincount(ix.ravel(),weights=np.mean((a-1)**2-(b-1)**2,axis=0).ravel(),minlength=len(count))
        label=np.full(len(count),np.nan);known=np.isfinite(y);label[ix[known]]=y[known]
        reflo=np.where(label==0,zero,np.where(label==1,one,np.minimum(zero,one)))
        refhi=np.where(label==0,zero,np.where(label==1,one,np.maximum(zero,one)))
        np.testing.assert_allclose(lo,reflo,atol=1e-14,rtol=0);np.testing.assert_allclose(hi,refhi,atol=1e-14,rtol=0)
        d=np.load(OUT/f/'target_time_draws.npz');anchor=pd.Timestamp(str(d['anchor']));draw=d['draws'];k=draw.shape[1]
        block=((times[z['single_keys'][:len(count),0]]-anchor).total_seconds()//(7*86400)).astype(int)
        ls=np.bincount(block,weights=lo,minlength=k);us=np.bincount(block,weights=hi,minlength=k);ns=np.bincount(block,weights=count,minlength=k)
        rl=np.bincount(block,weights=reflo,minlength=k);ru=np.bincount(block,weights=refhi,minlength=k)
        np.testing.assert_allclose(ls[draw].sum(1),rl[draw].sum(1),atol=1e-12,rtol=0);np.testing.assert_allclose(us[draw].sum(1),ru[draw].sum(1),atol=1e-12,rtol=0)
        data[f]=(lo.sum(),hi.sum(),int(count.sum()),ls[draw].sum(1),us[draw].sum(1),ns[draw].sum(1),int((ns>0).sum()))
        support.append(dict(fold=f,complete_24h_origins=int(complete.sum()),all_origins=len(ids),records=int(count.sum()),physical_windows=int((count>0).sum()),unknown_records=int(np.isnan(y).sum()),nonempty_target_blocks=int((ns>0).sum()),calendar_blocks=k,terminal_block_records=int(ns[-1]),empty_calendar_blocks=int((ns==0).sum())))
    rows=[]
    for scope in ['ALL',*FOLDS]:
        folds=FOLDS if scope=='ALL' else [scope];numlo=sum(data[f][0] for f in folds);numhi=sum(data[f][1] for f in folds);n=sum(data[f][2] for f in folds);bl=sum(data[f][3] for f in folds);bu=sum(data[f][4] for f in folds);bn=sum(data[f][5] for f in folds);valid=bn>0
        nonempty=min(data[f][6] for f in folds);supported=nonempty>=4 and valid.sum()>=1900
        rows.append(dict(scope=scope,cohort='all17_complete_last24h_input',method='refit64',regime='natural',kind='single',candidate='context1_graph',baseline='context1_mean',origins=sum(s['complete_24h_origins'] for s in support if s['fold'] in folds),n=n,lower=numlo/n,upper=numhi/n,outer_low=float(np.quantile((bl[valid]/bn[valid]),.025)) if supported else None,outer_high=float(np.quantile((bu[valid]/bn[valid]),.975)) if supported else None,minimum_fold_nonempty_blocks=nonempty,valid_draws=int(valid.sum()),supported=bool(supported)))
    pd.DataFrame(rows).to_csv(OUT/'complete_history_paired.csv',index=False,encoding='utf-8-sig')
    result=dict(status='PASS',selection='all 17 nodes x prior 24 hours have input validity 1; issue-time available only; no outcome selection',fixed_comparison='natural/refit64/single/context1_graph minus context1_mean',scope='prespecified review-required subgroup sensitivity, not a new main model-selection endpoint',origins=392,seed='reuse original target_time_draws seed20261103',resampling='retain original nine calendar blocks including zero subgroup-support blocks; keep terminal partial blocks; all physical-window forecasts share multiplicity',support=support,comparisons=rows,independent_verification='direct label-zero/label-one event losses and all bootstrap numerators agree with affine-coefficient core')
    (OUT/'complete_history_sensitivity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':run()

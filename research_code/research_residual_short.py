"""One fixed shorter-history sensitivity, proposed after the warm-start diagnosis."""
from pathlib import Path
from datetime import datetime,timezone
import shutil
import numpy as np
import pandas as pd
from seasonal_probability_core import FOLDS,score_atoms
from residual_shape_core import fit_residual,apply_residual,high_hour_mask,diagnose_marginals,BINS
from warm_residual_core import join_history
from window_error_attribution_core import summarize
from research_residual_shape import ART,BASE,SEEDS,read,sha,js,verify_manifest

OUT=ART/'residual_short_v1';WARM=ART/'residual_warm_v1';ARCH=ART/'seasonal_rolling_v1';SRC=ART/'seasonal_probability_v1'
PLAN=BASE/'研究笔记/第49轮_短历史适应性补充方案_2026-11-02.md';CASES=['B','W28','W7']

def run():
    assert read(WARM/'verification.json')['status']=='PASS';assert not OUT.exists();OUT.mkdir();shutil.copy2(PLAN,OUT/'protocol_frozen.md')
    inventory=read(WARM/'inventory.json');files=[Path(__file__),PLAN,ART.parent/'warm_residual_core.py',ART.parent/'residual_shape_core.py',WARM/'verification.json']
    for fold in FOLDS:
        files.extend([ARCH/fold/'archive_data.npz',SRC/fold/'data.npz',SRC/fold/'test_truth.npz'])
        for seed in SEEDS:files.extend([ARCH/fold/f'archive_predictions_gated_graph_atoms_{seed}.npz',WARM/fold/f'predictions_R_{seed}.npz',WARM/fold/f'predictions_WT_{seed}.npz'])
    identity={str(p):sha(p) for p in files};js(OUT/'source_identity.json',identity);js(OUT/'recipe.json',dict(cases=CASES,lookback_hours=168,exploratory_after_AB=True,primary='supplementary W7-W28/W7-B x single/group, ALL fixed baseline-high',utc=datetime.now(timezone.utc).isoformat()))
    audits=[];diag=[];outinventory=[]
    for fold in FOLDS:
        folder=OUT/fold;folder.mkdir()
        for name in ['population.npz','fixed_cohorts.npz','draws.npz']:shutil.copy2(WARM/fold/name,folder/name)
        pop=dict(np.load(folder/'population.npz'));ids=pop['ids'];blocks=pop['blocks'];y=np.load(SRC/fold/'test_truth.npz')['y'];scale=float(np.load(SRC/fold/'data.npz')['scale']);high=high_hour_mask(np.load(folder/'fixed_cohorts.npz')['single']);archive=np.load(ARCH/fold/'archive_data.npz')
        for seed in SEEDS:
            b=next(x for x in inventory if (x['fold'],x['seed'],x['case'])==(fold,seed,'B'));raw=np.load(WARM/fold/f'predictions_R_{seed}.npz')['q'];aq=np.load(ARCH/fold/f'archive_predictions_gated_graph_atoms_{seed}.npz')['q'];hist=join_history(aq,archive['truth'],archive['ids'],raw,y,ids);ranks=np.load(b['ranks_source'])['ranks'];changed=np.empty_like(raw)
            for block in np.unique(blocks):
                pos=np.flatnonzero(blocks==block);issue=int(ids[pos[0]]);fit=fit_residual(hist['q'],hist['y'],hist['ids'],issue,scale,lookback=168)
                future=np.broadcast_to(hist['ids'][:,None,None]+np.arange(1,73)[None,:,None]>=issue,hist['y'].shape);yp=hist['y'].copy();qp=hist['q'].copy();yp[future]=999;qp[future]=np.nan;check=fit_residual(qp,yp,hist['ids'],issue,scale,lookback=168)
                np.testing.assert_array_equal(fit['z_quantiles'],check['z_quantiles']);assert fit['audit']==check['audit'];changed[pos]=apply_residual(raw[pos],fit,scale,'T')
                np.savez_compressed(folder/f'fit_short_{seed}_w{block}.npz',z_quantiles=fit['z_quantiles'],fallback=fit['fallback'])
                for j,row in enumerate(fit['audit']):audits.append(dict(fold=fold,seed=seed,history='short',block=int(block),issue=issue,scale=scale,applied_origins=len(pos),z_first=float(fit['z_quantiles'][j,0]),z_median=float(fit['z_quantiles'][j,9]),z_last=float(fit['z_quantiles'][j,-1]),**row))
            for case,old in [('B','B'),('W28','WT')]:
                for stem in ['diagnostic','hourly_scores']:shutil.copy2(WARM/fold/f'{stem}_{old}_{seed}.npz',folder/f'{stem}_{case}_{seed}.npz')
                item=next(x for x in inventory if (x['fold'],x['seed'],x['case'])==(fold,seed,old)).copy();item['case']=case;outinventory.append(item)
            np.savez_compressed(folder/f'predictions_W7_{seed}.npz',q=changed,ids=ids);win=[summarize(qi,ranks) for qi in changed];np.savez_compressed(folder/f'diagnostic_W7_{seed}.npz',**{k:np.stack([v[k] for v in win]) for k in win[0]});scores=score_atoms(changed,y);np.savez_compressed(folder/f'hourly_scores_W7_{seed}.npz',crps=scores['crps'],brier=scores['brier_hourly'],mae=scores['mae'])
            for cohort,mask in [('all',np.ones_like(y,bool)),('baseline_high_hours',high)]:
                for a,end in [(0,72),*BINS]:
                    m=mask.copy();m[:,:a]=False;m[:,end:]=False;diag.append(dict(fold=fold,seed=seed,case='W7',cohort=cohort,lead_start=a+1,lead_end=end,**diagnose_marginals(changed,y,m,scale)))
            outinventory.append(dict(fold=fold,seed=seed,case='W7',q_source=str(folder/f'predictions_W7_{seed}.npz'),ranks_source=b['ranks_source'],origins=len(ids)))
            print('SHORT COMPLETE',fold,seed,flush=True)
    old=pd.read_csv(WARM/'marginal_diagnosis.csv');old=old[old.case.isin(['B','WT'])].copy();old['case']=old['case'].replace({'WT':'W28'});diag=pd.concat([old,pd.DataFrame(diag)],ignore_index=True);diag.to_csv(OUT/'marginal_diagnosis.csv',index=False,encoding='utf-8-sig');pd.DataFrame(audits).to_csv(OUT/'fit_audit.csv',index=False,encoding='utf-8-sig');js(OUT/'inventory.json',outinventory)
    for p,h in identity.items():assert sha(p)==h,p
    js(OUT/'experiment_complete.json',dict(status='COMPLETE',unique_origins=880,new_flows=12,all_flows=36,updates=108,group_checks=len(audits),future_pollution_checks=108,diagnosis_rows=len(diag),utc=datetime.now(timezone.utc).isoformat()));print('SHORT EXPERIMENT COMPLETE',flush=True)

if __name__=='__main__':run()

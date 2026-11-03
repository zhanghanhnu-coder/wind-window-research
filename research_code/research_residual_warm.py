"""Round49 post-diagnostic warm-start control; never substitutes the A experiment."""
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

OUT=ART/'residual_warm_v1';COLD=ART/'residual_shape_v1';ARCH=ART/'seasonal_rolling_v1';SRC=ART/'seasonal_probability_v1'
PLAN=BASE/'研究笔记/第49轮_启动历史补充对照_2026-11-02.md';CASES=['B','R','C','WL','WS','WT']

def run():
    assert read(COLD/'evaluation_complete.json')['status']=='COMPLETE';verify_manifest(COLD/'stage_a_completed_snapshot.json')
    assert not OUT.exists(),'preserve previous attempt';OUT.mkdir();shutil.copy2(PLAN,OUT/'protocol_frozen.md')
    inventory=read(COLD/'inventory.json');files=[Path(__file__),PLAN,ART.parent/'warm_residual_core.py',ART.parent/'tests/test_warm_residual.py',ART.parent/'residual_shape_core.py',COLD/'stage_a_completed_snapshot.json']
    for fold in FOLDS:
        files.extend([ARCH/fold/'archive_data.npz',SRC/fold/'data.npz',SRC/fold/'test_truth.npz'])
        for seed in SEEDS:files.extend([ARCH/fold/f'archive_predictions_gated_graph_atoms_{seed}.npz',SRC/fold/f'predictions_gated_graph_atoms_{seed}_budget40.npz'])
    identity={str(p):sha(p) for p in files};js(OUT/'source_identity.json',identity)
    js(OUT/'recipe.json',dict(status='PREDECLARED_BEFORE_B_RESULTS',cases=CASES,exploratory_after_A=True,history='same frozen raw network; pre-first-test archive + exact test origins',primary='supplementary focus only: WT-B/WT-C/WT-WL/WT-WS x single/group ALL fixed baseline-high',utc=datetime.now(timezone.utc).isoformat()))
    audits=[];diag=[];outinventory=[];history_rows=[]
    for fold in FOLDS:
        folder=OUT/fold;folder.mkdir()
        for name in ['population.npz','fixed_cohorts.npz','draws.npz']:shutil.copy2(COLD/fold/name,folder/name)
        pop=dict(np.load(folder/'population.npz'));ids=pop['ids'];blocks=pop['blocks'];y=np.load(SRC/fold/'test_truth.npz')['y'];scale=float(np.load(SRC/fold/'data.npz')['scale']);high=high_hour_mask(np.load(folder/'fixed_cohorts.npz')['single']);archive=np.load(ARCH/fold/'archive_data.npz')
        for seed in SEEDS:
            b=next(x for x in inventory if (x['fold'],x['seed'],x['case'])==(fold,seed,'B'));raw=np.load(SRC/fold/f'predictions_gated_graph_atoms_{seed}_budget40.npz')['test'].astype(float);aq=np.load(ARCH/fold/f'archive_predictions_gated_graph_atoms_{seed}.npz')['q'];hist=join_history(aq,archive['truth'],archive['ids'],raw,y,ids);ranks=np.load(b['ranks_source'])['ranks'];changed={c:np.empty_like(raw) for c in ['C','WL','WS','WT']}
            history_rows.append(dict(fold=fold,seed=seed,pretest_origins=hist['pretest_origins'],test_origins=len(ids),archive_first=int(hist['ids'][0]),test_first=int(ids[0])))
            for block in np.unique(blocks):
                pos=np.flatnonzero(blocks==block);issue=int(ids[pos[0]])
                for history,qq,yy,ii in [('cold',raw,y,ids),('warm',hist['q'],hist['y'],hist['ids'])]:
                    fit=fit_residual(qq,yy,ii,issue,scale)
                    future=np.broadcast_to(ii[:,None,None]+np.arange(1,73)[None,:,None]>=issue,yy.shape);yp=yy.copy();qp=qq.copy();yp[future]=999;qp[future]=np.nan;check=fit_residual(qp,yp,ii,issue,scale)
                    np.testing.assert_array_equal(fit['z_quantiles'],check['z_quantiles']);assert fit['audit']==check['audit']
                    np.savez_compressed(folder/f'fit_{history}_{seed}_w{block}.npz',z_quantiles=fit['z_quantiles'],fallback=fit['fallback'])
                    for j,row in enumerate(fit['audit']):audits.append(dict(fold=fold,seed=seed,history=history,block=int(block),issue=issue,scale=scale,applied_origins=len(pos),z_first=float(fit['z_quantiles'][j,0]),z_median=float(fit['z_quantiles'][j,9]),z_last=float(fit['z_quantiles'][j,-1]),**row))
                    for case,method in ([('C','T')] if history=='cold' else [('WL','L'),('WS','S'),('WT','T')]):changed[case][pos]=apply_residual(raw[pos],fit,scale,method)
            for name in [f'diagnostic_B_{seed}.npz',f'hourly_scores_B_{seed}.npz']:shutil.copy2(COLD/fold/name,folder/name)
            outinventory.append(b)
            for case,qq in [('R',raw),*changed.items()]:
                np.savez_compressed(folder/f'predictions_{case}_{seed}.npz',q=qq,ids=ids)
                win=[summarize(qi,ranks) for qi in qq];np.savez_compressed(folder/f'diagnostic_{case}_{seed}.npz',**{k:np.stack([v[k] for v in win]) for k in win[0]})
                scores=score_atoms(qq,y);np.savez_compressed(folder/f'hourly_scores_{case}_{seed}.npz',crps=scores['crps'],brier=scores['brier_hourly'],mae=scores['mae'])
                for cohort,mask in [('all',np.ones_like(y,bool)),('baseline_high_hours',high)]:
                    for a,end in [(0,72),*BINS]:
                        m=mask.copy();m[:,:a]=False;m[:,end:]=False
                        diag.append(dict(fold=fold,seed=seed,case=case,cohort=cohort,lead_start=a+1,lead_end=end,**diagnose_marginals(qq,y,m,scale)))
                outinventory.append(dict(fold=fold,seed=seed,case=case,q_source=str(folder/f'predictions_{case}_{seed}.npz'),ranks_source=b['ranks_source'],origins=len(ids)))
            print('WARM COMPLETE',fold,seed,flush=True)
    old=pd.read_csv(COLD/'marginal_diagnosis.csv');diag=pd.concat([old[old.case=='B'],pd.DataFrame(diag)],ignore_index=True)
    diag.to_csv(OUT/'marginal_diagnosis.csv',index=False,encoding='utf-8-sig');pd.DataFrame(audits).to_csv(OUT/'fit_audit.csv',index=False,encoding='utf-8-sig');pd.DataFrame(history_rows).to_csv(OUT/'history_support.csv',index=False,encoding='utf-8-sig');js(OUT/'inventory.json',outinventory)
    for p,h in identity.items():assert sha(p)==h,p
    verify_manifest(COLD/'stage_a_completed_snapshot.json');verify_manifest(ART/'v7_release_v1/final_manifest.json')
    js(OUT/'experiment_complete.json',dict(status='COMPLETE',unique_origins=880,new_flows=60,all_flows=72,updates=216,group_checks=len(audits),future_pollution_checks=216,diagnosis_rows=len(diag),utc=datetime.now(timezone.utc).isoformat()))
    print('WARM EXPERIMENT COMPLETE',flush=True)

if __name__=='__main__':run()

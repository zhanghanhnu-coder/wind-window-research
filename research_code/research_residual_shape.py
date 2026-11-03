"""Round49 fixed residual-distribution experiment; frozen parents are read-only."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
import numpy as np
import pandas as pd
from model import ART
from seasonal_probability_core import FOLDS,score_atoms
from residual_shape_core import BINS,fit_residual,apply_residual,high_hour_mask,diagnose_marginals
from window_error_attribution_core import summarize

OUT=ART/'residual_shape_v1';BASE=ART.parent.parent
PARENT=ART/'window_error_attribution_v1'
PLAN=BASE/'研究笔记/第49轮_远期风险与误差分布校正方案_2026-11-02.md'
SEEDS=[20241101,20241102,20241103];CASES=['B','L','S','T']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def verify_manifest(p):
    m=read(p)
    for file,h in m.items():assert sha(file)==h,file
    return len(m)

def run():
    assert not OUT.exists(),'preserve earlier attempts'
    OUT.mkdir();counts={}
    for name in ['window_error_attribution_v1','v7_release_v1']:
        counts[name]=verify_manifest(ART/name/'final_manifest.json')
    inventory=read(PARENT/'source_inventory.json')
    files=[Path(__file__),PLAN,ART.parent/'residual_shape_core.py',ART.parent/'tests/test_residual_shape.py',ART.parent/'tests/test_residual_shape_diagnostics.py',ART.parent/'window_error_attribution_core.py',ART.parent/'seasonal_probability_core.py',ART/'data.npz']
    for item in inventory:
        fold=item['fold'];b=item['base'];files.extend([Path(b['source']),Path(b['ranks_source']),ART/'seasonal_probability_v1'/fold/'data.npz',ART/'seasonal_probability_v1'/fold/'test_truth.npz'])
        for name in ['population.npz','fixed_cohorts.npz','draws.npz']:files.append(PARENT/fold/name)
    identity={str(p):sha(p) for p in files};js(OUT/'source_identity.json',identity);shutil.copy2(PLAN,OUT/'protocol_frozen.md')
    js(OUT/'recipe.json',dict(cases=CASES,folds=list(FOLDS),seeds=SEEDS,lookback_hours=672,minimum_origins=20,minimum_labels=100,scale_floor=.1,threshold=10,primary='T-B/T-L/T-S x single/group, ALL fixed baseline-high cohort',future_labels_fitted=False,parents=counts,utc=datetime.now(timezone.utc).isoformat()))
    diag=[];audits=[];outinventory=[]
    for fold in FOLDS:
        folder=OUT/fold;folder.mkdir()
        for name in ['population.npz','fixed_cohorts.npz','draws.npz']:shutil.copy2(PARENT/fold/name,folder/name)
        pop=dict(np.load(folder/'population.npz'));ids=pop['ids'];blocks=pop['blocks'];y=np.load(ART/'seasonal_probability_v1'/fold/'test_truth.npz')['y'];scale=float(np.load(ART/'seasonal_probability_v1'/fold/'data.npz')['scale']);high=high_hour_mask(np.load(folder/'fixed_cohorts.npz')['single'])
        for seed in SEEDS:
            item=next(v for v in inventory if v['fold']==fold and v['seed']==seed);b=item['base'];q=np.load(b['source'])['q'].astype(float);ranks=np.load(b['ranks_source'])['ranks']
            assert len(q)==len(ids) and q.shape==(*y.shape,19)
            changed={c:np.empty_like(q) for c in ['L','S','T']}
            for block in np.unique(blocks):
                positions=np.flatnonzero(blocks==block);issue=int(ids[positions[0]])
                fit=fit_residual(q,y,ids,issue,scale)
                future=np.broadcast_to(ids[:,None,None]+np.arange(1,73)[None,:,None]>=issue,y.shape);yp=y.copy();qp=q.copy();yp[future]=999;qp[future]=np.nan
                check=fit_residual(qp,yp,ids,issue,scale)
                np.testing.assert_array_equal(fit['z_quantiles'],check['z_quantiles']);np.testing.assert_array_equal(fit['fallback'],check['fallback']);assert fit['audit']==check['audit']
                np.savez_compressed(folder/f'fit_{seed}_w{block}.npz',z_quantiles=fit['z_quantiles'],fallback=fit['fallback'])
                for j,row in enumerate(fit['audit']):audits.append(dict(fold=fold,seed=seed,block=int(block),issue=issue,scale=scale,applied_origins=len(positions),z05=float(fit['z_quantiles'][j,0]),z50=float(fit['z_quantiles'][j,9]),z95=float(fit['z_quantiles'][j,-1]),**row))
                for case in changed:changed[case][positions]=apply_residual(q[positions],fit,scale,case)
            for case,qq in [('B',q),*changed.items()]:
                if case!='B':np.savez_compressed(folder/f'predictions_{case}_{seed}.npz',q=qq,ids=ids)
                window=[summarize(qi,ranks) for qi in qq]
                np.savez_compressed(folder/f'diagnostic_{case}_{seed}.npz',**{k:np.stack([v[k] for v in window]) for k in window[0]})
                if case=='B':
                    old=np.load(PARENT/fold/f'diagnostic_B_{seed}.npz')
                    for key in old.files:np.testing.assert_array_equal(old[key],np.stack([v[key] for v in window]))
                scores=score_atoms(qq,y);np.savez_compressed(folder/f'hourly_scores_{case}_{seed}.npz',crps=scores['crps'],brier=scores['brier_hourly'],mae=scores['mae'])
                for cohort,mask in [('all',np.ones_like(y,bool)),('baseline_high_hours',high)]:
                    for a,end in [(0,72),*BINS]:
                        m=mask.copy();m[:,:a]=False;m[:,end:]=False
                        diag.append(dict(fold=fold,seed=seed,case=case,cohort=cohort,lead_start=a+1,lead_end=end,**diagnose_marginals(qq,y,m,scale)))
                outinventory.append(dict(fold=fold,seed=seed,case=case,q_source=b['source'] if case=='B' else str(folder/f'predictions_{case}_{seed}.npz'),ranks_source=b['ranks_source'],origins=len(ids)))
            print('COMPLETE',fold,seed,'origins',len(ids),'updates',len(np.unique(blocks)),flush=True)
        pd.DataFrame(diag).to_csv(OUT/'marginal_diagnosis_partial.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(diag).to_csv(OUT/'marginal_diagnosis.csv',index=False,encoding='utf-8-sig');pd.DataFrame(audits).to_csv(OUT/'fit_audit.csv',index=False,encoding='utf-8-sig');js(OUT/'inventory.json',outinventory)
    for p,h in identity.items():assert sha(p)==h,p
    verify_manifest(ART/'v7_release_v1/final_manifest.json')
    js(OUT/'experiment_complete.json',dict(status='COMPLETE',unique_origins=880,seed_origins=2640,new_flows=36,all_flows=48,updates=108,group_checks=len(audits),future_pollution_checks=108,diagnosis_rows=len(diag),utc=datetime.now(timezone.utc).isoformat()))
    print('EXPERIMENT COMPLETE',flush=True)

if __name__=='__main__':run()

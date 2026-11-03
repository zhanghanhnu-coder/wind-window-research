"""Round48 stage B: fixed targeted marginal correction and a 2x2 ablation."""
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from seasonal_probability_core import FOLDS,score_atoms
from threshold_shift_calibration_core import fit_shift,apply_shift,BINS
from window_error_attribution_core import summarize
from research_window_error_attribution import OUT,ART,SEEDS,sources,rank_set,parents,read,sha,js

def improve():
 assert read(OUT/'diagnosis_complete.json')['status']=='COMPLETE' and (OUT/'diagnosis_decision.md').exists()
 assert not (OUT/'improvement_identity.json').exists(),'preserve previous attempt'
 for p,h in read(OUT/'diagnostic_identity.json').items():assert sha(p)==h
 files=[Path(__file__),ART.parent/'threshold_shift_calibration_core.py',ART.parent/'tests/test_threshold_shift_calibration.py',OUT/'diagnosis_complete.json',OUT/'diagnosis.csv',OUT/'diagnosis_decision.md']
 identity={str(p):sha(p) for p in files};js(OUT/'improvement_identity.json',identity)
 audits=[];marginals=[]
 for fold in FOLDS:
  folder=OUT/fold;pop=np.load(folder/'population.npz');ids=pop['ids'];blocks=pop['blocks'];scale=float(np.load(ART/'seasonal_probability_v1'/fold/'data.npz')['scale']);y=np.load(ART/'seasonal_probability_v1'/fold/'test_truth.npz')['y']
  for seed in SEEDS:
   b,j=sources(fold,seed);q=np.load(b['source'])['q'].astype(float);changed=np.empty_like(q);r0=np.load(b['ranks_source'])['ranks'];rr,which=rank_set(j)
   for block in np.unique(blocks):
    pos=np.flatnonzero(blocks==block);issue=int(ids[pos[0]]);fit=fit_shift(q,y,ids,issue,scale);changed[pos]=apply_shift(q[pos],fit['delta'])
    # Future observations/forecasts are poisoned before repeating the fit.
    future=np.broadcast_to(ids[:,None,None]+np.arange(1,73)[None,:,None]>=issue,y.shape);yp=y.copy();qp=q.copy();yp[future]=999;qp[future]=np.nan;check=fit_shift(qp,yp,ids,issue,scale)
    for key in ['delta','objectives','grid']:np.testing.assert_array_equal(fit[key],check[key])
    np.savez_compressed(folder/f'shift_{seed}_w{block}.npz',delta=fit['delta'],objectives=fit['objectives'],grid=fit['grid'])
    for i,row in enumerate(fit['audit']):audits.append(dict(fold=fold,seed=seed,block=int(block),issue=issue,applied_origins=len(pos),scale=scale,delta=float(fit['delta'][i]),**row))
   result_m=[];result_mj=[]
   for pos,qi in enumerate(changed):result_m.append(summarize(qi,r0));result_mj.append(summarize(qi,rr[which[pos]]))
   for case,pack in [('M',result_m),('MJ',result_mj)]:np.savez_compressed(folder/f'diagnostic_{case}_{seed}.npz',**{k:np.stack([r[k] for r in pack]) for k in pack[0]})
   for key in result_m[0]:
    if key not in ['single','group']:np.testing.assert_array_equal(np.stack([r[key] for r in result_m]),np.stack([r[key] for r in result_mj]))
   for case,qq in [('B',q),('M',changed)]:
    score=score_atoms(qq,y);np.savez_compressed(folder/f'hourly_scores_{case}_{seed}.npz',crps=score['crps'],brier=score['brier_hourly'])
    for start,end in [(0,72),*BINS]:
     for metric,key in [('crps','crps'),('brier','brier_hourly')]:
      z=score[key][:,start:end];marginals.append(dict(fold=fold,seed=seed,case=case,lead_start=start+1,lead_end=end,metric=metric,value=float(np.nanmean(z)),n=int(np.isfinite(z).sum())))
   print('IMPROVE',fold,seed,len(ids),flush=True)
 pd.DataFrame(audits).to_csv(OUT/'shift_audit.csv',index=False,encoding='utf-8-sig');pd.DataFrame(marginals).to_csv(OUT/'marginal_metrics.csv',index=False,encoding='utf-8-sig')
 for p,h in identity.items():assert sha(p)==h
 parents();js(OUT/'improvement_complete.json',dict(status='COMPLETE',new_flows=24,all_flows=48,unique_origins=880,seed_origins=2640,updates=108,lead_group_fits=len(audits),future_pollution_checks=108,utc=datetime.now(timezone.utc).isoformat()));print('IMPROVE COMPLETE',flush=True)

if __name__=='__main__':improve()

"""Round50: fixed raw/calibrated blend and honest past-performance adaptation."""
from pathlib import Path
from datetime import datetime,timezone
import shutil
import numpy as np
import pandas as pd
from seasonal_probability_core import FOLDS,score_atoms
from residual_shape_core import fit_residual,apply_residual,high_hour_mask,diagnose_marginals,BINS
from calibration_gate_core import choose_weight,blend,update_blocks
from warm_residual_core import join_history
from window_error_attribution_core import summarize
from research_residual_shape import ART,BASE,SEEDS,read,sha,js

OUT=ART/'calibration_gate_v1';WARM=ART/'residual_warm_v1';ARCH=ART/'seasonal_rolling_v1';SRC=ART/'seasonal_probability_v1'
CASES=['B','R','W28','M50','G7','G28'];PLAN=BASE/'研究笔记/第50轮_校准力度与历史适用性方案_2026-11-02.md'

def run():
 assert not OUT.exists(),'preserve previous attempt';OUT.mkdir();shutil.copy2(PLAN,OUT/'protocol_frozen.md')
 inventory=read(WARM/'inventory.json');files=[Path(__file__),PLAN,ART.parent/'calibration_gate_core.py',ART.parent/'tests/test_calibration_gate.py',ART.parent/'residual_shape_core.py',ART.parent/'warm_residual_core.py',WARM/'inventory.json',ART/'residual_shape_v1/final_manifest.json']
 for fold in FOLDS:
  files.extend([ARCH/fold/'archive_data.npz',SRC/fold/'data.npz',SRC/fold/'test_truth.npz'])
  files.extend(WARM.joinpath(fold).glob('fit_warm_*.npz'))
  for s in SEEDS:files.extend([ARCH/fold/f'archive_predictions_gated_graph_atoms_{s}.npz'])
 for x in inventory:
  if x['case'] in ['B','R','WT']:files.extend([Path(x['q_source']),Path(x['ranks_source'])])
 identity={str(p):sha(p) for p in set(files)};js(OUT/'source_identity.json',identity)
 # Consumed parent forecasts must still match their already-frozen content.
 expected={}
 for m in ART.glob('*/final_manifest.json'):
  expected.update({str(Path(p)):h for p,h in read(m).items() if Path(p).is_absolute()})
 for p,h in identity.items():
  if Path(p).is_relative_to(ART) and not p.endswith('final_manifest.json'):assert expected.get(p)==h,('unfrozen parent',p)
 js(OUT/'recipe.json',dict(cases=CASES,grid=[0,.25,.5,.75,1],gate_history_hours=[168,672],minimum_origins=20,minimum_labels=100,gate_update_hours=24,shadow_update_hours=168,exploratory_reused_2024=True,utc=datetime.now(timezone.utc).isoformat()))
 audits=[];shadow_audits=[];diag=[];outinventory=[];shadowchecks=0;gatechecks=0
 for fold in FOLDS:
  folder=OUT/fold;folder.mkdir()
  for name in ['population.npz','fixed_cohorts.npz','draws.npz']:shutil.copy2(WARM/fold/name,folder/name)
  pop=dict(np.load(folder/'population.npz'));ids=pop['ids'];y=np.load(SRC/fold/'test_truth.npz')['y'];scale=float(np.load(SRC/fold/'data.npz')['scale']);high=high_hour_mask(np.load(folder/'fixed_cohorts.npz')['single']);arch=np.load(ARCH/fold/'archive_data.npz');daily=update_blocks(ids,24)
  for seed in SEEDS:
   source={c:next(x for x in inventory if (x['fold'],x['seed'],x['case'])==(fold,seed,c)) for c in ['B','R','WT']}
   raw=np.load(source['R']['q_source'])['q'];wt=np.load(source['WT']['q_source'])['q'];aq=np.load(ARCH/fold/f'archive_predictions_gated_graph_atoms_{seed}.npz')['q']
   hist=join_history(aq,arch['truth'],arch['ids'],raw,y,ids);hi=hist['ids'];hy=hist['y'];hq=hist['q'];pre=hist['pretest_origins'];shadow=np.empty_like(hq);shadow[pre:]=wt
   sb=(hi[:pre]-ids[0])//168
   for block in np.unique(sb):
    pos=np.flatnonzero(sb==block);issue=int(hi[pos[0]]);fit=fit_residual(hq,hy,hi,issue,scale)
    future=np.broadcast_to(hi[:,None,None]+np.arange(1,73)[None,:,None]>=issue,hy.shape);yp=hy.copy();qp=hq.copy();yp[future]=999;qp[future]=np.nan;poison=fit_residual(qp,yp,hi,issue,scale)
    np.testing.assert_array_equal(fit['z_quantiles'],poison['z_quantiles']);assert fit['audit']==poison['audit'];shadowchecks+=1
    shadow[pos]=apply_residual(hq[pos],fit,scale,'T');np.savez_compressed(folder/f'shadow_fit_{seed}_w{block}.npz',z_quantiles=fit['z_quantiles'],fallback=fit['fallback'])
    for r in fit['audit']:shadow_audits.append(dict(fold=fold,seed=seed,block=int(block),issue=issue,applied_origins=len(pos),**r))
   np.testing.assert_array_equal(shadow[pre:],wt);np.savez_compressed(folder/f'honest_history_{seed}.npz',q=shadow,ids=hi,pretest_origins=pre)
   changed={'M50':blend(raw,wt,[.5]*4),'G7':np.empty_like(raw),'G28':np.empty_like(raw)}
   for block in np.unique(daily):
    pos=np.flatnonzero(daily==block);issue=int(ids[pos[0]])
    for case,lookback in [('G7',168),('G28',672)]:
     fit=choose_weight(hq,shadow,hy,hi,issue,lookback)
     future=np.broadcast_to(hi[:,None,None]+np.arange(1,73)[None,:,None]>=issue,hy.shape);yp=hy.copy();rp=hq.copy();wp=shadow.copy();yp[future]=999;rp[future]=np.nan;wp[future]=np.nan;poison=choose_weight(rp,wp,yp,hi,issue,lookback)
     np.testing.assert_array_equal(fit['weights'],poison['weights']);assert fit['audit']==poison['audit'];gatechecks+=1
     changed[case][pos]=blend(raw[pos],wt[pos],fit['weights']);np.savez_compressed(folder/f'gate_{case}_{seed}_d{block}.npz',weights=fit['weights'],losses=fit['losses'])
     for r in fit['audit']:audits.append(dict(fold=fold,seed=seed,case=case,block=int(block),issue=issue,lookback=lookback,applied_origins=len(pos),**r))
   ranks=np.load(source['B']['ranks_source'])['ranks']
   for case,old in [('B','B'),('R','R'),('W28','WT')]:
    for stem in ['diagnostic','hourly_scores']:shutil.copy2(WARM/fold/f'{stem}_{old}_{seed}.npz',folder/f'{stem}_{case}_{seed}.npz')
    entry=source[old].copy();entry['case']=case;outinventory.append(entry)
   for case,qq in changed.items():
    np.savez_compressed(folder/f'predictions_{case}_{seed}.npz',q=qq,ids=ids);win=[summarize(qi,ranks) for qi in qq];np.savez_compressed(folder/f'diagnostic_{case}_{seed}.npz',**{k:np.stack([v[k] for v in win]) for k in win[0]});scores=score_atoms(qq,y);np.savez_compressed(folder/f'hourly_scores_{case}_{seed}.npz',crps=scores['crps'],brier=scores['brier_hourly'],mae=scores['mae'])
    for cohort,mask in [('all',np.ones_like(y,bool)),('baseline_high_hours',high)]:
     for a,b in [(0,72),*BINS]:
      m=mask.copy();m[:,:a]=False;m[:,b:]=False;diag.append(dict(fold=fold,seed=seed,case=case,cohort=cohort,lead_start=a+1,lead_end=b,**diagnose_marginals(qq,y,m,scale)))
    outinventory.append(dict(fold=fold,seed=seed,case=case,q_source=str(folder/f'predictions_{case}_{seed}.npz'),ranks_source=source['B']['ranks_source'],origins=len(ids)))
   print('GATE COMPLETE',fold,seed,'shadow',shadowchecks,'gate',gatechecks,flush=True)
 old=pd.read_csv(WARM/'marginal_diagnosis.csv');old=old[old.case.isin(['B','R','WT'])].copy();old['case']=old['case'].replace({'WT':'W28'});pd.concat([old,pd.DataFrame(diag)],ignore_index=True).to_csv(OUT/'marginal_diagnosis.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(audits).to_csv(OUT/'gate_audit.csv',index=False,encoding='utf-8-sig');pd.DataFrame(shadow_audits).to_csv(OUT/'shadow_audit.csv',index=False,encoding='utf-8-sig');js(OUT/'inventory.json',outinventory)
 for p,h in identity.items():assert sha(p)==h,p
 js(OUT/'experiment_complete.json',dict(status='COMPLETE',unique_origins=880,new_flows=36,all_flows=72,shadow_updates=shadowchecks,gate_updates=gatechecks,gate_groups=len(audits),shadow_groups=len(shadow_audits),future_poison_checks=shadowchecks+gatechecks,utc=datetime.now(timezone.utc).isoformat()))
 print('GATE EXPERIMENT COMPLETE',flush=True)

if __name__=='__main__':run()

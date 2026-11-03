"""Round51 conditional calibration, frozen candidates and causal features."""
from pathlib import Path
from datetime import datetime,timezone
import shutil
import numpy as np
import pandas as pd
from research_calibration_gate import BASE,ART,OUT as PARENT,SEEDS,FOLDS,read,sha,js
from diagnose_state_gate import OUT,PLAN
from state_gate_core import state_features,select_state_weight
from calibration_gate_core import blend,update_blocks
from warm_residual_core import join_history
from seasonal_probability_core import score_atoms
from residual_shape_core import BINS
from window_error_attribution_core import summarize
CASES=['B','R','W28','M50','G28','K4','K6']
def run():
 assert read(OUT/'diagnosis_complete.json')['status']=='COMPLETE';assert not (OUT/'source_identity.json').exists()
 inv=read(PARENT/'inventory.json');old=read(PARENT/'final_manifest.json');files=[Path(__file__),PLAN,ART.parent/'state_gate_core.py',ART.parent/'diagnose_state_gate.py',ART.parent/'tests/test_state_gate.py',ART/'data.npz',PARENT/'inventory.json',PARENT/'final_manifest.json']
 for fold in FOLDS:
  files.extend([ART/'seasonal_rolling_v1'/fold/'archive_data.npz',ART/'seasonal_probability_v1'/fold/'data.npz',ART/'seasonal_probability_v1'/fold/'test_truth.npz'])
  for seed in SEEDS:files.extend([PARENT/fold/f'honest_history_{seed}.npz',ART/'seasonal_rolling_v1'/fold/f'archive_predictions_gated_graph_atoms_{seed}.npz'])
 for x in inv:
  if x['case'] in CASES:files.extend([Path(x['q_source']),Path(x['ranks_source'])])
 files+=list(OUT.glob('*.csv'))+[OUT/'diagnosis_complete.json']
 identity={str(p):sha(p) for p in set(files)}
 for p,h in identity.items():
  if p in old:assert old[p]==h,p
 js(OUT/'source_identity.json',identity);shutil.copy2(PLAN,OUT/'protocol_frozen.md')
 rawobs=np.load(ART/'data.npz')['raw'];inventory=[];audit=[];poison_checks=0;det=[]
 for fold in FOLDS:
  folder=OUT/fold;folder.mkdir()
  for name in ['population.npz','fixed_cohorts.npz','draws.npz']:shutil.copy2(PARENT/fold/name,folder/name)
  ids=np.load(folder/'population.npz')['ids'];y=np.load(ART/'seasonal_probability_v1'/fold/'test_truth.npz')['y'];d=np.load(ART/'seasonal_probability_v1'/fold/'data.npz');center=float(d['center']);scale=float(d['scale']);ar=np.load(ART/'seasonal_rolling_v1'/fold/'archive_data.npz');daily=update_blocks(ids,24)
  for seed in SEEDS:
   source={c:next(x for x in inv if (x['fold'],x['seed'],x['case'])==(fold,seed,c)) for c in CASES if c not in ['K4','K6']}
   r=np.load(source['R']['q_source'])['q'];w=np.load(source['W28']['q_source'])['q'];aq=np.load(ART/'seasonal_rolling_v1'/fold/f'archive_predictions_gated_graph_atoms_{seed}.npz')['q'];h=join_history(aq,ar['truth'],ar['ids'],r,y,ids);hq=h['q'];hy=h['y'];hi=h['ids'];pre=h['pretest_origins'];hw=np.load(PARENT/fold/f'honest_history_{seed}.npz')['q'];np.testing.assert_array_equal(hw[pre:],w)
   features=state_features(rawobs,hi,hq,hw,center,scale);np.savez_compressed(folder/f'features_{seed}.npz',features=features,ids=hi,pretest_origins=pre)
   new={c:np.empty_like(r) for c in ['K4','K6']}
   for block in np.unique(daily):
    pos=np.flatnonzero(daily==block);issue=int(ids[pos[0]]);current=features[pre+pos[0]]
    obs_poison=rawobs.copy();obs_poison[issue+1:]=999
    np.testing.assert_array_equal(current,state_features(obs_poison,np.array([issue]),r[pos[:1]],w[pos[:1]],center,scale)[0])
    for case,dim in [('K4',4),('K6',6)]:
     fit=select_state_weight(hq,hw,hy,hi,features,current,issue,dim);future=np.broadcast_to(hi[:,None,None]+np.arange(1,73)[None,:,None]>=issue,hy.shape);yp=hy.copy();rp=hq.copy();wp=hw.copy();fp=features.copy();yp[future]=999;rp[future]=np.nan;wp[future]=np.nan;fp[hi>=issue]=np.nan
     check=select_state_weight(rp,wp,yp,hi,fp,current,issue,dim);np.testing.assert_array_equal(fit['weights'],check['weights']);np.testing.assert_array_equal(fit['losses'],check['losses']);poison_checks+=1
     new[case][pos]=blend(r[pos],w[pos],fit['weights']);sel=np.full((4,40),-1,int)
     for j,chosen in enumerate(fit['selected']):sel[j,:len(chosen)]=chosen
     np.savez_compressed(folder/f'state_{case}_{seed}_d{block}.npz',weights=fit['weights'],losses=fit['losses'],selected_positions=sel,issue=issue)
     sr=score_atoms(w[pos[:1]],y[pos[:1]])['crps'];sm=score_atoms((r[pos[:1]]+w[pos[:1]])*.5,y[pos[:1]])['crps']
     for j,row in enumerate(fit['audit']):
      audit.append(dict(fold=fold,seed=seed,case=case,block=int(block),issue=issue,applied_origins=len(pos),**row));a,b=BINS[j];valid=np.isfinite(sr[:,a:b]);observed=int(valid.sum());actual=float((sr[:,a:b]-sm[:,a:b])[valid].mean()) if observed else np.nan
      det.append(dict(fold=fold,seed=seed,case=case,issue=issue,lead_start=a+1,lead_end=b,observed=observed,estimated_harm=row['estimated_harm'],actual_harm=actual,weight=row['weight']))
   ranks=np.load(source['B']['ranks_source'])['ranks']
   for case,entry in source.items():
    for stem in ['diagnostic','hourly_scores']:shutil.copy2(PARENT/fold/f'{stem}_{case}_{seed}.npz',folder/f'{stem}_{case}_{seed}.npz')
    inventory.append(entry)
   for case,q in new.items():
    path=folder/f'predictions_{case}_{seed}.npz';np.savez_compressed(path,q=q,ids=ids);win=[summarize(qi,ranks) for qi in q];np.savez_compressed(folder/f'diagnostic_{case}_{seed}.npz',**{k:np.stack([v[k] for v in win]) for k in win[0]});s=score_atoms(q,y);np.savez_compressed(folder/f'hourly_scores_{case}_{seed}.npz',crps=s['crps'],brier=s['brier_hourly'],mae=s['mae']);inventory.append(dict(fold=fold,seed=seed,case=case,q_source=str(path),ranks_source=source['B']['ranks_source'],origins=len(ids)))
   print('STATE COMPLETE',fold,seed,poison_checks,flush=True)
 pd.DataFrame(audit).to_csv(OUT/'state_audit.csv',index=False,encoding='utf-8-sig');pd.DataFrame(det).to_csv(OUT/'advance_detection.csv',index=False,encoding='utf-8-sig');js(OUT/'inventory.json',inventory)
 for p,h in identity.items():assert sha(p)==h,p
 js(OUT/'experiment_complete.json',dict(status='COMPLETE',unique_origins=880,new_flows=24,all_flows=84,gate_groups=len(audit),future_poison_checks=poison_checks,utc=datetime.now(timezone.utc).isoformat()))
if __name__=='__main__':run()

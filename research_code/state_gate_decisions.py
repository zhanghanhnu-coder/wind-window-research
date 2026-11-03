"""Paired false acceptance and missed usable windows on the same population."""
from pathlib import Path
import numpy as np
import pandas as pd
from research_state_gate import OUT,PARENT,CASES,FOLDS,read,sha,js
from state_gate_tables import population,PAIRS
from state_gate_core import decision_counts
from window_error_attribution_core import selected_bounds,selected_bootstrap

def decision_coefficients(pa,pb,metric):
 a=(pa>=.8-1e-12).astype(float);b=(pb>=.8-1e-12).astype(float)
 return (a-b,b-a) if metric=='FP' else (np.zeros_like(a),b-a)

def run():
 assert read(OUT/'experiment_complete.json')['status']=='COMPLETE';assert not (OUT/'decisions_complete.json').exists()
 files=[Path(__file__),OUT.parent.parent/'state_gate_core.py']+list(OUT.glob('*/population.npz'))+list(OUT.glob('*/diagnostic_*.npz'))+list(OUT.glob('*/draws.npz'))
 identity={str(p):sha(p) for p in files};js(OUT/'decision_identity.json',identity)
 counts=[];rows=[]
 for scope in ['ALL',*FOLDS]:
  for kind in ['single','group']:
   p=population(scope,kind);y=p['labels'];mean={c:v.mean(0) for c,v in p['probabilities'].items()};mask=np.ones_like(y,bool)
   for threshold in [.6,.7,.8,.9]:
    for case in CASES:counts.append(dict(scope=scope,kind=kind,case=case,threshold=threshold,**decision_counts(mean[case],y,threshold)))
   for candidate,baseline in PAIRS:
    for metric in ['FP','FN']:
     c,d=decision_coefficients(mean[candidate],mean[baseline],metric);bound=selected_bounds(c,d,y,p['inverse'],mask);boot=selected_bootstrap(c,d,y,p['inverse'],mask,p['blocks'],p['draws']);valid=np.isfinite(boot['lower'])&np.isfinite(boot['upper'])
     support=valid.sum()>=1900
     rows.append(dict(scope=scope,kind=kind,candidate=candidate,baseline=baseline,metric=metric,threshold=.8,primary=scope=='ALL' and (candidate,baseline) in PAIRS[:2],n=int(mask.sum()),lower=bound['lower'],upper=bound['upper'],outer_low=float(np.quantile(boot['lower'][valid],.025)) if support else np.nan,outer_high=float(np.quantile(boot['upper'][valid],.975)) if support else np.nan,valid_draws=int(valid.sum()),supported=bool(support)))
     np.savez_compressed(OUT/f'decision_{scope}_{kind}_{candidate}_{baseline}_{metric}.npz',lower=boot['lower'],upper=boot['upper'],n=boot['n'],witness_lower=bound['witness_lower'],witness_upper=bound['witness_upper'])
   print('DECISIONS',scope,kind,flush=True)
 pd.DataFrame(counts).to_csv(OUT/'decision_counts.csv',index=False,encoding='utf-8-sig');pd.DataFrame(rows).to_csv(OUT/'decision_paired.csv',index=False,encoding='utf-8-sig')
 det=pd.read_csv(OUT/'advance_detection.csv');old=pd.read_csv(PARENT/'gate_audit.csv');old=old[old['case']=='G28'].copy();extra=det[det['case']=='K4'].drop(columns=['estimated_harm','weight']).merge(old[['fold','seed','issue','lead_start','weight','loss_2','loss_4']],on=['fold','seed','issue','lead_start'],validate='one_to_one');extra['case']='G28';extra['estimated_harm']=extra.loss_4-extra.loss_2
 det=pd.concat([det,extra[det.columns]],ignore_index=True);det.to_csv(OUT/'detection_with_control.csv',index=False,encoding='utf-8-sig')
 avg=det.groupby(['fold','case','issue','lead_start','lead_end'],as_index=False).mean(numeric_only=True);summary=[]
 for scope in ['ALL',*FOLDS]:
  for case in ['G28','K4','K6']:
   part=avg[(avg['case']==case)&(True if scope=='ALL' else avg.fold==scope)];v=part[np.isfinite(part.estimated_harm)&np.isfinite(part.actual_harm)];a=v.estimated_harm>0;b=v.actual_harm>0
   tp=int((a&b).sum());fp=int((a&~b).sum());fn=int((~a&b).sum());tn=int((~a&~b).sum())
   summary.append(dict(scope=scope,case=case,total=len(part),evaluable=len(v),tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else np.nan,recall=tp/(tp+fn) if tp+fn else np.nan,observed_labels_only=True))
 pd.DataFrame(summary).to_csv(OUT/'advance_detection_summary.csv',index=False,encoding='utf-8-sig')
 for p,h in identity.items():assert sha(p)==h,p
 assert len(rows)==100 and sum(r['primary'] for r in rows)==8 and len(counts)==280
 js(OUT/'decisions_complete.json',dict(status='COMPLETE',paired_rows=100,primary_rows=8,threshold_count_rows=280,decision='mean seed probability >= threshold; paired FP/FN per all records'))
if __name__=='__main__':run()

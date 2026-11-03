"""Describe the already-selected demonstration case; never selects model settings."""
import numpy as np
import pandas as pd
from research_state_gate import OUT,ART,SEEDS,CASES,read,js

def run():
 assert read(OUT/'experiment_complete.json')['status']=='COMPLETE'
 fold='M09';issue=5861;node=10;offset=66;pop=np.load(OUT/fold/'population.npz');j=int(np.flatnonzero(pop['ids']==issue)[0]);truth=np.load(ART/'seasonal_probability_v1'/fold/'test_truth.npz')['y'][j];inv=read(OUT/'inventory.json');rows=[];summary=[]
 for case in CASES:
  qq=np.stack([np.load(next(x['q_source'] for x in inv if (x['fold'],x['seed'],x['case'])==(fold,s,case)))['q'][j] for s in SEEDS]);diag=[np.load(OUT/fold/f'diagnostic_{case}_{s}.npz') for s in SEEDS]
  for h in range(66,72):
   rows.append(dict(case=case,lead=h+1,actual=float(truth[h,node]),mean_median=float(qq[:,h,node,9].mean()),mean_largest=float(qq[:,h,node,-1].mean()),largest_across_seeds=float(qq[:,h,node,-1].max()),hourly_safe=float((qq[:,h,node]<=10).mean()),observed_exceeds=bool(truth[h,node]>10)))
  p=float(np.mean([d['single'][j,3*17+node] for d in diag]));summary.append(dict(case=case,window_probability=p,actual_success=0,known_exceedances=6,mean_median=float(qq[:,66:,node,9].mean()),hourly_safe_mean=float((qq[:,66:,node]<=10).mean()),high_probability_error=bool(p>=.8-1e-12)))
 pd.DataFrame(rows).to_csv(OUT/'demonstration_hourly.csv',index=False,encoding='utf-8-sig');pd.DataFrame(summary).to_csv(OUT/'demonstration_window.csv',index=False,encoding='utf-8-sig')
 audit=pd.read_csv(OUT/'state_audit.csv');selected=audit[(audit.fold==fold)&(audit.issue==issue)&(audit.lead_start==49)];selected.to_csv(OUT/'demonstration_gate_history.csv',index=False,encoding='utf-8-sig')
 js(OUT/'demonstration_scope.json',dict(fold=fold,issue_index=issue,issue='2024-09-01T05:00:00+08:00',turbine_number=12,node_index=node,offset=offset,duration=6,selection='case already chosen before this round; diagnostic only, not independent success evidence',unitsConfirmed=False))
 print(pd.DataFrame(summary).to_string(index=False),flush=True)
if __name__=='__main__':run()

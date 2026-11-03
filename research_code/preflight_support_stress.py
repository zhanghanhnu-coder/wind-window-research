"""Synthetic table/seed alignment preflight; never use actual outcome data."""
from pathlib import Path
import json,tempfile
import numpy as np
import pandas as pd
from model import ART
from seasonal_probability_core import FOLDS
from support_stress_tables import make_tables
import verify_support_stress as audit

def run():
 times=pd.date_range('2024-01-01',periods=8785,freq='h',tz='Asia/Shanghai');rows=[]
 for fi,(fold,cfg) in enumerate(FOLDS.items()):
  first=times.get_loc(pd.Timestamp(cfg['test_start'],tz=times.tz))
  for j in range(12):
   issue=int(first+j*120)
   for si,seed in enumerate([20241101,20241102,20241103]):
    for case,delta in [('base_tv025',0),('base_tv000',-5),('stress_e010',2)]+([('stress_e001',1),('stress_e050',4)] if si==0 else []):
     lo=100*si+10*fi+j+10+delta;complete=j%2==0;hi=lo if complete else lo+2
     rows.append(dict(case=case,fold=fold,seed=seed,issue_id=issue,complete=complete,cost_lower=lo,cost_upper=hi,exact_cost=lo if complete else np.nan,exact_completed=2 if complete else np.nan,booking_cost=3.,objective=lo+1,nominal_total=lo+.5,no_booking=0))
 df=pd.DataFrame(rows).sample(frac=1,random_state=123);assert len(df)==528
 tables,draws=make_tables(df,times)
 with tempfile.TemporaryDirectory(prefix='stress_preflight_',dir=ART) as path:
  out=Path(path)
  for name,values in tables.items():pd.DataFrame(values).to_csv(out/(name+'.csv'),index=False)
  np.savez_compressed(out/'bootstrap_draws.npz',**draws);old=audit.OUT
  try:audit.OUT=out;counts=audit.check_tables(df,times)
  finally:audit.OUT=old
 main=[r for r in tables['paired_comparisons'] if r['primary']]
 for r in main:
  expected=(1.,3.) if r['baseline']=='base_tv025' else (6.,8.)
  np.testing.assert_allclose([r['difference_lower'],r['difference_upper']],expected,rtol=0,atol=1e-12)
 for case,delta in [('stress_e001',1),('stress_e050',4)]:
  a=next(r for r in tables['tail_risk'] if r['case']==case);b=next(r for r in tables['tail_risk'] if r['case']=='base_tv025' and r['seed_scope']=='first_seed')
  np.testing.assert_allclose([a['cvar_lower']-b['cvar_lower'],a['cvar_upper']-b['cvar_upper']],[delta,delta],rtol=0,atol=1e-12)
 proof=dict(status='PASS',synthetic_rows=528,shuffled_seed_origin_alignment=True,known_main_bounds_checked=2,matched_first_seed_tail_shifts_checked=2,actual_data_read=False,**counts)
 (ART/'support_stress_preflight.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof,indent=2))

if __name__=='__main__':run()

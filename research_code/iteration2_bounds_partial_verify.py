"""Independent partial-evidence witnesses; no optimization/core code imports."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from precision_truth_core import load_truth
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/review_iteration2_v1/shared_bounds'
OUT=BASE/'partial_evidence'
OLD=ROOT/'artifacts/review_followup_v1/cross_execution'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
 _,_,raw=load_truth();bad=np.load(ROOT/'artifacts/review_iteration2_v1/data_gaps/partial_hour_evidence.npz')['known_bad']
 cases=read(OLD/'protocol.json')['cases'];sources=read(OUT/'sources.json')
 for path,digest in sources.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
 arms={
 'prediction_myopic':('fixed_B__W28__myopic','fixed_B__B__myopic'),
 'prediction_lookahead':('fixed_B__W28__lookahead72','fixed_B__B__lookahead72'),
 'executor_B':('fixed_B__B__lookahead72','fixed_B__B__myopic'),
 'executor_W28':('fixed_B__W28__lookahead72','fixed_B__W28__myopic'),
 'combined':('own__W28__lookahead72','own__B__myopic'),
 'analog_graph_lookahead72':('fixed_B__AnalogFixed__lookahead72','fixed_B__B__lookahead72'),
 'analog_graph_myopic':('fixed_B__AnalogFixed__myopic','fixed_B__B__myopic')}
 cache={};frame=pd.read_csv(OUT/'partial_evidence_bounds.csv');maxerr=0.;maxchange=0.;maxgap=0.
 for _,row in frame.iterrows():
  saved=read(OUT/'certificates'/f'{row.comparison}__{row.scope}.json')['optimization']
  sub=cases if row.scope=='ALL' else [c for c in cases if c['fold']==row.scope or f"{c['fold']}_issue{c['issue']}"==row.scope]
  for side in ['lower','upper']:
   witness={(h,n):g for h,n,g in saved[side+'_witness']};total=0.
   assert all(g in [0,1] and not bad[h,n] for (h,n),g in witness.items())
   for c in sub:
    for sign,arm in zip([1,-1],arms[row.comparison]):
     folder=ROOT/'artifacts/review_iteration2_v1/analog_execution/evaluations' if 'AnalogFixed' in arm else OLD/'evaluations'
     path=folder/f"{c['key']}__{arm}.json"
     if path not in cache:cache[path]=read(path)
     e=cache[path];value=e['metrics']['booking_cost']
     for t in e['tasks']:
      value+=t['operation_cost'];success=False
      if t['choice'] is not None:
       hours=np.arange(c['issue']+t['start']+1,c['issue']+t['end']+1);values=raw[hours,t['node']].copy()
       for j,h in enumerate(hours):
        if not np.isfinite(values[j]):values[j]=11. if bad[h,t['node']] else (10. if witness.get((h,t['node']),0) else 11.)
       success=bool(np.all(values<=10.))
      if not success:value+=t['penalty']
     total+=sign*value/len(sub)
   err=abs(total-saved[side]);assert err<1e-7;maxerr=max(maxerr,err)
   cert=saved[side+'_certificate'];gap=abs(cert['primal']-cert['dual_bound']);assert cert['status']==0 and gap<1e-7 and cert['mip_gap']<=1e-9;maxgap=max(maxgap,gap)
  maxchange=max(maxchange,abs(row.partial_shared_lower-row.original_shared_lower),abs(row.partial_shared_upper-row.original_shared_upper))
  assert row.partial_shared_lower>=row.original_shared_lower-1e-7 and row.partial_shared_upper<=row.original_shared_upper+1e-7
 result=dict(status='PASS',scope_bounds=len(frame),max_witness_error=maxerr,max_primal_dual_gap=maxgap,max_absolute_bound_change=maxchange,unknown_to_failed_rows=int(frame.candidate_attempt_rows_unknown_to_failed.sum()+frame.baseline_attempt_rows_unknown_to_failed.sum()),method='No bound-core or partial-expression imports. Hourly 10/11 completions honor observed partial known_bad; task ledger costs independently recomputed.')
 (OUT/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()

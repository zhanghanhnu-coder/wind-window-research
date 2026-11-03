"""Independent saved-witness verification without importing bound construction."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
import pandas as pd
from precision_truth_core import load_truth, SOURCE

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/review_iteration2_v1/shared_bounds'
OLD=ROOT/'artifacts/review_followup_v1/cross_execution'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 _,_,raw=load_truth();protocol=read(OLD/'protocol.json');cases=protocol['cases']
 assert sha(SOURCE)==read(OLD/'evaluation_complete.json')['truth_sha256']
 comparisons={
  'prediction_myopic':('fixed_B__W28__myopic','fixed_B__B__myopic'),
  'prediction_lookahead':('fixed_B__W28__lookahead72','fixed_B__B__lookahead72'),
  'executor_B':('fixed_B__B__lookahead72','fixed_B__B__myopic'),
  'executor_W28':('fixed_B__W28__lookahead72','fixed_B__W28__myopic'),
  'combined':('own__W28__lookahead72','own__B__myopic')}
 origins=sorted({(c['fold'],c['issue']) for c in cases});intervals=[(f,i+1,i+72) for f,i in origins]
 overlaps=[(a,b) for a,b in itertools.combinations(intervals,2) if max(a[1],b[1])<=min(a[2],b[2])]
 cache={};count=0;maxgap=0.;maxerror=0.
 for prefix in ['cross','analog']:
  source=OUT/f'{prefix}_sources.json'
  if not source.exists():continue
  for path,digest in read(source).items():assert sha(path)==digest,path
  frame=pd.read_csv(OUT/f'{prefix}_bounds.csv')
  for _,row in frame.iterrows():
   saved=read(OUT/'certificates'/f'{row.comparison}__{row.scope}.json');opt=saved['optimization']
   sub=cases if row.scope=='ALL' else [c for c in cases if (f"{c['fold']}_issue{c['issue']}"==row.scope or c['fold']==row.scope)]
   n_origins=len({(c['fold'],c['issue']) for c in sub});assert n_origins==row.origins
   if prefix=='cross':aa,bb=comparisons[row.comparison]
   else:
    executor=row.comparison.replace('analog_graph_','');aa=f'fixed_B__AnalogFixed__{executor}';bb=f'fixed_B__B__{executor}'
   for side in ['lower','upper']:
    witness={(h,n):int(g) for h,n,g in opt[side+'_witness']};total=0.
    assert all(g in [0,1] for g in witness.values())
    for c in sub:
     for sign,arm in [(1,aa),(-1,bb)]:
      path=(ROOT/'artifacts/review_iteration2_v1/analog_execution/evaluations' if 'AnalogFixed' in arm else OLD/'evaluations')/f"{c['key']}__{arm}.json"
      if path not in cache:cache[path]=read(path)
      e=cache[path];cost=e['metrics']['booking_cost']
      for task in e['tasks']:
       cost+=task['operation_cost'];success=False
       if task['choice'] is not None:
        hours=list(range(c['issue']+task['start']+1,c['issue']+task['end']+1));z=raw[hours,task['node']].copy()
        for j,h in enumerate(hours):
         if not np.isfinite(z[j]):z[j]=10. if witness.get((h,task['node']),0) else 11.
        success=bool(np.all(z<=10.))
       cost+=(not success)*task['penalty']
      total+=sign*cost/(n_origins*3)
    error=abs(total-opt[side]);maxerror=max(maxerror,error);assert error<1e-7
    cert=opt[side+'_certificate'];assert cert['status']==0 and cert['mip_gap']<=1e-9
    gap=abs(cert['primal']-cert['dual_bound']);maxgap=max(maxgap,gap);assert gap<1e-7
   assert row.coarse_lower-1e-7<=opt['lower']<=opt['upper']+1e-7
   assert opt['upper']<=row.coarse_upper+1e-7
   count+=1
 result=dict(status='PASS',verified_scope_bounds=count,max_witness_recomputation_error=maxerror,max_primal_dual_gap=maxgap,origins=44,seeds_per_origin=3,overlapping_origin_horizons=len(overlaps),origin_horizon_intervals=intervals,truth_matches_previous_cross_sha256=True,method='No bound-core imports; fill missing raw64 hourly cells with 10/11 from saved common witnesses, recompute each task cost and aggregate all seeds/origins.')
 (OUT/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()

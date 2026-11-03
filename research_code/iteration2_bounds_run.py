"""Predeclared shared-weather re-evaluation; immutable historical input only."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse, hashlib, json
import numpy as np
import pandas as pd
from precision_truth_core import load_truth, SOURCE
from iteration2_bounds_core import window_term, solve_bounds

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'artifacts/review_followup_v1/cross_execution'
OUT=ROOT/'artifacts/review_iteration2_v1/shared_bounds'
COMPARISONS={
 'prediction_myopic':(('fixed_B','W28','myopic'),('fixed_B','B','myopic')),
 'prediction_lookahead':(('fixed_B','W28','lookahead72'),('fixed_B','B','lookahead72')),
 'executor_B':(('fixed_B','B','lookahead72'),('fixed_B','B','myopic')),
 'executor_W28':(('fixed_B','W28','lookahead72'),('fixed_B','W28','myopic')),
 'combined':(('own','W28','lookahead72'),('own','B','myopic')),
}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def freeze_protocol():
 protocol=dict(status='FROZEN_BEFORE_NEW_REAL_BOUND_SOLVES',created_utc=datetime.now(timezone.utc).isoformat(),
  comparisons=COMPARISONS,extension=['AnalogFixed minus GraphFixed with same B resources, lookahead72 primary, myopic secondary'],
  population='all 44 prespecified origins x all 3 frozen seeds; no outcome filtering',
  estimand='Mean candidate cost minus baseline cost: seed average within each origin, then equal origin average',
  unknown_model='For each missing (absolute_hour,node), one unrestricted binary threshold state shared across seeds, arms and origins. Known raw64 >10 fixes window failure before missingness; all known <=10 fixes success.',
  tight_scope='Conditional on saved forecasts, quality gates and saved single-attempt actions. Not a reconstructed-history forecasting/policy counterfactual, physical weather model, confidence interval or online decision rule.',
  optimization='Global scope MILP: w<=g_h for every unknown hour; w>=sum(g_h)-(k-1); duplicate unknown sets combined; coefficients signed penalties. Origins/folds/ALL solved with all shared variables simultaneously.',
  certificate='Require optimal status 0, integer feasible witness, mip_gap<=1e-9, absolute primal-dual gap<=1e-7; direct completed-weather ledger cost recomputation <=1e-7; original coarse containment.',
  tests='RED missing module observed, then 7 tests passed: overlap, cancellation, cross seed, known failure, all known, absolute identity and 30 randomized exhaustive problems.',
  outcome_dependence='execute receives forecasts/quality/prior choice IDs only; dispatch_now and plan_future never receive success outcomes; prior attempted tasks stay consumed; failure occupies full duration and spares. Forecast and quality inputs remain frozen.',
  truth_interface='precision_truth_core.load_truth original float64, hour issue+start+1 through issue+end inclusive')
 dest=OUT/'protocol.json'
 if dest.exists():
  previous=read(dest);protocol['created_utc']=previous['created_utc'];assert previous==json.loads(json.dumps(protocol))
 else:write(dest,protocol)

def arm_expression(evaluation,case,raw):
 m=evaluation['metrics'];tasks=evaluation['tasks']
 constant=m['booking_cost']+m['operating_cost']+sum(t['penalty'] for t in tasks)
 terms=defaultdict(float)
 for row in tasks:
  if row['choice'] is None:continue
  key=window_term(raw,case['issue'],row['start'],row['end'],row['node'])
  expected='failed' if key is False else 'successful' if key is True else 'unknown'
  assert row['status']==expected
  if key is True:constant-=row['penalty']
  elif key is not False:terms[key]-=row['penalty']
 assert abs(constant+sum(terms.values())-m['cost_lower'])<1e-7
 assert abs(constant-m['cost_upper'])<1e-7
 return constant,terms

def direct_cost(evaluation,case,raw,witness):
 """Independent of symbolic classification and MILP AND variables."""
 value=evaluation['metrics']['booking_cost']+sum(t['operation_cost'] for t in evaluation['tasks'])
 for t in evaluation['tasks']:
  success=False
  if t['choice'] is not None:
   success=True
   for h in range(case['issue']+t['start']+1,case['issue']+t['end']+1):
    z=raw[h,t['node']]
    good=bool(z<=10.) if np.isfinite(z) else bool(witness.get((h,t['node']),0))
    if not good:success=False;break
  if not success:value+=t['penalty']
 return value

def analyze(name,cases,pairs,raw):
 rows=[];scopes=[('ALL',cases)]+[(f,[c for c in cases if c['fold']==f]) for f in sorted({c['fold'] for c in cases})]
 scopes += [(f"{c['fold']}_issue{c['issue']}",[d for d in cases if d['fold']==c['fold'] and d['issue']==c['issue']]) for c in cases if c['seed']==min(d['seed'] for d in cases)]
 for scope,sub in scopes:
  origins=sorted({(c['fold'],c['issue']) for c in sub});constant=0.;terms=defaultdict(float);coarse_lo=0.;coarse_hi=0.;weighted=[]
  for case in sub:
   seeds=[d['seed'] for d in sub if d['fold']==case['fold'] and d['issue']==case['issue']]
   assert len(seeds)==3 and len(set(seeds))==3
   weight=1/(len(origins)*len(seeds));a,b=pairs[case['key']];weighted.append((case,a,b,weight))
   coarse_lo+=weight*(a['metrics']['cost_lower']-b['metrics']['cost_upper'])
   coarse_hi+=weight*(a['metrics']['cost_upper']-b['metrics']['cost_lower'])
   for sign,evaluation in [(1,a),(-1,b)]:
    const,armterms=arm_expression(evaluation,case,raw);constant+=sign*weight*const
    for ks,coef in armterms.items():terms[ks]+=sign*weight*coef
  result=solve_bounds(constant,terms)
  for side in ['lower','upper']:
   assignment={(h,node):g for h,node,g in result[side+'_witness']}
   value=sum(weight*(direct_cost(a,c,raw,assignment)-direct_cost(b,c,raw,assignment)) for c,a,b,weight in weighted)
   if abs(value-result[side])>1e-7:raise AssertionError('independent completed-weather recomputation failed')
   result[side+'_certificate']['direct_ledger_recomputed']=value
  assert coarse_lo-1e-7<=result['lower'] and result['lower']<=result['upper']+1e-7 and result['upper']<=coarse_hi+1e-7, (name,scope,coarse_lo,result['lower'],result['upper'],coarse_hi)
  width=coarse_hi-coarse_lo;newwidth=result['upper']-result['lower']
  record=dict(comparison=name,scope=scope,origins=len(origins),seed_cases=len(sub),seeds_per_origin=3,
   coarse_lower=coarse_lo,coarse_upper=coarse_hi,shared_lower=result['lower'],shared_upper=result['upper'],coarse_width=width,shared_width=newwidth,width_reduction=width-newwidth,width_reduction_fraction=(width-newwidth)/width if width>1e-9 else 0.,
   unknown_hours=result['unknown_hours'],window_terms=result['window_terms'],certificate='PASS',max_absolute_gap=max(result[s+'_certificate']['absolute_gap'] for s in ['lower','upper']))
  write(OUT/'certificates'/f'{name}__{scope}.json',dict(**record,optimization=result));rows.append(record)
 return rows

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--analog-evaluations',type=Path);args=parser.parse_args()
 freeze_protocol()
 protocol=read(OLD/'protocol.json');cases=protocol['cases'];ident={str(OLD/'protocol.json'):sha(OLD/'protocol.json'),str(SOURCE):sha(SOURCE)}
 for name in ['precision_truth_core.py','cross_execution_review_core.py','causal_execution_core.py','lookahead_execution_core.py','iteration2_bounds_core.py','iteration2_bounds_run.py','iteration2_bounds_test.py']:
  ident[str(ROOT/name)]=sha(ROOT/name)
 times,nodes,raw=load_truth();assert raw.dtype==np.float64
 inventory=read(OLD/'trajectory_inventory.json')
 for item in inventory:
  assert sha(item['path'])==item['sha256'];ident[item['path']]=item['sha256']
 cache={}
 def old(case,arm):
  path=OLD/'evaluations'/f"{case['key']}__{'__'.join(arm)}.json"
  if path not in cache:cache[path]=read(path);ident[str(path)]=sha(path)
  return cache[path]
 rows=[]
 if not args.analog_evaluations:
  for name,(aa,bb) in COMPARISONS.items():
   pairs={c['key']:(old(c,aa),old(c,bb)) for c in cases}
   rows+=analyze(name,cases,pairs,raw);print('DONE',name,flush=True)
  suffix='cross'
 else:
  for executor in ['lookahead72','myopic']:
   pairs={}
   for c in cases:
    path=args.analog_evaluations/f"{c['key']}__fixed_B__AnalogFixed__{executor}.json"
    a=read(path);ident[str(path)]=sha(path);pairs[c['key']]=(a,old(c,('fixed_B','B',executor)))
   name='analog_graph_'+executor;rows+=analyze(name,cases,pairs,raw);print('DONE',name,flush=True)
  suffix='analog'
 frame=pd.DataFrame(rows);frame.to_csv(OUT/f'{suffix}_bounds.csv',index=False,encoding='utf-8-sig')
 frame[~frame.scope.str.contains('_issue')].to_csv(OUT/f'{suffix}_summary.csv',index=False,encoding='utf-8-sig')
 write(OUT/f'{suffix}_sources.json',ident)
 for path,digest in ident.items():assert sha(path)==digest
 write(OUT/f'{suffix}_complete.json',dict(status='PASS',bounds=len(rows),origins=44,seed_cases=132,raw_dtype=str(raw.dtype),truth_sha256=sha(SOURCE),old_trajectories_unchanged=len(inventory),protocol_sha256=sha(OUT/'protocol.json'),scope='Tight only conditional on saved predictions, quality gates, actions and unrestricted shared missing-hour threshold states.'))
 print(frame[frame.scope=='ALL'].to_string(index=False),flush=True)

if __name__=='__main__':main()

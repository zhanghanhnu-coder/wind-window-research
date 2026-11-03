"""Separate sensitivity: observed 5-minute exceedance certifies hour failure.

Never fabricates a complete hourly maximum, never treats partial-safe as safe.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import pandas as pd
from precision_truth_core import load_truth,SOURCE
from iteration2_bounds_core import window_term,solve_bounds
from iteration2_bounds_run import COMPARISONS

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/review_iteration2_v1/shared_bounds'
OUT=BASE/'partial_evidence'
OLD=ROOT/'artifacts/review_followup_v1/cross_execution'
ANALOG=ROOT/'artifacts/review_iteration2_v1/analog_execution/evaluations'
EVIDENCE=ROOT/'artifacts/review_iteration2_v1/data_gaps/partial_hour_evidence.npz'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def classify_window(raw,known_bad,issue,start,end,node):
 if known_bad[issue+start+1:issue+end+1,node].any():return False
 return window_term(raw,issue,start,end,node)

def expression(e,c,raw,bad):
 const=e['metrics']['booking_cost']+sum(t['operation_cost']+t['penalty'] for t in e['tasks']);terms=defaultdict(float);resolved=0
 for t in e['tasks']:
  if t['choice'] is None:continue
  ks=classify_window(raw,bad,c['issue'],t['start'],t['end'],t['node'])
  if t['status']=='unknown' and ks is False:resolved+=1
  if ks is True:const-=t['penalty']
  elif ks is not False:terms[ks]-=t['penalty']
 return const,terms,resolved

def witness_cost(e,c,raw,bad,witness):
 # Independently evaluate observed partial failures before threshold completion.
 cost=e['metrics']['booking_cost']
 for t in e['tasks']:
  cost+=t['operation_cost'];success=False
  if t['choice'] is not None:
   success=True
   for hour in range(c['issue']+t['start']+1,c['issue']+t['end']+1):
    z=raw[hour,t['node']]
    good=bool(z<=10.) if np.isfinite(z) else (False if bad[hour,t['node']] else bool(witness.get((hour,t['node']),0)))
    if not good:success=False;break
  cost+=(not success)*t['penalty']
 return cost

def main():
 protocol=dict(status='FROZEN_BEFORE_PARTIAL_SENSITIVITY_SOLVES',created_utc=datetime.now(timezone.utc).isoformat(),
  scope='All seven original predeclared comparisons, 44 origins x3 seeds, every origin/fold/ALL; saved forecasts, quality and actions fixed.',
  evidence='A valid observed 5-minute wind >10 in an otherwise strictly missing hour forces that physical hour bad for every arm/seed/origin. A partial maximum <=10 never certifies success. No interpolation or fabricated hourly maximum.',
  optimization='Same global shared binary-hour AND MILP as original main bounds, restricted by known_bad, two objectives and saved feasible witnesses.',
  certificate='Optimal status0, mip_gap<=1e-9, absolute objective/dual gap<=1e-7, independent witness cost <=1e-7, containment in original shared bounds. Main bounds unchanged.',
  evidence_path=str(EVIDENCE),tests='3 tests RED missing module then GREEN; partial observed bad fixes failure, partial-safe stays unknown, known raw failure dominates.')
 target=OUT/'protocol.json'
 if target.exists():previous=read(target);protocol['created_utc']=previous['created_utc'];assert previous==protocol
 else:write(target,protocol)
 times,nodes,raw=load_truth();z=np.load(EVIDENCE,allow_pickle=False);bad=z['known_bad'];count=z['valid_count'];partial=z['partial_max']
 assert bad.dtype==bool and bad.shape==raw.shape and count.shape==raw.shape and partial.shape==raw.shape
 assert list(z['nodes'].astype(str))==nodes
 evidence_times=pd.DatetimeIndex(pd.to_datetime(z['timestamps'],utc=True));assert np.array_equal(evidence_times.asi8,times.asi8)
 assert not (bad&np.isfinite(raw)).any()
 assert np.array_equal(bad,~np.isfinite(raw)&(count>0)&np.isfinite(partial)&(partial>10.))
 identity={str(EVIDENCE):sha(EVIDENCE),str(SOURCE):sha(SOURCE)}
 for p in [Path(__file__),ROOT/'iteration2_bounds_core.py',ROOT/'iteration2_bounds_partial_test.py']:
  identity[str(p)]=sha(p)
 comparisons=dict(COMPARISONS)
 comparisons.update({f'analog_graph_{ex}':(('fixed_B','AnalogFixed',ex),('fixed_B','B',ex)) for ex in ['lookahead72','myopic']})
 cases=read(OLD/'protocol.json')['cases'];cache={};rows=[];maxerror=0.;maxgap=0.
 baseline=pd.concat([pd.read_csv(BASE/f'{kind}_bounds.csv') for kind in ['cross','analog']]).set_index(['comparison','scope'])
 for kind in ['cross','analog']:identity[str(BASE/f'{kind}_bounds.csv')]=sha(BASE/f'{kind}_bounds.csv')
 def load(c,arm):
  path=(ANALOG if arm[1]=='AnalogFixed' else OLD/'evaluations')/f"{c['key']}__{'__'.join(arm)}.json"
  if path not in cache:cache[path]=read(path);identity[str(path)]=sha(path)
  return cache[path]
 scopes=[('ALL',cases)]+[(f,[c for c in cases if c['fold']==f]) for f in sorted({c['fold'] for c in cases})]
 scopes += [(f"{c['fold']}_issue{c['issue']}",[d for d in cases if d['fold']==c['fold'] and d['issue']==c['issue']]) for c in cases if c['seed']==min(d['seed'] for d in cases)]
 for name,(aa,bb) in comparisons.items():
  for scope,sub in scopes:
   n=len({(c['fold'],c['issue']) for c in sub});weight=1/(n*3);constant=0.;terms=defaultdict(float);resolved_a=0;resolved_b=0;weighted=[]
   for c in sub:
    a=load(c,aa);b=load(c,bb);weighted.append((c,a,b))
    for sign,e in [(1,a),(-1,b)]:
     const,t,resolved=expression(e,c,raw,bad);constant+=weight*sign*const
     if sign==1:resolved_a+=resolved
     else:resolved_b+=resolved
     for ks,coef in t.items():terms[ks]+=weight*sign*coef
   opt=solve_bounds(constant,terms)
   for side in ['lower','upper']:
    witness={(h,node):g for h,node,g in opt[side+'_witness']}
    assert not any(bad[h,node] for h,node in witness)
    direct=sum(weight*(witness_cost(a,c,raw,bad,witness)-witness_cost(b,c,raw,bad,witness)) for c,a,b in weighted)
    err=abs(direct-opt[side]);assert err<1e-7;maxerror=max(maxerror,err)
    opt[side+'_certificate']['independent_evidence_ledger_recomputed']=direct
    maxgap=max(maxgap,opt[side+'_certificate']['absolute_gap'])
   original=baseline.loc[name,scope]
   assert original.shared_lower-1e-7<=opt['lower'] and opt['upper']<=original.shared_upper+1e-7
   row=dict(comparison=name,scope=scope,origins=n,seed_cases=len(sub),original_shared_lower=float(original.shared_lower),original_shared_upper=float(original.shared_upper),partial_shared_lower=opt['lower'],partial_shared_upper=opt['upper'],original_shared_width=float(original.shared_width),partial_shared_width=opt['upper']-opt['lower'],width_reduction=float(original.shared_width)-(opt['upper']-opt['lower']),candidate_attempt_rows_unknown_to_failed=resolved_a,baseline_attempt_rows_unknown_to_failed=resolved_b,certificate='PASS')
   rows.append(row);write(OUT/'certificates'/f'{name}__{scope}.json',dict(**row,optimization=opt))
  print('DONE',name,flush=True)
 frame=pd.DataFrame(rows);frame.to_csv(OUT/'partial_evidence_bounds.csv',index=False,encoding='utf-8-sig');frame[~frame.scope.str.contains('_issue')].to_csv(BASE/'partial_evidence_summary.csv',index=False,encoding='utf-8-sig')
 write(OUT/'sources.json',identity)
 for path,digest in identity.items():assert sha(path)==digest
 write(OUT/'complete.json',dict(status='PASS',scope_bounds=len(rows),max_witness_error=maxerror,max_primal_dual_gap=maxgap,known_bad_physical_cells=int(bad.sum()),main_bounds_unchanged=True,evidence_sha256=sha(EVIDENCE),source_truth_sha256=sha(SOURCE)))
 print(frame[frame.scope=='ALL'].to_string(index=False),flush=True)

if __name__=='__main__':main()

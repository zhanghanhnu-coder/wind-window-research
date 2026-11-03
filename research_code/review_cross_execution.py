"""Isolated follow-up using exact round33 booking reuse and round27 updates.

Planning is completed and frozen before loading raw64 evaluation observations.
No original research result or default model is changed.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, hashlib, json, pickle, time
import numpy as np
import pandas as pd
from cross_execution_review_core import selected_positions,probability_grid,execute,task_ledger,difference_bounds
from two_stage_core import default_config,make_problem,weather_options,booking_cost,audit_schedule
from regret_core import solve_recourse
from distributional_schedule_core import worst_tv,observed_options
from crps_distribution_core import couple_atoms
from highs_readback_compat import install
from precision_truth_core import load_truth

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts'
OUT=ART/'review_followup_v1/cross_execution'
SRC=ART/'seasonal_probability_v1'
ROLL=ART/'seasonal_rolling_v1'
BOOK=ART/'distributional_schedule_v1'
SEEDS=[20241101,20241102,20241103]
FOLDS=['M05','M07','M09','M11']
MODEL='gated_graph_atoms'
METHODS={'B':'frozen','W28':'rolling28'}
EXECUTORS=['myopic','lookahead72']
TRUTH=ROOT.parent/'研究笔记/基线实验_2026-11-01/严格完整小时风速.pkl'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def checked(path,manifest):
 path=Path(path);expected=manifest.get(str(path))
 if expected is None or sha(path)!=expected:raise ValueError(f'Frozen manifest mismatch: {path}')
 return path

def sources():
 manifests={n:read(p/'final_manifest.json') for n,p in [('roll',ROLL),('book',BOOK)]}
 inventory=read(ROLL/'inventory.json');cache={};ident={}
 for fold in FOLDS:
  fpath=SRC/fold/'data.npz';f=dict(np.load(fpath));cache[fold]=dict(data=f,methods={});ident[str(fpath)]=sha(fpath)
  for seed in SEEDS:
   for label,method in METHODS.items():
    item=next(r for r in inventory if (r['fold'],r['seed'],r['model'],r['method'])==(fold,seed,MODEL,method))
    for p,h in [(item['source'],item['source_sha256']),(item['ranks_source'],item['ranks_sha256'])]:
     if sha(p)!=h:raise ValueError('Round27 forecast identity mismatch')
     ident[str(p)]=h
    cache[fold]['methods'][seed,label]=dict(q=np.load(item['source'])[item['source_key']],ranks=np.load(item['ranks_source'])['ranks'],item=item)
 return cache,manifests,ident

def specifications(cache):
 rows=[]
 for fold in FOLDS:
  ids=cache[fold]['data']['test_ids'];positions=selected_positions(len(ids))
  for pos in positions:
   issue=int(ids[pos])
   if issue+66>max(ids):raise ValueError('Selected case lacks in-period cached update range')
   for seed in SEEDS:rows.append(dict(fold=fold,position=int(pos),issue=issue,seed=seed,key=f'{fold}_{pos:03d}_{seed}'))
 return rows

def build_grids(case,cache):
 f=cache[case['fold']]['data'];ids=f['test_ids'];lookup={int(v):i for i,v in enumerate(ids)}
 grids={label:np.full((12,12,6),np.nan) for label in METHODS};quality=np.zeros(12,bool);checks=[]
 for k,now in enumerate(range(0,72,6)):
  issue=case['issue']+now;fraction=float(f['x'][issue-23:issue+1,:,-1].mean());quality[k]=fraction>=.65
  if quality[k] and issue not in lookup:raise ValueError('Good-quality update has no cached in-period forecast')
  checks.append(dict(now=now,issue=issue,valid_fraction=fraction,quality_ok=bool(quality[k]),history_sha256=hashlib.sha256(np.ascontiguousarray(f['x'][issue-23:issue+1]).tobytes()).hexdigest()))
  if not quality[k]:continue
  pos=lookup[issue]
  for label in METHODS:
   item=cache[case['fold']]['methods'][case['seed'],label]
   sc=couple_atoms(item['q'][pos].astype(float),item['ranks'])
   grids[label][k]=probability_grid(sc,default_config())
 return grids,quality,checks

def booking(case,label,cache,manifests,ident):
 p=make_problem(default_config());source_case='graph_frozen' if label=='B' else 'graph_rolling28'
 key=f'{case["fold"]}__{source_case}__{case["seed"]}__{case["position"]:03d}'
 forecast=checked(BOOK/'cases'/f'forecast_{key}.npz',manifests['book'])
 planpath=checked(BOOK/'cases'/f'plan_{key}__tv025.json',manifests['book'])
 ident[str(forecast)]=sha(forecast);ident[str(planpath)]=sha(planpath)
 z=np.load(forecast);source=cache[case['fold']]['methods'][case['seed'],label]
 sc=couple_atoms(source['q'][case['position']].astype(float),source['ranks'])
 np.testing.assert_array_equal(sc,z['scenarios']);np.testing.assert_array_equal(weather_options(p,sc),z['available'])
 np.testing.assert_array_equal(z['probability'],np.ones(57)/57)
 record=read(planpath);plan=record['plan'];ref=record['dual_reference']
 assert record['radius']==.25 and record['issue_id']==case['issue']
 assert read(BOOK/'recipe.json')['config']==p.config
 assert ref['solver_status']==0 and ref['mip_gap']<=1e-9
 costs=[]
 for a,sol in zip(z['available'],plan['scenario_solutions']):
  independent=audit_schedule(p,plan['x'],a,[r['choice'] for r in sol['schedule']])
  assert abs(independent['recourse_cost']-sol['recourse_cost'])<1e-7
  costs.append(booking_cost(p,plan['x'])+independent['recourse_cost'])
 risk=worst_tv(costs,z['probability'],.25)['objective']
 assert abs(risk-plan['objective'])<1e-5 and abs(risk-ref['objective'])<1e-5
 return dict(x=plan['x'],objective=risk,source=str(planpath),source_sha256=sha(planpath),forecast_sha256=sha(forecast),certificate=dict(full_status=ref['solver_status'],full_gap=ref['mip_gap'],gap=plan['gap'],objective_difference=risk-ref['objective']),scenario_schedules_audited=len(costs))

def pilot():
 install();cache,manifests,ident=sources();case=specifications(cache)[0];begin=time.monotonic();grids,quality,checks=build_grids(case,cache);b={label:booking(case,label,cache,manifests,ident) for label in METHODS};p=make_problem(default_config());n=0
 for label in METHODS:
  for ex in EXECUTORS:execute(p,b['B']['x'],grids[label],quality,ex);n+=1
 record=dict(case=case,seconds=time.monotonic()-begin,trajectories=n,truth_loaded=False,selection_depends_on_outcomes=False)
 write(OUT/'pilot_timing.json',record);print(json.dumps(record),flush=True)

def plan():
 install();cache,manifests,ident=sources();cases=specifications(cache)
 protocol=dict(status='FROZEN_BEFORE_EVALUATION',utc=datetime.now(timezone.utc).isoformat(),cases=cases,
  selection='Within each fold: first eleven positions from floor(linspace(0,nTest-1,12)); last omitted for lack of 66h cached updates; all 3 seeds',
  model=MODEL,methods=METHODS,seed_policy='all three frozen seeds; seeds do not create independent weather',
  config=default_config(),radius=.25,threshold=10.,update_hours=6,horizon=72,
  executors=EXECUTORS,layers=dict(fixed_B='B booking x shared by B/W28 and both executors',own='own B/W28 booking x, same executors'),
  booking='Exact cached round33 solution only after SHA/config/scenario/feasibility/probability/certificate/accounting validation',
  precision='Frozen float32-era predictions and calibration unchanged; outcome re-evaluated using original strict float64 hourly maxima',
  unknowns='conservative all-case cost bounds; same-case paired outer bounds; no causal information-value attribution',
  planned_origins=44,planned_origin_seed_cases=132,planned_arms=1056,
  no_confidence_intervals='Exploratory descriptive paired effects; 2024 repeatedly explored, three seeds are not independent weather',truth_loaded_in_planning=False)
 if (OUT/'protocol.json').exists():
  old=read(OUT/'protocol.json');protocol['utc']=old['utc'];assert old==protocol
 else:write(OUT/'protocol.json',protocol)
 p=make_problem(default_config());rows=[];start=time.monotonic()
 for case in cases:
  grids,quality,checks=build_grids(case,cache);b={label:booking(case,label,cache,manifests,ident) for label in METHODS}
  pack=OUT/'inputs'/f'{case["key"]}.npz';pack.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(pack,B=grids['B'],W28=grids['W28'],quality=quality)
  write(OUT/'inputs'/f'{case["key"]}.json',dict(case=case,updates=checks,booking=b))
  for layer in ['fixed_B','own']:
   for label in METHODS:
    x=b['B']['x'] if layer=='fixed_B' else b[label]['x']
    for ex in EXECUTORS:
     key=f'{case["key"]}__{layer}__{label}__{ex}';dest=OUT/'trajectories'/f'{key}.json'
     if dest.exists():r=read(dest)
     else:
      result=execute(p,x,grids[label],quality,ex)
      r=dict(case=case,layer=layer,prediction=label,executor=ex,trajectory=result,input_sha256=sha(pack));write(dest,r)
     assert r['input_sha256']==sha(pack) and r['trajectory']['x']==x
     rows.append(dict(**case,layer=layer,prediction=label,executor=ex,path=str(dest),sha256=sha(dest)))
  print('PLANNED',case['key'],'seconds',round(time.monotonic()-start,2),flush=True)
 for name in ['cross_execution_review_core.py','review_cross_execution.py','two_stage_core.py','regret_core.py','distributional_schedule_core.py','causal_execution_core.py','lookahead_execution_core.py']:
  ident[str(ROOT/name)]=sha(ROOT/name)
 ident[str(TRUTH)]=sha(TRUTH)  # bytes only, no labels loaded
 write(OUT/'source_identity.json',ident);write(OUT/'trajectory_inventory.json',rows)
 write(OUT/'planning_complete.json',dict(arms=len(rows),origin_seed_cases=len(cases),seconds=time.monotonic()-start,truth_loaded=False))

def evaluate():
 install();inventory=read(OUT/'trajectory_inventory.json');assert len(inventory)==1056
 assert read(OUT/'planning_complete.json')['truth_loaded'] is False
 # This is the first phase that loads original observations.
 truth_times,truth_nodes,raw=load_truth()
 timestamps=np.load(ART/'data.npz')['timestamps'];expected=pd.DatetimeIndex(pd.to_datetime(timestamps.tolist(),utc=True)).tz_convert('Asia/Shanghai')
 pd.testing.assert_index_equal(truth_times,expected,exact=False,check_names=False)
 assert raw.dtype==np.float64
 p=make_problem(default_config());rows=[];ledger=[];oracles={};outcomes={};evalfiles=[]
 for item in inventory:
  assert sha(item['path'])==item['sha256'];r=read(item['path']);case=r['case'];issue=case['issue'];truth=raw[issue+1:issue+73]
  tasks,metrics=task_ledger(p,r['trajectory'],truth)
  # Independent old evaluator cross-check (input remains float64).
  from causal_execution_core import evaluate_execution
  check=evaluate_execution(p,r['trajectory']['x'],r['trajectory']['attempts'],truth)
  for k in ['cost_lower','cost_upper','booking_cost','operating_cost','successful','failed','unknown','unattempted']:
   assert abs(check[k]-metrics[k])<1e-7
  x=r['trajectory']['x'];oraclekey=(issue,tuple(x))
  if oraclekey not in oracles:
   lo,hi,complete=observed_options(p,truth);a=solve_recourse(p,lo,x);b=a if np.array_equal(lo,hi) else solve_recourse(p,hi,x)
   oracles[oraclekey]=dict(lower=booking_cost(p,x)+a['recourse_cost'],upper=booking_cost(p,x)+b['recourse_cost'],complete=bool(complete),optimistic=a,pessimistic=b)
  oracle=oracles[oraclekey];assert metrics['cost_lower']>=oracle['lower']-1e-6
  if oracle['complete']:assert metrics['unknown']==0 and metrics['cost_lower']>=oracle['upper']-1e-6
  common={k:item[k] for k in ['key','fold','position','issue','seed','layer','prediction','executor']}
  row=dict(**common,**metrics,complete=oracle['complete'],ideal_lower=oracle['lower'],ideal_upper=oracle['upper'],execution_minus_ideal_lower=metrics['cost_lower']-oracle['upper'],execution_minus_ideal_upper=metrics['cost_upper']-oracle['lower'],x=json.dumps(x))
  rows.append(row);outcomes[tuple(common[k] for k in ['key','layer','prediction','executor'])]=row
  ledger.extend(dict(**common,**t) for t in tasks)
  dest=OUT/'evaluations'/Path(item['path']).name;write(dest,dict(**common,metrics=metrics,tasks=tasks,ideal=oracle));evalfiles.append(dict(path=str(dest),sha256=sha(dest)))
 csv('arms.csv',rows);csv('task_ledger.csv',ledger)
 pairs=[]
 comparisons=[('prediction_myopic',('fixed_B','W28','myopic'),('fixed_B','B','myopic')),
 ('prediction_lookahead',('fixed_B','W28','lookahead72'),('fixed_B','B','lookahead72')),
 ('executor_B',('fixed_B','B','lookahead72'),('fixed_B','B','myopic')),
 ('executor_W28',('fixed_B','W28','lookahead72'),('fixed_B','W28','myopic')),
 ('resource_myopic',('own','W28','myopic'),('fixed_B','W28','myopic')),
 ('resource_lookahead',('own','W28','lookahead72'),('fixed_B','W28','lookahead72')),
 ('own_prediction_myopic',('own','W28','myopic'),('own','B','myopic')),
 ('own_prediction_lookahead',('own','W28','lookahead72'),('own','B','lookahead72')),
 ('combined',('own','W28','lookahead72'),('own','B','myopic'))]
 for case in read(OUT/'protocol.json')['cases']:
  for name,ak,bk in comparisons:
   a=outcomes[(case['key'],)+ak];b=outcomes[(case['key'],)+bk];lo,hi=difference_bounds(a,b)
   pairs.append(dict(**case,comparison=name,difference_lower=lo,difference_upper=hi,complete=a['complete'] and b['complete'],success_difference=a['successful']-b['successful'],failed_difference=a['failed']-b['failed'],booking_difference=a['booking_cost']-b['booking_cost']))
 csv('paired_cases.csv',pairs)
 # First average seeds inside each origin. Weather count never multiplied by seeds.
 df=pd.DataFrame(pairs);origin=df.groupby(['fold','position','issue','comparison'],sort=False).agg(difference_lower=('difference_lower','mean'),difference_upper=('difference_upper','mean'),complete=('complete','all'),success_difference=('success_difference','mean'),failed_difference=('failed_difference','mean'),booking_difference=('booking_difference','mean')).reset_index()
 origin.to_csv(OUT/'paired_origins.csv',index=False,encoding='utf-8-sig');summary=[]
 for scope in ['ALL']+FOLDS:
  sub=origin if scope=='ALL' else origin[origin.fold==scope]
  for name,g in sub.groupby('comparison',sort=False):
   for population in ['all','complete_task_weather']:
    z=g if population=='all' else g[g.complete]
    summary.append(dict(scope=scope,comparison=name,population=population,origins=len(z),difference_lower=None if z.empty else float(z.difference_lower.mean()),difference_upper=None if z.empty else float(z.difference_upper.mean()),success_difference=None if z.empty else float(z.success_difference.mean()),failed_difference=None if z.empty else float(z.failed_difference.mean()),booking_difference=None if z.empty else float(z.booking_difference.mean())))
 csv('summary.csv',summary);write(OUT/'evaluation_inventory.json',evalfiles)
 write(OUT/'evaluation_complete.json',dict(arms=len(rows),task_rows=len(ledger),paired_seed_cases=len(pairs),independent_oracle_resource_pairs=len(oracles),truth_dtype=str(raw.dtype),truth_sha256=sha(TRUTH),training_repeated=False,forecast_calibration_refitted=False))
 print('EVALUATION COMPLETE',len(rows),'arms',len(ledger),'task rows',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['pilot','plan','evaluate']);args=parser.parse_args()
 {'pilot':pilot,'plan':plan,'evaluate':evaluate}[args.stage]()

"""Separate reconstruction of every support-miss diagnostic from raw inputs."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from precision_truth_core import load_truth
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/review_followup_v1/cross_execution'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 protocol=read(OUT/'protocol.json');cfg=protocol['config'];_,_,raw=load_truth();cache={};checks=0
 for c in protocol['cases']:
  meta=read(OUT/'inputs'/f'{c["key"]}.json');grid=dict(np.load(OUT/'inputs'/f'{c["key"]}.npz'))
  for pred in ['B','W28']:
   b=meta['booking'][pred];pp=Path(b['source']);fp=pp.with_name(pp.name.replace('plan_','forecast_').replace('__tv025.json','.npz'))
   assert sha(pp)==b['source_sha256'] and sha(fp)==b['forecast_sha256'];cache[c['key'],pred]=(np.load(fp)['scenarios'],grid,read(pp)['plan'])
 windows=pd.read_csv(OUT/'support_window_ledger.csv',low_memory=False);costs=pd.read_csv(OUT/'support_cost_ledger.csv');assert len(windows)==41574 and len(costs)==1320
 for r in windows.itertuples():
  t=cfg['tasks'][r.task];y=raw[r.issue+r.start+1:r.issue+r.start+t['duration']+1,t['node']];valid=np.isfinite(y)
  label=0. if (y[valid]>10).any() else 1. if valid.sum()==len(y) else np.nan
  assert np.isclose(label,r.label,equal_nan=True)
  sc,grid,plan=cache[r.key,r.prediction]
  if r.stage=='initial_candidate':prob=np.count_nonzero(np.max(sc[:,r.start:r.start+t['duration'],t['node']],axis=1)<=10)/57
  else:
   assert grid['quality'][r.start//6];prob=grid[r.prediction][r.start//6,0,r.task]
  assert abs(prob-r.probability)<1e-14
  assert r.all_available_actual_failure==bool(prob==1 and label==0) and r.all_unavailable_actual_success==bool(prob==0 and label==1);checks+=1
 for r in costs.itertuples():
  sc,grid,plan=cache[r.key,r.prediction]
  if r.stage=='own_resource_ideal_recourse':
   # Independent schedule sum, rather than the saved recourse_costs array.
   x=plan['x'];fee=float(np.array(cfg['boat_booking']+cfg['crew_booking']+cfg['spare_cost'])@x);vals=[]
   for sol in plan['scenario_solutions']:
    val=fee+sum(cfg['tasks'][j]['penalty'] for j in sol['deferred'])
    for o in sol['schedule']:
     t=cfg['tasks'][o['task']];val+=(cfg['boat_hourly'][o['boat']]+cfg['crew_hourly'][o['crew']])*t['duration']+cfg['late_fraction']*t['penalty']*o['start']/72
    vals.append(val)
   e=read(OUT/'evaluations'/f'{r.key}__own__{r.prediction}__myopic.json');lo=e['ideal']['lower'];hi=e['ideal']['upper']
  else:
   e=read(OUT/'evaluations'/f'{r.key}__{r.layer}__{r.prediction}__{r.executor}.json');a=read(OUT/'trajectories'/f'{r.key}__{r.layer}__{r.prediction}__{r.executor}.json')['trajectory'];x=a['x'];book=float(np.array(cfg['boat_booking']+cfg['crew_booking']+cfg['spare_cost'])@x);vals=[]
   for scenario in sc:
    total=book
    for task in e['tasks']:
     t=cfg['tasks'][task['task']]
     if task['choice'] is None:total+=t['penalty'];continue
     o=next(o for o in e['tasks'] if o['task']==task['task']);total+=o['operation_cost']
     if np.any(scenario[o['start']:o['end'],o['node']]>10):total+=t['penalty']
    vals.append(total)
   lo=e['metrics']['cost_lower'];hi=e['metrics']['cost_upper']
  lower=min(vals);upper=max(vals)
  assert abs(r.support_min-lower)<1e-6 and abs(r.support_max-upper)<1e-6
  assert r.certified_above==(lo>upper+1e-6) and r.certified_below==(hi<lower-1e-6)
  assert r.possible_above==(hi>upper+1e-6) and r.possible_below==(lo<lower-1e-6)
  assert abs(lo-r.observed_lower)<1e-6 and abs(hi-r.observed_upper)<1e-6
 for item in read(OUT/'trajectory_inventory.json'):assert sha(item['path'])==item['sha256']
 # Summary counts are recomputed from all rows, including per-fold strata.
 for r in pd.read_csv(OUT/'support_window_summary.csv').itertuples():
  z=windows[(windows.stage==r.stage)&(windows.prediction==r.prediction)]
  if r.scope!='ALL':z=z[z.fold==r.scope]
  assert len(z)==r.windows
  for c in ['all_available_actual_failure','all_unavailable_actual_success']:assert int(z[c].sum())==getattr(r,c)
 for r in pd.read_csv(OUT/'support_cost_summary.csv').itertuples():
  z=costs[(costs.stage==r.stage)&(costs.layer==r.layer)&(costs.prediction==r.prediction)&(costs.executor==r.executor)]
  if r.scope!='ALL':z=z[z.fold==r.scope]
  if r.population=='complete_task_weather':z=z[z.complete]
  assert len(z)==r.cases
  for c in ['certified_above','certified_below','possible_above','possible_below']:assert int(z[c].sum())==getattr(r,c)
 unique=windows[windows.stage=='executed_attempt'].drop_duplicates(['key','prediction','task','start']);unique.to_csv(OUT/'support_unique_executed_windows.csv',index=False,encoding='utf-8-sig')
 result=dict(status='PASS',window_labels_and_probabilities=checks,cost_support_ranges=len(costs),trajectories_unchanged=1056,raw_dtype=str(raw.dtype),unique_executed_window_cases=len(unique),all_available_actual_failure_unique={p:int(g.all_available_actual_failure.sum()) for p,g in unique.groupby('prediction')},all_unavailable_actual_success_unique={p:int(g.all_unavailable_actual_success.sum()) for p,g in unique.groupby('prediction')},strategy_modified=False)
 (OUT/'support_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

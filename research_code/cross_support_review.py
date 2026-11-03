"""Post-decision finite-support diagnostics; never changes forecasts or actions."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from precision_truth_core import load_truth

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/review_followup_v1/cross_execution'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def label_window(values,threshold=10.):
 z=np.asarray(values)
 if z.dtype!=np.float64:raise ValueError('original float64 values required')
 known=np.isfinite(z)
 return 0. if np.any(z[known]>threshold) else 1. if known.all() else np.nan

def interval_outside(lower,upper,support_min,support_max,tolerance=1e-6):
 if lower>upper or support_min>support_max:raise ValueError('reversed bounds')
 return dict(certified_above=bool(lower>support_max+tolerance),certified_below=bool(upper<support_min-tolerance),possible_above=bool(upper>support_max+tolerance),possible_below=bool(lower<support_min-tolerance))

def fixed_action_costs(config,x,tasks,scenarios):
 """Same saved attempts on every initial weather, without adaptive re-planning."""
 sc=np.asarray(scenarios,float);book=float(np.array(config['boat_booking']+config['crew_booking']+config['spare_cost'])@np.array(x));op=sum(t['operation_cost'] for t in tasks)
 costs=np.full(len(sc),book+op+sum(t['penalty'] for t in config['tasks']))
 for row in tasks:
  if row['choice'] is None:continue
  success=(sc[:,row['start']:row['end'],row['node']]<=10).all(1)
  costs-=config['tasks'][row['task']]['penalty']*success
 return costs

def candidate_record(common,stage,task,start,prob,label):
 return dict(**common,stage=stage,task=task,start=start,probability=float(prob),label=None if not np.isfinite(label) else float(label),all_available=bool(prob==1.),all_unavailable=bool(prob==0.),all_available_actual_failure=bool(prob==1. and label==0.),all_unavailable_actual_success=bool(prob==0. and label==1.))

def main():
 protocol=read(OUT/'protocol.json');cfg=protocol['config'];times,nodes,raw=load_truth();records=[];ranges=[];scache={}
 before={r['path']:r['sha256'] for r in read(OUT/'trajectory_inventory.json')}
 for case in protocol['cases']:
  meta=read(OUT/'inputs'/f'{case["key"]}.json');grid=np.load(OUT/'inputs'/f'{case["key"]}.npz')
  for pred in ['B','W28']:
   common={k:case[k] for k in ['key','fold','position','issue','seed']};common['prediction']=pred
   booking=meta['booking'][pred];planpath=Path(booking['source']);forecast=planpath.with_name(planpath.name.replace('plan_','forecast_').replace('__tv025.json','.npz'))
   assert sha(forecast)==booking['forecast_sha256'] and sha(planpath)==booking['source_sha256']
   sc=np.load(forecast)['scenarios'];scache[case['key'],pred]=sc
   for task,t in enumerate(cfg['tasks']):
    for start in range(0,73-t['duration'],6):
     label=label_window(raw[case['issue']+start+1:case['issue']+start+t['duration']+1,t['node']])
     prob=(sc[:,start:start+t['duration'],t['node']]<=10).all(1).mean()
     records.append(candidate_record(common,'initial_candidate',task,start,prob,label))
     if grid['quality'][start//6]:records.append(candidate_record(common,'current_candidate',task,start,grid[pred][start//6,0,task],label))
   # Same-x optimal recourse: comparable to actual ideal recourse, never execution cost.
   old=read(planpath)['plan'];values=old['booking_cost']+np.array(old['recourse_costs'])
   evaluation=read(OUT/'evaluations'/f'{case["key"]}__own__{pred}__myopic.json');ideal=evaluation['ideal']
   ranges.append(dict(**common,stage='own_resource_ideal_recourse',layer='own',executor='ideal',observed_lower=ideal['lower'],observed_upper=ideal['upper'],support_min=float(values.min()),support_max=float(values.max()),complete=bool(ideal['complete']),**interval_outside(ideal['lower'],ideal['upper'],float(values.min()),float(values.max()))))
 for item in read(OUT/'trajectory_inventory.json'):
  saved=read(item['path']);case=saved['case'];pred=saved['prediction'];layer=saved['layer'];ex=saved['executor'];common={k:case[k] for k in ['key','fold','position','issue','seed']};common['prediction']=pred
  evaluation=read(OUT/'evaluations'/Path(item['path']).name);grid=np.load(OUT/'inputs'/f'{case["key"]}.npz')
  for t in evaluation['tasks']:
   if t['choice'] is None:continue
   label=0. if t['status']=='failed' else 1. if t['status']=='successful' else np.nan
   row=candidate_record(common,'executed_attempt',t['task'],t['start'],grid[pred][t['start']//6,0,t['task']],label);row.update(layer=layer,executor=ex);records.append(row)
  costs=fixed_action_costs(cfg,saved['trajectory']['x'],evaluation['tasks'],scache[case['key'],pred]);lo=evaluation['metrics']['cost_lower'];hi=evaluation['metrics']['cost_upper']
  ranges.append(dict(**common,stage='fixed_saved_action_execution',layer=layer,executor=ex,observed_lower=lo,observed_upper=hi,support_min=float(costs.min()),support_max=float(costs.max()),complete=bool(evaluation['ideal']['complete']),**interval_outside(lo,hi,float(costs.min()),float(costs.max()))))
 pd.DataFrame(records).to_csv(OUT/'support_window_ledger.csv',index=False,encoding='utf-8-sig');pd.DataFrame(ranges).to_csv(OUT/'support_cost_ledger.csv',index=False,encoding='utf-8-sig')
 windows=pd.DataFrame(records);costs=pd.DataFrame(ranges);summary=[]
 for scope in ['ALL','M05','M07','M09','M11']:
  z=windows if scope=='ALL' else windows[windows.fold==scope]
  for (stage,pred),g in z.groupby(['stage','prediction'],sort=False):
   known=g.label.notna();summary.append(dict(scope=scope,stage=stage,prediction=pred,windows=len(g),known=int(known.sum()),unknown=int((~known).sum()),all_available=int(g.all_available.sum()),all_available_known=int((g.all_available&known).sum()),all_available_actual_failure=int(g.all_available_actual_failure.sum()),all_unavailable=int(g.all_unavailable.sum()),all_unavailable_known=int((g.all_unavailable&known).sum()),all_unavailable_actual_success=int(g.all_unavailable_actual_success.sum())))
 pd.DataFrame(summary).to_csv(OUT/'support_window_summary.csv',index=False,encoding='utf-8-sig');summary=[]
 for scope in ['ALL','M05','M07','M09','M11']:
  z=costs if scope=='ALL' else costs[costs.fold==scope]
  for (stage,layer,pred,ex),g in z.groupby(['stage','layer','prediction','executor'],sort=False):
   for population in ['all','complete_task_weather']:
    sub=g if population=='all' else g[g.complete]
    summary.append(dict(scope=scope,stage=stage,layer=layer,prediction=pred,executor=ex,population=population,cases=len(sub),**{c:int(sub[c].sum()) for c in ['certified_above','certified_below','possible_above','possible_below']}))
 pd.DataFrame(summary).to_csv(OUT/'support_cost_summary.csv',index=False,encoding='utf-8-sig')
 for p,h in before.items():assert sha(p)==h
 manifest=dict(trajectories_unchanged=len(before),window_rows=len(records),cost_rows=len(ranges),post_outcome_diagnostic=True,strategy_modified=False,scope='Raw counts include seeds and repeated arms, not independent weather trials; fixed-action support is retrospective and is not an adaptive-policy predictive distribution; ideal-recourse support is compared only to same-resource ideal recourse.')
 (OUT/'support_diagnostic_complete.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()

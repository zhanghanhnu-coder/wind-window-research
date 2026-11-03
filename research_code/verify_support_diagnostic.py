"""Independent per-hour windows and schedule accounting for finite-support audit."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from model import ART
from verify_two_stage import independent_schedule
from research_crps_distribution import guard
OUT=ART/'support_diagnostic_v1';PARENT=ART/'distributional_schedule_v1';FOLDS=['M05','M07','M09','M11']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,rtol=0,atol=3e-10,equal_nan=True)

def table(name,rows,keys):
 a=pd.read_csv(OUT/name,float_precision='round_trip').sort_values(keys).reset_index(drop=True);b=pd.DataFrame(rows).sort_values(keys).reset_index(drop=True);assert list(a.columns)==list(b.columns) and len(a)==len(b)
 for key in b:
  if b[key].dtype.kind in 'biufc':close(a[key],b[key])
  else:assert a[key].fillna('').tolist()==b[key].fillna('').tolist(),(name,key)
 return len(a)

def run():
 guard();assert read(OUT/'evaluation_complete.json')['status']=='COMPLETE';recipe=read(OUT/'recipe.json');assert recipe['tolerance']==1e-6 and recipe['no_policy_optimization'] and recipe['no_confidence_intervals'];manifest=read(PARENT/'final_manifest.json');assert sha(PARENT/'final_manifest.json')==recipe['parent_manifest_sha256']
 for file,h in manifest.items():assert sha(file)==h,file
 for file,h in read(OUT/'identity.json').items():assert sha(file)==h,file
 sources=read(OUT/'source_hashes.json')
 for file,h in sources.items():assert sha(file)==h,file
 cfg=read(PARENT/'recipe.json')['config'];raw=np.load(ART/'data.npz')['raw'];windows=[];costs=[];physical_labels={};schedule_count=0
 for item in read(PARENT/'inventory.json'):
  guard();sc=np.load(item['forecast_source'])['scenarios'];y=raw[item['issue_id']+1:item['issue_id']+73];common={k:item[k] for k in ['key','fold','case','seed','position','issue_id']};nodes={t['node'] for t in cfg['tasks']};complete=all(np.isfinite(y[:,n]).all() for n in nodes)
  for index,t in enumerate(cfg['tasks']):
   for start in range(0,73-t['duration'],6):
    end=start+t['duration'];count=sum(all(float(sc[s,h,t['node']])<=10 for h in range(start,end)) for s in range(57));actual=[float(y[h,t['node']]) for h in range(start,end)];known=[v for v in actual if np.isfinite(v)];label=0 if any(v>10 for v in known) else 1 if len(known)==len(actual) else -1;physical=f'{item["issue_id"]+start+1}:{t["node"]}:{t["duration"]}'
    if physical in physical_labels:assert physical_labels[physical]==label
    physical_labels[physical]=label;windows.append(dict(**common,physical_window=physical,task_index=index,task_id=t['id'],node=t['node'],start=start,end=end,scenario_count=57,scenario_successes=count,label=label))
  for policy in item['policies']:
   pf=PARENT/'cases'/f'plan_{item["key"]}__{policy}.json';of=PARENT/'cases'/f'outcome_{item["key"]}__{policy}.json';p=read(pf);o=read(of);x=p['plan']['x'];assert x==o['resources'] and complete==o['complete'];total=[]
   for s,solution in enumerate(p['plan']['scenario_solutions']):book,q=independent_schedule(cfg,x,sc[s],solution);total.append(book+q);schedule_count+=1
   lowerbook,lq=independent_schedule(cfg,x,y,o['optimistic_solution'],'optimistic');upperbook,uq=independent_schedule(cfg,x,y,o['pessimistic_solution'],'pessimistic');schedule_count+=2;assert book==lowerbook==upperbook;lo=book+lq;hi=book+uq;close([lo,hi],[o['cost_lower'],o['cost_upper']]);support_min=min(total);support_max=max(total);definite=lo>support_max+1e-6;possible=hi>support_max+1e-6
   costs.append(dict(**common,policy=policy,radius=p['radius'],booking_cost=book,resources=json.dumps(x,separators=(',',':')),complete=complete,actual_lower=lo,actual_upper=hi,support_min=support_min,support_max=support_max,definite_excess=definite,possible_excess=possible,uncertain_excess=possible and not definite,definite_within_support_max=not possible,excess_lower=max(0.,lo-support_max),excess_upper=max(0.,hi-support_max),complete_excess=int(definite) if complete else None,plan_source=str(pf),outcome_source=str(of)))
  print('SUPPORT_AUDIT',item['key'],flush=True)
 cost_count=table('cost_support.csv',costs,['key','policy']);window_count=table('task_windows.csv',windows,['key','task_index','start']);assert cost_count==1392 and window_count==30240;cm=[];wm=[]
 for scope in ['ALL',*FOLDS]:
  c=[r for r in costs if scope=='ALL' or r['fold']==scope];w=[r for r in windows if scope=='ALL' or r['fold']==scope]
  for case,policy in sorted({(r['case'],r['policy']) for r in c}):
   g=[r for r in c if r['case']==case and r['policy']==policy];comp=[r for r in g if r['complete']];n=len(g);d=sum(r['definite_excess'] for r in g);p=sum(r['possible_excess'] for r in g);ex=sum(r['complete_excess'] for r in comp);cm.append(dict(scope=scope,case=case,policy=policy,n=n,unique_origins=len({r['issue_id'] for r in g}),complete=len(comp),complete_unique_origins=len({r['issue_id'] for r in comp}),definite_excess=d,possible_excess=p,uncertain_excess=p-d,definite_excess_rate=d/n,possible_excess_rate=p/n,mean_excess_lower=sum(r['excess_lower'] for r in g)/n,mean_excess_upper=sum(r['excess_upper'] for r in g)/n,complete_excess=ex,complete_excess_rate=ex/len(comp) if comp else np.nan))
  for case in sorted({r['case'] for r in w}):
   g=[r for r in w if r['case']==case];row=dict(scope=scope,case=case,n=len(g),unique_origins=len({r['issue_id'] for r in g}),unique_physical_windows=len({r['physical_window'] for r in g}),known_success=sum(r['label']==1 for r in g),known_failure=sum(r['label']==0 for r in g),unknown=sum(r['label']==-1 for r in g))
   for prefix,k,opposite in [('all_safe',57,0),('all_unsafe',0,1)]:
    sub=[r for r in g if r['scenario_successes']==k];n=len(sub);success=sum(r['label']==1 for r in sub);failure=sum(r['label']==0 for r in sub);missing=sum(r['label']==-1 for r in sub);event=failure if opposite==0 else success;row.update({f'{prefix}_n':n,f'{prefix}_known_success':success,f'{prefix}_known_failure':failure,f'{prefix}_unknown':missing,f'{prefix}_opposite_lower':event/n if n else np.nan,f'{prefix}_opposite_upper':(event+missing)/n if n else np.nan})
   wm.append(row)
 assert table('cost_summary.csv',cm,['scope','case','policy'])==55 and table('window_summary.csv',wm,['scope','case'])==15
 assert read(OUT/'evaluation_complete.json')['unique_physical_windows']==len(physical_labels)
 for file,h in sources.items():assert sha(file)==h,file
 assert sha(PARENT/'final_manifest.json')==recipe['parent_manifest_sha256']
 result=dict(status='PASS',forecast_cases=432,cost_rows=cost_count,window_rows=window_count,schedules_independently_accounted=schedule_count,cost_summary_rows=55,window_summary_rows=15,unique_physical_windows=len(physical_labels),source_files=len(sources),parent_manifest_files=len(manifest),new_optimization_solves=0,inherited_optimality='Frozen round33 independent verification; this round independently reaccounts schedules.');(OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':run()

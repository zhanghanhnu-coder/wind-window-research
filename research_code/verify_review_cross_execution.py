"""Independent outcome and selected optimizer verification for the new cross study."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
import pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint,linprog
from precision_truth_core import load_truth
from highs_readback_compat import install

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/review_followup_v1/cross_execution'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def independent_reward(config,x,history,now,grid):
 """Rebuild binary scheduling directly from config; no production choice groups."""
 H=config['horizon'];B=len(config['boat_booking']);C=len(config['crew_booking']);seen={o['task'] for o in history};opts=[];gain=[]
 for task,t in enumerate(config['tasks']):
  if task in seen:continue
  for start in range(now,H-t['duration']+1,6):
   prob=grid[(start-now)//6,task]
   for b in range(B):
    for c in range(C):
     if not x[b] or not x[B+c] or t['skill'] not in config['crew_skills'][c]:continue
     value=t['penalty']*prob-(config['boat_hourly'][b]+config['crew_hourly'][c])*t['duration']-config['late_fraction']*t['penalty']*start/H
     if value>0:opts.append(dict(task=task,start=start,end=start+t['duration'],boat=b,crew=c,node=t['node'],spare=t['spare'],quantity=t['quantity']));gain.append(value)
 if not opts:return 0.
 rows=[];upper=[]
 def add(vals,cap):
  if any(vals):rows.append(vals);upper.append(cap)
 for j in range(len(config['tasks'])):add([int(o['task']==j) for o in opts],1)
 for key,size,offset in [('boat',B,0),('crew',C,B)]:
  for r in range(size):
   for h in range(now,H):
    add([int(o[key]==r and o['start']<=h<o['end']) for o in opts],x[offset+r]-sum(o[key]==r and o['start']<=h<o['end'] for o in history))
 for node in {t['node'] for t in config['tasks']}:
  for h in range(now,H):add([int(o['node']==node and o['start']<=h<o['end']) for o in opts],1-sum(o['node']==node and o['start']<=h<o['end'] for o in history))
 for spare in range(len(config['spare_cost'])):
  add([o['quantity'] if o['spare']==spare else 0 for o in opts],x[B+C+spare]-sum(o['quantity'] for o in history if o['spare']==spare))
 result=milp(-np.array(gain),integrality=np.ones(len(opts)),bounds=Bounds(np.zeros(len(opts)),np.ones(len(opts))),constraints=LinearConstraint(np.array(rows,float),np.full(len(rows),-np.inf),np.array(upper,float)),options=dict(mip_rel_gap=0.,time_limit=60.))
 assert result.status==0
 return float(-result.fun)

def independent_now(config,x,history,now,prob):
 B=len(config['boat_booking']);C=len(config['crew_booking']);seen={o['task'] for o in history};active=[o for o in history if o['end']>now];opts=[]
 for j,t in enumerate(config['tasks']):
  if j in seen or now+t['duration']>72:continue
  for b in range(B):
   for c in range(C):
    if not x[b] or not x[B+c] or t['skill'] not in config['crew_skills'][c]:continue
    if any(o['boat']==b or o['crew']==c or o['node']==t['node'] for o in active):continue
    gain=t['penalty']*prob[j]-(config['boat_hourly'][b]+config['crew_hourly'][c])*t['duration']-config['late_fraction']*t['penalty']*now/72
    if gain>0:opts.append(dict(task=j,boat=b,crew=c,node=t['node'],spare=t['spare'],quantity=t['quantity'],gain=gain))
 remaining=np.array(x[B+C:])-np.array([sum(o['quantity'] for o in history if o['spare']==s) for s in range(len(config['spare_cost']))]);best=0.
 for k in [1,2]:
  for chosen in itertools.combinations(opts,k):
   if any(len({o[key] for o in chosen})<k for key in ['task','boat','crew','node']):continue
   if any(sum(o['quantity'] for o in chosen if o['spare']==s)>remaining[s] for s in range(len(remaining))):continue
   best=max(best,sum(o['gain'] for o in chosen))
 return best

def main():
 install();protocol=read(OUT/'protocol.json');cfg=protocol['config'];times,nodes,raw=load_truth();assert raw.dtype==np.float64
 identities=read(OUT/'source_identity.json')
 for path,h in identities.items():assert sha(path)==h,path
 inventory=read(OUT/'trajectory_inventory.json');evaluations=read(OUT/'evaluation_inventory.json')
 assert len(inventory)==len(evaluations)==1056
 evalmap={Path(r['path']).name:r for r in evaluations};tested=0;taskchecks=0;stepchecks=0;independent_optima=0;causal_perturbations=0;batchchecks=0
 # Use production options only for identity lookup. Accounting/capacity rows below are independent.
 from two_stage_core import make_problem
 p=make_problem(cfg)
 for item in inventory:
  assert sha(item['path'])==item['sha256'];r=read(item['path']);a=r['trajectory'];x=a['x'];case=r['case'];truth=raw[case['issue']+1:case['issue']+73]
  eitem=evalmap[Path(item['path']).name];assert sha(eitem['path'])==eitem['sha256'];e=read(eitem['path'])
  inputpath=OUT/'inputs'/f'{case["key"]}.npz';assert sha(inputpath)==r['input_sha256'];inputs=np.load(inputpath);grids=inputs[r['prediction']];quality=inputs['quality'];history=[]
  for k,step in enumerate(a['steps']):
   now=step['now'];assert now==6*k
   if not quality[k]:assert not step['selected']
   for choice in step['selected']:
    o=p.choices[choice];t=cfg['tasks'][o['task']];assert o['start']==now and o['end']<=72 and o['task'] not in {z['task'] for z in history}
    assert t['skill'] in cfg['crew_skills'][o['crew']]
   if r['executor']=='lookahead72' and step.get('solver_called'):
    assert step['solver_status']==0 and step['mip_gap']<=1e-9 and abs(step['expected_benefit']-step['dual_bound'])<1e-6
   if r['executor']=='myopic' and quality[k]:
    best=independent_now(cfg,x,history,now,grids[k,0]);assert abs(best-step['expected_benefit'])<1e-7;batchchecks+=1
   # Fixed, outcome-independent audit subset: first position, seed1, both methods/layers.
   if case['position']==0 and case['seed']==20241101 and r['executor']=='lookahead72' and k in [0,4,8] and quality[k]:
    target=independent_reward(cfg,x,history,now,grids[k]);assert abs(target-step['expected_benefit'])<1e-6;independent_optima+=1
   history.extend(p.choices[c] for c in step['selected']);stepchecks+=1
  assert len(history)==len(a['attempts'])
  boats=np.zeros((2,72));crews=np.zeros((3,72));spares=np.zeros(2);occupied={t['node']:np.zeros(72) for t in cfg['tasks']};bytask={o['task']:o for o in history}
  book=float(np.array(cfg['boat_booking']+cfg['crew_booking']+cfg['spare_cost'])@np.array(x));op=0.;pl=0.;pu=0.;statuses=[]
  for j,t in enumerate(cfg['tasks']):
   if j not in bytask:status='unattempted';pl+=t['penalty'];pu+=t['penalty']
   else:
    o=bytask[j];boats[o['boat'],o['start']:o['end']]+=1;crews[o['crew'],o['start']:o['end']]+=1;spares[o['spare']]+=o['quantity'];occupied[o['node']][o['start']:o['end']]+=1
    y=truth[o['start']:o['end'],o['node']];known=np.isfinite(y)
    status='failed' if (y[known]>10).any() else 'successful' if known.all() else 'unknown'
    op+=(cfg['boat_hourly'][o['boat']]+cfg['crew_hourly'][o['crew']])*t['duration']+cfg['late_fraction']*t['penalty']*o['start']/72
    if status=='failed':pl+=t['penalty']
    if status!='successful':pu+=t['penalty']
   assert status==e['tasks'][j]['status'];statuses.append(status);taskchecks+=1
  assert np.all(boats<=np.array(x[:2])[:,None]) and np.all(crews<=np.array(x[2:5])[:,None]) and np.all(spares<=x[5:]) and all((z<=1).all() for z in occupied.values())
  assert abs(book+op+pl-e['metrics']['cost_lower'])<1e-6 and abs(book+op+pu-e['metrics']['cost_upper'])<1e-6
  tested+=1
  if case['position']==0 and case['seed']==20241101 and r['layer']=='fixed_B':
   from cross_execution_review_core import execute
   modified=grids.copy();modified[2:]=0;new=execute(p,x,modified,quality,r['executor'])
   assert [s['selected'] for s in new['steps'][:2]]==[s['selected'] for s in a['steps'][:2]];causal_perturbations+=1
 # Independently recompute CSV paired effects and summaries from saved arms.
 arms=pd.read_csv(OUT/'arms.csv');pairs=pd.read_csv(OUT/'paired_cases.csv');summary=pd.read_csv(OUT/'summary.csv');assert len(arms)==1056 and len(pairs)==1188
 mapping={
  'prediction_myopic':(('fixed_B','W28','myopic'),('fixed_B','B','myopic')),
  'prediction_lookahead':(('fixed_B','W28','lookahead72'),('fixed_B','B','lookahead72')),
  'executor_B':(('fixed_B','B','lookahead72'),('fixed_B','B','myopic')),
  'executor_W28':(('fixed_B','W28','lookahead72'),('fixed_B','W28','myopic')),
  'resource_myopic':(('own','W28','myopic'),('fixed_B','W28','myopic')),
  'resource_lookahead':(('own','W28','lookahead72'),('fixed_B','W28','lookahead72')),
  'own_prediction_myopic':(('own','W28','myopic'),('own','B','myopic')),
  'own_prediction_lookahead':(('own','W28','lookahead72'),('own','B','lookahead72')),
  'combined':(('own','W28','lookahead72'),('own','B','myopic'))}
 keyed={(r.key,r.layer,r.prediction,r.executor):r for r in arms.itertuples()}
 for pair in pairs.itertuples():
  ak,bk=mapping[pair.comparison];a=keyed[(pair.key,)+ak];b=keyed[(pair.key,)+bk]
  assert abs(pair.difference_lower-(a.cost_lower-b.cost_upper))<1e-7 and abs(pair.difference_upper-(a.cost_upper-b.cost_lower))<1e-7
 complete=arms.groupby(['fold','issue']).complete.first();support=[]
 for fold in ['M05','M07','M09','M11']:
  cases=[c for c in protocol['cases'] if c['fold']==fold and c['seed']==20241101];start=times[min(c['issue'] for c in cases)].normalize();weeks={int((times[c['issue']]-start).days//7) for c in cases}
  support.append(dict(fold=fold,origins=len(cases),weeks=len(weeks),complete_origins=int(complete.loc[fold].sum())))
 for _,s in summary.iterrows():
  subset=pairs[pairs.comparison==s.comparison]
  if s.scope!='ALL':subset=subset[subset.fold==s.scope]
  if s.population!='all':subset=subset[subset.complete]
  origin=subset.groupby(['fold','issue'])[['difference_lower','difference_upper']].mean()
  assert len(origin)==s.origins
  for metric in ['difference_lower','difference_upper']:
   assert np.isclose(origin[metric].mean(),s[metric],atol=1e-8,equal_nan=True)
 out=dict(status='PASS',arms=tested,task_labels_and_costs=taskchecks,steps=stepchecks,independent_current_batch_optima=batchchecks,independent_reward_optima=independent_optima,future_forecast_perturbations=causal_perturbations,source_hashes_unchanged=len(identities),per_fold_support=support,truth_dtype=str(raw.dtype),scope='Fixed prediction re-evaluation; three seeds do not triple independent weather; no confidence or multistage optimality claim')
 (OUT/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

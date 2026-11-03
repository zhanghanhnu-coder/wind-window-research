"""Round39: lock all artificial-support bookings before actual cost evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time
import numpy as np
import pandas as pd
from model import ART
from research_crps_distribution import guard,SEEDS
from two_stage_core import default_config,make_problem,weather_options,booking_cost,SolverFailure
from distributional_schedule_core import solve_distribution_ccg,solve_distribution_master,observed_options
from regret_core import solve_recourse
from support_stress_core import augment_closed,closed_support_bound
from support_stress_tables import make_tables
from highs_readback_compat import install

OUT=ART/'support_stress_v1';PARENT=ART/'distributional_schedule_v1';PARENTS=[PARENT,ART/'support_diagnostic_v1'];PLAN=ART.parent.parent/'研究笔记/支持集补充压力情景方案_2026-11-02.md'
EPS={'stress_e010':.01,'stress_e001':.001,'stress_e050':.05}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def js(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def save_csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def parents():
 result={}
 for root in PARENTS:
  m=read(root/'final_manifest.json')
  for file,h in m.items():assert sha(file)==h,file
  result[root.name]=len(m)
 return result

def prepare():
 guard();parents();assert read(ART/'v5_release_v1/final_manifest.json') and read(ART.parent/'v5/evidence/final_delivery.json')['status']=='PASS','finish V5 delivery first'
 OUT.mkdir(exist_ok=True);(OUT/'cases').mkdir(exist_ok=True);protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0];frozen=OUT/'protocol_frozen.md'
 if frozen.exists():assert frozen.read_text(encoding='utf-8')==protocol
 else:frozen.write_text(protocol,encoding='utf-8')
 files=[frozen,Path(__file__)]+[Path(__file__).parent/n for n in ['support_stress_core.py','support_stress_tables.py','verify_support_stress.py','preflight_support_stress.py','tests/test_support_stress.py','distributional_schedule_core.py','two_stage_core.py','regret_core.py','highs_readback_compat.py']]+[r/'final_manifest.json' for r in PARENTS]+[ART/'data.npz']
 identity={str(p):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==identity
 else:js(OUT/'identity.json',identity)
 if not (OUT/'recipe.json').exists():js(OUT/'recipe.json',dict(epsilon=EPS,radius=.25,threshold=10.,stress_value=11.,expected_plans=240,main_plans=144,sensitivity_plans=96,config=default_config(),solver_time_limit=60,bootstrap_seed=20261125,bootstrap_replicates=2000,scope='posthoc exploratory artificial support stress; no calibrated weather probability',start_utc=datetime.now(timezone.utc).isoformat()))
 js(OUT/'solver_compatibility.json',install())

def forecasts():
 inventory=[]
 for old in read(PARENT/'inventory.json'):
  if old['case']!='graph_rolling28':continue
  assert sha(old['forecast_source'])==old['forecast_sha256'];z=np.load(old['forecast_source'])
  for case in [*EPS] if old['seed']==SEEDS[0] else ['stress_e010']:
   key=old['key']+'__'+case;path=OUT/'cases'/('forecast_'+key+'.npz');sc,p=augment_closed(z['scenarios'],z['probability'],EPS[case],10.);available=weather_options(make_problem(default_config()),sc)
   if path.exists():
    saved=np.load(path)
    for k,v in [('scenarios',sc),('probability',p),('available',available)]:np.testing.assert_array_equal(saved[k],v)
   else:np.savez_compressed(path,scenarios=sc,probability=p,available=available)
   inventory.append(dict(key=key,parent_key=old['key'],fold=old['fold'],seed=old['seed'],position=old['position'],issue_id=old['issue_id'],timestamp=old['timestamp'],case=case,epsilon=EPS[case],source_forecast=old['forecast_source'],source_sha256=old['forecast_sha256'],forecast_source=str(path),forecast_sha256=sha(path)))
 assert len(inventory)==240;js(OUT/'inventory.json',inventory);js(OUT/'forecasts_complete.json',dict(status='COMPLETE',forecasts=240,utc=datetime.now(timezone.utc).isoformat()))

def plan_all():
 p=make_problem(default_config());rows=[];failures=[]
 for item in read(OUT/'inventory.json'):
  guard();path=OUT/'cases'/('plan_'+item['key']+'.json');failed=OUT/'cases'/('failure_'+item['key']+'.json');assert sha(item['forecast_source'])==item['forecast_sha256'];z=np.load(item['forecast_source'])
  if failed.exists():failures.append(read(failed));continue
  if path.exists():r=read(path)
  else:
   begin=time.monotonic();print('PLAN',item['key'],flush=True)
   try:
    plan=solve_distribution_ccg(p,z['available'],z['probability'],.25,guard=guard);full=solve_distribution_master(p,z['available'],z['probability'],.25);np.testing.assert_allclose(plan['objective'],full['objective'],rtol=0,atol=1e-5)
    bound=closed_support_bound(p.config,plan['x']);np.testing.assert_allclose(plan['recourse_costs'][-1],bound['all_defer'],rtol=0,atol=1e-6);assert max(plan['recourse_costs'])<=bound['all_defer']+1e-6
    r=dict(**item,radius=.25,plan=plan,dual_reference=full,closed_bound=bound,seconds=time.monotonic()-begin);js(path,r)
   except SolverFailure as exc:
    failure=dict(key=item['key'],error=exc.details,seconds=time.monotonic()-begin);js(failed,failure);failures.append(failure);print('FAILURE',item['key'],exc,flush=True);continue
  plan=r['plan'];rows.append(dict(key=item['key'],case=item['case'],fold=item['fold'],seed=item['seed'],issue_id=item['issue_id'],epsilon=item['epsilon'],objective=plan['objective'],booking_cost=plan['booking_cost'],nominal_total=plan['nominal_total'],no_booking=int(sum(plan['x'])==0),iterations=plan['iterations'],gap=plan['gap'],seconds=r['seconds'],plan_source=str(path),plan_sha256=sha(path)));print('PLANNED',len(rows),item['key'],round(r['seconds'],3),flush=True)
 save_csv('planning_summary.csv',rows);js(OUT/'failures.json',failures);js(OUT/'planning_complete.json',dict(status='COMPLETE' if len(rows)==240 and not failures else 'INCOMPLETE',plans=len(rows),failures=len(failures),utc=datetime.now(timezone.utc).isoformat()));assert not failures and len(rows)==240,'preserve failure; do not evaluate a selectively successful subset'

def evaluate():
 assert read(OUT/'planning_complete.json')['status']=='COMPLETE';raw=np.load(ART/'data.npz')['raw'];p=make_problem(default_config());nodes=sorted({t['node'] for t in p.config['tasks']});cache={};rows=[]
 for item in read(OUT/'inventory.json'):
  guard();r=read(OUT/'cases'/('plan_'+item['key']+'.json'));plan=r['plan'];x=plan['x'];truth=raw[item['issue_id']+1:item['issue_id']+73];lo,hi,complete=observed_options(p,truth);k=item['issue_id'],tuple(x)
  if k not in cache:
   a=solve_recourse(p,lo,x);b=a if np.array_equal(lo,hi) else solve_recourse(p,hi,x);cache[k]=a,b
  a,b=cache[k];book=booking_cost(p,x);lower=book+a['recourse_cost'];upper=book+b['recourse_cost'];assert lower<=upper+1e-6 and upper<=r['closed_bound']['upper_total']+1e-6
  record=dict(key=item['key'],case=item['case'],fold=item['fold'],seed=item['seed'],position=item['position'],issue_id=item['issue_id'],epsilon=item['epsilon'],complete=complete,known_fraction=float(np.isfinite(truth[:,nodes]).mean()),cost_lower=lower,cost_upper=upper,exact_cost=lower if complete else None,exact_completed=a['completed'] if complete else None,booking_cost=book,resources=x,optimistic_solution=a,pessimistic_solution=b)
  path=OUT/'cases'/('outcome_'+item['key']+'.json');js(path,record);rows.append({k:v for k,v in record.items() if k not in ['resources','optimistic_solution','pessimistic_solution']}|dict(objective=plan['objective'],nominal_total=plan['nominal_total'],no_booking=int(sum(x)==0),outcome_source=str(path),outcome_sha256=sha(path)))
 save_csv('new_outcomes.csv',rows);old=pd.read_csv(PARENT/'outcomes.csv',float_precision='round_trip');old=old[(old.case=='graph_rolling28')&old.policy.isin(['tv000','tv025'])].copy();assert len(old)==288;old['case']='base_'+old.policy;old['epsilon']=0.;combined=pd.concat([pd.DataFrame(rows),old],ignore_index=True);save_csv('outcomes.csv',combined)
 times=pd.DatetimeIndex(pd.to_datetime(np.load(ART/'data.npz')['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');tables,draws=make_tables(combined,times)
 for name,values in tables.items():save_csv(name+'.csv',values)
 np.savez_compressed(OUT/'bootstrap_draws.npz',**draws);parents()
 for file,h in read(OUT/'identity.json').items():assert sha(file)==h,file
 js(OUT/'evaluation_complete.json',dict(status='COMPLETE',new_plans=240,reference_plans=288,actual_resource_pairs=len(cache),utc=datetime.now(timezone.utc).isoformat()));print('COMPLETE',flush=True)

if __name__=='__main__':prepare();forecasts();plan_all();evaluate()

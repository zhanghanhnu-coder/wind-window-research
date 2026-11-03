"""Round17: finite-scenario two-stage booking, with complete-model certificates."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time
import numpy as np
import pandas as pd
import scipy
from model import ART
from two_stage_core import default_config,make_problem,weather_options,booking_cost,solve_master,solve_recourse,solve_ccg,SolverFailure
from research_crps_distribution import SEEDS,guard
OUT=ART/'two_stage_v1';PLAN=ART.parent.parent/'研究笔记/有限情景两阶段调度研究方案_2026-11-01.md'
POLICIES=['deterministic','stochastic','robust']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')

def prepare():
 guard();OUT.mkdir(exist_ok=True);f=OUT/'protocol_frozen.md';protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0]
 if f.exists():assert f.read_text(encoding='utf-8')==protocol
 else:f.write_text(protocol,encoding='utf-8')
 if not (OUT/'previous_hashes.json').exists():js('previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 files=[f,Path(__file__)]+[Path(__file__).with_name(n) for n in ['two_stage_core.py','two_stage_enumeration.py','research_crps_distribution.py','model.py']]+[ART/'data.npz',ART/'metadata.json',ART/'probability_v1/predictions.npz']
 identity={str(p):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read('identity.json')==identity
 else:js('identity.json',identity)
 if not (OUT/'recipe.json').exists():js('recipe.json',dict(config=default_config(),test_positions=list(range(0,232,10)),seeds=SEEDS,scenario_count=16,scenario_selection_seed=20261106,first_stage='resources common to every scenario',second_stage='full 72h window scenario revealed',cost_units='simulated points',scipy_version=scipy.__version__,solver_time_limit=60,mip_rel_gap=0,start_utc=datetime.now(timezone.utc).isoformat()))

def inputs():
 raw=np.load(ART/'data.npz');old=np.load(ART/'probability_v1/predictions.npz');positions=np.arange(0,len(raw['test_ids']),10);assert len(raw['test_ids'])==232 and len(positions)==24;complete=np.isfinite(old['cal_truth']).all((1,2));np.testing.assert_array_equal(old['cal_ids'][complete],raw['cal_ids']);pick32=np.random.default_rng(20241101).choice(58,32,replace=False);pick16=np.random.default_rng(20261106).choice(32,16,replace=False);pack=dict(positions=positions,issue_ids=raw['test_ids'][positions],timestamps=raw['timestamps'][raw['test_ids'][positions]],truth=old['test_truth'][positions],selected32_positions=pick16,source_cal_ids=raw['cal_ids'][pick32[pick16]])
 for seed in SEEDS:
  residual=(old['cal_truth']-old[f'cal_q_{seed}'][...,1])[complete];med=old[f'test_q_{seed}'][positions,...,1];pack[f'scenarios_{seed}']=np.maximum(0,med[:,None]+residual[pick32[pick16]][None])
 np.savez_compressed(OUT/'input_data.npz',**pack);return pack

def plan_case(p,sc,key):
 available=weather_options(p,sc);mean_available=weather_options(p,sc.mean(0)[None]);plans={}
 try:
  for policy,a,mode in [('deterministic',mean_available,'stochastic'),('stochastic',available,'stochastic')]:
   guard();plans[policy]=solve_master(p,a,mode)
  plans['robust']=solve_ccg(p,available,guard=guard,progress=lambda row:js(f'progress_{key}.json',row));guard();full=solve_master(p,available,'robust')
 except SolverFailure as exc:
  js(f'partial_plans_{key}.json',plans);raise
 if abs(plans['robust']['objective']-full['objective'])>1e-6:
  js(f'objective_mismatch_{key}.json',dict(plans=plans,full=full));raise RuntimeError('C&CG and complete robust objective mismatch')
 # Worst-case master recourses need not minimize nonbinding scenario costs.
 for policy in POLICIES:
  x=plans[policy]['x'];solutions=plans[policy]['scenario_solutions'] if policy=='robust' else []
  if policy!='robust':
   for a in available:guard();solutions.append(solve_recourse(p,a,x))
  costs=np.array([s['recourse_cost'] for s in solutions]);plans[policy]['evaluated_scenario_solutions']=solutions;plans[policy]['forecast_expected_cost']=booking_cost(p,x)+float(costs.mean());plans[policy]['forecast_worst_cost']=booking_cost(p,x)+float(costs.max())
 return dict(plans=plans,full_robust=full,no_task_window_fraction=float(np.mean(~available.any(1))),choice_count=len(p.choices))

def observed_outcomes(p,planned,truth):
 # Only called after all first-stage decisions and complete-model check are saved in memory.
 nodes=sorted({t['node'] for t in p.config['tasks']});complete=bool(np.isfinite(truth[:,nodes]).all());lower=weather_options(p,truth[None],unknown='optimistic')[0];upper=weather_options(p,truth[None],unknown='pessimistic')[0];out={}
 for policy in POLICIES:
  guard();x=planned['plans'][policy]['x'];lo=solve_recourse(p,lower,x);hi=lo if np.array_equal(lower,upper) else solve_recourse(p,upper,x);book=booking_cost(p,x);assert lo['recourse_cost']<=hi['recourse_cost']+1e-6
  out[policy]=dict(complete=complete,known_fraction=float(np.isfinite(truth[:,nodes]).mean()),cost_lower=book+lo['recourse_cost'],cost_upper=book+hi['recourse_cost'],exact_cost=book+lo['recourse_cost'] if complete else None,exact_completed=lo['completed'] if complete else None,optimistic_solution=lo,pessimistic_solution=hi)
 return out

def experiments(pack):
 p=make_problem(default_config());cases=[];failures=[]
 for seed in SEEDS:
  for index,position in enumerate(pack['positions']):
   guard();key=f'{seed}_{int(position):03d}';path=OUT/f'case_{key}.json'
   if path.exists():cases.append(read(path.name));print('resume',key,flush=True);continue
   begin=time.monotonic();print('START',key,flush=True)
   try:
    planned=plan_case(p,pack[f'scenarios_{seed}'][index],key);outcomes=observed_outcomes(p,planned,pack['truth'][index]);case=dict(seed=seed,position=int(position),input_index=index,issue_id=int(pack['issue_ids'][index]),timestamp=str(pack['timestamps'][index]),planned=planned,observed=outcomes,seconds=time.monotonic()-begin);js(path.name,case);cases.append(case)
    print('DONE',key,'seconds',round(case['seconds'],2),'iterations',planned['plans']['robust']['iterations'],'book',planned['plans']['robust']['booking_cost'],'no-window',planned['no_task_window_fraction'],flush=True)
   except SolverFailure as exc:
    failed=dict(seed=seed,position=int(position),error=exc.details,seconds=time.monotonic()-begin);js(f'failure_{key}.json',failed);failures.append(failed);print('SOLVER_FAILURE',key,exc,flush=True)
 js('failures.json',failures);return cases

def summarize(cases,pack):
 rows=[];certificates=[];traces=[]
 for case in cases:
  planned=case['planned'];r=planned['plans']['robust'];full=planned['full_robust'];common=dict(seed=case['seed'],position=case['position'],issue_id=case['issue_id'],timestamp=case['timestamp']);certificates.append(dict(**common,ccg_objective=r['objective'],full_objective=full['objective'],difference=r['objective']-full['objective'],iterations=r['iterations'],active_count=len(r['active_scenarios']),oracle_solves=r['oracle_solves'],ccg_seconds=r['seconds'],full_seconds=full['seconds'],ccg_lower=r['lower_bound'],ccg_upper=r['upper_bound'],full_gap=full['mip_gap']))
  for tr in r['trace']:traces.append(dict(**common,**tr))
  for policy in POLICIES:
   plan=planned['plans'][policy];obs=case['observed'][policy];rows.append(dict(**common,policy=policy,optimization_objective=plan['objective'],booking_cost=plan['booking_cost'],forecast_expected_cost=plan['forecast_expected_cost'],forecast_worst_cost=plan['forecast_worst_cost'],actual_cost=obs['exact_cost'],actual_completed=obs['exact_completed'],observed_complete=obs['complete'],observed_known_fraction=obs['known_fraction'],cost_lower=obs['cost_lower'],cost_upper=obs['cost_upper'],no_booking=int(not any(plan['x'])),no_task_window_fraction=planned['no_task_window_fraction'],resources=json.dumps(plan['x'])))
 csv('cases.csv',rows);csv('certificates.csv',certificates);csv('ccg_trace.csv',traces);frame=pd.DataFrame(rows);summary=[];frame['month']=pd.to_datetime(frame.timestamp,utc=True).dt.tz_convert('Asia/Shanghai').dt.month
 for seed in SEEDS:
  for policy in POLICIES:
   for month in [0,11,12]:
    f=frame[(frame.seed==seed)&(frame.policy==policy)];f=f if month==0 else f[f.month==month]
    for metric in ['booking_cost','forecast_expected_cost','forecast_worst_cost','actual_cost','actual_completed','cost_lower','cost_upper','no_booking','no_task_window_fraction']:
     value=f[metric].mean();summary.append(dict(seed=seed,policy=policy,month=month,metric=metric,value=float(value) if pd.notna(value) else np.nan,n=int(f[metric].notna().sum())))
 csv('metrics.csv',summary)
 # Pair by the same origin, requiring all three seeds and all policies.
 eligible=[]
 for position in pack['positions']:
  f=frame[frame.position==position]
  if len(f)==9 and f.groupby('policy').seed.nunique().eq(3).all():eligible.append(int(position))
 all_times=pd.DatetimeIndex(pd.to_datetime(pack['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');allweeks=((all_times-pd.Timestamp('2024-11-01',tz=all_times.tz)).days//7).to_numpy();keys=np.unique(allweeks);weekmap=dict(zip(pack['positions'].tolist(),allweeks.tolist()));draw=np.random.default_rng(20261106).integers(0,len(keys),(2000,len(keys)));pairs=[];bounds=[]
 for baseline in ['stochastic','deterministic']:
  differences=[];position_list=[];lows=[];highs=[]
  for position in eligible:
   f=frame[frame.position==position];r=f[f.policy=='robust'];b=f[f.policy==baseline];lows.append(r.cost_lower.mean()-b.cost_upper.mean());highs.append(r.cost_upper.mean()-b.cost_lower.mean())
   if r.actual_cost.notna().all() and b.actual_cost.notna().all():differences.append(r.actual_cost.mean()-b.actual_cost.mean());position_list.append(position)
  delta=np.array(differences);weeks=np.array([weekmap[p] for p in position_list]);counts=np.array([(weeks==k).sum() for k in keys]);sums=np.array([delta[weeks==k].sum() for k in keys]);den=counts[draw].sum(1);boot=np.divide(sums[draw].sum(1),den,out=np.full(len(draw),np.nan),where=den>0);valid=np.isfinite(boot);enough=len(delta)>=8 and (counts>0).sum()>=4 and valid.sum()>=1900;lo,hi=np.quantile(boot[valid],[.025,.975]) if enough else (np.nan,np.nan)
  pairs.append(dict(candidate='robust',baseline=baseline,difference=float(delta.mean()) if len(delta) else np.nan,low95=lo,high95=hi,complete_origins=len(delta),nonempty_weeks=int((counts>0).sum()),all_weeks=len(keys),valid_bootstrap=int(valid.sum()),interval_reported=bool(enough),primary=True));bounds.append(dict(candidate='robust',baseline=baseline,mean_difference_lower=float(np.mean(lows)) if lows else np.nan,mean_difference_upper=float(np.mean(highs)) if highs else np.nan,origins=len(eligible),interpretation='bounds from unknown historical weather, not confidence intervals'))
 csv('paired_comparisons.csv',pairs);csv('missing_observation_bounds.csv',bounds);js('study_counts.json',dict(planned_origins=24,planned_seed_cases=72,completed_seed_cases=len(cases),complete_origins_for_all_seeds=len(position_list),policies=3,scenarios=16,all_weeks=len(keys),eligible_all_seed_origins=len(eligible)))

if __name__=='__main__':
 prepare();pack=inputs();cases=experiments(pack);summarize(cases,pack);print('COMPLETE',flush=True)

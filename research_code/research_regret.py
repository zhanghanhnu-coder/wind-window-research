"""Round18: minimax regret with preregistered booking-cost sensitivity."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,time
import numpy as np
import pandas as pd
import scipy
from model import ART
from two_stage_core import default_config,make_problem,weather_options,booking_cost,SolverFailure
from regret_core import solve_master,solve_recourse,solve_ccg,perfect_information
from research_crps_distribution import SEEDS,guard
OLD=ART/'two_stage_v1';OUT=ART/'regret_v1';PLAN=ART.parent.parent/'研究笔记/最坏机会损失与成本敏感性研究方案_2026-11-02.md'
POLICIES=['deterministic','stochastic','robust','regret'];LEVELS=[1.,.5,2.]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def config(level):
 c=default_config()
 for key in ['boat_booking','crew_booking','spare_cost']:c[key]=[v*level for v in c[key]]
 return c
def key_for(level,seed,position):return f'{level:g}_{seed}_{position:03d}'
def prepare():
 guard();OUT.mkdir(exist_ok=True);f=OUT/'protocol_frozen.md';text=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0]
 if f.exists():assert f.read_text(encoding='utf-8')==text
 else:f.write_text(text,encoding='utf-8')
 if not (OUT/'previous_hashes.json').exists():js('previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 files=[f,Path(__file__)]+[Path(__file__).with_name(n) for n in ['regret_core.py','two_stage_core.py','two_stage_enumeration.py','research_crps_distribution.py','model.py']]+[OLD/'input_data.npz',OLD/'recipe.json',OLD/'final_manifest.json']+list(sorted(OLD.glob('case_*.json')))
 identities={str(p):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==identities
 else:js('identity.json',identities)
 if not (OUT/'recipe.json').exists():js('recipe.json',dict(configs={str(v):config(v) for v in LEVELS},levels=LEVELS,base_seeds=SEEDS,sensitivity_seeds=[SEEDS[0]],test_positions=list(range(0,232,10)),scenarios=16,planned_cases=120,perfect_information='same resource feasible set and total costs, separately optimized per predicted scenario',realized_perfect_information='evaluation only after all plans saved; complete task-node observations only',cost_units='simulated points',scipy_version=scipy.__version__,time_limit=60,mip_rel_gap=0,start_utc=datetime.now(timezone.utc).isoformat()))

def plan_case(p,sc,level,seed,position,key):
 a=weather_options(p,sc);benchmark=perfect_information(p,a,guard=guard);v=np.asarray(benchmark['values']);plans={};source=None
 if level==1:
  source=OLD/f'case_{seed}_{position:03d}.json';original=read(source)
  plans={k:copy.deepcopy(original['planned']['plans'][k]) for k in POLICIES[:-1]}
 else:
  plans['deterministic']=solve_master(p,weather_options(p,sc.mean(0)[None]),'stochastic');guard();plans['stochastic']=solve_master(p,a,'stochastic');guard();plans['robust']=solve_ccg(p,a,guard=guard)
 plans['regret']=solve_ccg(p,a,offsets=v,guard=guard,progress=lambda row:js(f'progress_{key}.json',row));guard();full=solve_master(p,a,offsets=v)
 if abs(full['objective']-plans['regret']['objective'])>1e-6:raise SolverFailure(4,'regret full model and CCG mismatch')
 for policy in POLICIES:
  plan=plans[policy];x=plan['x']
  if level==1 and policy!='regret':solutions=plan['evaluated_scenario_solutions']
  elif policy in ['robust','regret']:solutions=plan['scenario_solutions']
  else:
   solutions=[]
   for row in a:guard();solutions.append(solve_recourse(p,row,x))
  q=np.array([r['recourse_cost'] for r in solutions]);total=booking_cost(p,x)+q;regrets=total-v
  if regrets.min()<-1e-6:raise SolverFailure(4,'policy cost below perfect information benchmark')
  plan.update(evaluated_scenario_solutions=solutions,forecast_expected_cost=float(total.mean()),forecast_worst_cost=float(total.max()),forecast_mean_regret=float(regrets.mean()),forecast_worst_regret=float(regrets.max()),scenario_regrets=regrets.tolist())
 for policy in POLICIES[:-1]:assert plans['regret']['forecast_worst_regret']<=plans[policy]['forecast_worst_regret']+1e-6
 assert abs(plans['regret']['objective']-plans['regret']['forecast_worst_regret'])<1e-6
 return dict(plans=plans,benchmark=benchmark,full_regret=full,no_task_window_fraction=float((~a.any(1)).mean()),source_round17=str(source) if source else None)

def plan_all(pack):
 cases=[];failures=[]
 for level in LEVELS:
  p=make_problem(config(level))
  for seed in SEEDS if level==1 else [SEEDS[0]]:
   for idx,pos in enumerate(pack['positions']):
    guard();pos=int(pos);key=key_for(level,seed,pos);path=OUT/f'planned_{key}.json'
    if path.exists():cases.append(read(path));print('RESUME PLAN',key,flush=True);continue
    begin=time.monotonic();print('PLAN',key,flush=True)
    try:
     planned=plan_case(p,pack[f'scenarios_{seed}'][idx],level,seed,pos,key);case=dict(level=level,seed=seed,position=pos,input_index=idx,issue_id=int(pack['issue_ids'][idx]),timestamp=str(pack['timestamps'][idx]),planned=planned,planning_seconds=time.monotonic()-begin);js(path.name,case);cases.append(case);r=planned['plans']['regret'];print('PLANNED',key,'seconds',round(case['planning_seconds'],2),'iterations',r['iterations'],'book',r['booking_cost'],'regret',r['objective'],flush=True)
    except SolverFailure as exc:
     failed=dict(level=level,seed=seed,position=pos,error=exc.details);js(f'failure_{key}.json',failed);failures.append(failed);print('FAIL',key,exc,flush=True)
 js('failures.json',failures);js('planning_complete.json',dict(completed=len(cases),planned=120,failures=len(failures),utc=datetime.now(timezone.utc).isoformat()));return cases

def evaluate_all(cases,pack):
 # Entire planning pass is complete before any realized-weather benchmark is optimized.
 results=[]
 for original in cases:
  guard();key=key_for(original['level'],original['seed'],original['position']);path=OUT/f'case_{key}.json'
  if path.exists():results.append(read(path));continue
  case=copy.deepcopy(original);p=make_problem(config(case['level']));truth=pack['truth'][case['input_index']];nodes=sorted({t['node'] for t in p.config['tasks']});known=bool(np.isfinite(truth[:,nodes]).all());lo=weather_options(p,truth[None],unknown='optimistic')[0];hi=weather_options(p,truth[None],unknown='pessimistic')[0];benchmark=solve_master(p,lo[None],'stochastic') if known else None;out={}
  for policy in POLICIES:
   guard();x=case['planned']['plans'][policy]['x'];a=solve_recourse(p,lo,x);b=a if np.array_equal(lo,hi) else solve_recourse(p,hi,x);book=booking_cost(p,x);lower=book+a['recourse_cost'];upper=book+b['recourse_cost'];assert lower<=upper+1e-6
   out[policy]=dict(complete=known,known_fraction=float(np.isfinite(truth[:,nodes]).mean()),cost_lower=lower,cost_upper=upper,exact_cost=lower if known else None,exact_completed=a['completed'] if known else None,exact_regret=lower-benchmark['objective'] if known else None,optimistic_solution=a,pessimistic_solution=b)
  case['observed']=out;case['realized_benchmark']=benchmark;js(path.name,case);results.append(case);print('EVALUATED',key,flush=True)
 return results

def summarize(cases,pack):
 rows=[];cert=[];traces=[];bench=[]
 for case in cases:
  common={k:case[k] for k in ['level','seed','position','issue_id','timestamp']};planned=case['planned'];r=planned['plans']['regret'];full=planned['full_regret'];cert.append(dict(**common,ccg_objective=r['objective'],full_objective=full['objective'],difference=r['objective']-full['objective'],iterations=r['iterations'],active_count=len(r['active_scenarios']),ccg_lower=r['lower_bound'],ccg_upper=r['upper_bound'],ccg_seconds=r['seconds'],full_seconds=full['seconds'],full_gap=full['mip_gap']))
  for tr in r['trace']:traces.append(dict(**common,**tr))
  for s,b in enumerate(planned['benchmark']['solutions']):bench.append(dict(**common,scenario=s,perfect_cost=b['objective'],booking_cost=b['booking_cost'],resources=json.dumps(b['x']),gap=b['mip_gap']))
  for policy in POLICIES:
   plan=planned['plans'][policy];o=case['observed'][policy];rows.append(dict(**common,policy=policy,optimization_objective=plan['objective'],booking_cost=plan['booking_cost'],forecast_expected_cost=plan['forecast_expected_cost'],forecast_worst_cost=plan['forecast_worst_cost'],forecast_mean_regret=plan['forecast_mean_regret'],forecast_worst_regret=plan['forecast_worst_regret'],actual_cost=o['exact_cost'],actual_completed=o['exact_completed'],actual_regret=o['exact_regret'],observed_complete=o['complete'],known_fraction=o['known_fraction'],cost_lower=o['cost_lower'],cost_upper=o['cost_upper'],no_booking=int(not any(plan['x'])),resources=json.dumps(plan['x']),no_task_window_fraction=planned['no_task_window_fraction']))
 csv('cases.csv',rows);csv('certificates.csv',cert);csv('ccg_trace.csv',traces);csv('benchmarks.csv',bench);f=pd.DataFrame(rows);metrics=[]
 for (level,seed,policy),g in f.groupby(['level','seed','policy']):
  for metric in ['booking_cost','forecast_expected_cost','forecast_worst_cost','forecast_mean_regret','forecast_worst_regret','actual_cost','actual_completed','actual_regret','cost_lower','cost_upper','no_booking']:
   metrics.append(dict(level=level,seed=seed,policy=policy,metric=metric,value=g[metric].mean(),n=int(g[metric].notna().sum())))
 csv('metrics.csv',metrics);times=pd.DatetimeIndex(pd.to_datetime(pack['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');weeks=((times-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();keys=np.unique(weeks);wm=dict(zip(pack['positions'],weeks));draw=np.random.default_rng(20261106).integers(0,len(keys),(2000,len(keys)));pairs=[];bounds=[]
 for level in LEVELS:
  sub=f[f.level==level];seeds=SEEDS if level==1 else [SEEDS[0]];eligible=[]
  for pos in pack['positions']:
   g=sub[sub.position==pos]
   if len(g)==len(seeds)*4 and set(zip(g.policy,g.seed))=={(p,s) for p in POLICIES for s in seeds}:eligible.append(pos)
  for baseline in POLICIES[:-1]:
   delta=[];ww=[];lower=[];upper=[]
   for pos in eligible:
    g=sub[sub.position==pos];a=g[g.policy=='regret'];b=g[g.policy==baseline];lower.append(a.cost_lower.mean()-b.cost_upper.mean());upper.append(a.cost_upper.mean()-b.cost_lower.mean())
    if a.actual_cost.notna().all() and b.actual_cost.notna().all():delta.append(a.actual_cost.mean()-b.actual_cost.mean());ww.append(wm[pos])
   delta=np.array(delta);ww=np.array(ww);n=np.array([(ww==k).sum() for k in keys]);sums=np.array([delta[ww==k].sum() for k in keys]);den=n[draw].sum(1);valid=den>0;boot=sums[draw][valid].sum(1)/den[valid];enough=len(delta)>=8 and sum(n>0)>=4 and sum(valid)>=1900;lo,hi=np.quantile(boot,[.025,.975]) if enough else (np.nan,np.nan)
   pairs.append(dict(level=level,candidate='regret',baseline=baseline,difference=float(delta.mean()) if len(delta) else np.nan,low95=lo,high95=hi,complete_origins=len(delta),nonempty_weeks=int(sum(n>0)),all_weeks=len(keys),valid_bootstrap=int(sum(valid)),interval_reported=bool(enough),seeds=len(seeds),primary=level==1))
   bounds.append(dict(level=level,candidate='regret',baseline=baseline,mean_difference_lower=float(np.mean(lower)) if lower else np.nan,mean_difference_upper=float(np.mean(upper)) if upper else np.nan,origins=len(eligible),interpretation='unknown-observation bounds, not confidence interval'))
 csv('paired_comparisons.csv',pairs);csv('missing_observation_bounds.csv',bounds);js('study_counts.json',dict(planned_cases=120,completed_cases=len(cases),policy_rows=len(rows),benchmark_rows=len(bench),certificates=len(cert),levels=LEVELS))

if __name__=='__main__':
 prepare();pack=np.load(OLD/'input_data.npz');plans=plan_all(pack);cases=evaluate_all(plans,pack);summarize(cases,pack);print('COMPLETE',flush=True)

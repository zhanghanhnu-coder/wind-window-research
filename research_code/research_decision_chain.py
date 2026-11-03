"""Round20 preregistered forecast-family/resource-decision comparison."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,time
import numpy as np
import pandas as pd
from model import ART
from decision_chain_core import ScenarioBank,FAMILIES,POLICIES,plan_scenarios
from two_stage_core import default_config,make_problem,weather_options,booking_cost,SolverFailure
from regret_core import solve_master,solve_recourse
from probability_core import score_scenarios
from research_crps_distribution import SEEDS,guard
OUT=ART/'decision_chain_v1';PLAN=ART.parent.parent/'研究笔记/联合预测到资源决策对照方案_2026-11-02.md';POSITIONS=list(range(0,232,10))
PAIRS=[('gated_graph57','old57',True),('gated_graph57','gated_temporal57',True),('old57_time_broken','old57',False),('old57_space_broken','old57',False)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def key_for(family,seed,pos):return f'{family}_{seed}_{pos:03d}'

def prepare():
 guard();OUT.mkdir(exist_ok=True);f=OUT/'protocol_frozen.md';protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0]
 if f.exists():assert f.read_text(encoding='utf-8')==protocol
 else:f.write_text(protocol,encoding='utf-8')
 if not (OUT/'previous_hashes.json').exists():js('previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 files=[f,Path(__file__)]+[Path(__file__).with_name(n) for n in ['decision_chain_core.py','regret_core.py','two_stage_core.py','probability_core.py','crps_distribution_core.py','lead_calibration_core.py','research_crps_distribution.py','model.py']]+[ART/'metadata.json',ART/'probability_v1/predictions.npz',ART/'crps_distribution_v1/evaluation_data.npz',ART/'two_stage_v1/input_data.npz']
 for seed in SEEDS:
  for model in ['gated_graph_atoms','gated_temporal_atoms']:files.extend([ART/'crps_distribution_v1'/f'predictions_{model}_{seed}_budget40.npz',ART/'crps_distribution_v1'/f'bank_{model}_{seed}.npz',ART/'crps_lead_calibration_v1'/f'parameters_{model}_{seed}.npz'])
 identity={str(p):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==identity
 else:js('identity.json',identity)
 if not (OUT/'recipe.json').exists():js('recipe.json',dict(families=FAMILIES,policies=POLICIES,seeds=SEEDS,positions=POSITIONS,scenario_count=57,planned_cases=360,config=default_config(),solver_time_limit=60,failure_stop=10,start_utc=datetime.now(timezone.utc).isoformat()))

def inputs():
 bank=ScenarioBank();old=np.load(ART/'two_stage_v1/input_data.npz');np.testing.assert_array_equal(old['positions'],POSITIONS);np.testing.assert_array_equal(old['issue_ids'],bank.test_ids[POSITIONS]);info=dict(positions=np.array(POSITIONS),issue_ids=bank.test_ids[POSITIONS],timestamps=old['timestamps'],source_cal_ids=bank.source_ids)
 for seed in SEEDS:
  for family in FAMILIES:
   guard();path=OUT/f'scenarios_{family}_{seed}.npz'
   if not path.exists():np.savez_compressed(path,scenarios=np.stack([bank.get(family,seed,pos) for pos in POSITIONS]),**info)
 if not (OUT/'input_identity.json').exists():js('input_identity.json',{str(p):sha(p) for p in sorted(OUT.glob('scenarios_*.npz'))})
 else:
  for p,h in read(OUT/'input_identity.json').items():assert sha(Path(p))==h
 return info

def plan_all():
 p=make_problem(default_config());cases=[];failures=[];stopped=False
 for family in FAMILIES:
  for seed in SEEDS:
   pack=np.load(OUT/f'scenarios_{family}_{seed}.npz')
   for idx,pos in enumerate(POSITIONS):
    guard();key=key_for(family,seed,pos);path=OUT/f'planned_{key}.json';failed=OUT/f'failure_{key}.json'
    if path.exists():cases.append(read(path));continue
    if failed.exists():failures.append(read(failed))
    elif len(failures)<10:
     begin=time.monotonic();print('PLAN',key,flush=True)
     try:
      planned=plan_scenarios(p,pack['scenarios'][idx],guard=guard,progress=lambda row:js(f'progress_{key}.json',row),partial=lambda obj:js(f'partial_{key}.json',obj));case=dict(family=family,seed=seed,position=pos,input_index=idx,issue_id=int(pack['issue_ids'][idx]),timestamp=str(pack['timestamps'][idx]),planned=planned,planning_seconds=time.monotonic()-begin);js(path.name,case);cases.append(case);print('PLANNED',key,'seconds',round(case['planning_seconds'],2),'regret',planned['plans']['regret']['objective'],flush=True)
     except SolverFailure as exc:
      failure=dict(family=family,seed=seed,position=pos,error=exc.details,seconds=time.monotonic()-begin);js(failed.name,failure);failures.append(failure);print('FAIL',key,exc,flush=True)
    if len(failures)>=10:stopped=True;break
   if stopped:break
  if stopped:break
 js('failures.json',failures);js('planning_complete.json',dict(completed=len(cases),planned=360,failures=len(failures),stopped_after_failures=stopped,utc=datetime.now(timezone.utc).isoformat()));return cases

def evaluate_all(cases):
 assert (OUT/'planning_complete.json').exists();pack=np.load(ART/'two_stage_v1/input_data.npz');p=make_problem(default_config());nodes=sorted({t['node'] for t in p.config['tasks']});benchmarks={};results=[]
 for original in cases:
  guard();key=key_for(original['family'],original['seed'],original['position']);path=OUT/f'case_{key}.json'
  if path.exists():results.append(read(path));continue
  case=copy.deepcopy(original);idx=case['input_index'];truth=pack['truth'][idx];known=bool(np.isfinite(truth[:,nodes]).all());lo=weather_options(p,truth[None],unknown='optimistic')[0];hi=weather_options(p,truth[None],unknown='pessimistic')[0]
  if known and idx not in benchmarks:benchmarks[idx]=solve_master(p,lo[None],'stochastic')
  benchmark=benchmarks.get(idx) if known else None;observed={}
  for policy in POLICIES:
   guard();x=case['planned']['plans'][policy]['x'];a=solve_recourse(p,lo,x);b=a if np.array_equal(lo,hi) else solve_recourse(p,hi,x);book=booking_cost(p,x);lower=book+a['recourse_cost'];upper=book+b['recourse_cost'];assert lower<=upper+1e-6
   observed[policy]=dict(complete=known,known_fraction=float(np.isfinite(truth[:,nodes]).mean()),cost_lower=lower,cost_upper=upper,exact_cost=lower if known else None,exact_completed=a['completed'] if known else None,exact_regret=lower-benchmark['objective'] if known else None,optimistic_solution=a,pessimistic_solution=b)
  case.update(observed=observed,realized_benchmark=benchmark);js(path.name,case);results.append(case);print('EVALUATED',key,flush=True)
 return results

def summarize(cases,info):
 rows=[];cert=[];trace=[];bench=[]
 for case in cases:
  common={k:case[k] for k in ['family','seed','position','issue_id','timestamp']};planned=case['planned'];r=planned['plans']['regret'];full=planned['full_regret'];cert.append(dict(**common,ccg_objective=r['objective'],full_objective=full['objective'],difference=r['objective']-full['objective'],iterations=r['iterations'],lower_bound=r['lower_bound'],upper_bound=r['upper_bound'],ccg_seconds=r['seconds'],full_seconds=full['seconds']))
  for tr in r['trace']:trace.append(dict(**common,**tr))
  for s,b in enumerate(planned['benchmark']['solutions']):bench.append(dict(**common,scenario=s,perfect_cost=b['objective'],resources=json.dumps(b['x']),solver_status=b['solver_status'],gap=b['mip_gap']))
  for policy in POLICIES:
   plan=planned['plans'][policy];o=case['observed'][policy];rows.append(dict(**common,policy=policy,objective=plan['objective'],booking_cost=plan['booking_cost'],forecast_expected_cost=plan['forecast_expected_cost'],forecast_worst_cost=plan['forecast_worst_cost'],forecast_worst_regret=plan['forecast_worst_regret'],actual_cost=o['exact_cost'],actual_completed=o['exact_completed'],actual_regret=o['exact_regret'],observed_complete=o['complete'],known_fraction=o['known_fraction'],cost_lower=o['cost_lower'],cost_upper=o['cost_upper'],no_booking=int(not any(plan['x'])),resources=json.dumps(plan['x']),no_task_window_fraction=planned['no_task_window_fraction']))
 csv('cases.csv',rows);csv('certificates.csv',cert);csv('ccg_trace.csv',trace);csv('benchmarks.csv',bench);f=pd.DataFrame(rows);metrics=[]
 for (family,seed,policy),g in f.groupby(['family','seed','policy']):
  for metric in ['booking_cost','forecast_expected_cost','forecast_worst_cost','forecast_worst_regret','actual_cost','actual_completed','actual_regret','cost_lower','cost_upper','no_booking','no_task_window_fraction']:metrics.append(dict(family=family,seed=seed,policy=policy,metric=metric,value=g[metric].mean(),n=int(g[metric].notna().sum())))
 csv('metrics.csv',metrics);times=pd.DatetimeIndex(pd.to_datetime(info['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');weeks=((times-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();keys=np.unique(weeks);wm=dict(zip(POSITIONS,weeks));draw=np.random.default_rng(20261106).integers(0,len(keys),(2000,len(keys)));pairs=[];bounds=[]
 for candidate,baseline,primary in PAIRS:
  for policy in ['stochastic','regret']:
   sub=f[(f.policy==policy)&f.family.isin([candidate,baseline])];delta=[];ww=[];lower=[];upper=[]
   for pos in POSITIONS:
    g=sub[sub.position==pos]
    if set(zip(g.family,g.seed))!={(family,s) for family in [candidate,baseline] for s in SEEDS}:continue
    a=g[g.family==candidate];b=g[g.family==baseline];lower.append(a.cost_lower.mean()-b.cost_upper.mean());upper.append(a.cost_upper.mean()-b.cost_lower.mean())
    if a.actual_cost.notna().all() and b.actual_cost.notna().all():delta.append(a.actual_cost.mean()-b.actual_cost.mean());ww.append(wm[pos])
   delta=np.array(delta);ww=np.array(ww);n=np.array([(ww==k).sum() for k in keys]);sums=np.array([delta[ww==k].sum() for k in keys]);den=n[draw].sum(1);valid=den>0;boot=sums[draw][valid].sum(1)/den[valid];enough=len(delta)>=8 and sum(n>0)>=4 and sum(valid)>=1900;lo,hi=np.quantile(boot,[.025,.975]) if enough else (np.nan,np.nan)
   pairs.append(dict(candidate=candidate,baseline=baseline,policy=policy,primary=primary,difference=float(delta.mean()) if len(delta) else np.nan,low95=lo,high95=hi,complete_origins=len(delta),nonempty_weeks=int(sum(n>0)),all_weeks=len(keys),valid_bootstrap=int(sum(valid)),interval_reported=bool(enough)))
   bounds.append(dict(candidate=candidate,baseline=baseline,policy=policy,mean_difference_lower=float(np.mean(lower)) if lower else np.nan,mean_difference_upper=float(np.mean(upper)) if upper else np.nan,origins=len(lower),interpretation='observation bounds, not confidence interval'))
 csv('paired_comparisons.csv',pairs);csv('missing_observation_bounds.csv',bounds);js('study_counts.json',dict(completed_cases=len(cases),planned_cases=360,policy_rows=len(rows),certificates=len(cert),benchmark_rows=len(bench)))

def prediction_scores():
 assert (OUT/'planning_complete.json').exists();truth=np.load(ART/'two_stage_v1/input_data.npz')['truth'];meta=read(ART/'metadata.json');rows=[];metrics=['crps','brier6','length_crps72','group_brier','temporal_variogram','spatial_variogram']
 for family in FAMILIES:
  for seed in SEEDS:
   guard();dest=OUT/f'scores_{family}_{seed}.npz'
   if not dest.exists():
    paths=np.load(OUT/f'scenarios_{family}_{seed}.npz')['scenarios'];scores=[score_scenarios(sc,y,train_scale=meta['scale']) for sc,y in zip(paths,truth)];np.savez_compressed(dest,**{k:np.stack([s[k] for s in scores]) for k in scores[0]})
   data=np.load(dest)
   if family in FAMILIES[:3]:
    model='old_graph' if family=='old57' else ('gated_graph_atoms' if family=='gated_graph57' else 'gated_temporal_atoms');method='old57' if family=='old57' else 'smooth_affine';frozen=np.load(ART/'crps_lead_calibration_v1'/f'scores_{model}__{method}__{seed}.npz')
    for metric in metrics:np.testing.assert_allclose(data[metric],frozen[metric][POSITIONS],atol=1e-12,rtol=0,equal_nan=True)
   for metric in metrics:rows.append(dict(family=family,seed=seed,metric=metric,value=float(np.nanmean(data[metric])),n=int(np.isfinite(data[metric]).sum())))
 csv('prediction_metrics.csv',rows)

if __name__=='__main__':
 prepare();info=inputs();plans=plan_all();cases=evaluate_all(plans);summarize(cases,info);prediction_scores();print('COMPLETE',flush=True)

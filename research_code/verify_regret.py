"""Independent accounting, perfect-information benchmarks, and table checks."""
from pathlib import Path
from itertools import product
import hashlib,json
import numpy as np
import pandas as pd
from model import ART
from two_stage_core import make_problem,weather_options,solve_master as original_master,solve_recourse as original_recourse
from two_stage_enumeration import brute_recourse
from regret_core import solve_master,solve_ccg
from verify_two_stage import independent_schedule,tiny_config
OUT=ART/'regret_v1';OLD=ART/'two_stage_v1';POLICIES=['deterministic','stochastic','robust','regret'];SEEDS=[20241101,20241102,20241103]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-6):assert np.isclose(a,b,atol=tol,rtol=0),(a,b)
def run():
 old=read(OUT/'previous_hashes.json');identity=read(OUT/'identity.json')
 for n,h in old.items():assert sha(ART/n)==h,n
 for n,h in identity.items():assert sha(Path(n))==h,n
 recipe=read(OUT/'recipe.json');pack=np.load(OLD/'input_data.npz');files=sorted(OUT.glob('case_*.json'));records={};audits=0;resolves=0;copied=0;certs=0;benchmarks=0;realized=0
 # Round17 input identity is frozen; additionally verify issue/label maturity.
 assert max(pack['source_cal_ids']+72)<min(pack['issue_ids']);np.testing.assert_array_equal(pack['positions'],np.arange(0,232,10))
 completion=read(OUT/'planning_complete.json');assert completion['completed']==len(files)
 for path in files:
  case=read(path);level=case['level'];seed=case['seed'];pos=case['position'];idx=int(np.where(pack['positions']==pos)[0][0]);key=(level,seed,pos);assert key not in records;assert level in [1.,.5,2.] and seed in (SEEDS if level==1 else SEEDS[:1]);assert idx==case['input_index'] and case['issue_id']==int(pack['issue_ids'][idx]) and case['timestamp']==str(pack['timestamps'][idx]);cfg=recipe['configs'][str(float(level))];p=make_problem(cfg);sc=pack[f'scenarios_{seed}'][idx];truth=pack['truth'][idx];a=weather_options(p,sc);planned=case['planned'];v=np.asarray(planned['benchmark']['values']);assert v.shape==(16,) and np.isfinite(v).all() and (v>=0).all();assert len(planned['benchmark']['solutions'])==16;selected=pos in [0,110,230]
  # All candidate planning is written before outcome evaluation. Check the saved planning object exactly.
  baseplan=read(OUT/f'planned_{level:g}_{seed}_{pos:03d}.json');assert baseplan['planned']==planned
  for s,b in enumerate(planned['benchmark']['solutions']):
   assert b['solver_status']==0 and b['mip_gap']<1e-9;close(b['objective'],b['dual_bound']);book,q=independent_schedule(cfg,b['x'],sc[s],b['scenario_solutions'][0]);close(book+q,v[s]);close(b['objective'],v[s]);audits+=1;benchmarks+=1
   if selected:again=original_master(p,a[s:s+1],'stochastic');close(again['objective'],v[s]);resolves+=1
  full=planned['full_regret'];r=planned['plans']['regret'];assert r['solver_status']==full['solver_status']==0 and full['mip_gap']<1e-9;close(full['objective'],full['dual_bound']);close(full['objective'],r['objective']);close(r['objective'],r['upper_bound']);assert r['upper_bound']-r['lower_bound']<=1e-7*(1+abs(r['upper_bound'])) and r['lower_bound']<=r['upper_bound']+1e-6;np.testing.assert_allclose(r['offsets'],v,atol=1e-7,rtol=0);certs+=1
  qfull=[]
  for s,sol in enumerate(full['scenario_solutions']):book,q=independent_schedule(cfg,full['x'],sc[s],sol);qfull.append(q);audits+=1
  close(book+max(np.asarray(qfull)-v),full['objective']);close(full['eta'],max(np.asarray(qfull)-v));assert full['objective']>=-1e-6
  history=r['trace'];assert len(history)==r['iterations'];lb=np.array([x['lower_bound'] for x in history]);ub=np.array([x['upper_bound'] for x in history]);assert np.all(np.diff(lb)>=-1e-6) and np.all(np.diff(ub)<=1e-6) and np.all(lb<=full['objective']+1e-6) and np.all(ub>=full['objective']-1e-6)
  for j,h in enumerate(history):
   active=h['active_scenarios'];assert active[0]==0 and len(active)==len(set(active)) and all(0<=s<16 for s in active);assert h['iteration']==j+1
   if j+1<len(history):assert history[j+1]['active_scenarios']==active+[h['worst_scenario']]
  nodes=sorted({t['node'] for t in cfg['tasks']});known=bool(np.isfinite(truth[:,nodes]).all());actualbenchmark=case['realized_benchmark']
  if known:
   assert actualbenchmark['solver_status']==0 and actualbenchmark['mip_gap']<1e-9;close(actualbenchmark['objective'],actualbenchmark['dual_bound']);book,q=independent_schedule(cfg,actualbenchmark['x'],truth,actualbenchmark['scenario_solutions'][0]);close(book+q,actualbenchmark['objective']);audits+=1;realized+=1
   if selected:again=original_master(p,weather_options(p,truth[None],unknown='optimistic'),'stochastic');close(again['objective'],actualbenchmark['objective']);resolves+=1
  else:assert actualbenchmark is None
  for policy in POLICIES:
   plan=planned['plans'][policy];assert plan['solver_status']==0;forecast=[];solutions=plan['evaluated_scenario_solutions'];assert len(solutions)==16
   for s,sol in enumerate(solutions):
    book,q=independent_schedule(cfg,plan['x'],sc[s],sol);forecast.append(book+q);audits+=1
    if selected:again=original_recourse(p,a[s],plan['x']);close(q,again['recourse_cost']);resolves+=1
   forecast=np.array(forecast);reg=forecast-v;assert reg.min()>=-1e-6;np.testing.assert_allclose(plan['scenario_regrets'],reg,atol=1e-6,rtol=0);close(plan['booking_cost'],book);close(plan['forecast_expected_cost'],forecast.mean());close(plan['forecast_worst_cost'],forecast.max());close(plan['forecast_mean_regret'],reg.mean());close(plan['forecast_worst_regret'],reg.max())
   if policy=='regret':close(plan['objective'],reg.max())
   elif policy=='robust':close(plan['objective'],forecast.max())
   elif policy=='stochastic':close(plan['objective'],forecast.mean())
   else:
    b,q=independent_schedule(cfg,plan['x'],sc.mean(0),plan['scenario_solutions'][0]);close(b+q,plan['objective']);audits+=1
   obs=case['observed'][policy];assert obs['complete']==known;close(obs['known_fraction'],np.isfinite(truth[:,nodes]).mean());_,lo=independent_schedule(cfg,plan['x'],truth,obs['optimistic_solution'],'optimistic');_,hi=independent_schedule(cfg,plan['x'],truth,obs['pessimistic_solution'],'pessimistic');audits+=2;assert lo<=hi+1e-6;close(book+lo,obs['cost_lower']);close(book+hi,obs['cost_upper'])
   if known:close(obs['exact_cost'],book+lo);close(lo,hi);close(obs['exact_regret'],book+lo-actualbenchmark['objective']);assert obs['exact_regret']>=-1e-6 and obs['exact_completed']==obs['optimistic_solution']['completed']
   else:assert obs['exact_cost'] is None and obs['exact_completed'] is None and obs['exact_regret'] is None
   if level==1 and policy!='regret':
    source=read(OLD/f'case_{seed}_{pos:03d}.json');original=source['planned']['plans'][policy]
    for k,value in original.items():
     if k=='forecast_expected_cost':close(plan[k],value,4*np.spacing(max(abs(value),abs(plan[k]),1.)))
     else:assert plan[k]==value,(key,policy,k)
    copied+=1
  for policy in POLICIES[:-1]:assert r['forecast_worst_regret']<=planned['plans'][policy]['forecast_worst_regret']+1e-6
  for policy in ['robust','regret','deterministic']:assert planned['plans']['stochastic']['forecast_expected_cost']<=planned['plans'][policy]['forecast_expected_cost']+1e-6
  close(planned['no_task_window_fraction'],(~a.any(1)).mean())
  if selected:again=solve_master(p,a,offsets=v);close(again['objective'],full['objective']);resolves+=1
  records[key]=case;print('audited',level,seed,pos,flush=True)
 tables=verify_tables(records,pack)
 rng=np.random.default_rng(20261109);tiny=tiny_config();enumerations=0
 for j in range(24):
  c=json.loads(json.dumps(tiny));c['boat_booking']=[float(rng.choice([.5,1.,2.]))];c['crew_booking']=[float(rng.choice([.5,1.,2.])) for _ in range(2)];c['tasks'][1]['node']=int(rng.integers(0,2));p=make_problem(c);weather=np.where(rng.random((3,4,2))>.35,5.,15.);a=weather_options(p,weather);xs=list(product(range(2),range(2),range(2),range(3)));fee=np.array(c['boat_booking']+c['crew_booking']+c['spare_cost']);cost=np.array([[fee@x+brute_recourse(c,w,x)['recourse_cost'] for w in weather] for x in xs]);v=cost.min(0);true=float((cost-v).max(1).min());full=solve_master(p,a,offsets=v);ccg=solve_ccg(p,a,offsets=v);close(full['objective'],true);close(ccg['objective'],true);enumerations+=1
 summary=dict(status='PASS' if len(records)==120 else 'PARTIAL_VERIFIED',previous_unchanged=len(old),frozen_identities=len(identity),case_certificates=certs,independent_schedules=audits,predicted_perfect_benchmarks=benchmarks,realized_perfect_benchmarks=realized,reoptimized_problems=resolves,round17_plans_reused=copied,round17_resources_and_schedules_exact=True,mean_cost_reassociation_tolerance='4 float64 ulps; every other inherited field exact',additional_tiny_enumerations=enumerations,table_rows=tables,negative_eta_supported=True,all_plans_before_outcome_benchmarks=True)
 (OUT/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

def verify_tables(records,pack):
 f=pd.read_csv(OUT/'cases.csv');cert=pd.read_csv(OUT/'certificates.csv');tr=pd.read_csv(OUT/'ccg_trace.csv');bm=pd.read_csv(OUT/'benchmarks.csv');metrics=pd.read_csv(OUT/'metrics.csv');pairs=pd.read_csv(OUT/'paired_comparisons.csv');bounds=pd.read_csv(OUT/'missing_observation_bounds.csv');assert len(f)==len(records)*4 and len(cert)==len(records) and len(bm)==len(records)*16
 def eq(a,b):
  if b is None or not np.isfinite(b):assert pd.isna(a)
  else:close(a,b,1e-7)
 for row in f.itertuples():
  c=records[(row.level,row.seed,row.position)];p=c['planned']['plans'][row.policy];o=c['observed'][row.policy];assert row.issue_id==c['issue_id'] and row.timestamp==c['timestamp'] and json.loads(row.resources)==p['x'];eq(row.optimization_objective,p['objective'])
  for k in ['booking_cost','forecast_expected_cost','forecast_worst_cost','forecast_mean_regret','forecast_worst_regret']:eq(getattr(row,k),p[k])
  for k,target in [('actual_cost','exact_cost'),('actual_completed','exact_completed'),('actual_regret','exact_regret'),('cost_lower','cost_lower'),('cost_upper','cost_upper'),('known_fraction','known_fraction')]:eq(getattr(row,k),o[target])
  assert row.observed_complete==o['complete'] and row.no_booking==int(not any(p['x']));eq(row.no_task_window_fraction,c['planned']['no_task_window_fraction'])
 for row in cert.itertuples():
  c=records[(row.level,row.seed,row.position)];r=c['planned']['plans']['regret'];full=c['planned']['full_regret']
  for k,val in [('ccg_objective',r['objective']),('full_objective',full['objective']),('difference',r['objective']-full['objective']),('iterations',r['iterations']),('active_count',len(r['active_scenarios'])),('ccg_lower',r['lower_bound']),('ccg_upper',r['upper_bound']),('ccg_seconds',r['seconds']),('full_seconds',full['seconds']),('full_gap',full['mip_gap'])]:eq(getattr(row,k),val)
 assert len(tr)==sum(c['planned']['plans']['regret']['iterations'] for c in records.values())
 for row in tr.itertuples():
  h=records[(row.level,row.seed,row.position)]['planned']['plans']['regret']['trace'][row.iteration-1]
  for k in ['lower_bound','upper_bound','candidate_upper','worst_scenario','oracle_solves','master_seconds','master_gap']:eq(getattr(row,k),h[k])
  assert json.loads(row.resources)==h['resources'] and json.loads(row.active_scenarios)==h['active_scenarios']
 for row in bm.itertuples():
  b=records[(row.level,row.seed,row.position)]['planned']['benchmark']['solutions'][row.scenario];eq(row.perfect_cost,b['objective']);eq(row.booking_cost,b['booking_cost']);eq(row.gap,b['mip_gap']);assert json.loads(row.resources)==b['x']
 for row in metrics.itertuples():
  group=f[(f.level==row.level)&(f.seed==row.seed)&(f.policy==row.policy)];eq(row.value,group[row.metric].mean());assert row.n==group[row.metric].notna().sum()
 times=pd.DatetimeIndex(pd.to_datetime(pack['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');ww=((times-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();keys=np.unique(ww);wm=dict(zip(pack['positions'],ww));draw=np.random.default_rng(20261106).integers(0,len(keys),(2000,len(keys)))
 for row in pairs.itertuples():
  sub=f[f.level==row.level];seeds=SEEDS if row.level==1 else SEEDS[:1];delta=[];weeks=[];low=[];high=[]
  for pos in pack['positions']:
   group=sub[sub.position==pos]
   if len(group)!=len(seeds)*4 or set(zip(group.policy,group.seed))!={(p,s) for p in POLICIES for s in seeds}:continue
   a=group[group.policy=='regret'];b=group[group.policy==row.baseline];low.append(a.cost_lower.mean()-b.cost_upper.mean());high.append(a.cost_upper.mean()-b.cost_lower.mean())
   if a.actual_cost.notna().all() and b.actual_cost.notna().all():delta.append(a.actual_cost.mean()-b.actual_cost.mean());weeks.append(wm[pos])
  delta=np.array(delta);weeks=np.array(weeks);n=np.array([(weeks==k).sum() for k in keys]);ss=np.array([delta[weeks==k].sum() for k in keys]);den=n[draw].sum(1);valid=den>0;boot=ss[draw][valid].sum(1)/den[valid];enough=len(delta)>=8 and sum(n>0)>=4 and sum(valid)>=1900;l,h=np.quantile(boot,[.025,.975]) if enough else (np.nan,np.nan);eq(row.difference,delta.mean() if len(delta) else np.nan);eq(row.low95,l);eq(row.high95,h);assert row.complete_origins==len(delta) and row.nonempty_weeks==sum(n>0) and row.all_weeks==len(keys) and row.valid_bootstrap==sum(valid) and row.interval_reported==enough and row.seeds==len(seeds) and row.primary==(row.level==1)
  b=bounds[(bounds.level==row.level)&(bounds.baseline==row.baseline)].iloc[0];eq(b.mean_difference_lower,np.mean(low));eq(b.mean_difference_upper,np.mean(high));assert b.origins==len(low)
 return dict(cases=len(f),certificates=len(cert),trace=len(tr),benchmarks=len(bm),metrics=len(metrics),paired=len(pairs),missing_bounds=len(bounds))

if __name__=='__main__':run()

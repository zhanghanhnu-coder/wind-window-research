"""Independent resource/time/weather accounting and finite-scenario certificates."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
from two_stage_core import make_problem,weather_options,solve_master,solve_recourse
from two_stage_enumeration import brute_two_stage,brute_recourse
OUT=ART/'two_stage_v1';SEEDS=[20241101,20241102,20241103]
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def independent_schedule(c,x,weather,solution,unknown='error'):
 B=len(c['boat_booking']);C=len(c['crew_booking']);K=len(c['spare_cost']);H=c['horizon'];x=np.asarray(x);assert x.shape==(B+C+K,) and (x==np.rint(x)).all() and (x>=0).all() and (x<=np.array([1]*(B+C)+c['spare_max'])).all();book=float(np.array(c['boat_booking']+c['crew_booking']+c['spare_cost'])@x);seen=set();used=np.zeros(K,int);operating=0.;accepted=[]
 for row in solution['schedule']:
  j=row['task'];assert j not in seen and 0<=j<len(c['tasks']);seen.add(j);t=c['tasks'][j];a=row['start'];b=row['end'];boat=row['boat'];crew=row['crew'];assert row['id']==t['id'] and row['node']==t['node'] and row['spare']==t['spare'] and row['quantity']==t['quantity'];assert int(a)==a and a%c['step']==0 and b==a+t['duration'] and 0<=a<b<=H;assert 0<=boat<B and 0<=crew<C and x[boat]==1 and x[B+crew]==1 and t['skill'] in c['crew_skills'][crew]
  w=weather[a:b,t['node']];known=np.isfinite(w)
  if unknown=='error':assert known.all()
  good=np.where(known,w<=10,unknown=='optimistic');assert good.all()
  for prev in accepted:
   overlap=max(a,prev['start'])<min(b,prev['end']);assert not (overlap and (boat==prev['boat'] or crew==prev['crew'] or t['node']==prev['node']))
  accepted.append(row);used[t['spare']]+=t['quantity'];cost=(c['boat_hourly'][boat]+c['crew_hourly'][crew])*t['duration']+c['late_fraction']*t['penalty']*a/H;assert abs(cost-row['cost'])<1e-8;operating+=cost
 assert (used<=x[B+C:]).all();np.testing.assert_array_equal(used,solution['spares_used']);deferred=[j for j in range(len(c['tasks'])) if j not in seen];assert deferred==solution['deferred'] and solution['completed']==len(seen);penalty=sum(c['tasks'][j]['penalty'] for j in deferred);assert abs(operating-solution['operating_cost'])<1e-7 and abs(penalty-solution['unserved_penalty'])<1e-7 and abs(operating+penalty-solution['recourse_cost'])<1e-7
 if 'solver_status' in solution:assert solution['solver_status']==0
 return book,operating+penalty

def tiny_config():
 return dict(horizon=4,step=1,tasks=[dict(id='a',node=0,duration=2,skill='A',spare=0,quantity=1,penalty=10.),dict(id='b',node=1,duration=1,skill='B',spare=0,quantity=1,penalty=14.)],boat_booking=[1.],boat_hourly=[.1],crew_skills=[['A'],['B']],crew_booking=[1.,1.],crew_hourly=[0.,.05],spare_cost=[.5],spare_max=[2],late_fraction=.03)

def run():
 old=read('previous_hashes.json');identity=read('identity.json')
 for path,h in old.items():assert sha(ART/path)==h,path
 for path,h in identity.items():assert sha(Path(path))==h,path
 cfg=read('recipe.json')['config'];problem=make_problem(cfg);pack=np.load(OUT/'input_data.npz');raw=np.load(ART/'data.npz');pred=np.load(ART/'probability_v1/predictions.npz');positions=np.arange(0,232,10);np.testing.assert_array_equal(pack['positions'],positions);np.testing.assert_array_equal(pack['issue_ids'],raw['test_ids'][positions]);np.testing.assert_array_equal(pack['truth'],np.stack([raw['raw'][i+1:i+73] for i in pack['issue_ids']]));complete=np.isfinite(pred['cal_truth']).all((1,2));np.testing.assert_array_equal(pred['cal_ids'][complete],raw['cal_ids']);choose32=np.random.default_rng(20241101).choice(58,32,replace=False);choose16=np.random.default_rng(20261106).choice(32,16,replace=False);np.testing.assert_array_equal(pack['selected32_positions'],choose16);np.testing.assert_array_equal(pack['source_cal_ids'],raw['cal_ids'][choose32[choose16]]);assert max(pack['source_cal_ids']+72)<min(pack['issue_ids'])
 for seed in SEEDS:
  residual=(pred['cal_truth']-pred[f'cal_q_{seed}'][...,1])[complete];sc=np.maximum(0,pred[f'test_q_{seed}'][positions,...,1][:,None]+residual[choose32[choose16]][None]);np.testing.assert_array_equal(sc,pack[f'scenarios_{seed}'])
 files=sorted(OUT.glob('case_*.json'));records={};schedule_checks=0;reoptimized=0;certificate_count=0;nodeids=[t['node'] for t in cfg['tasks']];total_penalty=sum(t['penalty'] for t in cfg['tasks']);selected_positions=[0,110,230]
 for path in files:
  case=json.loads(path.read_text(encoding='utf-8'));seed=case['seed'];position=case['position'];idx=list(positions).index(position);assert idx==case['input_index'] and case['issue_id']==int(pack['issue_ids'][idx]) and case['timestamp']==str(pack['timestamps'][idx]);sc=pack[f'scenarios_{seed}'][idx];truth=pack['truth'][idx];planned=case['planned'];available=weather_options(problem,sc);outage=float((~available.any(1)).mean());assert outage==planned['no_task_window_fraction'];r=planned['plans']['robust'];full=planned['full_robust'];assert full['solver_status']==r['solver_status']==0 and abs(full['objective']-r['objective'])<1e-6
  assert abs(full['objective']-full['dual_bound'])<1e-6 and full['mip_gap']<1e-9;assert r['lower_bound']<=r['upper_bound']+1e-6 and r['upper_bound']-r['lower_bound']<=1e-7*(1+abs(r['upper_bound']));assert abs(r['objective']-r['upper_bound'])<1e-7
  hist=r['trace'];assert len(hist)==r['iterations'];lb=np.array([h['lower_bound'] for h in hist]);ub=np.array([h['upper_bound'] for h in hist]);assert (np.diff(lb)>=-1e-6).all() and (np.diff(ub)<=1e-6).all();assert np.all(lb<=ub+1e-6)
  for i,h in enumerate(hist):
   assert len(h['active_scenarios'])==len(set(h['active_scenarios'])) and all(0<=s<16 for s in h['active_scenarios'])
   if i:assert set(hist[i-1]['active_scenarios']).issubset(h['active_scenarios']) and hist[i-1]['worst_scenario'] in h['active_scenarios']
  fullcost=[]
  for s,solution in enumerate(full['scenario_solutions']):book,q=independent_schedule(cfg,full['x'],sc[s],solution);fullcost.append(q);schedule_checks+=1
  assert abs(book+max(fullcost)-full['objective'])<1e-6;certificate_count+=1
  for policy in ['deterministic','stochastic','robust']:
   plan=planned['plans'][policy];x=plan['x'];assert plan['solver_status']==0;forecastcost=[]
   for s,solution in enumerate(plan['evaluated_scenario_solutions']):book,q=independent_schedule(cfg,x,sc[s],solution);forecastcost.append(q);schedule_checks+=1
   assert len(forecastcost)==16;assert abs(book-plan['booking_cost'])<1e-7 and abs(book+np.mean(forecastcost)-plan['forecast_expected_cost'])<1e-6 and abs(book+max(forecastcost)-plan['forecast_worst_cost'])<1e-6
   if policy=='deterministic':
    _,q=independent_schedule(cfg,x,sc.mean(0),plan['scenario_solutions'][0]);assert abs(book+q-plan['objective'])<1e-6;schedule_checks+=1
   if policy=='stochastic':assert abs(plan['objective']-plan['forecast_expected_cost'])<1e-6
   if policy=='robust':assert abs(plan['objective']-plan['forecast_worst_cost'])<1e-6;np.testing.assert_allclose(forecastcost,plan['recourse_costs'],atol=1e-7,rtol=0)
   obs=case['observed'][policy];known=bool(np.isfinite(truth[:,nodeids]).all());assert obs['complete']==known;assert abs(obs['known_fraction']-np.isfinite(truth[:,nodeids]).mean())<1e-15
   _,lo=independent_schedule(cfg,x,truth,obs['optimistic_solution'],'optimistic');_,hi=independent_schedule(cfg,x,truth,obs['pessimistic_solution'],'pessimistic');schedule_checks+=2;assert lo<=hi+1e-6 and abs(book+lo-obs['cost_lower'])<1e-7 and abs(book+hi-obs['cost_upper'])<1e-7
   if known:assert abs(obs['exact_cost']-(book+lo))<1e-7 and abs(lo-hi)<1e-7 and obs['exact_completed']==obs['optimistic_solution']['completed']
   else:assert obs['exact_cost'] is None and obs['exact_completed'] is None
   if position in selected_positions:
    fresh=[]
    for s in range(16):fresh.append(solve_recourse(problem,available[s],x)['recourse_cost']);reoptimized+=1
    np.testing.assert_allclose(fresh,forecastcost,atol=1e-6,rtol=0)
  for policy in ['deterministic','stochastic']:assert r['forecast_worst_cost']<=planned['plans'][policy]['forecast_worst_cost']+1e-6
  for policy in ['deterministic','robust']:assert planned['plans']['stochastic']['forecast_expected_cost']<=planned['plans'][policy]['forecast_expected_cost']+1e-6
  if outage>0:assert not any(r['x']) and abs(r['objective']-total_penalty)<1e-6
  if position in selected_positions:
   again=solve_master(problem,available,'robust');assert abs(again['objective']-r['objective'])<1e-6;reoptimized+=1
  records[(seed,position)]=case
  print('audited',seed,position,flush=True)
 verify_tables(records,pack)
 small=tiny_config();p=make_problem(small);rng=np.random.default_rng(20261108);enumerations=0
 for _ in range(18):
  w=np.where(rng.random((3,4,2))>.4,5.,15.);a=weather_options(p,w)
  for mode in ['stochastic','robust']:
   brute=brute_two_stage(small,w,mode);sol=solve_master(p,a,mode);assert abs(brute['objective']-sol['objective'])<1e-6;enumerations+=1
 summary=dict(status='PASS' if len(records)==72 else 'PARTIAL_VERIFIED',previous_unchanged=len(old),frozen_identities=len(identity),case_certificates=certificate_count,independent_schedules=schedule_checks,reoptimized_problems=reoptimized,additional_tiny_enumerations=enumerations,all_resources_shared=True,no_unknown_as_observed_failure=True,all_table_checks=True)
 (OUT/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

def verify_tables(records,pack):
 frame=pd.read_csv(OUT/'cases.csv');cert=pd.read_csv(OUT/'certificates.csv');traces=pd.read_csv(OUT/'ccg_trace.csv');metrics=pd.read_csv(OUT/'metrics.csv');pairs=pd.read_csv(OUT/'paired_comparisons.csv');bounds=pd.read_csv(OUT/'missing_observation_bounds.csv');assert len(frame)==3*len(records) and len(cert)==len(records)
 def close(value,expected):
  if expected is None or not np.isfinite(expected):assert pd.isna(value)
  else:assert np.isclose(value,expected,atol=1e-7,rtol=0),(value,expected)
 for r in frame.itertuples():
  case=records[(r.seed,r.position)];p=case['planned']['plans'][r.policy];o=case['observed'][r.policy];assert r.issue_id==case['issue_id'] and r.timestamp==case['timestamp'];assert json.loads(r.resources)==p['x'];assert r.observed_complete==o['complete'];close(r.optimization_objective,p['objective']);close(r.booking_cost,p['booking_cost']);close(r.forecast_expected_cost,p['forecast_expected_cost']);close(r.forecast_worst_cost,p['forecast_worst_cost']);close(r.actual_cost,o['exact_cost']);close(r.actual_completed,o['exact_completed']);close(r.cost_lower,o['cost_lower']);close(r.cost_upper,o['cost_upper']);close(r.observed_known_fraction,o['known_fraction']);assert r.no_booking==int(not any(p['x']));close(r.no_task_window_fraction,case['planned']['no_task_window_fraction'])
 for r in cert.itertuples():
  case=records[(r.seed,r.position)];a=case['planned']['plans']['robust'];b=case['planned']['full_robust']
  for name,value in [('ccg_objective',a['objective']),('full_objective',b['objective']),('difference',a['objective']-b['objective']),('iterations',a['iterations']),('active_count',len(a['active_scenarios'])),('oracle_solves',a['oracle_solves']),('ccg_seconds',a['seconds']),('full_seconds',b['seconds']),('ccg_lower',a['lower_bound']),('ccg_upper',a['upper_bound']),('full_gap',b['mip_gap'])]:close(getattr(r,name),value)
 assert len(traces)==sum(c['planned']['plans']['robust']['iterations'] for c in records.values())
 for r in traces.itertuples():
  target=records[(r.seed,r.position)]['planned']['plans']['robust']['trace'][r.iteration-1]
  for key in ['lower_bound','upper_bound','candidate_upper','worst_scenario','oracle_solves','master_seconds','master_gap']:close(getattr(r,key),target[key])
  assert json.loads(r.resources)==target['resources'] and json.loads(r.active_scenarios)==target['active_scenarios']
 frame['month']=pd.to_datetime(frame.timestamp,utc=True).dt.tz_convert('Asia/Shanghai').dt.month
 for r in metrics.itertuples():
  f=frame[(frame.seed==r.seed)&(frame.policy==r.policy)];f=f if r.month==0 else f[f.month==r.month];close(r.value,f[r.metric].mean());assert r.n==f[r.metric].notna().sum()
 times=pd.DatetimeIndex(pd.to_datetime(pack['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');allweeks=((times-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();keys=np.unique(allweeks);weekmap=dict(zip(pack['positions'],allweeks));draw=np.random.default_rng(20261106).integers(0,len(keys),(2000,len(keys)));eligible=[]
 for pos in pack['positions']:
  f=frame[frame.position==pos]
  if set(zip(f.policy,f.seed))=={(p,s) for p in ['deterministic','stochastic','robust'] for s in SEEDS} and len(f)==9:eligible.append(pos)
 for r in pairs.itertuples():
  diff=[];weeks=[];lows=[];highs=[]
  for pos in eligible:
   f=frame[frame.position==pos];a=f[f.policy=='robust'];b=f[f.policy==r.baseline];lows.append(a.cost_lower.mean()-b.cost_upper.mean());highs.append(a.cost_upper.mean()-b.cost_lower.mean())
   if a.actual_cost.notna().all() and b.actual_cost.notna().all():diff.append(a.actual_cost.mean()-b.actual_cost.mean());weeks.append(weekmap[pos])
  diff=np.array(diff);weeks=np.array(weeks);n=np.array([(weeks==k).sum() for k in keys]);ss=np.array([diff[weeks==k].sum() for k in keys]);den=n[draw].sum(1);valid=den>0;boot=ss[draw][valid].sum(1)/den[valid];enough=len(diff)>=8 and (n>0).sum()>=4 and valid.sum()>=1900;lo,hi=np.quantile(boot,[.025,.975]) if enough else (np.nan,np.nan);close(r.difference,diff.mean() if len(diff) else np.nan);close(r.low95,lo);close(r.high95,hi);assert r.complete_origins==len(diff) and r.nonempty_weeks==(n>0).sum() and r.all_weeks==len(keys) and r.valid_bootstrap==valid.sum() and r.interval_reported==enough
  br=bounds[bounds.baseline==r.baseline].iloc[0];close(br.mean_difference_lower,np.mean(lows));close(br.mean_difference_upper,np.mean(highs));assert br.origins==len(eligible)

if __name__=='__main__':run()

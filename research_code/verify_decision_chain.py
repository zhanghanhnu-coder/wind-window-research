"""Independent audit of round20 accounting, temporal boundaries and scores."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
from two_stage_core import make_problem,weather_options,solve_master as fresh_master,solve_recourse as fresh_recourse
from regret_core import solve_master as full_master
from verify_two_stage import independent_schedule
from research_crps_distribution import guard
from research_lead_calibration import paths
from decision_chain_core import FAMILIES,POLICIES
OUT=ART/'decision_chain_v1';SEEDS=[20241101,20241102,20241103];POSITIONS=list(range(0,232,10))
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-6):assert np.isclose(a,b,atol=tol,rtol=0),(a,b)
def empirical(x,y):
 x=np.sort(np.asarray(x,float),axis=0);k=len(x);return np.mean(abs(x-y),0)-np.sum(x*(2*np.arange(k)-k+1).reshape((k,)+(1,)*(x.ndim-1)),0)/k**2
def rows_sorted(x):return np.array(sorted(map(tuple,x)))

def runtime_identity():
 from highs_readback_compat import build_wrapper
 root=OUT/'runtime_compatibility';cut=read(root/'execution_cutover.json');proof=read(root/'validation.json');_,meta=build_wrapper()
 assert cut['compatibility']==proof['compatibility']==meta
 assert cut['hypothesis_or_solver_settings_changed'] is False and proof['status']=='PASS'
 assert sha(root/'validation.json')==cut['validation_sha256']
 for n,h in cut['additional_source_identities'].items():assert sha(Path(n))==h,n
 for n,h in cut['preserved_completed'].items():assert sha(OUT/n)==h,n
 assert len(proof['cases'])==13 and len(cut['preserved_completed'])==14
 for row in proof['cases']:
  assert row['all_nontiming_fields_exact'] is True and sha(OUT/row['file'])==row['sha256']
 return dict(additional_sources=len(cut['additional_source_identities']),preserved_cases=14,equivalent_recomputations=13)

def sources():
 meta=read(ART/'metadata.json');ev=np.load(ART/'crps_distribution_v1/evaluation_data.npz');original=np.load(ART/'two_stage_v1/input_data.npz');pack={};checks=0
 for seed in SEEDS:
  for family in FAMILIES:
   d=np.load(OUT/f'scenarios_{family}_{seed}.npz');np.testing.assert_array_equal(d['positions'],POSITIONS);np.testing.assert_array_equal(d['issue_ids'],ev['test_ids'][POSITIONS]);np.testing.assert_array_equal(d['timestamps'],original['timestamps']);np.testing.assert_array_equal(d['source_cal_ids'],ev['selected57_ids']);assert max(d['source_cal_ids']+72)<min(d['issue_ids']);sc=d['scenarios'];assert sc.shape==(24,57,72,17) and np.isfinite(sc).all() and (sc>=0).all();pack[(family,seed)]=sc
   if family in FAMILIES[:3]:
    model='old_graph' if family=='old57' else ('gated_graph_atoms' if family=='gated_graph57' else 'gated_temporal_atoms');method='old57' if family=='old57' else 'smooth_affine';wanted=set(POSITIONS)
    for pos,s in enumerate(paths(dict(model=model,method=method,seed=seed),meta,ev)):
     if pos in wanted:np.testing.assert_array_equal(sc[POSITIONS.index(pos)],s);checks+=1
   else:
    kind=family.split('_')[1];base=pack[('old57',seed)]
    for a,b in zip(base,sc):
     np.testing.assert_array_equal(np.sort(a,0),np.sort(b,0));np.testing.assert_array_equal(a.mean(0,dtype=np.float64),b.mean(0,dtype=np.float64))
     if kind=='time':
      for h in range(72):np.testing.assert_array_equal(rows_sorted(a[:,h]),rows_sorted(b[:,h]))
     else:
      for n in range(17):np.testing.assert_array_equal(rows_sorted(a[:,:,n]),rows_sorted(b[:,:,n]))
     checks+=1
 return pack,original,checks

def run():
 guard();runtime=runtime_identity();previous=read(OUT/'previous_hashes.json');identity=read(OUT/'identity.json');inputs=read(OUT/'input_identity.json')
 for n,h in previous.items():assert sha(ART/n)==h,n
 for n,h in {**identity,**inputs}.items():assert sha(Path(n))==h,n
 pack,truthpack,sourcechecks=sources();cfg=read(OUT/'recipe.json')['config'];p=make_problem(cfg);records={};audits=0;resolves=0;certs=0;selected_count=0;benchcount=0
 completion=read(OUT/'planning_complete.json');files=sorted(OUT.glob('case_*.json'));assert completion['completed']==len(files)
 for file in files:
  guard();c=read(file);family=c['family'];seed=c['seed'];pos=c['position'];idx=POSITIONS.index(pos);assert c['input_index']==idx and c['issue_id']==int(truthpack['issue_ids'][idx]) and c['timestamp']==str(truthpack['timestamps'][idx]);key=(family,seed,pos);assert key not in records;sc=pack[(family,seed)][idx];truth=truthpack['truth'][idx];a=weather_options(p,sc);planned=c['planned'];assert read(OUT/f'planned_{family}_{seed}_{pos:03d}.json')['planned']==planned;v=np.array(planned['benchmark']['values']);assert v.shape==(57,);selected=pos==0 or (seed==SEEDS[0] and pos in [110,230]);fresh_values=[];fresh_q={}
  for s,b in enumerate(planned['benchmark']['solutions']):
   assert b['solver_status']==0 and b['mip_gap']<1e-9;close(b['objective'],b['dual_bound']);book,q=independent_schedule(cfg,b['x'],sc[s],b['scenario_solutions'][0]);close(book+q,v[s]);close(b['objective'],v[s]);audits+=1;benchcount+=1
  full=planned['full_regret'];r=planned['plans']['regret'];assert full['solver_status']==r['solver_status']==0 and full['mip_gap']<1e-9;close(full['objective'],full['dual_bound']);close(full['objective'],r['objective']);close(r['objective'],r['upper_bound']);assert 0<=r['upper_bound']-r['lower_bound']+1e-6<=1e-7*(1+abs(r['upper_bound']))+1e-6;np.testing.assert_allclose(r['offsets'],v,atol=1e-8,rtol=0);qfull=[]
  for s,sol in enumerate(full['scenario_solutions']):book,q=independent_schedule(cfg,full['x'],sc[s],sol);qfull.append(q);audits+=1
  close(book+max(np.array(qfull)-v),full['objective']);close(full['eta'],max(np.array(qfull)-v));certs+=1;history=r['trace'];assert len(history)==r['iterations'];lb=np.array([h['lower_bound'] for h in history]);ub=np.array([h['upper_bound'] for h in history]);assert (np.diff(lb)>=-1e-6).all() and (np.diff(ub)<=1e-6).all() and (lb<=full['objective']+1e-6).all() and (ub>=full['objective']-1e-6).all()
  for i,h in enumerate(history):
   active=h['active_scenarios'];assert active[0]==0 and len(active)==len(set(active)) and all(0<=x<57 for x in active)
   if i+1<len(history):assert history[i+1]['active_scenarios']==active+[h['worst_scenario']]
  nodes=sorted({t['node'] for t in cfg['tasks']});known=bool(np.isfinite(truth[:,nodes]).all());actual=c['realized_benchmark']
  if known:
   assert actual['solver_status']==0 and actual['mip_gap']<1e-9;close(actual['objective'],actual['dual_bound']);b,q=independent_schedule(cfg,actual['x'],truth,actual['scenario_solutions'][0]);close(b+q,actual['objective']);audits+=1
  else:assert actual is None
  for policy in POLICIES:
   plan=planned['plans'][policy];assert plan['solver_status']==0;costs=[];sols=plan['evaluated_scenario_solutions'];assert len(sols)==57
   for s,sol in enumerate(sols):book,q=independent_schedule(cfg,plan['x'],sc[s],sol);costs.append(book+q);audits+=1
   costs=np.array(costs);reg=costs-v;assert reg.min()>=-1e-6;np.testing.assert_allclose(costs,plan['scenario_costs'],atol=1e-6,rtol=0);np.testing.assert_allclose(reg,plan['scenario_regrets'],atol=1e-6,rtol=0);close(plan['booking_cost'],book);close(plan['forecast_expected_cost'],costs.mean());close(plan['forecast_worst_cost'],costs.max());close(plan['forecast_worst_regret'],reg.max());close(plan['forecast_mean_regret'],reg.mean())
   if policy=='regret':close(plan['objective'],reg.max())
   else:
    assert plan['mip_gap']<1e-9;close(plan['objective'],plan['dual_bound'])
    if policy=='stochastic':close(plan['objective'],costs.mean())
    else:b,q=independent_schedule(cfg,plan['x'],sc.mean(0,dtype=np.float64),plan['scenario_solutions'][0]);close(plan['objective'],b+q);audits+=1
   o=c['observed'][policy];assert o['complete']==known;close(o['known_fraction'],np.isfinite(truth[:,nodes]).mean());_,lo=independent_schedule(cfg,plan['x'],truth,o['optimistic_solution'],'optimistic');_,hi=independent_schedule(cfg,plan['x'],truth,o['pessimistic_solution'],'pessimistic');audits+=2;assert lo<=hi+1e-6;close(book+lo,o['cost_lower']);close(book+hi,o['cost_upper'])
   if known:close(o['exact_cost'],book+lo);close(lo,hi);close(o['exact_regret'],book+lo-actual['objective']);assert o['exact_completed']==o['optimistic_solution']['completed']
   else:assert o['exact_cost'] is None and o['exact_completed'] is None and o['exact_regret'] is None
  close(planned['no_task_window_fraction'],(~a.any(1)).mean())
  for plan in planned['plans'].values():assert r['forecast_worst_regret']<=plan['forecast_worst_regret']+1e-6 and planned['plans']['stochastic']['forecast_expected_cost']<=plan['forecast_expected_cost']+1e-6
  if selected:
   saved=OUT/f'independent_resolve_{family}_{seed}_{pos:03d}.json';source_hash=sha(file)
   if saved.exists():fresh=read(saved);assert fresh['source_hash']==source_hash
   else:
    for row in a:guard();fresh_values.append(fresh_master(p,row[None],'stochastic')['objective'])
    for policy in POLICIES:
     vals=[]
     for row in a:guard();vals.append(fresh_recourse(p,row,planned['plans'][policy]['x'])['recourse_cost'])
     fresh_q[policy]=vals
    guard();fm=full_master(p,a,offsets=v);fresh=dict(source_hash=source_hash,benchmark=fresh_values,recourse=fresh_q,full=fm['objective'],full_status=fm['solver_status']);saved.write_text(json.dumps(fresh,indent=2),encoding='utf-8')
   np.testing.assert_allclose(fresh['benchmark'],v,atol=1e-6,rtol=0)
   for policy in POLICIES:np.testing.assert_allclose(fresh['recourse'][policy],[s['recourse_cost'] for s in planned['plans'][policy]['evaluated_scenario_solutions']],atol=1e-6,rtol=0)
   close(fresh['full'],full['objective']);assert fresh['full_status']==0;resolves+=229;selected_count+=1
  records[key]=c;print('AUDIT',family,seed,pos,flush=True)
 # The marginal-preserving controls must have the same deterministic problem optimum.
 comparisons=0
 for seed in SEEDS:
  for pos in POSITIONS:
   base=records.get(('old57',seed,pos))
   for family in FAMILIES[3:]:
    control=records.get((family,seed,pos))
    if base and control:close(base['planned']['plans']['deterministic']['objective'],control['planned']['plans']['deterministic']['objective']);comparisons+=1
 tables=verify_tables(records,truthpack);scorechecks=verify_scores(pack,truthpack['truth']);summary=dict(status='PASS' if len(records)==360 else 'PARTIAL_VERIFIED',previous_unchanged=len(previous),frozen_identities=len(identity),input_identities=len(inputs),source_paths_checked=sourcechecks,case_certificates=certs,independent_schedules=audits,perfect_benchmarks=benchcount,selected_fresh_cases=selected_count,reoptimized_problems=resolves,deterministic_control_equalities=comparisons,prediction_score_checks=scorechecks,table_rows=tables)
 summary['runtime_compatibility']=runtime
 (OUT/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

def verify_tables(records,pack):
 f=pd.read_csv(OUT/'cases.csv');cert=pd.read_csv(OUT/'certificates.csv');bm=pd.read_csv(OUT/'benchmarks.csv');tr=pd.read_csv(OUT/'ccg_trace.csv');metrics=pd.read_csv(OUT/'metrics.csv');pairs=pd.read_csv(OUT/'paired_comparisons.csv');bounds=pd.read_csv(OUT/'missing_observation_bounds.csv');assert len(f)==len(records)*3 and len(cert)==len(records) and len(bm)==len(records)*57
 def eq(a,b):
  if b is None or not np.isfinite(b):assert pd.isna(a)
  else:close(a,b,1e-7)
 for row in f.itertuples():
  c=records[(row.family,row.seed,row.position)];p=c['planned']['plans'][row.policy];o=c['observed'][row.policy];assert row.issue_id==c['issue_id'] and row.timestamp==c['timestamp'] and json.loads(row.resources)==p['x']
  for k in ['objective','booking_cost','forecast_expected_cost','forecast_worst_cost','forecast_worst_regret']:eq(getattr(row,k),p[k])
  for k,v in [('actual_cost','exact_cost'),('actual_completed','exact_completed'),('actual_regret','exact_regret'),('known_fraction','known_fraction'),('cost_lower','cost_lower'),('cost_upper','cost_upper')]:eq(getattr(row,k),o[v])
  assert row.observed_complete==o['complete'] and row.no_booking==int(not any(p['x']));eq(row.no_task_window_fraction,c['planned']['no_task_window_fraction'])
 for row in cert.itertuples():
  c=records[(row.family,row.seed,row.position)];r=c['planned']['plans']['regret'];full=c['planned']['full_regret']
  for k,v in [('ccg_objective',r['objective']),('full_objective',full['objective']),('difference',r['objective']-full['objective']),('iterations',r['iterations']),('lower_bound',r['lower_bound']),('upper_bound',r['upper_bound']),('ccg_seconds',r['seconds']),('full_seconds',full['seconds'])]:eq(getattr(row,k),v)
 assert len(tr)==sum(c['planned']['plans']['regret']['iterations'] for c in records.values())
 for row in tr.itertuples():
  t=records[(row.family,row.seed,row.position)]['planned']['plans']['regret']['trace'][row.iteration-1]
  for k in ['lower_bound','upper_bound','candidate_upper','worst_scenario','oracle_solves','master_seconds','master_gap']:eq(getattr(row,k),t[k])
  assert json.loads(row.resources)==t['resources'] and json.loads(row.active_scenarios)==t['active_scenarios']
 for row in bm.itertuples():
  b=records[(row.family,row.seed,row.position)]['planned']['benchmark']['solutions'][row.scenario];eq(row.perfect_cost,b['objective']);eq(row.gap,b['mip_gap']);assert row.solver_status==b['solver_status'] and json.loads(row.resources)==b['x']
 for row in metrics.itertuples():
  g=f[(f.family==row.family)&(f.seed==row.seed)&(f.policy==row.policy)];eq(row.value,g[row.metric].mean());assert row.n==g[row.metric].notna().sum()
 times=pd.DatetimeIndex(pd.to_datetime(pack['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');ww=((times-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();keys=np.unique(ww);wm=dict(zip(POSITIONS,ww));draw=np.random.default_rng(20261106).integers(0,len(keys),(2000,len(keys)))
 expected={(a,b,p,primary) for a,b,primary in [('gated_graph57','old57',True),('gated_graph57','gated_temporal57',True),('old57_time_broken','old57',False),('old57_space_broken','old57',False)] for p in ['stochastic','regret']};assert set(zip(pairs.candidate,pairs.baseline,pairs.policy,pairs.primary))==expected and len(pairs)==8 and len(bounds)==8
 for row in pairs.itertuples():
  sub=f[(f.policy==row.policy)&f.family.isin([row.candidate,row.baseline])];delta=[];weeks=[];lower=[];upper=[]
  for pos in POSITIONS:
   g=sub[sub.position==pos]
   if set(zip(g.family,g.seed))!={(family,seed) for family in [row.candidate,row.baseline] for seed in SEEDS}:continue
   a=g[g.family==row.candidate];b=g[g.family==row.baseline];lower.append(a.cost_lower.mean()-b.cost_upper.mean());upper.append(a.cost_upper.mean()-b.cost_lower.mean())
   if a.actual_cost.notna().all() and b.actual_cost.notna().all():delta.append(a.actual_cost.mean()-b.actual_cost.mean());weeks.append(wm[pos])
  delta=np.array(delta);weeks=np.array(weeks);n=np.array([(weeks==k).sum() for k in keys]);sums=np.array([delta[weeks==k].sum() for k in keys]);den=n[draw].sum(1);valid=den>0;boot=sums[draw][valid].sum(1)/den[valid];enough=len(delta)>=8 and sum(n>0)>=4 and sum(valid)>=1900;lo,hi=np.quantile(boot,[.025,.975]) if enough else (np.nan,np.nan);eq(row.difference,delta.mean() if len(delta) else np.nan);eq(row.low95,lo);eq(row.high95,hi);assert row.complete_origins==len(delta) and row.nonempty_weeks==sum(n>0) and row.all_weeks==len(keys) and row.valid_bootstrap==sum(valid) and row.interval_reported==enough
  b=bounds[(bounds.candidate==row.candidate)&(bounds.baseline==row.baseline)&(bounds.policy==row.policy)].iloc[0];eq(b.mean_difference_lower,np.mean(lower) if lower else np.nan);eq(b.mean_difference_upper,np.mean(upper) if upper else np.nan);assert b.origins==len(lower)
 return dict(cases=len(f),certificates=len(cert),benchmarks=len(bm),trace=len(tr),metrics=len(metrics),paired=len(pairs),missing_bounds=len(bounds))

def verify_scores(pack,truth):
 count=0;summary=pd.read_csv(OUT/'prediction_metrics.csv')
 for key,scenarios in pack.items():
  family,seed=key;saved=np.load(OUT/f'scores_{family}_{seed}.npz')
  for idx,sc in enumerate(scenarios):
   y=truth[idx];np.testing.assert_allclose(empirical(sc,y),saved['crps'][idx],atol=3e-12,rtol=0,equal_nan=True);errors=[]
   for off in [0,24,48,66]:
    obs=y[off:off+6];valid=np.isfinite(obs).all(0);prob=(sc[:,off:off+6]<=10).all(1).mean(0);errors.append(np.where(valid,(prob-(obs<=10).all(0))**2,np.nan))
   np.testing.assert_allclose(errors,saved['brier6'][idx],atol=1e-12,rtol=0,equal_nan=True)
   def runlength(row):return np.diff(np.r_[-1,np.flatnonzero(~row),len(row)]).max()-1
   lengths=np.array([[runlength(row[:,j]<=10) for j in range(17)] for row in sc]);actual=np.array([runlength(y[:,j]<=10) for j in range(17)]);loss=np.where(np.isfinite(y).all(0),empirical(lengths,actual),np.nan);np.testing.assert_allclose(loss,saved['length_crps72'][idx],atol=1e-12,rtol=0,equal_nan=True)
   group=[]
   for off in [0,24,48,66]:
    for nodes in [[0,1,2],[8,9,10],[13,15,16]]:
     obs=y[off:off+6][:,nodes];prob=np.mean([bool((row[off:off+6][:,nodes]<=10).all()) for row in sc]);group.append((prob-float((obs<=10).all()))**2 if np.isfinite(obs).all() else np.nan)
   np.testing.assert_allclose(group,saved['group_brier'][idx],atol=1e-12,rtol=0,equal_nan=True);scale=read(ART/'metadata.json')['scale'];space=[];time=[]
   for h in [5,23,47,71]:
    for i in range(17):
     for j in range(i+1,17):
      expected=np.sqrt(abs(sc[:,h,i]-sc[:,h,j])/scale).mean(dtype=np.float64);actual=np.sqrt(abs(y[h,i]-y[h,j])/scale);space.append((actual-expected)**2)
   hours=[0,5,11,23,47,71]
   for i,h in enumerate(hours):
    for h2 in hours[i+1:]:
     expected=np.sqrt(abs(sc[:,h]-sc[:,h2])/scale).mean(axis=0,dtype=np.float64);actual=np.sqrt(abs(y[h]-y[h2])/scale);time.extend((actual-expected)**2)
   np.testing.assert_allclose(space,saved['spatial_variogram'][idx],atol=1e-12,rtol=0,equal_nan=True);np.testing.assert_allclose(time,saved['temporal_variogram'][idx],atol=1e-12,rtol=0,equal_nan=True);count+=1
  for row in summary[(summary.family==family)&(summary.seed==seed)].itertuples():close(row.value,np.nanmean(saved[row.metric]),1e-10);assert row.n==np.isfinite(saved[row.metric]).sum()
 return count

if __name__=='__main__':run()

"""Independent LP adversaries, schedule accounting, certificates and outcome bounds."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from model import ART
from two_stage_core import make_problem
from regret_core import solve_recourse
from distributional_schedule_core import solve_distribution_master
from verify_two_stage import independent_schedule
from highs_readback_compat import install
from research_crps_distribution import guard
OUT=ART/'distributional_schedule_v1';SRC=ART/'seasonal_probability_v1';COP=ART/'partial_copula_v1';ROLLING=ART/'seasonal_rolling_v1';FOLDS={'M05':'2024-05-01','M07':'2024-07-01','M09':'2024-09-01','M11':'2024-11-01'};SEEDS=[20241101,20241102,20241103];CASES={'graph_frozen':('gated_graph_atoms','frozen'),'graph_rolling28':('gated_graph_atoms','rolling28'),'own_rolling28':('gated_temporal_atoms','rolling28')};RADII={'tv000':0.,'tv025':.25,'tv100':1.,'tv010':.1,'tv050':.5};PAIRS=[('graph_rolling28','tv025','graph_rolling28','tv000'),('graph_rolling28','tv025','graph_rolling28','tv100'),('graph_rolling28','tv025','graph_frozen','tv025'),('graph_rolling28','tv025','own_rolling28','tv025')]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-6):np.testing.assert_allclose(a,b,atol=tol,rtol=0,equal_nan=True)
def lp(values,p,rho):
 n=len(p);res=linprog(np.r_[-np.asarray(values),np.zeros(n)],A_ub=np.vstack([np.c_[np.eye(n),-np.eye(n)],np.c_[-np.eye(n),-np.eye(n)],np.r_[np.zeros(n),np.ones(n)][None]]),b_ub=np.r_[p,-p,2*rho],A_eq=np.r_[np.ones(n),np.zeros(n)][None],b_eq=[1.],bounds=[(0,None)]*(2*n),method='highs');assert res.status==0 and res.success;return float(-res.fun),res.x[:n]
def options_reference(config,weather,unknown=False):
 rows=[]
 for t in config['tasks']:
  for start in range(0,config['horizon']-t['duration']+1,config['step']):
   for boat in range(len(config['boat_booking'])):
    for crew in range(len(config['crew_booking'])):
     if t['skill'] not in config['crew_skills'][crew]:continue
     x=weather[:,start:start+t['duration'],t['node']];good=np.where(np.isfinite(x),x<=10,unknown);rows.append(good.all(1))
 return np.stack(rows,1)
def compare_table(file,rows,keys):
 a=pd.read_csv(OUT/file).sort_values(keys).reset_index(drop=True);b=pd.DataFrame(rows).sort_values(keys).reset_index(drop=True);assert len(a)==len(b)
 for k in b:
  if b[k].dtype.kind in 'biufc':close(a[k],b[k],3e-10)
  else:assert a[k].fillna('').tolist()==b[k].fillna('').tolist(),(file,k)
 return len(a)
def run():
 guard();install();assert read(OUT/'evaluation_complete.json')['status']=='COMPLETE' and read(OUT/'planning_complete.json')['status']=='COMPLETE' and read(OUT/'failures.json')==[];parents={}
 for root in [ART/'two_stage_v1',ART/'decision_chain_v1',SRC,COP,ROLLING]:
  manifest=read(root/'final_manifest.json')
  for file,h in manifest.items():assert sha(Path(file))==h,file
  parents[root.name]=len(manifest)
 for file,h in read(OUT/'identity.json').items():assert sha(Path(file))==h,file
 for file,h in read(OUT/'previous_hashes.json').items():assert sha(ART/file)==h,file
 cfg=read(OUT/'recipe.json')['config'];p=make_problem(cfg);inventory=read(OUT/'inventory.json');assert len(inventory)==432;raw=np.load(ART/'data.npz')['raw'];times=pd.DatetimeIndex(pd.to_datetime(np.load(ART/'data.npz')['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');source_inv=read(ROLLING/'inventory.json');counts=dict(forecast_cases=0,plans=0,scenario_schedule_audits=0,probability_lp_oracles=0,master_certificates=0,endpoint_certificates=0,full_dual_certificates=0,selected_fresh_dual_solves=0,selected_fresh_recourse_solves=0,actual_recourse_solves=0);outcomes=[];planning=[];actual_cache={};fresh=[]
 for item in inventory:
  guard();name=item['fold'];case=item['case'];seed=item['seed'];pos=item['position'];f=np.load(SRC/name/'data.npz');positions=np.floor(np.linspace(0,len(f['test_ids'])-1,12)).astype(int);assert pos in positions and item['issue_id']==f['test_ids'][pos];model,method=CASES[case];source=next(r for r in source_inv if (r['fold'],r['model'],r['method'],r['seed'])==(name,model,method,seed));assert item['source_item']==source;assert sha(Path(item['forecast_source']))==item['forecast_sha256'];q=np.load(source['source'])[source['source_key']][pos];ranks=np.load(source['ranks_source'])['ranks'];expected=q[np.arange(72)[None,:,None],np.arange(17)[None,None,:],ranks//3];pack=np.load(item['forecast_source']);sc=pack['scenarios'];np.testing.assert_array_equal(sc,expected);available=options_reference(cfg,sc);np.testing.assert_array_equal(available,pack['available']);prob=np.ones(57)/57;np.testing.assert_array_equal(prob,pack['probability']);prob/=prob.sum();policies=['tv000','tv025','tv100']+(['tv010','tv050'] if case=='graph_rolling28' and seed==SEEDS[0] else []);assert item['policies']==policies;cost_by_radius=[];counts['forecast_cases']+=1
  truth=raw[item['issue_id']+1:item['issue_id']+73];nodes=sorted({t['node'] for t in cfg['tasks']});complete=bool(np.isfinite(truth[:,nodes]).all());lo=options_reference(cfg,truth[None],True)[0];hi=options_reference(cfg,truth[None],False)[0]
  for policy in policies:
   guard();rho=RADII[policy];path=OUT/'cases'/f'plan_{item["key"]}__{policy}.json';record=read(path);r=record['plan'];full=record['dual_reference'];x=r['x'];assert record['radius']==rho and r['radius']==rho and r['solver_status']==0;close(r['objective'],r['upper_bound']);assert 0<=r['gap']<=1e-7*(1+abs(r['upper_bound']));close(r['upper_bound']-r['lower_bound'],r['gap']);close(r['objective'],full['objective'],1e-5);assert full['solver_status']==0 and full['mip_gap']<1e-9;close(full['objective'],full['dual_bound']);costs=[]
   for j,sol in enumerate(r['scenario_solutions']):book,cost=independent_schedule(cfg,x,sc[j],sol);costs.append(cost);counts['scenario_schedule_audits']+=1
   costs=np.array(costs);close(costs,r['recourse_costs']);worst,_=lp(costs,prob,rho);weights=np.array(r['adversary']['weights']);assert weights.min()>=0;close(weights.sum(),1,1e-12);tv=.5*abs(weights-prob).sum();assert tv<=rho+1e-12;close(tv,r['adversary']['tv'],1e-12);close(weights@costs,worst);close(book+worst,r['objective']);close(book+prob@costs,r['nominal_total']);counts['probability_lp_oracles']+=1
   fullcost=[]
   for j,sol in enumerate(full['scenario_solutions']):fullbook,cost=independent_schedule(cfg,full['x'],sc[j],sol);fullcost.append(cost);counts['scenario_schedule_audits']+=1
   fullcost=np.array(fullcost);dual=full['dual_variables'];u=np.array(dual['u']);eta=dual['eta'];lam=dual['lambda_value'];assert lam>=-1e-8 and (abs(u)<=lam+1e-6).all() and (eta+u>=fullcost-1e-6).all();primal,_=lp(fullcost,prob,rho);close(eta+prob@u+2*rho*lam,primal);close(fullbook+primal,full['objective']);counts['probability_lp_oracles']+=1;counts['full_dual_certificates']+=1;history=r['trace'];assert len(history)==r['iterations']<=513;lbs=[];ubs=[]
   for i,h in enumerate(history):
    master=h['master'];assert master['solver_status']==0 and master['mip_gap']<1e-9;close(master['objective'],master['dual_bound']);close(master['objective'],h['lower_bound']);mc=[]
    for j,sol in enumerate(master['scenario_solutions']):mb,cost=independent_schedule(cfg,master['x'],sc[j],sol);mc.append(cost);counts['scenario_schedule_audits']+=1
    cuts=np.asarray(h['cuts']);assert len(cuts)==i+1;close(cuts[0],prob,1e-14);assert cuts.min()>=0;close(cuts.sum(1),1,1e-12);assert (.5*abs(cuts-prob).sum(1)<=rho+1e-12).all();close(mb+np.max(cuts@np.asarray(mc)),master['objective']);wv=np.asarray(h['worst_weights']);hv=np.asarray(h['recourse_costs']);assert wv.min()>=0;close(wv.sum(),1,1e-12);assert .5*abs(wv-prob).sum()<=rho+1e-12;opt,_=lp(hv,prob,rho);close(wv@hv,opt);close(mb+opt,h['candidate_upper']);counts['probability_lp_oracles']+=1;counts['master_certificates']+=1;lbs.append(h['lower_bound']);ubs.append(h['upper_bound'])
    if i+1<len(history):close(history[i+1]['cuts'],np.vstack([cuts,wv]),1e-14)
   assert (np.diff(lbs)>=-1e-6).all() and (np.diff(ubs)<=1e-6).all();assert (np.array(lbs)<=r['objective']+1e-6).all() and (np.array(ubs)>=r['objective']-1e-6).all();close(lbs[-1],r['lower_bound']);close(ubs[-1],r['upper_bound'])
   if rho in [0,1]:
    endpoint=record['endpoint_reference'];assert endpoint['solver_status']==0 and endpoint['mip_gap']<1e-9;close(endpoint['objective'],endpoint['dual_bound']);ec=[]
    for j,sol in enumerate(endpoint['scenario_solutions']):eb,cost=independent_schedule(cfg,endpoint['x'],sc[j],sol);ec.append(cost);counts['scenario_schedule_audits']+=1
    close(eb+(np.mean(ec) if rho==0 else np.max(ec)),endpoint['objective']);close(endpoint['objective'],r['objective'],1e-5);counts['endpoint_certificates']+=1
   else:assert record['endpoint_reference'] is None
   if seed==SEEDS[0] and pos in [positions[0],positions[-1]] and policy=='tv025':
    saved=OUT/'cases'/f'fresh_{item["key"]}.json'
    if saved.exists():fresh_record=read(saved);assert fresh_record['plan_sha256']==sha(path)
    else:
     freshfull=solve_distribution_master(p,available,prob,rho);freshq=[solve_recourse(p,a,x)['recourse_cost'] for a in available];fresh_record=dict(plan_sha256=sha(path),objective=freshfull['objective'],status=freshfull['solver_status'],costs=freshq);saved.write_text(json.dumps(fresh_record,indent=2),encoding='utf-8')
    close(fresh_record['objective'],r['objective'],1e-5);close(fresh_record['costs'],costs);assert fresh_record['status']==0;counts['selected_fresh_dual_solves']+=1;counts['selected_fresh_recourse_solves']+=57;fresh.append(dict(key=item['key'],objective_difference=fresh_record['objective']-r['objective']))
   outcome_path=OUT/'cases'/f'outcome_{item["key"]}__{policy}.json';out=read(outcome_path);assert outcome_path.stat().st_mtime>=(OUT/'planning_complete.json').stat().st_mtime;assert out['complete']==complete and out['resources']==x;key=item['issue_id'],tuple(x)
   if key not in actual_cache:actual_cache[key]=(solve_recourse(p,lo,x),solve_recourse(p,hi,x));counts['actual_recourse_solves']+=2
   actual_lo,actual_hi=actual_cache[key];_,al=independent_schedule(cfg,x,truth,out['optimistic_solution'],'optimistic');_,ah=independent_schedule(cfg,x,truth,out['pessimistic_solution'],'pessimistic');close(al,actual_lo['recourse_cost']);close(ah,actual_hi['recourse_cost']);close(out['cost_lower'],book+al);close(out['cost_upper'],book+ah);assert al<=ah+1e-6;close(out['known_fraction'],np.isfinite(truth[:,nodes]).mean(),1e-14)
   if complete:close(out['exact_cost'],book+al);close(al,ah);assert out['exact_completed']==out['optimistic_solution']['completed']
   else:assert out['exact_cost'] is None and out['exact_completed'] is None
   planning.append(dict(key=item['key'],fold=name,case=case,seed=seed,position=pos,issue_id=item['issue_id'],policy=policy,radius=rho,objective=r['objective'],nominal_total=r['nominal_total'],booking_cost=r['booking_cost'],boats=sum(x[:2]),crews=sum(x[2:5]),spares=sum(x[5:]),no_booking=int(sum(x)==0),iterations=r['iterations'],gap=r['gap'],oracle_solves=r['oracle_solves'],dual_objective=full['objective'],seconds=record['seconds'],plan_source=str(path),plan_sha256=sha(path)));outcomes.append(dict(fold=name,case=case,seed=seed,position=pos,**{k:out[k] for k in ['key','policy','issue_id','complete','known_fraction','cost_lower','cost_upper','exact_cost','exact_completed','booking_cost']},objective=r['objective'],nominal_total=r['nominal_total'],no_booking=int(sum(x)==0),outcome_source=str(outcome_path),outcome_sha256=sha(outcome_path)));cost_by_radius.append((rho,r['objective']));counts['plans']+=1
  cost_by_radius.sort();assert (np.diff([v for _,v in cost_by_radius])>=-1e-5).all();print('CASE_AUDIT',item['key'],flush=True)
 counts['planning_rows']=compare_table('planning_summary.csv',planning,['key','policy']);counts['outcome_rows']=compare_table('outcomes.csv',outcomes,['key','policy']);counts.update(verify_tables(pd.DataFrame(outcomes),times));pd.DataFrame(fresh).to_csv(OUT/'independent_resolves.csv',index=False,encoding='utf-8-sig');assert counts['plans']==1392 and counts['forecast_cases']==432 and counts['selected_fresh_dual_solves']==24;counts.update(status='PASS',parent_manifest_files=parents,previous_unchanged=len(read(OUT/'previous_hashes.json')),frozen_identities=len(read(OUT/'identity.json')));(OUT/'verification.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(json.dumps(counts,indent=2),flush=True)
def verify_tables(df,times):
 summaries=[]
 for scope in ['ALL']+list(FOLDS):
  selected=df if scope=='ALL' else df[df.fold==scope]
  for (case,policy),g in selected.groupby(['case','policy'],sort=False):summaries.append(dict(scope=scope,case=case,policy=policy,n=len(g),unique_origins=g.issue_id.nunique(),complete=int(g.complete.sum()),complete_unique_origins=g[g.complete].issue_id.nunique(),**{m:float(g[m].mean()) for m in ['cost_lower','cost_upper','exact_cost','exact_completed','booking_cost','objective','nominal_total','no_booking']}))
 metricrows=compare_table('metrics.csv',summaries,['scope','case','policy']);rng=np.random.default_rng(20261122);draws={};weeks={};ids={};saved=np.load(OUT/'bootstrap_draws.npz')
 for name,start in FOLDS.items():
  ids[name]=np.sort(df[df.fold==name].issue_id.unique());weeks[name]=((times[ids[name]]-pd.Timestamp(start,tz=times.tz)).days//7).to_numpy();k=int(weeks[name].max())+1;draws[name]=rng.integers(0,k,(2000,k));np.testing.assert_array_equal(draws[name],saved[name])
 pairs=pd.read_csv(OUT/'paired_comparisons.csv');assert len(pairs)==60 and pairs.primary.sum()==4;assert set(map(tuple,pairs[['candidate','policy','baseline','baseline_policy']].drop_duplicates().values))==set(PAIRS)
 for row in pairs.itertuples():
  selected=list(FOLDS) if row.scope=='ALL' else [row.scope];numlo=np.zeros(2000);numhi=np.zeros(2000);den=np.zeros(2000);slo=0.;shi=0.;n=0;blocks=[]
  for name in selected:
   lower=[];upper=[]
   for issue in ids[name]:
    a=df[(df.fold==name)&(df.issue_id==issue)&(df.case==row.candidate)&(df.policy==row.policy)].sort_values('seed');b=df[(df.fold==name)&(df.issue_id==issue)&(df.case==row.baseline)&(df.policy==row.baseline_policy)].sort_values('seed');assert len(a)==len(b)==3
    if row.metric=='all_cost_bounds':lower.append(float(np.mean(a.cost_lower.to_numpy()-b.cost_upper.to_numpy())));upper.append(float(np.mean(a.cost_upper.to_numpy()-b.cost_lower.to_numpy())))
    else:lower.append(float(np.mean(a[row.metric].to_numpy()-b[row.metric].to_numpy())));upper.append(lower[-1])
   lower=np.asarray(lower);upper=np.asarray(upper);valid=np.isfinite(lower)&np.isfinite(upper);k=draws[name].shape[1];sl=np.array([lower[valid&(weeks[name]==j)].sum() for j in range(k)]);sh=np.array([upper[valid&(weeks[name]==j)].sum() for j in range(k)]);cn=np.array([(valid&(weeks[name]==j)).sum() for j in range(k)]);numlo+=sl[draws[name]].sum(1);numhi+=sh[draws[name]].sum(1);den+=cn[draws[name]].sum(1);slo+=sl.sum();shi+=sh.sum();n+=cn.sum();blocks.append(int((cn>0).sum()))
  valid=den>0;supported=min(blocks)>=4 and valid.sum()>=1900;lq=np.quantile(numlo[valid]/den[valid],[.025,.975]) if supported else [np.nan,np.nan];uq=np.quantile(numhi[valid]/den[valid],[.025,.975]) if supported else [np.nan,np.nan];close(row.difference_lower,slo/n if n else np.nan,3e-10);close(row.difference_upper,shi/n if n else np.nan,3e-10);close([row.outer_low,row.lower_percentile_high,row.upper_percentile_low,row.outer_high],[lq[0],lq[1],uq[0],uq[1]],3e-10);assert row.n==n and row.minimum_blocks_in_fold==min(blocks) and row.valid_draws==valid.sum() and row.resampling_supported==supported and row.primary==(row.scope=='ALL' and row.metric=='all_cost_bounds')
 return dict(metric_rows=metricrows,paired_rows=len(pairs))
if __name__=='__main__':run()

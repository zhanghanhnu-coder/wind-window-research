"""Independent constraints, receding commitment, realized cost and tables."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import csr_matrix
from model import ART
from two_stage_core import make_problem
from v3.forecasting import ResearchForecaster
from verify_causal_execution import outcome_audit,close
from highs_readback_compat import install
from research_crps_distribution import guard
OUT=ART/'lookahead_execution_v1';SRC=ART/'causal_execution_v1';BOOK=ART/'decision_chain_v1';SEEDS=[20241101,20241102,20241103];MODELS=['old57','gated_graph57','gated_temporal57'];POLICIES=['deterministic','stochastic','regret'];RULES=['economic','guard80'];SPANS=[24,72];POSITIONS=list(range(0,232,10))
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def verify_grids(cfg):
 f=ResearchForecaster();cache={};checked=0
 for m in MODELS:
  for s in SEEDS:
   source=np.load(SRC/f'forecasts_{m}_{s}.npz');saved=dict(np.load(OUT/f'window_grids_{m}_{s}.npz'));np.testing.assert_array_equal(saved['quality'],source['quality']);np.testing.assert_array_equal(saved['grids'][:,0],source['probabilities'])
   for j,q in enumerate(source['q']):
    if not source['quality'][j]:assert np.isnan(saved['grids'][j]).all();continue
    if m=='old57':sc=np.maximum(0,q[...,1][None]+f.residual[s][f.selected57])
    else:
     theta=f.params[(m,s)];pp=np.stack([np.interp(np.arange(1,73),[3.5,15.5,36.5,60.5],theta[:,k]) for k in range(2)],-1);q=q.astype(float);med=q[...,9,None];atoms=np.maximum(0,med+pp[:,0,None,None]*f.meta['scale']+pp[:,1,None,None]*(q-med));sc=np.take_along_axis(np.repeat(atoms,3,axis=-1),f.ranks[(m,s)].transpose(1,2,0),axis=-1).transpose(2,0,1)
    for h,offset in enumerate(range(0,72,6)):
     for k,t in enumerate(cfg['tasks']):
      expected=np.nan if offset+t['duration']>72 else (sc[:,offset:offset+t['duration'],t['node']]<=10).all(1).mean();close(saved['grids'][j,h,k],expected,1e-14);checked+=1
   cache[(m,s)]=saved
 return cache,checked

def candidates(p,x,history,now,grid,span,rule):
 attempted={p.choices[i]['task'] for i in history};out=[];gains=[]
 for i,o in enumerate(p.choices):
  if o['task'] in attempted or not now<=o['start']<now+span or x[o['boat']]==0 or x[p.B+o['crew']]==0:continue
  pr=grid[(o['start']-now)//6,o['task']];gain=p.config['tasks'][o['task']]['penalty']*pr-o['cost']
  if gain>0 and (rule=='economic' or pr>=.8):out.append(i);gains.append(gain)
 return out,np.asarray(gains)

def independent_optimum(p,x,history,now,grid,span,rule):
 choices,gains=candidates(p,x,history,now,grid,span,rule)
 if not choices:return 0.,False
 oo=[p.choices[i] for i in choices];prior=[p.choices[i] for i in history];rows=[];bounds=[]
 def row(values,capacity):
  if any(values):rows.append(values);bounds.append(capacity)
 for j in range(len(p.config['tasks'])):row([int(o['task']==j) for o in oo],1)
 for kind,size,offset in [('boat',p.B,0),('crew',p.C,p.B)]:
  for resource in range(size):
   for hour in range(72):row([int(o[kind]==resource and o['start']<=hour<o['end']) for o in oo],x[offset+resource]-sum(o[kind]==resource and o['start']<=hour<o['end'] for o in prior))
 for node in sorted({t['node'] for t in p.config['tasks']}):
  for hour in range(72):row([int(o['node']==node and o['start']<=hour<o['end']) for o in oo],1-sum(o['node']==node and o['start']<=hour<o['end'] for o in prior))
 for k in range(p.K):row([o['quantity'] if o['spare']==k else 0 for o in oo],x[p.B+p.C+k]-sum(o['quantity'] for o in prior if o['spare']==k))
 result=milp(-gains,integrality=np.ones(len(choices)),bounds=Bounds(np.zeros(len(choices)),np.ones(len(choices))),constraints=LinearConstraint(csr_matrix(np.array(rows,float)),np.full(len(rows),-np.inf),np.array(bounds,float)),options={'mip_rel_gap':0.,'time_limit':60.});assert result.status==0 and result.mip_gap<1e-9;close(result.fun,result.mip_dual_bound,1e-7);return float(-result.fun),True

def bootstrap(values,weeks):
 values=np.asarray(values,float);weeks=np.asarray(weeks);draw=np.random.default_rng(20261112).integers(0,9,(2000,9));samples=[]
 for block in draw:
  sample=np.concatenate([values[weeks==k] for k in block])
  if len(sample):samples.append(sample.mean())
 enough=len(values)>=8 and len(set(weeks))>=4 and len(samples)>=1900;lo,hi=np.quantile(samples,[.025,.975]) if enough else (np.nan,np.nan)
 return dict(difference=values.mean() if len(values) else np.nan,low95=lo,high95=hi,complete_origins=len(values),nonempty_weeks=len(set(weeks)),valid_draws=len(samples),interval_reported=enough)

def tables(df):
 metrics=pd.read_csv(OUT/'metrics.csv');pairs=pd.read_csv(OUT/'paired_comparisons.csv');bounds=pd.read_csv(OUT/'missing_bounds.csv');assert len(metrics)==6156 and len(pairs)==52 and len(bounds)==52 and pairs.primary.sum()==5
 for r in metrics.itertuples():
  g=df[(df.family==r.family)&(df.policy==r.policy)&(df.rule==r.rule)&(df.planner==r.planner)&(df.seed==r.seed)]
  if r.scope=='complete':g=g[g.complete]
  else:assert r.scope=='all'
  close(r.value,g[r.metric].mean());assert r.n==g[r.metric].notna().sum()
 times=pd.to_datetime(df.drop_duplicates('position').set_index('position').timestamp,utc=True).dt.tz_convert('Asia/Shanghai');weeks=((times-pd.Timestamp('2024-11-01',tz='Asia/Shanghai')).dt.days//7).to_dict()
 for r in pairs.itertuples():
  values=[];ww=[];lower=[];upper=[]
  for pos in POSITIONS:
   a=df[(df.family==r.candidate)&(df.policy==r.policy)&(df.rule==r.rule)&(df.planner==r.planner)&(df.position==pos)];b=df[(df.family==r.baseline)&(df.policy==r.policy)&(df.rule==r.rule)&(df.planner==r.baseline_planner)&(df.position==pos)];assert len(a)==len(b)==3;lower.append(a.cost_lower.mean()-b.cost_upper.mean());upper.append(a.cost_upper.mean()-b.cost_lower.mean())
   if a.complete.all() and b.complete.all():values.append(a.actual_cost.mean()-b.actual_cost.mean());ww.append(weeks[pos])
  for k,value in bootstrap(values,ww).items():close(getattr(r,k),value)
  primary=r.planner=='lookahead72' and r.rule=='economic' and ((r.kind=='planner' and ((r.candidate=='old57' and r.policy=='stochastic') or (r.candidate=='gated_graph57' and r.policy in ['stochastic','regret']))) or (r.kind=='model' and r.baseline=='gated_temporal57'));assert r.primary==primary;bb=bounds[(bounds.kind==r.kind)&(bounds.candidate==r.candidate)&(bounds.baseline==r.baseline)&(bounds.policy==r.policy)&(bounds.rule==r.rule)&(bounds.planner==r.planner)&(bounds.baseline_planner==r.baseline_planner)].iloc[0];close(bb.difference_lower,np.mean(lower));close(bb.difference_upper,np.mean(upper));assert bb.origins==24
 return dict(metrics=len(metrics),paired=len(pairs),missing_bounds=len(bounds))

def run():
 guard();runtime=install();previous=read(OUT/'previous_hashes.json');identity=read(OUT/'identity.json');modelids=read(OUT/'model_identity.json')
 for n,h in previous.items():assert sha(ART/n)==h,n
 for n,h in {**identity,**modelids}.items():assert sha(Path(n))==h,n
 for n,h in read(SRC/'final_manifest.json').items():assert sha(Path(n))==h,n
 assert runtime==read(OUT/'recipe.json')['runtime_compatibility'];assert read(OUT/'trajectory_complete.json')['completed']==2592;cfg=read(OUT/'recipe.json')['config'];p=make_problem(cfg);cache,probchecks=verify_grids(cfg);ids=np.load(SRC/'update_inputs.npz')['issue_ids'];raw=np.load(ART/'data.npz')['raw'];source={(r['family'],r['seed'],r['position']):r for r in read(SRC/'booking_inputs.json')};df=pd.read_csv(OUT/'cases.csv');cert=pd.read_csv(OUT/'planning_certificates.csv');attempts=pd.read_csv(OUT/'attempts.csv');expected={(m,s,pos,policy,rule,planner) for m in MODELS for s in SEEDS for pos in POSITIONS for policy in POLICIES for rule in RULES for planner in ['myopic','lookahead24','lookahead72']};assert len(df)==len(expected)==3888 and set(zip(df.family,df.seed,df.position,df.policy,df.rule,df.planner))==expected and len(cert)==31104
 base=df[df.planner=='myopic'].reset_index(drop=True);old=pd.read_csv(SRC/'cases.csv');pd.testing.assert_frame_equal(base[old.columns],old,check_exact=False,atol=1e-9,rtol=0);oldattempts=pd.read_csv(SRC/'attempts.csv');pd.testing.assert_frame_equal(attempts[attempts.planner=='myopic'][oldattempts.columns].reset_index(drop=True),oldattempts,check_exact=False,atol=1e-9,rtol=0)
 ci=cert.set_index(['family','seed','position','policy','rule','planner','now']).sort_index();ai=attempts.set_index(['family','seed','position','policy','rule','planner']).sort_index();casecount=stepcount=freshcount=freshcalls=attemptcount=0;oraclecache={}
 for row in df[df.planner!='myopic'].itertuples():
  guard();key=(row.family,row.seed,row.position);src=source[key];name=f'{row.family}_{row.seed}_{row.position:03d}_{row.policy}_{row.rule}_s{row.planner.replace("lookahead", "")}.json';file=OUT/f'trajectory_{name}';record=read(file);a=record['result'];x=src['resources'][row.policy];span=int(row.planner.replace('lookahead',''));assert a['resources']==x and a['rule']==row.rule and a['span']==record['span']==span;assert record['booking_source_sha256']==src['source_sha256'];indices=np.searchsorted(ids,src['issue_id']+np.arange(0,72,6));np.testing.assert_array_equal(record['update_indices'],indices);history=[];previous_plan={};postponed=future_only=skipped=0;selected_fresh=row.position==0 or (row.seed==SEEDS[0] and row.position in [110,230]);freshfile=OUT/f'fresh_optima_{name}';fresh=read(freshfile) if freshfile.exists() else dict(trajectory_sha256=sha(file),steps=[]);assert fresh['trajectory_sha256']==sha(file)
  for k in ['family','seed','position','issue_id','timestamp']:assert record[k]==src[k]==getattr(row,k)
  assert len(a['steps'])==12
  for j,now in enumerate(range(0,72,6)):
   step=a['steps'][j];data=cache[(row.family,row.seed)];grid=data['grids'][indices[j]];quality=bool(data['quality'][indices[j]]);assert step['now']==now and step['quality_ok']==quality;possible,gain=candidates(p,x,history,now,grid,span,row.rule) if quality else ([],np.array([]));mapping=dict(zip(possible,gain));assert step['candidate_count']==len(possible) and step['solver_called']==bool(possible);assert all(i in mapping for i in step['planned']);outcome_audit(p,x,history+step['planned'],np.zeros((72,17)));value=sum(mapping[i] for i in step['planned']);close(value,step['expected_benefit'],1e-7);close(value,step['dual_bound'],1e-7);assert step['solver_status']==0 and step['mip_gap']<1e-9;assert step['selected']==[i for i in step['planned'] if p.choices[i]['start']==now]
   if not quality:assert step['planned']==step['selected']==[];skipped+=1
   if selected_fresh:
    if len(fresh['steps'])<=j:
     optimum,called=independent_optimum(p,x,history,now,grid,span,row.rule) if quality else (0.,False);fresh['steps'].append(dict(now=now,optimum=optimum,solver_called=called));freshfile.write_text(json.dumps(fresh,indent=2),encoding='utf-8')
    check=fresh['steps'][j];assert check['now']==now;close(value,check['optimum'],1e-6);freshcount+=1;freshcalls+=check['solver_called']
   current={p.choices[i]['task']:p.choices[i]['start'] for i in step['planned']};postponed+=sum(t in previous_plan and start>previous_plan[t] for t,start in current.items());future_only+=bool(step['planned'] and not step['selected']);previous_plan=current;history.extend(step['selected']);cr=ci.loc[(row.family,row.seed,row.position,row.policy,row.rule,row.planner,now)]
   for k in ['quality_ok','expected_benefit','dual_bound','mip_gap','solver_status','solver_called','candidate_count','seconds']:close(cr[k],step[k])
   assert json.loads(cr.planned)==step['planned'] and json.loads(cr.selected)==step['selected'];stepcount+=1
  assert history==a['attempts'] and a['postponed_task_plans']==row.postponed_task_plans==postponed and a['future_only_updates']==row.future_only_updates==future_only and a['skipped_updates']==row.skipped_updates==skipped;truth=raw[src['issue_id']+1:src['issue_id']+73];complete=np.isfinite(truth[:,sorted({t['node'] for t in cfg['tasks']})]).all();assert row.complete==complete;e=outcome_audit(p,x,history,truth);ev=read(OUT/f'evaluated_{name}');assert ev['trajectory_sha256']==sha(file)
  for k,value in e.items():
   if k in ['outcomes','spares_used']:assert ev['evaluation'][k]==value
   else:close(ev['evaluation'][k],value);close(getattr(row,k),value)
  if key not in oraclecache:oraclecache[key]=read(BOOK/f'case_{row.family}_{row.seed}_{row.position:03d}.json')
  oracle=oraclecache[key]['observed'][row.policy];close(row.actual_cost,e['cost_lower'] if complete else None);close(row.oracle_cost,oracle['exact_cost'] if complete else None);close(row.execution_gap,e['cost_lower']-oracle['exact_cost'] if complete else None);close(row.gap_lower,e['cost_lower']-oracle['cost_upper']);close(row.gap_upper,e['cost_upper']-oracle['cost_lower'])
  if complete:assert row.execution_gap>=-1e-6
  dur=sum(p.choices[i]['end']-p.choices[i]['start'] for i in history);assert row.boat_hours==row.crew_hours==dur;close(row.boat_utilization,dur/(sum(x[:p.B])*72) if sum(x[:p.B]) else None);close(row.crew_utilization,dur/(sum(x[p.B:p.B+p.C])*72) if sum(x[p.B:p.B+p.C]) else None)
  groupkey=(row.family,row.seed,row.position,row.policy,row.rule,row.planner);at=ai.loc[groupkey] if groupkey in ai.index else pd.DataFrame();assert len(at)==len(history)
  for z in e['outcomes']:
   rr=at[at.choice==z['choice']].iloc[0];o=p.choices[z['choice']];assert rr.outcome==z['status'] and rr.known_hours==z['known_hours']
   for k in ['task','start','end','boat','crew']:assert rr[k]==o[k]
   close(rr.operating_cost,o['cost']);close(rr.probability,cache[(row.family,row.seed)]['grids'][indices[o['start']//6],0,o['task']]);close(rr.success,1 if z['status']=='successful' else 0 if z['status']=='failed' else None);attemptcount+=1
  casecount+=1
  if casecount%100==0:print('AUDITED',casecount,'fresh optima',freshcount,flush=True)
 result=dict(status='PASS',previous_unchanged=len(previous),frozen_identities=len(identity),model_identities=len(modelids),probability_grid_entries=probchecks,baseline_rows_unchanged=len(base),new_trajectories=casecount,planning_certificates=stepcount,selected_fresh_optima=freshcount,independent_milp_calls=freshcalls,new_attempts_audited=attemptcount,table_rows=tables(df));(OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':run()

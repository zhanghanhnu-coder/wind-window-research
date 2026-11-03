"""Independent causal dispatch, resource, outcome and statistical verification."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import torch
from model import ART
from v3.forecasting import ResearchForecaster
from two_stage_core import make_problem
from research_crps_distribution import guard
OUT=ART/'causal_execution_v1';SRC=ART/'decision_chain_v1';SEEDS=[20241101,20241102,20241103];MODELS=['old57','gated_graph57','gated_temporal57'];POLICIES=['deterministic','stochastic','regret'];RULES=['economic','guard80'];POSITIONS=list(range(0,232,10))
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=1e-9):
 if b is None or not np.isfinite(b):assert a is None or pd.isna(a)
 else:assert np.isclose(a,b,atol=tol,rtol=0),(a,b)

def forecasts(cfg):
 f=ResearchForecaster();info=np.load(OUT/'update_inputs.npz');ids=info['issue_ids'];hist=np.stack([f.x[i-23:i+1] for i in ids]);quality=hist[...,2].mean((1,2))>=.65;np.testing.assert_array_equal(info['source_cal_ids'],f.cal_ids[f.selected57]);assert max(info['source_cal_ids']+72)<min(ids);cache={};checks=0;maximum=0.;initial=[];pdict={};raw=np.load(ART/'data.npz')['raw']
 for model in MODELS:
  for seed in SEEDS:
   guard();saved=dict(np.load(OUT/f'forecasts_{model}_{seed}.npz'));key=('old' if model=='old57' else model,seed);net=f.models[key]
   np.testing.assert_array_equal(saved['quality'],quality);np.testing.assert_array_equal(saved['valid_fraction'],hist[...,2].mean((1,2)));np.testing.assert_array_equal(saved['valid_hours'],(hist[...,2]>0).sum(1));np.testing.assert_array_equal(saved['history_sha256'],[hashlib.sha256(np.ascontiguousarray(h).tobytes()).hexdigest() for h in hist]);assert np.isnan(saved['q'][~quality]).all() and np.isnan(saved['probabilities'][~quality]).all()
   for j in np.unique(np.linspace(0,len(ids)-1,7,dtype=int)):
    if not quality[j]:continue
    with torch.no_grad():q=net(torch.from_numpy(hist[j:j+1])).numpy()[0]
    if model=='old57':q=q*f.meta['scale']+f.meta['center']
    maximum=max(maximum,float(abs(q-saved['q'][j]).max()));np.testing.assert_allclose(q,saved['q'][j],atol=1e-5,rtol=0);checks+=1
   old=np.load(SRC/f'scenarios_{model}_{seed}.npz');im={int(issue):i for i,issue in enumerate(old['issue_ids'])};maxinitial=0.
   for j in np.flatnonzero(quality):
    q=saved['q'][j]
    if model=='old57':sc=np.maximum(0,q[...,1][None]+f.residual[seed][f.selected57])
    else:
     theta=f.params[key];param=np.stack([np.interp(np.arange(1,73),[3.5,15.5,36.5,60.5],theta[:,k]) for k in range(2)],-1);q=q.astype(float);med=q[...,9,None];atoms=np.maximum(0,med+param[:,0,None,None]*f.meta['scale']+param[:,1,None,None]*(q-med));repeated=np.repeat(atoms,3,axis=-1);sc=np.take_along_axis(repeated,f.ranks[key].transpose(1,2,0),axis=-1).transpose(2,0,1)
    prob=np.array([(sc[:,:t['duration'],t['node']]<=10).all(1).mean() for t in cfg['tasks']]);np.testing.assert_array_equal(prob,saved['probabilities'][j])
    if int(ids[j]) in im:
     ref=old['scenarios'][im[int(ids[j])]];maxinitial=max(maxinitial,float(abs(sc-ref).max()));np.testing.assert_allclose(sc,ref,atol=1e-5,rtol=0);np.testing.assert_array_equal(prob,[(ref[:,:t['duration'],t['node']]<=10).all(1).mean() for t in cfg['tasks']])
   initial.append(dict(model=model,seed=seed,origins=24,max_initial_scenario_difference=maxinitial,unique_updates=len(ids),skipped=int((~quality).sum())));cache[(model,seed)]=saved
 assert read(OUT/'initial_forecast_checks.json')==initial
 return ids,cache,raw,dict(scalar_network_checks=checks,maximum_scalar_difference=maximum,full_probability_updates=int(quality.sum())*9,initial_scenario_groups=216)

def best_current(p,x,history,now,prob,rule):
 previous=[p.choices[i] for i in history];seen={o['task'] for o in previous};active=[o for o in previous if o['end']>now];occupied_b={o['boat'] for o in active};occupied_c={o['crew'] for o in active};occupied_n={o['node'] for o in active};spares=np.array(x[p.B+p.C:],int)
 for o in previous:spares[o['spare']]-=o['quantity']
 options={j:[] for j in range(len(p.config['tasks']))};candidates=[]
 for i,o in enumerate(p.choices):
  if o['start']!=now or o['task'] in seen or o['boat'] in occupied_b or o['crew'] in occupied_c or o['node'] in occupied_n:continue
  if x[o['boat']]!=1 or x[p.B+o['crew']]!=1 or spares[o['spare']]<o['quantity']:continue
  value=float(p.config['tasks'][o['task']]['penalty']*prob[o['task']]-o['cost'])
  if value<=0 or (rule=='guard80' and prob[o['task']]<.8):continue
  options[o['task']].append((i,value));candidates.append(dict(choice=i,expected_benefit=value))
 best=();bestvalue=0.
 def visit(task,chosen,benefit,boats,crews,nodes,stock):
  nonlocal best,bestvalue
  if task==len(p.config['tasks']):
   ids=tuple(sorted(chosen))
   if benefit>bestvalue or (benefit==bestvalue and ids<best):best=ids;bestvalue=benefit
   return
  visit(task+1,chosen,benefit,boats,crews,nodes,stock)
  for i,value in options[task]:
   o=p.choices[i]
   if o['boat'] in boats or o['crew'] in crews or o['node'] in nodes or stock[o['spare']]<o['quantity']:continue
   nextstock=stock.copy();nextstock[o['spare']]-=o['quantity'];visit(task+1,chosen+[i],benefit+value,boats|{o['boat']},crews|{o['crew']},nodes|{o['node']},nextstock)
 visit(0,[],0.,set(),set(),set(),spares)
 return list(best),bestvalue,candidates,spares.tolist()

def outcome_audit(p,x,attempts,truth):
 boats=np.zeros((p.B,72),int);crews=np.zeros((p.C,72),int);nodes={t['node']:np.zeros(72,int) for t in p.config['tasks']};stock=np.zeros(p.K,int);tasks=set();operating=0.;outcomes=[];success=failed=unknown=0;penalty=0.;uncertain=0.
 for i in attempts:
  assert isinstance(i,int) and 0<=i<len(p.choices);o=p.choices[i];t=p.config['tasks'][o['task']];assert o['task'] not in tasks and t['skill'] in p.config['crew_skills'][o['crew']];tasks.add(o['task']);boats[o['boat'],o['start']:o['end']]+=1;crews[o['crew'],o['start']:o['end']]+=1;nodes[o['node']][o['start']:o['end']]+=1;stock[o['spare']]+=o['quantity'];cost=(p.config['boat_hourly'][o['boat']]+p.config['crew_hourly'][o['crew']])*t['duration']+p.config['late_fraction']*t['penalty']*o['start']/72;close(cost,o['cost']);operating+=cost;y=truth[o['start']:o['end'],o['node']]
  if any(np.isfinite(y)&(y>10)):status='failed';failed+=1;penalty+=t['penalty']
  elif np.isfinite(y).all():status='successful';success+=1
  else:status='unknown';unknown+=1;uncertain+=t['penalty']
  outcomes.append(dict(choice=i,task=o['task'],status=status,revealed_at=o['end'],known_hours=int(np.isfinite(y).sum())))
 assert (boats<=np.array(x[:p.B])[:,None]).all() and (crews<=np.array(x[p.B:p.B+p.C])[:,None]).all() and (stock<=x[p.B+p.C:]).all() and all((z<=1).all() for z in nodes.values());penalty+=sum(t['penalty'] for j,t in enumerate(p.config['tasks']) if j not in tasks);book=sum(a*b for a,b in zip(x,p.config['boat_booking']+p.config['crew_booking']+p.config['spare_cost']));lower=book+operating+penalty
 return dict(booking_cost=book,operating_cost=operating,penalty_lower=penalty,penalty_upper=penalty+uncertain,cost_lower=lower,cost_upper=lower+uncertain,successful=success,failed=failed,unknown=unknown,unattempted=len(p.config['tasks'])-len(tasks),attempted=len(attempts),spares_used=stock.tolist(),outcomes=outcomes)

def independent_interval(values,weeks):
 values=np.asarray(values,float);weeks=np.asarray(weeks);draw=np.random.default_rng(20261111).integers(0,9,(2000,9));samples=[]
 for keys in draw:
  sample=np.concatenate([values[weeks==k] for k in keys])
  if len(sample):samples.append(sample.mean())
 enough=len(values)>=8 and len(set(weeks))>=4 and len(samples)>=1900;lo,hi=np.quantile(samples,[.025,.975]) if enough else (np.nan,np.nan)
 return dict(difference=float(values.mean()) if len(values) else np.nan,low95=lo,high95=hi,complete_origins=len(values),nonempty_weeks=len(set(weeks)),valid_draws=len(samples),interval_reported=enough)

def tables(df):
 m=pd.read_csv(OUT/'metrics.csv');pairs=pd.read_csv(OUT/'paired_comparisons.csv');bounds=pd.read_csv(OUT/'missing_bounds.csv');gaps=pd.read_csv(OUT/'oracle_gaps.csv');rel=pd.read_csv(OUT/'attempt_reliability.csv');attempts=pd.read_csv(OUT/'attempts.csv');assert len(df)==1296 and len(m)==1836 and len(rel)==360 and len(pairs)==17 and len(bounds)==17 and len(gaps)==18 and pairs.primary.sum()+gaps.primary.sum()==5
 for r in m.itertuples():
  g=df[(df.family==r.family)&(df.policy==r.policy)&(df.rule==r.rule)&(df.seed==r.seed)]
  if r.scope=='complete':g=g[g.complete]
  else:assert r.scope=='all'
  close(r.value,g[r.metric].mean());assert r.n==g[r.metric].notna().sum()
 times=pd.to_datetime(df.drop_duplicates('position').set_index('position').timestamp,utc=True).dt.tz_convert('Asia/Shanghai');weeks=((times-pd.Timestamp('2024-11-01',tz='Asia/Shanghai')).dt.days//7).to_dict()
 for r in pairs.itertuples():
  values=[];ww=[];low=[];high=[]
  for pos in POSITIONS:
   a=df[(df.family==r.candidate)&(df.policy==r.policy)&(df.rule==r.rule)&(df.position==pos)];b=df[(df.family==r.baseline)&(df.policy==r.policy)&(df.rule==r.baseline_rule)&(df.position==pos)];assert len(a)==len(b)==3;low.append(a.cost_lower.mean()-b.cost_upper.mean());high.append(a.cost_upper.mean()-b.cost_lower.mean())
   if a.complete.all() and b.complete.all():values.append(a.actual_cost.mean()-b.actual_cost.mean());ww.append(weeks[pos])
  expected=independent_interval(values,ww)
  for k,value in expected.items():close(getattr(r,k),value)
  bound=bounds[(bounds.kind==r.kind)&(bounds.candidate==r.candidate)&(bounds.baseline==r.baseline)&(bounds.policy==r.policy)&(bounds.rule==r.rule)&(bounds.baseline_rule==r.baseline_rule)].iloc[0];close(bound.difference_lower,np.mean(low));close(bound.difference_upper,np.mean(high));assert bound.origins==24 and r.primary==(r.kind=='model' and r.rule=='economic')
 for r in gaps.itertuples():
  g=df[(df.family==r.family)&(df.policy==r.policy)&(df.rule==r.rule)&df.complete].groupby('position').execution_gap.mean();expected=independent_interval(g.to_numpy(),[weeks[p] for p in g.index])
  for k,value in expected.items():close(getattr(r,k),value)
  assert r.primary==(r.family=='old57' and r.policy=='stochastic' and r.rule=='economic')
 for r in rel.itertuples():
  g=attempts[(attempts.family==r.family)&(attempts.policy==r.policy)&(attempts.rule==r.rule)];g=g[g.success.notna()] if r.scope=='all_known' else g[g.position.isin(df[df.complete].position.unique())];g=g[(g.probability>=r.bin/10)&((g.probability<(r.bin+1)/10) if r.bin<9 else g.probability<=1)];assert r.n==len(g);close(r.mean_probability,g.probability.mean());close(r.observed_rate,g.success.mean())
 return dict(cases=len(df),attempts=len(attempts),metrics=len(m),paired=len(pairs),bounds=len(bounds),oracle_gaps=len(gaps),attempt_reliability=len(rel))

def run():
 guard();old=read(OUT/'previous_hashes.json');ident=read(OUT/'identity.json');models=read(OUT/'model_identity.json');revision=read(OUT/'evaluation_audit_revision.json')
 for n,h in old.items():assert sha(ART/n)==h,n
 for n,h in {**ident,**models,**revision['additional_source_identities']}.items():assert sha(Path(n))==h,n
 for n,h in revision['preserved_predictions_and_trajectories'].items():assert sha(OUT/n)==h,n
 assert revision['forecast_or_decision_changed'] is False
 for n,h in read(SRC/'final_manifest.json').items():assert sha(Path(n))==h,n
 cfg=read(OUT/'recipe.json')['config'];p=make_problem(cfg);records=read(OUT/'booking_inputs.json');source={}
 for r in records:
  assert sha(Path(r['source_file']))==r['source_sha256'];original=read(Path(r['source_file']));assert r['resources']=={k:original['planned']['plans'][k]['x'] for k in POLICIES};source[(r['family'],r['seed'],r['position'])]=r
 assert len(source)==216 and read(OUT/'trajectory_complete.json')['completed']==1296;ids,cache,raw,forecastcounts=forecasts(cfg);df=pd.read_csv(OUT/'cases.csv');attempttable=pd.read_csv(OUT/'attempts.csv');decisions=0;casechecks=0;attemptchecks=0
 expected={(m,s,pos,policy,rule) for m in MODELS for s in SEEDS for pos in POSITIONS for policy in POLICIES for rule in RULES};assert set(zip(df.family,df.seed,df.position,df.policy,df.rule))==expected and len(df)==len(expected)
 for row in df.itertuples():
  guard();key=(row.family,row.seed,row.position);r=source[key];name=f'{row.family}_{row.seed}_{row.position:03d}_{row.policy}_{row.rule}.json';path=OUT/f'trajectory_{name}';saved=read(path);a=saved['result'];x=r['resources'][row.policy];assert a['resources']==x and saved['booking_source_sha256']==r['source_sha256'];data=cache[(row.family,row.seed)];indices=np.searchsorted(ids,r['issue_id']+np.arange(0,72,6));np.testing.assert_array_equal(saved['update_indices'],indices);history=[];skipped=0
  for k in ['family','seed','position','issue_id','timestamp']:assert saved[k]==r[k]==getattr(row,k)
  assert saved['policy']==row.policy and saved['rule']==a['rule']==row.rule and len(a['steps'])==12
  for j,now in enumerate(range(0,72,6)):
   step=a['steps'][j];idx=indices[j];assert step['now']==now and step['quality_ok']==bool(data['quality'][idx]);assert step['active_attempts']==[i for i in history if p.choices[i]['end']>now]
   if data['quality'][idx]:
    prob=data['probabilities'][idx];np.testing.assert_array_equal(step['probabilities'],prob);selected,value,candidates,stock=best_current(p,x,history,now,prob,row.rule);assert selected==step['selected'] and candidates==step['candidates'] and stock==step['remaining_spares'];close(step['expected_benefit'],value);decisions+=1
   else:
    assert step['selected']==[] and step['probabilities'] is None and step['candidates']==[] and step['expected_benefit']==0;stock=np.array(x[p.B+p.C:],int)
    for i in history:stock[p.choices[i]['spare']]-=p.choices[i]['quantity']
    np.testing.assert_array_equal(stock,step['remaining_spares']);skipped+=1
   history.extend(step['selected'])
  assert history==a['attempts'] and skipped==a['skipped_updates']==row.skipped_updates;truth=raw[r['issue_id']+1:r['issue_id']+73];complete=np.isfinite(truth[:,sorted({t['node'] for t in cfg['tasks']})]).all();assert row.complete==complete;e=outcome_audit(p,x,history,truth);evaluation=read(OUT/f'evaluated_{name}');assert evaluation['trajectory_sha256']==sha(path)
  for k,value in e.items():
   if k in ['outcomes','spares_used']:assert evaluation['evaluation'][k]==value
   else:close(evaluation['evaluation'][k],value);close(getattr(row,k),value)
  oracle=read(SRC/f'case_{row.family}_{row.seed}_{row.position:03d}.json')['observed'][row.policy];close(row.actual_cost,e['cost_lower'] if complete else None);close(row.oracle_cost,oracle['exact_cost'] if complete else None);close(row.execution_gap,e['cost_lower']-oracle['exact_cost'] if complete else None);close(row.gap_lower,e['cost_lower']-oracle['cost_upper']);close(row.gap_upper,e['cost_upper']-oracle['cost_lower'])
  if complete:assert row.execution_gap>=-1e-6
  dur=sum(p.choices[i]['end']-p.choices[i]['start'] for i in history);assert row.boat_hours==row.crew_hours==dur;close(row.boat_utilization,dur/(sum(x[:p.B])*72) if sum(x[:p.B]) else None);close(row.crew_utilization,dur/(sum(x[p.B:p.B+p.C])*72) if sum(x[p.B:p.B+p.C]) else None)
  at=attempttable[(attempttable.family==row.family)&(attempttable.seed==row.seed)&(attempttable.position==row.position)&(attempttable.policy==row.policy)&(attempttable.rule==row.rule)];assert len(at)==len(history)
  for z in e['outcomes']:
   rr=at[at.choice==z['choice']].iloc[0];o=p.choices[z['choice']];assert rr.outcome==z['status'] and rr.known_hours==z['known_hours']
   for k in ['task','start','end','boat','crew']:assert rr[k]==o[k]
   close(rr.operating_cost,o['cost']);close(rr.probability,a['steps'][o['start']//6]['probabilities'][o['task']]);close(rr.success,1 if z['status']=='successful' else 0 if z['status']=='failed' else None);attemptchecks+=1
  casechecks+=1
  if casechecks%100==0:print('AUDITED TRAJECTORIES',casechecks,flush=True)
 result=dict(status='PASS',previous_unchanged=len(old),frozen_sources=len(ident),model_identities=len(models),preserved_revision_files=len(revision['preserved_predictions_and_trajectories']),verified_trajectories=casechecks,independent_batch_optimizations=decisions,independent_attempt_outcomes=attemptchecks,forecast_checks=forecastcounts,tables=tables(df));(OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':run()

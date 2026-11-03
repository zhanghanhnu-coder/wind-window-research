"""Round24: expected-reward lookahead versus frozen current-batch dispatch."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
import numpy as np
import pandas as pd
from model import ART
from v3.forecasting import ResearchForecaster
from causal_execution_forecast import scenarios_from_quantiles
from causal_execution_core import evaluate_execution
from causal_evaluation_scope import completed_task_weather_options
from lookahead_execution_core import window_grid,roll_execution
from two_stage_core import make_problem,audit_schedule,SolverFailure
from highs_readback_compat import install
from research_crps_distribution import guard
OUT=ART/'lookahead_execution_v1';SRC=ART/'causal_execution_v1';BOOK=ART/'decision_chain_v1';SEEDS=[20241101,20241102,20241103];MODELS=['old57','gated_graph57','gated_temporal57'];POLICIES=['deterministic','stochastic','regret'];RULES=['economic','guard80'];SPANS=[24,72];POSITIONS=list(range(0,232,10));PLAN=ART.parent.parent/'研究笔记/考虑后续窗口的滚动执行对照方案_2026-11-02.md'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(n,obj):(OUT/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(n,rows):pd.DataFrame(rows).to_csv(OUT/n,index=False,encoding='utf-8-sig')

def prepare():
 guard();assert (SRC/'final_manifest.json').exists();assert read(SRC/'verification.json')['status']=='PASS';OUT.mkdir(exist_ok=True);protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0];f=OUT/'protocol_frozen.md'
 if f.exists():assert f.read_text(encoding='utf-8')==protocol
 else:f.write_text(protocol,encoding='utf-8')
 if not (OUT/'previous_hashes.json').exists():js('previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 files=[f,Path(__file__),SRC/'final_manifest.json',SRC/'recipe.json',SRC/'booking_inputs.json',SRC/'update_inputs.npz',ART/'data.npz',ART/'metadata.json']+[ART.parent/n for n in ['lookahead_execution_core.py','causal_execution_core.py','causal_execution_forecast.py','causal_evaluation_scope.py','two_stage_core.py','v3/forecasting.py','highs_readback_compat.py','research_crps_distribution.py']]+[SRC/f'forecasts_{m}_{s}.npz' for m in MODELS for s in SEEDS];ident={str(p):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==ident
 else:js('identity.json',ident)
 cfg=read(SRC/'recipe.json')['config'];runtime=install()
 if not (OUT/'recipe.json').exists():js('recipe.json',dict(models=MODELS,seeds=SEEDS,positions=POSITIONS,policies=POLICIES,rules=RULES,spans=SPANS,config=cfg,threshold=10.,planned_new_trajectories=2592,bootstrap_seed=20261112,solver_time_limit=60.,runtime_compatibility=runtime,start_utc=datetime.now(timezone.utc).isoformat()))
 return cfg,read(SRC/'booking_inputs.json')

def probabilities(cfg):
 f=ResearchForecaster();identity={str(ART/n):h for n,h in f.file_ids.items()}
 if (OUT/'model_identity.json').exists():assert read(OUT/'model_identity.json')==identity
 else:js('model_identity.json',identity)
 for m in MODELS:
  for s in SEEDS:
   guard();target=OUT/f'window_grids_{m}_{s}.npz';source=np.load(SRC/f'forecasts_{m}_{s}.npz')
   if not target.exists():
    grids=np.full((len(source['q']),12,6),np.nan)
    for j in np.flatnonzero(source['quality']):grids[j]=window_grid(scenarios_from_quantiles(f,source['q'][j],m,s),cfg)
    np.savez_compressed(target,grids=grids,quality=source['quality'])
   result=np.load(target);np.testing.assert_array_equal(result['grids'][:,0],source['probabilities']);np.testing.assert_array_equal(result['quality'],source['quality']);print('WINDOW GRIDS',m,s,flush=True)
 js('probability_complete.json',dict(flows=9,utc=datetime.now(timezone.utc).isoformat()))

def trajectories(cfg,records):
 assert (OUT/'probability_complete.json').exists();ids=np.load(SRC/'update_inputs.npz')['issue_ids'];cache={(m,s):dict(np.load(OUT/f'window_grids_{m}_{s}.npz')) for m in MODELS for s in SEEDS};failures=read(OUT/'failures.json') if (OUT/'failures.json').exists() else [];completed=0;stopped=False
 for r in records:
  data=cache[(r['family'],r['seed'])];indices=np.searchsorted(ids,r['issue_id']+np.arange(0,72,6));np.testing.assert_array_equal(ids[indices],r['issue_id']+np.arange(0,72,6))
  for policy in POLICIES:
   for rule in RULES:
    for span in SPANS:
     guard();key=f'{r["family"]}_{r["seed"]}_{r["position"]:03d}_{policy}_{rule}_s{span}';file=OUT/f'trajectory_{key}.json'
     if not file.exists():
      try:result=roll_execution(cfg,r['resources'][policy],data['grids'][indices],data['quality'][indices],span,rule,guard)
      except SolverFailure as exc:
       failures.append(dict(key=key,error=exc.details,utc=datetime.now(timezone.utc).isoformat()));js('failures.json',failures)
       if len(failures)>=10:stopped=True;break
       continue
      js(file.name,dict(family=r['family'],seed=r['seed'],position=r['position'],issue_id=r['issue_id'],timestamp=r['timestamp'],policy=policy,rule=rule,span=span,planner=f'lookahead{span}',update_indices=indices.tolist(),booking_source_sha256=r['source_sha256'],result=result))
     completed+=1
    if stopped:break
   if stopped:break
  print('PLANNED LOOKAHEAD',r['family'],r['seed'],r['position'],'completed',completed,flush=True)
  if stopped:break
 js('trajectory_complete.json',dict(completed=completed,planned=2592,failures=len(failures),stopped=stopped,utc=datetime.now(timezone.utc).isoformat()));assert completed==2592,'partial planning; do not publish complete comparisons'

def evaluate(cfg):
 assert read(OUT/'trajectory_complete.json')['completed']==2592;p=make_problem(cfg);raw=np.load(ART/'data.npz')['raw'];rows=[];attempts=[];certs=[];oracles={};nodes=sorted({t['node'] for t in cfg['tasks']})
 for file in sorted(OUT.glob('trajectory_*.json')):
  if file.name=='trajectory_complete.json':continue
  guard();r=read(file);a=r['result'];truth=raw[r['issue_id']+1:r['issue_id']+73];complete=bool(np.isfinite(truth[:,nodes]).all());e=evaluate_execution(p,a['resources'],a['attempts'],truth);key=(r['family'],r['seed'],r['position'])
  if key not in oracles:oracles[key]=read(BOOK/f'case_{key[0]}_{key[1]}_{key[2]:03d}.json')
  oracle=oracles[key]['observed'][r['policy']];assert complete==oracle['complete']
  if complete:assert e['cost_lower']>=oracle['exact_cost']-1e-6;audit_schedule(p,a['resources'],completed_task_weather_options(p,truth[None])[0],[z['choice'] for z in e['outcomes'] if z['status']=='successful'])
  common={k:r[k] for k in ['family','seed','position','issue_id','timestamp','policy','rule','planner']};dur=sum(p.choices[i]['end']-p.choices[i]['start'] for i in a['attempts']);boats=sum(a['resources'][:p.B]);crews=sum(a['resources'][p.B:p.B+p.C]);rows.append(dict(**common,**{k:v for k,v in e.items() if k not in ['outcomes','spares_used']},complete=complete,actual_cost=e['cost_lower'] if complete else None,oracle_cost=oracle['exact_cost'] if complete else None,execution_gap=e['cost_lower']-oracle['exact_cost'] if complete else None,gap_lower=e['cost_lower']-oracle['cost_upper'],gap_upper=e['cost_upper']-oracle['cost_lower'],skipped_updates=a['skipped_updates'],boat_hours=dur,crew_hours=dur,boat_utilization=dur/(boats*72) if boats else None,crew_utilization=dur/(crews*72) if crews else None,future_only_updates=a['future_only_updates'],postponed_task_plans=a['postponed_task_plans']))
  sourcegrids=np.load(OUT/f'window_grids_{r["family"]}_{r["seed"]}.npz')['grids']
  for z in e['outcomes']:
   o=p.choices[z['choice']];update=o['start']//6;attempts.append(dict(**common,choice=z['choice'],task=o['task'],start=o['start'],end=o['end'],boat=o['boat'],crew=o['crew'],operating_cost=o['cost'],probability=float(sourcegrids[r['update_indices'][update],0,o['task']]),outcome=z['status'],success=1 if z['status']=='successful' else 0 if z['status']=='failed' else None,known_hours=z['known_hours']))
  for step in a['steps']:certs.append(dict(**common,now=step['now'],quality_ok=step['quality_ok'],expected_benefit=step['expected_benefit'],dual_bound=step['dual_bound'],mip_gap=step['mip_gap'],solver_status=step['solver_status'],solver_called=step['solver_called'],candidate_count=step['candidate_count'],seconds=step['seconds'],planned=json.dumps(step['planned']),selected=json.dumps(step['selected'])))
  js(file.name.replace('trajectory_','evaluated_'),dict(trajectory_sha256=sha(file),evaluation=e,oracle_cost_lower=oracle['cost_lower'],oracle_cost_upper=oracle['cost_upper']))
 base=pd.read_csv(SRC/'cases.csv');base['planner']='myopic';base['future_only_updates']=np.nan;base['postponed_task_plans']=np.nan;new=pd.DataFrame(rows);allcases=pd.concat([base,new],ignore_index=True);allcases.to_csv(OUT/'cases.csv',index=False,encoding='utf-8-sig');baseattempts=pd.read_csv(SRC/'attempts.csv');baseattempts['planner']='myopic';allattempts=pd.concat([baseattempts,pd.DataFrame(attempts)],ignore_index=True);allattempts.to_csv(OUT/'attempts.csv',index=False,encoding='utf-8-sig');csv('planning_certificates.csv',certs);js('evaluation_complete.json',dict(new_cases=len(rows),combined_cases=len(allcases),new_attempts=len(attempts),combined_attempts=len(allattempts),utc=datetime.now(timezone.utc).isoformat()));return allcases

def interval(values,weeks):
 values=np.asarray(values,float);weeks=np.asarray(weeks);draw=np.random.default_rng(20261112).integers(0,9,(2000,9));nn=np.array([(weeks==k).sum() for k in range(9)]);ss=np.array([values[weeks==k].sum() for k in range(9)]);den=nn[draw].sum(1);valid=den>0;enough=len(values)>=8 and sum(nn>0)>=4 and sum(valid)>=1900;boot=ss[draw][valid].sum(1)/den[valid];lo,hi=np.quantile(boot,[.025,.975]) if enough else (np.nan,np.nan);return dict(difference=float(values.mean()) if len(values) else np.nan,low95=lo,high95=hi,complete_origins=len(values),nonempty_weeks=int(sum(nn>0)),valid_draws=int(sum(valid)),interval_reported=bool(enough))

def summarize(df):
 fields=['actual_cost','oracle_cost','execution_gap','booking_cost','operating_cost','penalty_lower','penalty_upper','cost_lower','cost_upper','successful','failed','unknown','unattempted','attempted','skipped_updates','boat_utilization','crew_utilization','future_only_updates','postponed_task_plans'];keys=['family','policy','rule','planner','seed'];metrics=[]
 for key,g in df.groupby(keys):
  for scope,sub in [('all',g),('complete',g[g.complete])]:
   for metric in fields:metrics.append(dict(**dict(zip(keys,key)),scope=scope,metric=metric,value=sub[metric].mean(),n=int(sub[metric].notna().sum())))
 csv('metrics.csv',metrics);times=pd.to_datetime(df.drop_duplicates('position').set_index('position').timestamp,utc=True).dt.tz_convert('Asia/Shanghai');weeks=((times-pd.Timestamp('2024-11-01',tz='Asia/Shanghai')).dt.days//7).to_dict();specs=[]
 for model in MODELS:
  for policy in POLICIES:
   for rule in RULES:
    for span in SPANS:
     primary=span==72 and rule=='economic' and ((model=='old57' and policy=='stochastic') or (model=='gated_graph57' and policy in ['stochastic','regret']));specs.append(('planner',model,model,policy,rule,f'lookahead{span}','myopic',primary))
 for base in ['old57','gated_temporal57']:
  for policy in ['stochastic','regret']:
   for rule in RULES:
    for span in SPANS:specs.append(('model','gated_graph57',base,policy,rule,f'lookahead{span}',f'lookahead{span}',base=='gated_temporal57' and rule=='economic' and span==72))
 pairs=[];bounds=[]
 for kind,cand,base,policy,rule,planner,bplanner,primary in specs:
  values=[];ww=[];lower=[];upper=[]
  for pos in POSITIONS:
   a=df[(df.family==cand)&(df.policy==policy)&(df.rule==rule)&(df.planner==planner)&(df.position==pos)];b=df[(df.family==base)&(df.policy==policy)&(df.rule==rule)&(df.planner==bplanner)&(df.position==pos)];assert len(a)==len(b)==3;lower.append(a.cost_lower.mean()-b.cost_upper.mean());upper.append(a.cost_upper.mean()-b.cost_lower.mean())
   if a.complete.all() and b.complete.all():values.append(a.actual_cost.mean()-b.actual_cost.mean());ww.append(weeks[pos])
  common=dict(kind=kind,candidate=cand,baseline=base,policy=policy,rule=rule,planner=planner,baseline_planner=bplanner,primary=primary);pairs.append(dict(**common,**interval(values,ww)));bounds.append(dict(**common,difference_lower=float(np.mean(lower)),difference_upper=float(np.mean(upper)),origins=24))
 csv('paired_comparisons.csv',pairs);csv('missing_bounds.csv',bounds);js('study_counts.json',dict(new_trajectories=2592,combined_trajectories=len(df),complete_rows=int(df.complete.sum()),complete_origins=int(df[df.complete].position.nunique()),paired_rows=len(pairs),primary_comparisons=int(sum(r['primary'] for r in pairs))))

if __name__=='__main__':
 cfg,records=prepare();probabilities(cfg);trajectories(cfg,records);df=evaluate(cfg);summarize(df);print('COMPLETE',flush=True)

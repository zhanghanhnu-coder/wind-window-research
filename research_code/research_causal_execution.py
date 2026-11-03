"""Round23: past-only forecast updates and irrevocable simulated operations."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from model import ART
from v3.forecasting import ResearchForecaster
from causal_execution_forecast import forecast_batch,scenarios_from_quantiles,completion_probabilities,MODELS
from causal_execution_core import plan_execution,evaluate_execution
from two_stage_core import make_problem,audit_schedule,weather_options
from research_crps_distribution import guard
OUT=ART/'causal_execution_v1';SRC=ART/'decision_chain_v1';SEEDS=[20241101,20241102,20241103];POSITIONS=list(range(0,232,10));POLICIES=['deterministic','stochastic','regret'];RULES=['economic','guard80'];PLAN=ART.parent.parent/'研究笔记/逐步获知天气的维修执行回放方案_2026-11-02.md'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(n,obj):(OUT/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(n,rows):pd.DataFrame(rows).to_csv(OUT/n,index=False,encoding='utf-8-sig')

def prepare():
 guard();assert (SRC/'final_manifest.json').exists() and read(SRC/'verification.json')['status']=='PASS';OUT.mkdir(exist_ok=True);protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0];f=OUT/'protocol_frozen.md'
 if f.exists():assert f.read_text(encoding='utf-8')==protocol
 else:f.write_text(protocol,encoding='utf-8')
 if not (OUT/'previous_hashes.json').exists():js('previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 code=['causal_execution_core.py','causal_execution_forecast.py','two_stage_core.py','v3/forecasting.py','model.py','crps_distribution_core.py','lead_calibration_core.py','research_crps_distribution.py'];files=[f,Path(__file__),SRC/'final_manifest.json',SRC/'recipe.json',ART/'data.npz',ART/'metadata.json']+[ART.parent/n for n in code];ident={str(p):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==ident
 else:js('identity.json',ident)
 records=[]
 for family in MODELS:
  for seed in SEEDS:
   for pos in POSITIONS:
    path=SRC/f'planned_{family}_{seed}_{pos:03d}.json';row=read(path);records.append({**{k:row[k] for k in ['family','seed','position','input_index','issue_id','timestamp']},'resources':{p:row['planned']['plans'][p]['x'] for p in POLICIES},'source_file':str(path),'source_sha256':sha(path)})
 if (OUT/'booking_inputs.json').exists():assert read(OUT/'booking_inputs.json')==records
 else:js('booking_inputs.json',records)
 cfg=read(SRC/'recipe.json')['config']
 if not (OUT/'recipe.json').exists():js('recipe.json',dict(models=MODELS,seeds=SEEDS,positions=POSITIONS,policies=POLICIES,rules=RULES,config=cfg,threshold=10.,updates=list(range(0,72,6)),planned_trajectories=1296,bootstrap_seed=20261111,start_utc=datetime.now(timezone.utc).isoformat()))
 return cfg,records

def forecasts(cfg,records):
 f=ResearchForecaster();unique=np.unique([r['issue_id']+h for r in records for h in range(0,72,6)]);assert max(unique)<len(f.x);ident={str(ART/n):h for n,h in f.file_ids.items()}
 if (OUT/'model_identity.json').exists():assert read(OUT/'model_identity.json')==ident
 else:js('model_identity.json',ident)
 inputfile=OUT/'update_inputs.npz'
 if not inputfile.exists():np.savez_compressed(inputfile,issue_ids=unique,timestamps=np.array([str(f.times[i]) for i in unique]),source_cal_ids=f.cal_ids[f.selected57])
 else:np.testing.assert_array_equal(np.load(inputfile)['issue_ids'],unique)
 checks=[]
 for model in MODELS:
  for seed in SEEDS:
   guard();dest=OUT/f'forecasts_{model}_{seed}.npz'
   if not dest.exists():
    data=forecast_batch(f,unique,model,seed);prob=np.full((len(unique),len(cfg['tasks'])),np.nan)
    for j in np.flatnonzero(data['quality']):prob[j]=completion_probabilities(scenarios_from_quantiles(f,data['q'][j],model,seed),cfg)
    np.savez_compressed(dest,**data,probabilities=prob)
   data=np.load(dest);old=np.load(SRC/f'scenarios_{model}_{seed}.npz');maxdiff=0.
   for k,issue in enumerate(old['issue_ids']):
    j=int(np.searchsorted(unique,issue));assert data['quality'][j];sc=scenarios_from_quantiles(f,data['q'][j],model,seed);maxdiff=max(maxdiff,float(abs(sc-old['scenarios'][k]).max()));np.testing.assert_allclose(sc,old['scenarios'][k],atol=1e-5,rtol=0);np.testing.assert_array_equal(data['probabilities'][j],completion_probabilities(old['scenarios'][k],cfg))
   checks.append(dict(model=model,seed=seed,origins=24,max_initial_scenario_difference=maxdiff,unique_updates=len(unique),skipped=int((~data['quality']).sum())));print('FORECASTED',model,seed,len(unique),'initial maxdiff',maxdiff,flush=True)
 js('initial_forecast_checks.json',checks);js('forecast_complete.json',dict(flows=9,updates=len(unique),utc=datetime.now(timezone.utc).isoformat()))

def trajectories(cfg,records):
 assert (OUT/'forecast_complete.json').exists();unique=np.load(OUT/'update_inputs.npz')['issue_ids'];cache={(m,s):dict(np.load(OUT/f'forecasts_{m}_{s}.npz')) for m in MODELS for s in SEEDS};done=0
 for r in records:
  guard();data=cache[(r['family'],r['seed'])];indices=np.searchsorted(unique,r['issue_id']+np.arange(0,72,6));np.testing.assert_array_equal(unique[indices],r['issue_id']+np.arange(0,72,6))
  for policy in POLICIES:
   for rule in RULES:
    path=OUT/f'trajectory_{r["family"]}_{r["seed"]}_{r["position"]:03d}_{policy}_{rule}.json'
    if not path.exists():
     result=plan_execution(cfg,r['resources'][policy],data['probabilities'][indices],data['quality'][indices],rule);js(path.name,dict(family=r['family'],seed=r['seed'],position=r['position'],issue_id=r['issue_id'],timestamp=r['timestamp'],policy=policy,rule=rule,update_indices=indices.tolist(),booking_source_sha256=r['source_sha256'],result=result))
    done+=1
  print('PLANNED EXECUTION',r['family'],r['seed'],r['position'],flush=True)
 assert done==1296;js('trajectory_complete.json',dict(completed=done,utc=datetime.now(timezone.utc).isoformat()))

def evaluate(cfg):
 assert read(OUT/'trajectory_complete.json')['completed']==1296;p=make_problem(cfg);raw=np.load(ART/'data.npz')['raw'];rows=[];attempts=[];oraclecache={};nodes=sorted({t['node'] for t in cfg['tasks']})
 for path in sorted(OUT.glob('trajectory_*.json')):
  if path.name=='trajectory_complete.json':continue
  guard();r=read(path);a=r['result'];truth=raw[r['issue_id']+1:r['issue_id']+73];complete=bool(np.isfinite(truth[:,nodes]).all());e=evaluate_execution(p,a['resources'],a['attempts'],truth);key=(r['family'],r['seed'],r['position'])
  if key not in oraclecache:oraclecache[key]=read(SRC/f'case_{key[0]}_{key[1]}_{key[2]:03d}.json')
  oracle=oraclecache[key]['observed'][r['policy']];assert oracle['complete']==complete
  if complete:
   assert e['cost_lower']>=oracle['exact_cost']-1e-6 and abs(e['cost_lower']-e['cost_upper'])<1e-9;success=[z['choice'] for z in e['outcomes'] if z['status']=='successful'];audit_schedule(p,a['resources'],weather_options(p,truth[None])[0],success)
  common={k:r[k] for k in ['family','seed','position','issue_id','timestamp','policy','rule']};dur=sum(p.choices[i]['end']-p.choices[i]['start'] for i in a['attempts']);boats=sum(a['resources'][:p.B]);crews=sum(a['resources'][p.B:p.B+p.C]);row=dict(**common,**{k:v for k,v in e.items() if k not in ['outcomes','spares_used']},complete=complete,actual_cost=e['cost_lower'] if complete else None,oracle_cost=oracle['exact_cost'] if complete else None,execution_gap=e['cost_lower']-oracle['exact_cost'] if complete else None,gap_lower=e['cost_lower']-oracle['cost_upper'],gap_upper=e['cost_upper']-oracle['cost_lower'],skipped_updates=a['skipped_updates'],boat_hours=dur,crew_hours=dur,boat_utilization=dur/(boats*72) if boats else None,crew_utilization=dur/(crews*72) if crews else None);rows.append(row)
  for outcome in e['outcomes']:
   choice=p.choices[outcome['choice']];step=a['steps'][choice['start']//6];attempts.append(dict(**common,choice=outcome['choice'],task=choice['task'],start=choice['start'],end=choice['end'],boat=choice['boat'],crew=choice['crew'],operating_cost=choice['cost'],probability=step['probabilities'][choice['task']],outcome=outcome['status'],success=1 if outcome['status']=='successful' else 0 if outcome['status']=='failed' else None,known_hours=outcome['known_hours']))
  js(path.name.replace('trajectory_','evaluated_'),dict(trajectory_sha256=sha(path),evaluation=e,oracle_cost_lower=oracle['cost_lower'],oracle_cost_upper=oracle['cost_upper']))
 csv('cases.csv',rows);csv('attempts.csv',attempts);js('evaluation_complete.json',dict(cases=len(rows),attempts=len(attempts),utc=datetime.now(timezone.utc).isoformat()));return pd.DataFrame(rows)

def interval(delta,weeks):
 delta=np.asarray(delta,float);weeks=np.asarray(weeks);keys=np.arange(9);draw=np.random.default_rng(20261111).integers(0,9,(2000,9));nn=np.array([(weeks==k).sum() for k in keys]);ss=np.array([delta[weeks==k].sum() for k in keys]);den=nn[draw].sum(1);valid=den>0;enough=len(delta)>=8 and sum(nn>0)>=4 and sum(valid)>=1900;boot=ss[draw][valid].sum(1)/den[valid];lo,hi=np.quantile(boot,[.025,.975]) if enough else (np.nan,np.nan);return dict(difference=float(delta.mean()) if len(delta) else np.nan,low95=lo,high95=hi,complete_origins=len(delta),nonempty_weeks=int(sum(nn>0)),valid_draws=int(sum(valid)),interval_reported=bool(enough))

def summarize(df):
 metrics=[];keys=['actual_cost','oracle_cost','execution_gap','booking_cost','operating_cost','penalty_lower','penalty_upper','cost_lower','cost_upper','successful','failed','unknown','unattempted','attempted','skipped_updates','boat_utilization','crew_utilization'];group=['family','policy','rule','seed']
 for key,g in df.groupby(group):
  for scope,sub in [('all',g),('complete',g[g.complete])]:
   for metric in keys:metrics.append(dict(**dict(zip(group,key)),scope=scope,metric=metric,value=sub[metric].mean(),n=int(sub[metric].notna().sum())))
 csv('metrics.csv',metrics);times=pd.to_datetime(df.drop_duplicates('position').set_index('position').timestamp,utc=True).dt.tz_convert('Asia/Shanghai');weeks=((times-pd.Timestamp('2024-11-01',tz='Asia/Shanghai')).dt.days//7).to_dict();comparisons=[]
 for rule in RULES:
  for baseline in ['old57','gated_temporal57']:
   for policy in ['stochastic','regret']:comparisons.append(('model','gated_graph57',baseline,policy,rule,rule,rule=='economic'))
 for model in MODELS:
  for policy in POLICIES:comparisons.append(('rule',model,model,policy,'guard80','economic',False))
 pairs=[];bounds=[]
 for kind,cand,base,policy,rule,brule,primary in comparisons:
  delta=[];ww=[];lower=[];upper=[]
  for pos in POSITIONS:
   a=df[(df.family==cand)&(df.policy==policy)&(df.rule==rule)&(df.position==pos)];b=df[(df.family==base)&(df.policy==policy)&(df.rule==brule)&(df.position==pos)];assert len(a)==len(b)==3;lower.append(a.cost_lower.mean()-b.cost_upper.mean());upper.append(a.cost_upper.mean()-b.cost_lower.mean())
   if a.complete.all() and b.complete.all():delta.append(a.actual_cost.mean()-b.actual_cost.mean());ww.append(weeks[pos])
  common=dict(kind=kind,candidate=cand,baseline=base,policy=policy,rule=rule,baseline_rule=brule,primary=primary);pairs.append(dict(**common,**interval(delta,ww)));bounds.append(dict(**common,difference_lower=float(np.mean(lower)),difference_upper=float(np.mean(upper)),origins=24))
 csv('paired_comparisons.csv',pairs);csv('missing_bounds.csv',bounds);gaps=[]
 for (family,policy,rule),g in df.groupby(['family','policy','rule']):
  complete=g[g.complete].groupby('position').execution_gap.mean();gaps.append(dict(family=family,policy=policy,rule=rule,primary=family=='old57' and policy=='stochastic' and rule=='economic',**interval(complete.to_numpy(),[weeks[p] for p in complete.index])))
 csv('oracle_gaps.csv',gaps);attempts=pd.read_csv(OUT/'attempts.csv');reliability=[]
 for (family,policy,rule),g in attempts.groupby(['family','policy','rule']):
  for scope,sub in [('all_known',g[g.success.notna()]),('complete_origins',g[g.position.isin(df[df.complete].position.unique())])]:
   for b in range(10):
    chosen=sub[(sub.probability>=b/10)&((sub.probability<(b+1)/10) if b<9 else sub.probability<=1)];reliability.append(dict(family=family,policy=policy,rule=rule,scope=scope,bin=b,n=len(chosen),mean_probability=chosen.probability.mean(),observed_rate=chosen.success.mean()))
 csv('attempt_reliability.csv',reliability);js('study_counts.json',dict(trajectories=len(df),complete_rows=int(df.complete.sum()),complete_origins=int(df[df.complete].position.nunique()),attempts=len(attempts),paired_rows=len(pairs),gap_rows=len(gaps),primary_comparisons=int(sum(r['primary'] for r in pairs)+sum(r['primary'] for r in gaps))))

if __name__=='__main__':
 cfg,records=prepare();forecasts(cfg,records);trajectories(cfg,records);frame=evaluate(cfg);summarize(frame);print('COMPLETE',flush=True)

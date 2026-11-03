"""Round33: freeze every resource booking before observing actual recourse outcomes."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time
import numpy as np
import pandas as pd
from model import ART
from seasonal_probability_core import FOLDS
from research_crps_distribution import guard,SEEDS
from crps_distribution_core import couple_atoms
from two_stage_core import default_config,make_problem,weather_options,booking_cost,SolverFailure
from regret_core import solve_master as original_master,solve_recourse
from distributional_schedule_core import solve_distribution_master,solve_distribution_ccg,observed_options
from highs_readback_compat import install
OUT=ART/'distributional_schedule_v1';SRC=ART/'seasonal_probability_v1';COP=ART/'partial_copula_v1';ROLLING=ART/'seasonal_rolling_v1';PLAN=ART.parent.parent/'研究笔记/概率权重不确定性的两阶段调度方案_2026-11-02.md'
CASES={'graph_frozen':('gated_graph_atoms','frozen'),'graph_rolling28':('gated_graph_atoms','rolling28'),'own_rolling28':('gated_temporal_atoms','rolling28')};RADII={'tv000':0.,'tv025':.25,'tv100':1.,'tv010':.1,'tv050':.5};PAIRS=[('graph_rolling28','tv025','graph_rolling28','tv000'),('graph_rolling28','tv025','graph_rolling28','tv100'),('graph_rolling28','tv025','graph_frozen','tv025'),('graph_rolling28','tv025','own_rolling28','tv025')]
PARENTS=[ART/'two_stage_v1',ART/'decision_chain_v1',SRC,COP,ROLLING]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def js(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def verify_parents():
 counts={}
 for root in PARENTS:
  manifest=read(root/'final_manifest.json')
  for file,h in manifest.items():assert sha(Path(file))==h,file
  counts[root.name]=len(manifest)
 return counts
def prepare():
 guard();verify_parents();OUT.mkdir(exist_ok=True);(OUT/'cases').mkdir(exist_ok=True);protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0];frozen=OUT/'protocol_frozen.md'
 if frozen.exists():assert frozen.read_text(encoding='utf-8')==protocol
 else:frozen.write_text(protocol,encoding='utf-8')
 files=[frozen,Path(__file__)]+[Path(__file__).parent/n for n in ['distributional_schedule_core.py','two_stage_core.py','regret_core.py','highs_readback_compat.py','crps_distribution_core.py','tests/test_distributional_schedule.py']]+[p/'final_manifest.json' for p in PARENTS]+[ART/'data.npz'];identity={str(p):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==identity
 else:js(OUT/'identity.json',identity)
 active=[OUT]+[ART/n for n in ['window_pooling_v1','missing_window_bounds_v1','multiscale_context_v1']]
 if not (OUT/'previous_hashes.json').exists():js(OUT/'previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and not any(a in p.parents for a in active) and p.suffix in ['.npz','.pt','.csv','.joblib']})
 if not (OUT/'recipe.json').exists():js(OUT/'recipe.json',dict(cases=CASES,radii=RADII,seeds=SEEDS,folds=FOLDS,positions_per_fold=12,position_rule='floor(linspace(0,B-1,12))',cases_total=432,main_plans=1296,sensitivity_plans=96,plans_total=1392,config=default_config(),solver_time_limit=60,bootstrap_seed=20261122,bootstrap_replicates=2000,concurrently_active_excluded=[str(p) for p in active],start_utc=datetime.now(timezone.utc).isoformat()))
 js(OUT/'solver_compatibility.json',install())
def forecasts():
 inventory=[];source_inventory=read(ROLLING/'inventory.json');data=np.load(ART/'data.npz');times=data['timestamps'];p=make_problem(default_config())
 for name in FOLDS:
  f=np.load(SRC/name/'data.npz');positions=np.floor(np.linspace(0,len(f['test_ids'])-1,12)).astype(int);assert len(np.unique(positions))==12
  for case,(model,method) in CASES.items():
   for seed in SEEDS:
    guard();source=next(r for r in source_inventory if (r['fold'],r['model'],r['method'],r['seed'])==(name,model,method,seed));assert sha(Path(source['source']))==source['source_sha256'] and sha(Path(source['ranks_source']))==source['ranks_sha256'];q=np.load(source['source'])[source['source_key']];ranks=np.load(source['ranks_source'])['ranks']
    for pos in positions:
     key=f'{name}__{case}__{seed}__{pos:03d}';dest=OUT/'cases'/f'forecast_{key}.npz';sc=couple_atoms(q[pos].astype(float),ranks);available=weather_options(p,sc);np.savez_compressed(dest,scenarios=sc,available=available,probability=np.ones(57)/57);issue=int(f['test_ids'][pos]);policies=['tv000','tv025','tv100']+(['tv010','tv050'] if case=='graph_rolling28' and seed==SEEDS[0] else []);inventory.append(dict(key=key,fold=name,case=case,seed=seed,position=int(pos),issue_id=issue,timestamp=str(times[issue]),source_item=source,forecast_source=str(dest),forecast_sha256=sha(dest),policies=policies))
 assert len(inventory)==432 and sum(len(r['policies']) for r in inventory)==1392;js(OUT/'inventory.json',inventory);js(OUT/'forecasts_complete.json',dict(status='COMPLETE',cases=432,plans=1392,utc=datetime.now(timezone.utc).isoformat()))
def plan_all():
 p=make_problem(default_config());rows=[];failures=[]
 for item in read(OUT/'inventory.json'):
  guard();assert sha(Path(item['forecast_source']))==item['forecast_sha256'];z=np.load(item['forecast_source']);available=z['available'];prob=z['probability']
  for policy in item['policies']:
   guard();dest=OUT/'cases'/f'plan_{item["key"]}__{policy}.json';failed=OUT/'cases'/f'failure_{item["key"]}__{policy}.json';rho=RADII[policy]
   if failed.exists():failures.append(read(failed));continue
   if dest.exists():record=read(dest)
   else:
    start=time.monotonic();print('PLAN',item['key'],policy,flush=True)
    try:
     plan=solve_distribution_ccg(p,available,prob,rho,guard=guard);dual=solve_distribution_master(p,available,prob,rho);np.testing.assert_allclose(plan['objective'],dual['objective'],rtol=0,atol=1e-5);endpoint=original_master(p,available,'stochastic' if rho==0 else 'robust') if rho in [0,1] else None
     if endpoint is not None:np.testing.assert_allclose(plan['objective'],endpoint['objective'],rtol=0,atol=1e-5)
     record=dict(key=item['key'],fold=item['fold'],case=item['case'],seed=item['seed'],position=item['position'],issue_id=item['issue_id'],policy=policy,radius=rho,plan=plan,dual_reference=dual,endpoint_reference=endpoint,seconds=time.monotonic()-start);js(dest,record)
    except SolverFailure as exc:
     failure=dict(key=item['key'],policy=policy,error=exc.details,seconds=time.monotonic()-start);js(failed,failure);failures.append(failure);print('FAILURE',item['key'],policy,exc,flush=True);continue
   plan=record['plan'];rows.append(dict(key=item['key'],fold=item['fold'],case=item['case'],seed=item['seed'],position=item['position'],issue_id=item['issue_id'],policy=policy,radius=rho,objective=plan['objective'],nominal_total=plan['nominal_total'],booking_cost=plan['booking_cost'],boats=sum(plan['x'][:2]),crews=sum(plan['x'][2:5]),spares=sum(plan['x'][5:]),no_booking=int(sum(plan['x'])==0),iterations=plan['iterations'],gap=plan['gap'],oracle_solves=plan['oracle_solves'],dual_objective=record['dual_reference']['objective'],seconds=record['seconds'],plan_source=str(dest),plan_sha256=sha(dest)));print('PLANNED',item['key'],policy,'iterations',plan['iterations'],'seconds',round(record['seconds'],2),flush=True)
 csv('planning_summary.csv',rows);js(OUT/'failures.json',failures);js(OUT/'planning_complete.json',dict(status='COMPLETE' if not failures and len(rows)==1392 else 'INCOMPLETE',plans=len(rows),expected=1392,failures=len(failures),utc=datetime.now(timezone.utc).isoformat()));assert not failures and len(rows)==1392,'retain failures; no selectively filtered outcome evaluation'
def evaluate():
 assert read(OUT/'planning_complete.json')['status']=='COMPLETE';raw=np.load(ART/'data.npz')['raw'];p=make_problem(default_config());rows=[];cache={}
 for item in read(OUT/'inventory.json'):
  y=raw[item['issue_id']+1:item['issue_id']+73];lo,hi,complete=observed_options(p,y);nodes=sorted({t['node'] for t in p.config['tasks']})
  for policy in item['policies']:
   guard();path=OUT/'cases'/f'plan_{item["key"]}__{policy}.json';record=read(path);plan=record['plan'];x=plan['x'];key=item['issue_id'],tuple(x)
   if key not in cache:
    a=solve_recourse(p,lo,x);b=a if np.array_equal(lo,hi) else solve_recourse(p,hi,x);cache[key]=(a,b)
   a,b=cache[key];book=booking_cost(p,x);lower=book+a['recourse_cost'];upper=book+b['recourse_cost'];assert lower<=upper+1e-6;out=dict(key=item['key'],policy=policy,issue_id=item['issue_id'],complete=complete,known_fraction=float(np.isfinite(y[:,nodes]).mean()),cost_lower=lower,cost_upper=upper,exact_cost=lower if complete else None,exact_completed=a['completed'] if complete else None,optimistic_solution=a,pessimistic_solution=b,resources=x,booking_cost=book);dest=OUT/'cases'/f'outcome_{item["key"]}__{policy}.json';js(dest,out);rows.append(dict(fold=item['fold'],case=item['case'],seed=item['seed'],position=item['position'],**{k:out[k] for k in ['key','policy','issue_id','complete','known_fraction','cost_lower','cost_upper','exact_cost','exact_completed','booking_cost']},objective=plan['objective'],nominal_total=plan['nominal_total'],no_booking=int(sum(x)==0),outcome_source=str(dest),outcome_sha256=sha(dest)))
 csv('outcomes.csv',rows);tables(rows);verify_parents();js(OUT/'evaluation_complete.json',dict(status='COMPLETE',plans=len(rows),actual_recourse_resource_pairs=len(cache),utc=datetime.now(timezone.utc).isoformat()));print('COMPLETE',flush=True)
def tables(rows):
 df=pd.DataFrame(rows);summary=[]
 for scope in ['ALL']+list(FOLDS):
  selected=df if scope=='ALL' else df[df.fold==scope]
  for (case,policy),g in selected.groupby(['case','policy'],sort=False):summary.append(dict(scope=scope,case=case,policy=policy,n=len(g),unique_origins=g.issue_id.nunique(),complete=int(g.complete.sum()),complete_unique_origins=g[g.complete].issue_id.nunique(),**{m:float(g[m].mean()) for m in ['cost_lower','cost_upper','exact_cost','exact_completed','booking_cost','objective','nominal_total','no_booking']}))
 csv('metrics.csv',summary);times=pd.DatetimeIndex(pd.to_datetime(np.load(ART/'data.npz')['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');rng=np.random.default_rng(20261122);draws={};week={};ids={};bootstrap={};paired=[]
 for name,cfg in FOLDS.items():
  ids[name]=np.sort(df[df.fold==name].issue_id.unique());week[name]=((times[ids[name]]-pd.Timestamp(cfg['test_start'],tz=times.tz)).days//7).to_numpy();k=int(week[name].max())+1;draws[name]=rng.integers(0,k,(2000,k));bootstrap[name]=np.stack([np.bincount(row,minlength=k)[week[name]] for row in draws[name]])
 np.savez_compressed(OUT/'bootstrap_draws.npz',**{name:draw for name,draw in draws.items()})
 for case,policy,base,bpolicy in PAIRS:
  for metric in ['all_cost_bounds','exact_cost','exact_completed']:
   values={}
   for name in FOLDS:
    a=df[(df.fold==name)&(df.case==case)&(df.policy==policy)].sort_values(['seed','issue_id']);b=df[(df.fold==name)&(df.case==base)&(df.policy==bpolicy)].sort_values(['seed','issue_id']);assert len(a)==len(b)==36;np.testing.assert_array_equal(a[['seed','issue_id']],b[['seed','issue_id']])
    if metric=='all_cost_bounds':lower=(a.cost_lower.to_numpy()-b.cost_upper.to_numpy()).reshape(3,12).mean(0);upper=(a.cost_upper.to_numpy()-b.cost_lower.to_numpy()).reshape(3,12).mean(0)
    else:lower=upper=(a[metric].to_numpy()-b[metric].to_numpy()).reshape(3,12).mean(0)
    values[name]=(lower,upper)
   for scope in ['ALL']+list(FOLDS):
    selected=list(FOLDS) if scope=='ALL' else [scope];numlo=np.zeros(2000);numhi=np.zeros(2000);den=np.zeros(2000);sumlo=0.;sumhi=0.;n=0;blocks=[]
    for name in selected:
     lower,upper=values[name];valid=np.isfinite(lower)&np.isfinite(upper);numlo+=bootstrap[name]@np.where(valid,lower,0);numhi+=bootstrap[name]@np.where(valid,upper,0);den+=bootstrap[name]@valid.astype(int);sumlo+=lower[valid].sum();sumhi+=upper[valid].sum();n+=valid.sum();blocks.append(len(np.unique(week[name][valid])))
    ok=den>0;supported=min(blocks)>=4 and ok.sum()>=1900;lq=np.quantile(numlo[ok]/den[ok],[.025,.975]) if supported else [np.nan,np.nan];uq=np.quantile(numhi[ok]/den[ok],[.025,.975]) if supported else [np.nan,np.nan];paired.append(dict(scope=scope,candidate=case,policy=policy,baseline=base,baseline_policy=bpolicy,metric=metric,primary=scope=='ALL' and metric=='all_cost_bounds',difference_lower=float(sumlo/n) if n else np.nan,difference_upper=float(sumhi/n) if n else np.nan,outer_low=float(lq[0]),outer_high=float(uq[1]),lower_percentile_high=float(lq[1]),upper_percentile_low=float(uq[0]),n=int(n),minimum_blocks_in_fold=min(blocks),valid_draws=int(ok.sum()),resampling_supported=supported))
 assert len(paired)==60 and sum(r['primary'] for r in paired)==4;csv('paired_comparisons.csv',paired)
if __name__=='__main__':prepare();forecasts();plan_all();evaluate()

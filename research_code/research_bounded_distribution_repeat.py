"""Fixed numerical representation / exact formulation study after V6 failures."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time
import numpy as np
import pandas as pd
from model import ART
from two_stage_core import make_problem,SolverFailure
from support_dominance_core import solve_dominance
from bounded_distribution_core import solve_bounded_dominance
from highs_readback_compat import install
from research_crps_distribution import guard
OUT=ART/'bounded_distribution_repeat_v1';BASE=ART.parent.parent;PARENT=ART/'support_dominance_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def run():
 guard();assert read(ART/'bounded_distribution_v1/complete.json')['status']=='COMPLETE';assert read(ART.parent/'v6/evidence/package_smoke_final.json')['status']=='PASS';assert not OUT.exists(),'preserve prior attempt';OUT.mkdir();(OUT/'cases').mkdir();install()
 parent=read(PARENT/'final_manifest.json')
 for f,h in parent.items():assert sha(f)==h,f
 plan=BASE/'研究笔记/有界费用鲁棒模型数值稳定性方案_2026-11-02.md';(OUT/'protocol_frozen.md').write_bytes(plan.read_bytes());old=read(ART/'scenario_equivalence_v1/inventory.json');cfg=read(ART/'scenario_equivalence_v1/recipe.json')['config'];p=make_problem(cfg)
 files=[Path(__file__),ART.parent/'bounded_distribution_core.py',ART.parent/'verify_bounded_distribution.py',ART.parent/'verify_bounded_distribution_repeat.py',ART/'bounded_distribution_v1/complete.json',ART/'bounded_distribution_v1/recipe.json',ART.parent/'tests/test_bounded_distribution.py',OUT/'protocol_frozen.md',PARENT/'final_manifest.json'];identity={str(f.resolve()):sha(f) for f in files};js(OUT/'identity.json',identity)
 inventory=[];jobs=[]
 for case in ['M07_235','M09_206']:
  item=next(i for i in old if i['case_id']==case);assert sha(item['forecast_source'])==item['forecast_sha256'];z=np.load(item['forecast_source'])
  for representation in ['raw','ui_normalized']:
   key=case+'__'+representation;w=z['probability'].copy()
   if representation=='ui_normalized':w=w/w.sum()
   path=OUT/'cases'/('input_'+key+'.npz');np.savez_compressed(path,scenarios=z['scenarios'],available=z['available'],probability=w)
   inventory.append(dict(key=key,case_id=case,representation=representation,source=item['forecast_source'],source_sha256=item['forecast_sha256'],input=str(path),input_sha256=sha(path),max_difference_from_raw=float(np.max(abs(w-z['probability'])))))
   for algorithm in ['ccg','full']:
    order=['original','bounded'] if (len(inventory)+(algorithm=='full'))%2 else ['bounded','original']
    jobs.extend(dict(key=key+'__'+algorithm+'__'+form,case_id=case,representation=representation,algorithm=algorithm,formulation=form,input_key=key) for form in order)
 js(OUT/'inventory.json',inventory);js(OUT/'recipe.json',dict(jobs=jobs,config=cfg,radius=.25,solver_time_limit=60,start_utc=datetime.now(timezone.utc).isoformat(),scope='full16 repeated once after first run and package MILP finished; all first outcomes retained; no selection or probability rounding'))
 rows=[]
 for index,job in enumerate(jobs):
  guard();item=next(i for i in inventory if i['key']==job['input_key']);z=np.load(item['input']);start=time.monotonic();print('SOLVE',index,job['key'],flush=True)
  try:
   fn=solve_dominance if job['formulation']=='original' else solve_bounded_dominance;r=fn(p,z['available'],z['probability'],.25,job['algorithm'],60.,guard);record=dict(**job,order=index,status='COMPLETE',result=r,seconds=time.monotonic()-start)
  except SolverFailure as exc:record=dict(**job,order=index,status='SOLVER_FAILURE',error=exc.details,seconds=time.monotonic()-start)
  path=OUT/'cases'/('solve_'+job['key']+'.json');js(path,record);r=record.get('result',{});rows.append(dict(**job,order=index,status=record['status'],seconds=record['seconds'],objective=r.get('objective',np.nan),gap=r.get('gap',np.nan),source=str(path),source_sha256=sha(path)));print('SOLVED',index,record['status'],round(record['seconds'],3),flush=True)
 df=pd.DataFrame(rows);df.to_csv(OUT/'runs.csv',index=False,encoding='utf-8-sig');comparisons=[]
 for item in inventory:
  for algorithm in ['ccg','full']:
   pair=df[(df.input_key==item['key'])&(df.algorithm==algorithm)];a=pair[pair.formulation=='original'].iloc[0];b=pair[pair.formulation=='bounded'].iloc[0];both=a.status==b.status=='COMPLETE';diff=b.objective-a.objective if both else np.nan
   if both:assert abs(diff)<=1e-5
   comparisons.append(dict(input_key=item['key'],algorithm=algorithm,original_status=a.status,bounded_status=b.status,both_complete=both,objective_difference=diff))
 pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False,encoding='utf-8-sig');summary=[]
 for (form,rep,alg),g in df.groupby(['formulation','representation','algorithm']):summary.append(dict(formulation=form,representation=rep,algorithm=alg,runs=len(g),completed=int((g.status=='COMPLETE').sum()),failed=int((g.status!='COMPLETE').sum()),seconds_total=float(g.seconds.sum())))
 pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
 for f,h in {**identity,**parent}.items():assert sha(f)==h,f
 js(OUT/'complete.json',dict(status='COMPLETE',runs=16,successful=int((df.status=='COMPLETE').sum()),solver_failures=int((df.status!='COMPLETE').sum()),future_observations_used=False));print('COMPLETE',flush=True)
if __name__=='__main__':run()

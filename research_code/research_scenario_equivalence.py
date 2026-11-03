"""Fixed raw/strictly equivalent aggregation benchmark; all failures retained."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,platform,sys
import numpy as np
import pandas as pd
import scipy
from model import ART
from v5.store import StudyStore
from two_stage_core import default_config,make_problem,weather_options,SolverFailure
from scenario_equivalence_core import group_conditions,solve_equivalent
from research_crps_distribution import guard
from highs_readback_compat import install

OUT=ART/'scenario_equivalence_v1';PARENT=ART/'v5_release_v1';PLAN=ART.parent.parent/'研究笔记/同作业条件情景等价合并方案_2026-11-02.md'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def run():
 guard();assert read(ART/'support_stress_v1/verification.json')['status']=='PASS','wait for round39 high-load planning and independent audit to finish'
 assert not (OUT/'final_manifest.json').exists();manifest=read(PARENT/'final_manifest.json')
 for file,h in manifest.items():assert sha(file)==h,file
 OUT.mkdir(exist_ok=True);(OUT/'cases').mkdir(exist_ok=True);protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0];frozen=OUT/'protocol_frozen.md'
 if frozen.exists():assert frozen.read_text(encoding='utf-8')==protocol
 else:frozen.write_text(protocol,encoding='utf-8')
 files=[Path(__file__),frozen,PARENT/'final_manifest.json']+[Path(__file__).parent/n for n in ['scenario_equivalence_core.py','verify_scenario_equivalence.py','tests/test_scenario_equivalence.py','distributional_schedule_core.py','two_stage_core.py','regret_core.py','highs_readback_compat.py','v5/store.py']]+list((ART.parent/'v5/data').glob('*'))
 identity={str(p):sha(p) for p in files if p.is_file()}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==identity
 else:js(OUT/'identity.json',identity)
 install();store=StudyStore();store.observations=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('future observations forbidden'));meta=store.metadata();cases=[]
 for fold in ['M05','M07','M09','M11']:
  choices=[c for c in meta['cases'] if c['fold']==fold];cases.extend([choices[0],choices[-1]])
 inventory=[];p=make_problem(default_config())
 for index,case in enumerate(cases):
  req=dict(case_id=case['id'],model='window_pool',seed=meta['seeds'][0],threshold=10);sc,w,info=store.scenarios(req);a=weather_options(p,sc,threshold=10);groups=group_conditions(a,w);dest=OUT/'cases'/('forecast_'+case['id']+'.npz')
  if dest.exists():
   z=np.load(dest)
   for name,value in [('scenarios',sc),('probability',w),('available',a)]:np.testing.assert_array_equal(z[name],value)
  else:np.savez_compressed(dest,scenarios=sc,probability=w,available=a)
  inventory.append(dict(index=index,case_id=case['id'],fold=case['fold'],position='first' if index%2==0 else 'last',request=req,forecast_source=str(dest),forecast_sha256=sha(dest),scenario_count=len(w),classes=len(groups['available']),zero_mass_classes=int((groups['probability']==0).sum()),nonuniform=bool(np.ptp(w)>1e-14),expert_weights=info['expert_weights'].tolist(),prediction_identity=info['prediction_identity']))
 assert len(inventory)==8;js(OUT/'inventory.json',inventory)
 jobs=[]
 for item in inventory:
  for ai,algorithm in enumerate(['ccg','full']):
   for compressed in ([False,True] if (item['index']+ai)%2==0 else [True,False]):jobs.append(dict(case_id=item['case_id'],algorithm=algorithm,compressed=compressed))
 if not (OUT/'recipe.json').exists():js(OUT/'recipe.json',dict(jobs=jobs,config=default_config(),radius=.25,solver_time_limit=60,sequential=True,python=sys.version,scipy=scipy.__version__,numpy=np.__version__,platform=platform.platform(),timing_scope='same local machine; not a strictly isolated performance experiment',start_utc=datetime.now(timezone.utc).isoformat()))
 rows=[]
 for ji,job in enumerate(jobs):
  guard();item=next(r for r in inventory if r['case_id']==job['case_id']);assert sha(item['forecast_source'])==item['forecast_sha256'];z=np.load(item['forecast_source']);key=f'{job["case_id"]}__{job["algorithm"]}__'+('grouped' if job['compressed'] else 'raw');target=OUT/'cases'/('solve_'+key+'.json')
  if target.exists():record=read(target)
  else:
   start=time.monotonic();print('SOLVE',ji,key,flush=True)
   try:r=solve_equivalent(p,z['available'],z['probability'],.25,job['algorithm'],job['compressed'],60.,guard);record=dict(key=key,order=ji,**job,status='COMPLETE',result=r,seconds=time.monotonic()-start)
   except SolverFailure as exc:record=dict(key=key,order=ji,**job,status='SOLVER_FAILURE',error=exc.details,seconds=time.monotonic()-start)
   js(target,record)
  r=record.get('result',{});rows.append(dict(key=key,order=ji,case_id=item['case_id'],fold=item['fold'],position=item['position'],algorithm=job['algorithm'],compressed=job['compressed'],nonuniform=item['nonuniform'],status=record['status'],seconds=record['seconds'],scenario_count=228,classes=item['classes'],zero_mass_classes=item['zero_mass_classes'],optimization_scenarios=item['classes'] if job['compressed'] else 228,objective=r.get('objective',np.nan),booking_cost=r.get('booking_cost',np.nan),iterations=r.get('iterations'),gap=r.get('gap',np.nan),source=str(target),source_sha256=sha(target)));print('SOLVED',ji,key,record['status'],round(record['seconds'],3),flush=True)
 df=pd.DataFrame(rows);assert len(df)==32;df.to_csv(OUT/'runs.csv',index=False,encoding='utf-8-sig');paired=[]
 for item in inventory:
  for alg in ['ccg','full']:
   g=df[(df.case_id==item['case_id'])&(df.algorithm==alg)];a=g[~g.compressed].iloc[0];b=g[g.compressed].iloc[0];both=a.status==b.status=='COMPLETE';diff=b.objective-a.objective if both else np.nan
   if both:assert abs(diff)<=1e-5
   paired.append(dict(case_id=item['case_id'],fold=item['fold'],position=item['position'],algorithm=alg,nonuniform=item['nonuniform'],classes=item['classes'],raw_status=a.status,grouped_status=b.status,both_complete=both,objective_difference=diff,raw_seconds=a.seconds,grouped_seconds=b.seconds,raw_over_grouped_time=a.seconds/b.seconds if both else np.nan))
 pd.DataFrame(paired).to_csv(OUT/'paired.csv',index=False,encoding='utf-8-sig');summary=[]
 for scope in ['ALL','uniform','nonuniform']:
  selected=df if scope=='ALL' else df[df.nonuniform==(scope=='nonuniform')]
  for (alg,comp),g in selected.groupby(['algorithm','compressed']):summary.append(dict(scope=scope,algorithm=alg,compressed=bool(comp),runs=len(g),completed=int((g.status=='COMPLETE').sum()),failed=int((g.status!='COMPLETE').sum()),seconds_total=float(g.seconds.sum()),median_seconds_including_failures=float(g.seconds.median())))
 pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
 for file,h in identity.items():assert sha(file)==h,file
 for file,h in manifest.items():assert sha(file)==h,file
 js(OUT/'complete.json',dict(status='COMPLETE',runs=32,successful=int((df.status=='COMPLETE').sum()),solver_failures=int((df.status!='COMPLETE').sum()),future_observations_used=False,utc=datetime.now(timezone.utc).isoformat()));print('COMPLETE',flush=True)

if __name__=='__main__':run()

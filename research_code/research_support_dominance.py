"""Fixed eight-origin exact zero-mass dominance experiment, after round41."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,sys,platform
import numpy as np
import pandas as pd
from model import ART
from two_stage_core import make_problem,SolverFailure
from support_dominance_core import reduce_zero_dominated,solve_dominance
from highs_readback_compat import install
from research_crps_distribution import guard
OUT=ART/'support_dominance_v1';PARENT=ART/'scenario_equivalence_v1';BASE=ART.parent.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def run():
 guard();assert read(PARENT/'verification.json')['status']=='PASS';parent=read(PARENT/'final_manifest.json');assert not (OUT/'final_manifest.json').exists()
 for file,h in parent.items():assert sha(file)==h,file
 OUT.mkdir(exist_ok=True);(OUT/'cases').mkdir(exist_ok=True);plan=BASE/'研究笔记/零名义质量支配情景约简方案_2026-11-02.md';protocol=OUT/'protocol_frozen.md'
 if protocol.exists():assert protocol.read_bytes()==plan.read_bytes()
 else:protocol.write_bytes(plan.read_bytes())
 names=['support_dominance_core.py','scenario_equivalence_core.py','verify_support_dominance.py','tests/test_support_dominance.py','two_stage_core.py','regret_core.py','distributional_schedule_core.py','verify_distributional_schedule.py','verify_two_stage.py','highs_readback_compat.py']
 identity={str(p.resolve()):sha(p) for p in [Path(__file__),protocol,PARENT/'final_manifest.json',*[ART.parent/n for n in names]]}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==identity
 else:js(OUT/'identity.json',identity)
 inventory=[];old=read(PARENT/'inventory.json');cfg=read(PARENT/'recipe.json')['config'];p=make_problem(cfg);install()
 for item in old:
  assert sha(item['forecast_source'])==item['forecast_sha256'];z=np.load(item['forecast_source']);g=reduce_zero_dominated(z['available'],z['probability']);inventory.append(dict(item,kept_classes=g['kept_classes'].tolist(),removed_classes=g['removed_classes'].tolist(),dominance_witness=g['witness'].tolist(),optimization_scenarios=len(g['available']),kept_zero_mass_classes=int((g['probability']==0).sum())))
 js(OUT/'inventory.json',inventory);jobs=[dict(case_id=item['case_id'],algorithm=alg) for item in inventory for alg in ['ccg','full']]
 if not (OUT/'recipe.json').exists():js(OUT/'recipe.json',dict(jobs=jobs,config=cfg,radius=.25,solver_time_limit=60,python=sys.version,platform=platform.platform(),start_utc=datetime.now(timezone.utc).isoformat(),scope='posthoc exact engineering improvement; no new actual weather evaluation; no fresh paired timing baseline'))
 rows=[];comparisons=[]
 for ji,job in enumerate(jobs):
  guard();item=next(r for r in inventory if r['case_id']==job['case_id']);z=np.load(item['forecast_source']);key=job['case_id']+'__'+job['algorithm'];dest=OUT/'cases'/('solve_'+key+'.json')
  if dest.exists():record=read(dest)
  else:
   start=time.monotonic();print('SOLVE',ji,key,'K',item['optimization_scenarios'],flush=True)
   try:r=solve_dominance(p,z['available'],z['probability'],.25,job['algorithm'],60.,guard);record=dict(key=key,order=ji,**job,status='COMPLETE',result=r,seconds=time.monotonic()-start)
   except SolverFailure as exc:record=dict(key=key,order=ji,**job,status='SOLVER_FAILURE',error=exc.details,seconds=time.monotonic()-start)
   js(dest,record)
  r=record.get('result',{});rows.append(dict(key=key,order=ji,case_id=job['case_id'],fold=item['fold'],position=item['position'],algorithm=job['algorithm'],nonuniform=item['nonuniform'],scenario_count=228,classes=item['classes'],optimization_scenarios=item['optimization_scenarios'],removed=len(item['removed_classes']),kept_zero_mass_classes=item['kept_zero_mass_classes'],status=record['status'],seconds=record['seconds'],objective=r.get('objective',np.nan),gap=r.get('gap',np.nan),source=str(dest),source_sha256=sha(dest)))
  for mode in ['raw','grouped']:
   oldrec=read(PARENT/'cases'/('solve_'+key+'__'+mode+'.json'));both=record['status']==oldrec['status']=='COMPLETE';diff=r['objective']-oldrec['result']['objective'] if both else np.nan
   if both:assert abs(diff)<=1e-5
   comparisons.append(dict(key=key,case_id=job['case_id'],algorithm=job['algorithm'],reference_mode=mode,reference_status=oldrec['status'],current_status=record['status'],both_complete=both,objective_difference=diff))
  print('SOLVED',ji,key,record['status'],round(record['seconds'],3),flush=True)
 df=pd.DataFrame(rows);df.to_csv(OUT/'runs.csv',index=False,encoding='utf-8-sig');pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False,encoding='utf-8-sig');summary=[]
 for scope in ['ALL','uniform','nonuniform']:
  part=df if scope=='ALL' else df[df.nonuniform==(scope=='nonuniform')]
  for alg,g in part.groupby('algorithm'):summary.append(dict(scope=scope,algorithm=alg,runs=len(g),completed=int((g.status=='COMPLETE').sum()),failed=int((g.status!='COMPLETE').sum()),median_seconds_including_failures=float(g.seconds.median()),seconds_total=float(g.seconds.sum())))
 pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
 for file,h in identity.items():assert sha(file)==h,file
 for file,h in parent.items():assert sha(file)==h,file
 js(OUT/'complete.json',dict(status='COMPLETE',runs=16,successful=int((df.status=='COMPLETE').sum()),solver_failures=int((df.status!='COMPLETE').sum()),future_observations_used=False,utc=datetime.now(timezone.utc).isoformat()));print('COMPLETE',flush=True)
if __name__=='__main__':run()

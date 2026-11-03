"""Prespecified diagnosis of finite weather support; frozen round33 decisions only."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from model import ART
from research_crps_distribution import guard
from support_diagnostic_core import cost_support,task_windows

OUT=ART/'support_diagnostic_v1';PARENT=ART/'distributional_schedule_v1';BASE=ART.parent.parent
FOLDS=['M05','M07','M09','M11'];TOL=1e-6

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def parent_check():
 assert read(PARENT/'verification.json')['status']=='PASS'
 manifest=read(PARENT/'final_manifest.json')
 for file,h in manifest.items():assert sha(file)==h,file
 return sha(PARENT/'final_manifest.json'),len(manifest)

def run():
 guard();parent_hash,parent_files=parent_check();assert read(PARENT/'evaluation_complete.json')['status']=='COMPLETE';OUT.mkdir(exist_ok=True);assert not (OUT/'identity.json').exists(),'preserve prior diagnostic';plan=BASE/'研究笔记/有限情景支持范围诊断方案_2026-11-02.md';protocol=OUT/'protocol_frozen.md';protocol.write_text(plan.read_text(encoding='utf-8').split('## 执行记录')[0],encoding='utf-8');code=[Path(__file__),ART.parent/'support_diagnostic_core.py',ART.parent/'verify_support_diagnostic.py',ART.parent/'tests/test_support_diagnostic.py',protocol,BASE/'研究笔记/有限情景支持范围诊断_预设时间记录.json'];js(OUT/'identity.json',{str(p):sha(p) for p in code});recipe=dict(parent_manifest_sha256=parent_hash,parent_manifest_files=parent_files,tolerance=TOL,forecast_cases=432,plans=1392,windows_per_case=70,unique_origins=48,no_policy_optimization=True,no_confidence_intervals=True,utc=datetime.now(timezone.utc).isoformat());js(OUT/'recipe.json',recipe)
 paths=[PARENT/'inventory.json',PARENT/'recipe.json',PARENT/'outcomes.csv',ART/'data.npz'];sources={str(p):sha(p) for p in paths};cfg=read(PARENT/'recipe.json')['config'];assert cfg['horizon']==72 and cfg['step']==6;raw=np.load(ART/'data.npz')['raw'];costs=[];windows=[];labels={}
 for item in read(PARENT/'inventory.json'):
  guard();file=Path(item['forecast_source']);sources[str(file)]=sha(file);assert sources[str(file)]==item['forecast_sha256'];z=np.load(file);sc=z['scenarios'];assert sc.shape==(57,72,17);common={k:item[k] for k in ['key','fold','case','seed','position','issue_id']};truth=raw[item['issue_id']+1:item['issue_id']+73];rows=task_windows(sc,truth,cfg['tasks'],cfg['step'],10.);assert len(rows)==70
  for row in rows:
   physical=f'{item["issue_id"]+row["start"]+1}:{row["node"]}:{row["end"]-row["start"]}'
   if physical in labels:assert labels[physical]==row['label']
   labels[physical]=row['label'];windows.append(dict(**common,physical_window=physical,**row))
  for policy in item['policies']:
   pf=PARENT/'cases'/f'plan_{item["key"]}__{policy}.json';of=PARENT/'cases'/f'outcome_{item["key"]}__{policy}.json';sources[str(pf)]=sha(pf);sources[str(of)]=sha(of);p=read(pf);o=read(of);assert p['plan']['x']==o['resources'];assert p['plan']['booking_cost']==o['booking_cost'];diagnostic=cost_support(o['booking_cost'],p['plan']['recourse_costs'],o['cost_lower'],o['cost_upper'],o['complete'],TOL);diagnostic['complete_excess']=int(diagnostic['complete_excess']) if diagnostic['complete_excess'] is not None else None;costs.append(dict(**common,policy=policy,radius=p['radius'],booking_cost=o['booking_cost'],resources=json.dumps(o['resources'],separators=(',',':')),complete=o['complete'],actual_lower=o['cost_lower'],actual_upper=o['cost_upper'],**diagnostic,plan_source=str(pf),outcome_source=str(of)))
  print('SUPPORT_DIAGNOSTIC',item['key'],flush=True)
 assert len(costs)==1392 and len(windows)==30240;cdf=pd.DataFrame(costs);wdf=pd.DataFrame(windows);cm=[];wm=[]
 for scope in ['ALL',*FOLDS]:
  c=cdf if scope=='ALL' else cdf[cdf.fold==scope];w=wdf if scope=='ALL' else wdf[wdf.fold==scope]
  for (case,policy),g in c.groupby(['case','policy'],sort=False):
   complete=g[g.complete];cm.append(dict(scope=scope,case=case,policy=policy,n=len(g),unique_origins=g.issue_id.nunique(),complete=int(g.complete.sum()),complete_unique_origins=complete.issue_id.nunique(),definite_excess=int(g.definite_excess.sum()),possible_excess=int(g.possible_excess.sum()),uncertain_excess=int(g.uncertain_excess.sum()),definite_excess_rate=float(g.definite_excess.mean()),possible_excess_rate=float(g.possible_excess.mean()),mean_excess_lower=float(g.excess_lower.mean()),mean_excess_upper=float(g.excess_upper.mean()),complete_excess=int(complete.complete_excess.sum()),complete_excess_rate=float(complete.complete_excess.mean()) if len(complete) else np.nan))
  for case,g in w.groupby('case',sort=False):
   row=dict(scope=scope,case=case,n=len(g),unique_origins=g.issue_id.nunique(),unique_physical_windows=g.physical_window.nunique(),known_success=int((g.label==1).sum()),known_failure=int((g.label==0).sum()),unknown=int((g.label==-1).sum()))
   for prefix,count,event in [('all_safe',57,0),('all_unsafe',0,1)]:
    sub=g[g.scenario_successes==count];n=len(sub);success=int((sub.label==1).sum());failure=int((sub.label==0).sum());unknown=int((sub.label==-1).sum());observed=failure if event==0 else success;row.update({f'{prefix}_n':n,f'{prefix}_known_success':success,f'{prefix}_known_failure':failure,f'{prefix}_unknown':unknown,f'{prefix}_opposite_lower':observed/n if n else np.nan,f'{prefix}_opposite_upper':(observed+unknown)/n if n else np.nan})
   wm.append(row)
 assert len(cm)==55 and len(wm)==15
 for name,df in [('cost_support.csv',cdf),('task_windows.csv',wdf),('cost_summary.csv',pd.DataFrame(cm)),('window_summary.csv',pd.DataFrame(wm))]:df.to_csv(OUT/name,index=False,encoding='utf-8-sig')
 js(OUT/'source_hashes.json',sources)
 for file,h in sources.items():assert sha(file)==h,file
 assert parent_check()==(parent_hash,parent_files)
 js(OUT/'evaluation_complete.json',dict(status='COMPLETE',forecast_cases=432,plans=1392,window_records=30240,unique_physical_windows=len(labels),unique_origins=int(cdf.issue_id.nunique()),utc=datetime.now(timezone.utc).isoformat()));print('COMPLETE',flush=True)

if __name__=='__main__':run()

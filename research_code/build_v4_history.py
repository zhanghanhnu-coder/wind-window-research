"""Compile every frozen round23/24 trajectory into one portable read-only asset."""
from pathlib import Path
import gzip,hashlib,json
import numpy as np
import pandas as pd
from model import ART
from two_stage_core import make_problem
from v4.history import case_key,canonical,HistoryStore

OUT=ART/'v4_release_v1';DEST=ART.parent/'v4/data/history.json.gz';R23=ART/'causal_execution_v1';R24=ART/'lookahead_execution_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
 if isinstance(v,np.generic):return v.item()
 return v

def main():
 OUT.mkdir(exist_ok=True);DEST.parent.mkdir(exist_ok=True);sources={};parents={}
 def source(p):sources[str(p.relative_to(ART))]=sha(p);return p
 for root in [R23,R24]:
  manifest=read(root/'final_manifest.json')
  for file,h in manifest.items():assert sha(Path(file))==h,file
  parents[root.name]=len(manifest);source(root/'final_manifest.json')
 config=read(source(R24/'recipe.json'))['config'];p=make_problem(config);frame=pd.read_csv(source(R24/'cases.csv'));assert len(frame)==3888;nodes=read(source(ART/'metadata.json'))['nodes'];flows={}
 for family in ['old57','gated_graph57','gated_temporal57']:
  for seed in [20241101,20241102,20241103]:
   z=dict(np.load(source(R24/f'window_grids_{family}_{seed}.npz')));flows[f'{family}_{seed}']=clean(z)
 cases={};origins=[]
 for position,g in frame.groupby('position',sort=True):
  assert g.complete.nunique()==1 and g.timestamp.nunique()==1;origins.append(dict(position=int(position),timestamp=g.iloc[0].timestamp,complete=bool(g.iloc[0].complete)))
 for row in frame.itertuples():
  base=f'{row.family}_{row.seed}_{row.position:03d}_{row.policy}_{row.rule}';folder=R23 if row.planner=='myopic' else R24;suffix='' if row.planner=='myopic' else '_s'+row.planner.removeprefix('lookahead');tpath=source(folder/f'trajectory_{base}{suffix}.json');epath=source(folder/f'evaluated_{base}{suffix}.json');t=read(tpath);e=read(epath);assert e['trajectory_sha256']==sha(tpath);result=t['result'];identity={k:t[k] for k in ['family','seed','position','issue_id','timestamp','policy','rule']};identity['planner']=row.planner;key=case_key(identity);assert key not in cases;steps=[]
  for step in result['steps']:steps.append({k:step[k] for k in ['now','quality_ok','selected','expected_benefit']}|{'planned':step.get('planned',[])})
  assert [s['now'] for s in steps]==list(range(0,72,6));assert [i for s in steps for i in s['selected']]==result['attempts']
  for field in ['cost_lower','cost_upper','booking_cost','operating_cost','attempted','successful','failed','unknown','unattempted']:np.testing.assert_allclose(e['evaluation'][field],getattr(row,field),rtol=0,atol=2e-10)
  cases[key]=dict(identity=identity,resources=result['resources'],attempts=result['attempts'],steps=steps,update_indices=t['update_indices'],complete=bool(row.complete),evaluation=e['evaluation'],oracle_cost_lower=e['oracle_cost_lower'],oracle_cost_upper=e['oracle_cost_upper'],trajectory_source=str(tpath.relative_to(ART)),evaluation_source=str(epath.relative_to(ART)))
 asset=dict(schema='wind-history-v4',config=config,choices=p.choices,nodes=nodes,origins=origins,flows=flows,cases=cases,source_files=sources);encoded=gzip.compress(canonical(asset),compresslevel=9,mtime=0)
 if DEST.exists():assert DEST.read_bytes()==encoded,'Existing asset differs; preserve and investigate'
 else:DEST.write_bytes(encoded)
 store=HistoryStore(DEST);assert store.metadata()['cases']==3888;summary=dict(status='BUILT_NOT_YET_INDEPENDENTLY_VERIFIED',cases=len(cases),origins=len(origins),complete_origins=sum(o['complete'] for o in origins),flows=len(flows),source_files=len(sources),parent_manifest_files=parents,bytes=len(encoded),asset_sha256=sha(DEST));(OUT/'history_build.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');(OUT/'history_source_identity.json').write_text(json.dumps(sources,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

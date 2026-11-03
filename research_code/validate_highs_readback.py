"""Recompute every already-completed round20 plan with cached result readback."""
from pathlib import Path
import json,time,hashlib
import numpy as np
from model import ART
from highs_readback_compat import install
from decision_chain_core import plan_scenarios
from two_stage_core import make_problem,default_config
from research_crps_distribution import guard
OUT=ART/'decision_chain_v1';DEST=OUT/'runtime_compatibility';DEST.mkdir(exist_ok=True)
def clean(value):
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k not in ['seconds','master_seconds']}
 if isinstance(value,list):return [clean(v) for v in value]
 return value
meta=install();p=make_problem(default_config());files=sorted(OUT.glob('planned_*.json'));rows=[]
for path in files:
 guard();case=json.loads(path.read_text(encoding='utf-8'));sc=np.load(OUT/f'scenarios_{case["family"]}_{case["seed"]}.npz')['scenarios'][case['input_index']];start=time.monotonic();fresh=plan_scenarios(p,sc,guard=guard);assert clean(fresh)==clean(case['planned']),path.name;rows.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),original_seconds=case['planning_seconds'],cached_seconds=time.monotonic()-start,all_nontiming_fields_exact=True));print('EXACT',path.name,round(rows[-1]['cached_seconds'],4),'original',round(case['planning_seconds'],4),flush=True)
record=dict(status='PASS',compatibility=meta,cases=rows,scipy_files_edited=False,scope='all complete plans present at validation start; all non-timing fields recursively exact')
(DEST/'validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');print('VALIDATION COMPLETE',len(rows),flush=True)

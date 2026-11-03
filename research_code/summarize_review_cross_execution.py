"""Compact descriptive exports; no refitting, sample selection, or new solver calls."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/review_followup_v1/cross_execution'

def main():
 assert json.loads((OUT/'verification.json').read_text())['status']=='PASS'
 arms=pd.read_csv(OUT/'arms.csv');pairs=pd.read_csv(OUT/'summary.csv')
 metrics=['cost_lower','cost_upper','booking_cost','operating_cost','successful','failed','unknown','unattempted','ideal_lower','ideal_upper','execution_minus_ideal_lower','execution_minus_ideal_upper']
 # Seed averaging first; origin is the statistical unit.
 origin=arms.groupby(['fold','position','issue','layer','prediction','executor'],sort=False).agg({**{m:'mean' for m in metrics},'complete':'all'}).reset_index()
 origin.to_csv(OUT/'arm_origins.csv',index=False,encoding='utf-8-sig');rows=[]
 for scope in ['ALL','M05','M07','M09','M11']:
  subset=origin if scope=='ALL' else origin[origin.fold==scope]
  for (layer,prediction,executor),g in subset.groupby(['layer','prediction','executor'],sort=False):
   for population in ['all','complete_task_weather']:
    z=g if population=='all' else g[g.complete]
    rows.append(dict(scope=scope,layer=layer,prediction=prediction,executor=executor,population=population,origins=len(z),**{m:z[m].mean() for m in metrics}))
 pd.DataFrame(rows).to_csv(OUT/'arm_summary.csv',index=False,encoding='utf-8-sig')
 pairs.to_csv(OUT/'paired.csv',index=False,encoding='utf-8-sig')
 status=dict(status='COMPLETE',origins=44,origin_seed_cases=132,arms=1056,tasks=6336,
             complete_weather_origins=15,solver_failures=0,planning_assertion_failures=1,
             assertion_failure_reason='Frozen MIP gap 7.36e-16 rejected by initial exact-zero check; corrected to existing <=1e-9 certificate tolerance before evaluation',
             statistical_scope='Descriptive 2024 fixed-forecast re-evaluation; no CI or confirmation of H3',
             default_models_or_historical_outputs_changed=False,
             support_diagnostics_verification=json.loads((OUT/'support_verification.json').read_text())['status'] if (OUT/'support_verification.json').exists() else 'not_yet_added')
 (OUT/'complete.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'failures.json').write_text(json.dumps(dict(solver_failures=[],implementation_revision='implementation_revision.json'),ensure_ascii=False,indent=2),encoding='utf-8')
 files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='final_manifest.json']
 files += [ROOT/n for n in ['cross_execution_review_core.py','review_cross_execution.py','verify_review_cross_execution.py','summarize_review_cross_execution.py','precision_truth_core.py','tests/test_cross_execution_review.py']]
 files += [ROOT/n for n in ['cross_support_review.py','verify_cross_support_review.py','tests/test_cross_support_review.py'] if (ROOT/n).exists()]
 files += [ROOT.parent/'研究笔记/复审后实证推进_2026-11-02/预测执行交叉实验.md']
 manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 (OUT/'final_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(status,ensure_ascii=False))
if __name__=='__main__':main()

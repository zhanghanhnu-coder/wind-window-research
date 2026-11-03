"""Continue the frozen round20 protocol with equivalent cached SciPy readback."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
from highs_readback_compat import install
import research_decision_chain as study

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
 out=study.OUT/'runtime_compatibility';proof=json.loads((out/'validation.json').read_text(encoding='utf-8'));assert proof['status']=='PASS' and len(proof['cases'])>=10
 for r in proof['cases']:assert sha(study.OUT/r['file'])==r['sha256']
 meta=install();files=[Path(__file__),Path(__file__).with_name('highs_readback_compat.py'),Path(__file__).with_name('validate_highs_readback.py')];record=dict(utc=datetime.now(timezone.utc).isoformat(),compatibility=meta,additional_source_identities={str(p):sha(p) for p in files},preserved_completed={p.name:sha(p) for p in sorted(study.OUT.glob('planned_*.json'))},validation_sha256=sha(out/'validation.json'),hypothesis_or_solver_settings_changed=False)
 path=out/'execution_cutover.json'
 if path.exists():
  previous=json.loads(path.read_text(encoding='utf-8'));assert previous['additional_source_identities']==record['additional_source_identities'] and previous['compatibility']==meta
 else:path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
 study.prepare();info=study.inputs();plans=study.plan_all();cases=study.evaluate_all(plans);study.summarize(cases,info);study.prediction_scores();print('COMPLETE CACHED READBACK',flush=True)

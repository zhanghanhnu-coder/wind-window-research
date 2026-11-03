"""Resume frozen trajectories after narrowing an audit to the task nodes."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import research_causal_execution as study
from causal_evaluation_scope import completed_task_weather_options
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if __name__=='__main__':
 out=study.OUT;assert json.loads((out/'trajectory_complete.json').read_text())['completed']==1296
 for n,h in json.loads((out/'identity.json').read_text(encoding='utf-8')).items():assert sha(Path(n))==h,n
 files=list(out.glob('trajectory_*.json'))+list(out.glob('forecasts_*.npz'))+[out/'booking_inputs.json',out/'update_inputs.npz',out/'initial_forecast_checks.json'];preserved={p.name:sha(p) for p in files};sources={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('causal_evaluation_scope.py')]};record=dict(utc=datetime.now(timezone.utc).isoformat(),reason='unused node missing caused overly broad success-schedule audit; task-node completeness criterion unchanged',additional_source_identities=sources,preserved_predictions_and_trajectories=preserved,forecast_or_decision_changed=False)
 path=out/'evaluation_audit_revision.json'
 if path.exists():
  old=json.loads(path.read_text(encoding='utf-8'));assert old['preserved_predictions_and_trajectories']==preserved and old['additional_source_identities']==sources
 else:path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
 study.weather_options=completed_task_weather_options
 cfg=study.read(out/'recipe.json')['config'];frame=study.evaluate(cfg);study.summarize(frame)
 for n,h in preserved.items():assert sha(out/n)==h,n
 print('COMPLETE EVALUATION AUDIT REVISION',flush=True)

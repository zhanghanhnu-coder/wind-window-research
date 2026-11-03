"""Resume27 only after predictions complete; correct raw-score float32 arithmetic."""
from pathlib import Path
from datetime import datetime,timezone
import json
import research_seasonal_rolling as run
from seasonal_score_precision import scoring_coupling

def main():
 out=run.OUT;phase=run.read(out/'prediction_complete.json');assert phase['flows']==144
 for file,h in run.read(out/'identity.json').items():assert run.sha(Path(file))==h,file
 preserved={str(p):run.sha(p) for p in out.rglob('*') if p.is_file() and p.suffix in ['.npz','.json','.md'] and p.name not in ['resume_precision_identity.json','resume_precision_verification.json']}
 destination=out/'resume_precision_identity.json';identity=dict(sources={str(p):run.sha(p) for p in [Path(__file__),Path(__file__).with_name('seasonal_score_precision.py')]},preserved=preserved,reason='Raw atoms are float32; width/interval subtraction previously rounded in float32. Cast the identical coupled forecast values to float64 before existing scoring; unchanged targets, parameters, paths and score definition.',started_utc=datetime.now(timezone.utc).isoformat())
 if destination.exists():
  saved=run.read(destination)
  for file,h in saved['sources'].items():assert run.sha(Path(file))==h
  preserved=saved['preserved']
 else:run.js(destination,identity)
 for file,h in preserved.items():assert run.sha(Path(file))==h,file
 run.couple_atoms=scoring_coupling
 run.evaluate();run.compare();run.verify_parents()
 for file,h in preserved.items():assert run.sha(Path(file))==h,file
 run.js(out/'resume_precision_verification.json',dict(status='PASS',preserved_files=len(preserved),completed_utc=datetime.now(timezone.utc).isoformat()));print('COMPLETE_WITH_PRECISION_CORRECTION',flush=True)

if __name__=='__main__':main()

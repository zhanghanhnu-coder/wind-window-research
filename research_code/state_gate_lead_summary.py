"""Deliver the predeclared lead-group diagnosis without changing predictions."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

def summarize_leads(df):
 unit=df.groupby(['fold','issue','lead_start','lead_end'],as_index=False)['harm'].mean();rows=[]
 for scope in ['ALL',*sorted(unit.fold.unique())]:
  part=unit if scope=='ALL' else unit[unit.fold==scope]
  for (a,b),g in part.groupby(['lead_start','lead_end']):
   valid=g[np.isfinite(g.harm)]
   rows.append(dict(scope=scope,lead_start=int(a),lead_end=int(b),n=len(g),observed_origins=len(valid),mean_harm=float(valid.harm.mean()),harm_fraction=float((valid.harm>0).mean())))
 return pd.DataFrame(rows)

def run():
 out=Path(__file__).parent/'artifacts/state_gate_v1';source=out/'origin_diagnosis.csv';digest=hashlib.sha256(source.read_bytes()).hexdigest();df=pd.read_csv(source);result=summarize_leads(df)
 # Independent per-record dictionary aggregation checks each exported cell.
 cells={}
 for r in df.itertuples():cells.setdefault((r.fold,r.issue,r.lead_start,r.lead_end),[]).append(r.harm)
 for r in result.itertuples():
  values=[]
  for (f,i,a,b),v in cells.items():
   if (r.scope=='ALL' or f==r.scope) and a==r.lead_start and b==r.lead_end:
    known=[x for x in v if np.isfinite(x)];values.append(sum(known)/len(known) if known else np.nan)
  good=[v for v in values if np.isfinite(v)];assert r.n==len(values) and r.observed_origins==len(good)
  np.testing.assert_allclose([r.mean_harm,r.harm_fraction],[sum(good)/len(good),sum(x>0 for x in good)/len(good)],rtol=0,atol=1e-12)
 assert len(result)==20 and result[result.scope=='ALL'].n.sum()==3520
 result.to_csv(out/'lead_failure_diagnosis.csv',index=False,encoding='utf-8-sig');assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
 (out/'lead_diagnosis_verification.json').write_text(json.dumps(dict(status='PASS',rows=20,source_sha=digest,predeclared_not_new_analysis=True),indent=2),encoding='utf-8')
 print(result[result.scope=='ALL'].to_string(index=False))
if __name__=='__main__':run()

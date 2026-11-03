"""Versioned evidence package; never overwrite v1 summaries."""
from pathlib import Path
import json,hashlib,shutil,unittest,io,subprocess,sys
import pandas as pd
import numpy as np
from model import ROOT,ART

def run():
 v=ART/'v2';out=ROOT.parent/'核心结果/demo_v2';out.mkdir(parents=True,exist_ok=True)
 stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py');res=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 (v/'unit_tests.txt').write_text(stream.getvalue(),encoding='utf-8')
 if not res.wasSuccessful():raise RuntimeError(stream.getvalue())
 ab=pd.read_csv(v/'ablation_metrics.csv');assert len(ab)==12 and ab.groupby('model').size().eq(3).all()
 summary=pd.read_csv(v/'replay_summary.csv');assert len(summary)==24
 for p in v.glob('replay_*.json'):
  r=json.loads(p.read_text(encoding='utf-8'));s=r['summary'];assert s['arrived']==sum(s[k] for k in ['pending','running','completed','unknown'])
  for key in ['crew','node']:
   for value in {a[key] for a in r['attempts']}:
    group=sorted([a for a in r['attempts'] if a[key]==value],key=lambda a:a['start_index'])
    assert all(a['end_index']<=b['start_index'] for a,b in zip(group,group[1:])),(p.name,key,value)
  for a in r['attempts']:
   assert a['issued_index']<=a['start_index']<a['end_index']<=a['issued_index']+72
   if a['outcome']!='running':assert a['revealed_index']>=a['end_index']
 browser=json.loads((v/'browser_validation.json').read_text());assert browser['passed']
 regression=subprocess.run([sys.executable,str(ROOT/'tests/browser_v2_regressions.py'),'-v'],capture_output=True,text=True,encoding='utf-8',errors='replace',cwd=ROOT)
 (v/'browser_regressions.txt').write_text(regression.stdout+regression.stderr,encoding='utf-8')
 if regression.returncode:raise RuntimeError(regression.stdout+regression.stderr)
 export=json.loads((v/'regression_export.json').read_text(encoding='utf-8'));assert export['displayed_replay']['kind']=='fixed_v2' and len(export['displayed_replay']['rows'])==24
 for p in v.iterdir():
  if p.is_file() and p.suffix in ['.csv','.json','.txt'] and p.name not in ['manifest.json','verification.json','browser_export.json','regression_export.json']:
   shutil.copy2(p,out/p.name)
 files=[]
 sources=list(ROOT.glob('*.py'))+list((ROOT/'static').glob('*'))+list((ROOT/'tests').glob('*.py'))+list(v.glob('*.pt'))+list(v.glob('*.npz'))+list(v.glob('*.csv'))
 for p in sources:
  if p.is_file():files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 report=ROOT.parent/'07_Demo第二版与研究项目细化_2026-11-01.pdf'
 manifest={'version':'2.0','files':files,'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest()}
 result={'version':'2.0','unit_tests_passed':res.testsRun,'new_training_runs':len(ab),'fixed_replay_runs':len(summary),'browser_smoke_passed':True,'browser_regression_cases':4,'node_and_crew_conservation_audited':True,'scope':'2024 only; exploratory; simulated tasks and resources; not field dispatch validation'}
 for name,data in [('manifest.json',manifest),('verification.json',result)]:
  (v/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(v/name,out/name)
 print(json.dumps(result,ensure_ascii=False));print('fingerprints',len(files))

if __name__=='__main__':run()

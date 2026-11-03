"""Copy reviewable outputs and create an explicit reproducibility manifest."""
from pathlib import Path
import json,hashlib,shutil,unittest,io,contextlib,sys
import numpy as np
import pandas as pd
from model import ART,ROOT

def run():
 out=ROOT.parent/'核心结果/demo';out.mkdir(parents=True,exist_ok=True)
 suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py');stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 (ART/'unit_test_results.txt').write_text(stream.getvalue(),encoding='utf-8')
 if not result.wasSuccessful():raise RuntimeError(stream.getvalue())
 for p in ART.glob('*.csv'):shutil.copy2(p,out/p.name)
 for name in ['metadata.json','paired_graph_comparison.json','unit_test_results.txt']:
  shutil.copy2(ART/name,out/name)
 # Cross-file checks on the summaries and fixed runs.
 for p in ART.glob('replay_*.json'):
  d=json.loads(p.read_text(encoding='utf-8'));s=d['summary'];assert s['arrived']==s['pending']+s['running']+s['completed']+s['unknown']
  assert len(d['tasks'])==s['arrived'] and len(d['attempts'])==s['attempts']
  for a in d['attempts']:
   assert a['start_index']>=a['issued_index'] and a['end_index']<=a['issued_index']+72
   if a['outcome']!='running':assert a['revealed_index']>=a['end_index']
  for crew in range(1,d['settings']['crews']+1):
   intervals=sorted((a['start_index'],a['end_index']) for a in d['attempts'] if a['crew']==crew)
   assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
  shutil.copy2(p,out/p.name)
 files=[]
 for p in sorted(ROOT.rglob('*')):
  if p.is_file() and '__pycache__' not in str(p) and p.suffix in ['.py','.js','.html','.css','.pt','.npz','.csv','.json','.cmd','.md','.txt'] and p.name!='manifest.json':
   files.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 report=ROOT.parent/'06_项目Demo_使用说明与本轮研究结果_2026-11-01.pdf'
 manifest={'date':'2026-11-01','unit_tests':result.testsRun,'unit_passed':result.wasSuccessful(),'fixed_replay_runs':len(list(ART.glob('replay_*.json'))),'sensitivity_runs':len(pd.read_csv(ART/'sensitivity.csv')),'data_year':2024,'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'files':files}
 (ART/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(ART/'manifest.json',out/'manifest.json')
 print(json.dumps({k:v for k,v in manifest.items() if k!='files'},ensure_ascii=False));print('hashes',len(files))

if __name__=='__main__':run()

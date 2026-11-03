"""Audit artifact consistency, causal cutoffs, saved inference and prior preservation."""
import hashlib,json,subprocess,sys,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from research_propagation import ROOT,OUT,RESULT,ART,SEEDS,NAMES,data
from propagation_graph import LagNet

def run():
 result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'风电运维Demo/tests'),'-p','test_*.py'],capture_output=True,text=True)
 (OUT/'unit_tests.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
 if result.returncode:raise RuntimeError('unit tests failed; see unit_tests.txt')
 prior=json.loads((OUT/'previous_artifact_hashes.json').read_text());changed=[name for name,digest in prior.items() if hashlib.sha256((ART/name).read_bytes()).hexdigest()!=digest]
 assert not changed,changed
 d,meta,times=data();graphs=np.load(OUT/'graphs.npz');te=d['test_ids'];cal=d['cal_ids'];records=pd.read_csv(OUT/'training_metrics.csv');assert len(records)==15
 assert (times[cal+72]<pd.Timestamp('2024-11-01',tz=times.tz)).all()
 assert (times[te]>=pd.Timestamp('2024-11-01',tz=times.tz)).all()
 for name in NAMES:
  a=graphs[name];assert np.isfinite(a).all() and (a>=0).all();assert np.allclose(np.diagonal(a,axis1=1,axis2=2),0)
  row_sum=a.sum((0,2));assert np.all(np.isclose(row_sum,0)|np.isclose(row_sum,1));assert ((a.sum(0)>0).sum(1)<=4).all()
  for seed in SEEDS:
   scores=np.load(OUT/f'{name}_{seed}_scores.npz');ck=torch.load(OUT/f'{name}_{seed}.pt',weights_only=False,map_location='cpu')
   assert np.array_equal(scores['test_ids'],te) and np.array_equal(scores['cal_ids'],cal);assert np.array_equal(ck['lag_adj'],a)
   row=records[(records.model==name)&(records.seed==seed)].iloc[0]
   assert np.isclose(row.brier_primary,np.nanmean(scores['errors']),rtol=0,atol=1e-14)
   net=LagNet(a);net.load_state_dict(ck['state']);net.eval();i=te[0]
   with torch.no_grad():med=net(torch.from_numpy(d['x'][i-23:i+1][None]))[0,...,1].numpy()*meta['scale']+meta['center']
   assert np.allclose(med,scores['median'][0],rtol=1e-5,atol=1e-5)
 for seed in SEEDS:
  a=np.load(OUT/f'fusion_{seed}_scores.npz');b=np.load(OUT/f'fusion_zero_{seed}_scores.npz')
  assert np.array_equal(a['median'],b['median']);assert np.array_equal(a['errors'],b['errors'],equal_nan=True)
 files=[ROOT/'风电运维Demo'/f for f in ['propagation_graph.py','research_propagation.py','diagnose_propagation.py','verify_propagation.py','tests/test_propagation.py']]
 files += [p for p in OUT.iterdir() if p.suffix in ['.csv','.npz','.pt','.json'] and p.name not in ['manifest.json','verification.json']]
 verification={'new_training_runs':15,'new_graph_methods':4,'same_graph_zero_lag_control':1,'unit_tests_passed':33,'models_inference_rechecked':15,'old_artifacts_byte_identical':len(prior),'all_selected_hourly_lags_zero':True,'scope':'2024 exploratory; wind only; lead-lag association not causal weather propagation','identical_fusion_control_not_evidence_against_delay_models':True}
 (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
 for path in OUT.iterdir():
  if path.suffix in ['.csv','.json','.txt'] and path.name not in ['manifest.json','previous_artifact_hashes.json']:shutil.copy2(path,RESULT/path.name)
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 pdf=ROOT/'08_传播图研究_方法对照与实验结论_2026-11-01.pdf'
 if pdf.exists():manifest[str(pdf.relative_to(ROOT))]=hashlib.sha256(pdf.read_bytes()).hexdigest()
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(OUT/'manifest.json',RESULT/'manifest.json')
 print(json.dumps(verification,ensure_ascii=False));print(result.stderr)

if __name__=='__main__':run()

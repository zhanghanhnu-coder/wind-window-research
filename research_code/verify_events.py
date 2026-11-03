"""Fresh, data-derived verification of event propagation artifacts."""
import hashlib,json,subprocess,sys,re,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from research_events import ROOT,ART,OUT,RESULT,MODELS,SEEDS,load,identity,event_batches
from event_graph import EventNet,require_identity,KINDS,state_at,event_scores,crossings

def verify_prior(root,manifest_path):
 previous=json.loads(manifest_path.read_text(encoding='utf-8'))
 changed=[name for name,digest in previous.items() if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest]
 if changed:raise ValueError('Previous artifacts changed: '+str(changed))
 return len(previous)

def run():
 tested=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'风电运维Demo/tests'),'-p','test_*.py'],capture_output=True,text=True)
 text=tested.stdout+tested.stderr;(OUT/'unit_tests.txt').write_text(text,encoding='utf-8')
 if tested.returncode:raise RuntimeError(text)
 count=int(re.search(r'Ran (\d+) tests?',text).group(1))
 previous_count=verify_prior(ROOT,OUT/'previous_hashes.json')
 d,meta,f,times,ft=load();inputs=np.load(OUT/'model_inputs.npz');graphs=np.load(OUT/'event_graphs.npz');token=identity();records=pd.read_csv(OUT/'training_metrics.csv');te=d['test_ids'];cal=d['cal_ids'];state=inputs['state'];x=d['x'];y=((d['raw']-meta['center'])/meta['scale']).astype(np.float32)
 events=json.loads((OUT/'events.json').read_text());assert len(records)==len(MODELS)*len(SEEDS)
 assert (times[cal+72]<pd.Timestamp('2024-11-01',tz=times.tz)).all()
 assert (times[te]>=pd.Timestamp('2024-11-01',tz=times.tz)).all()
 # Independently recompute causal prefix events with the frozen training thresholds.
 limit=next(i for i,t in enumerate(ft) if t>=pd.Timestamp('2024-11-15',tz=ft.tz));s=event_scores(f['wind'][:limit],f['direction'][:limit]);thresholds=json.loads((OUT/'recipe.json').read_text())['thresholds'];prefix=[]
 for kind,values,threshold in [('up',s['wind_change'],thresholds['wind']),('down',-s['wind_change'],thresholds['wind']),('turn',abs(s['turn']),thresholds['turn'])]:
  prefix.extend((i,kind) for i in crossings(values,threshold,72))
 assert sorted(prefix)==sorted((e['index'],e['kind']) for e in events if e['index']<limit)
 for i in te:assert state[i]==KINDS.index(state_at(events,int(i)*12))
 nonzero={};counts={}
 for name in graphs.files:
  a=graphs[name];assert np.isfinite(a).all() and (a>=0).all();assert np.allclose(np.diagonal(a,axis1=1,axis2=2),0);sums=a.sum((0,2));assert (np.isclose(sums,0)|np.isclose(sums,1)).all();assert ((a.sum(0)>0).sum(1)<=4).all();nonzero[name]=int((a[1:]>0).sum());counts[name]=int((a>0).sum())
 oldmask=np.isfinite(np.load(ART/'v2/paired_losses.npz')['graph24'])
 for model in MODELS:
  np.testing.assert_array_equal(inputs[model][0],inputs['reference'][0])
  for seed in SEEDS:
   ck=torch.load(OUT/f'{model}_{seed}.pt',map_location='cpu',weights_only=False);require_identity(ck,token);score=np.load(OUT/f'{model}_{seed}_scores.npz');assert str(score['identity'])==token
   assert np.array_equal(score['test_ids'],te) and np.array_equal(score['cal_ids'],cal);assert np.array_equal(np.isfinite(score['errors']),oldmask[0])
   row=records[(records.model==model)&(records.seed==seed)].iloc[0];assert row.identity==token;assert np.isclose(row.brier_primary,np.nanmean(score['errors']),rtol=0,atol=1e-14)
   net=EventNet();net.load_state_dict(ck['state']);net.eval()
   # Check all four online regimes that occur in evaluation, not just one quiet origin.
   selected=[int(np.flatnonzero(state[te]==k)[0]) for k in range(4) if (state[te]==k).any()]
   xx,_=next(event_batches(x,inputs[model],state,y,te[selected],batch=len(selected)))
   with torch.no_grad():pred=net(xx)[...,1].numpy()*meta['scale']+meta['center']
   np.testing.assert_allclose(pred,score['median'][selected],rtol=1e-5,atol=1e-5)
 verification={'new_training_runs':len(records),'models':MODELS,'seeds':SEEDS,'unit_tests_passed':count,'reloaded_models_checked':len(records),'online_regimes_checked':int(len(np.unique(state[te]))),'prefix_event_detection_matches':True,'previous_artifacts_unchanged':previous_count,'nonzero_graph_edges':nonzero,'graph_edge_counts':counts,'test_origins':len(te),'event_test_origins':int((state[te]>0).sum()),'valid_windows_per_model_seed':int(oldmask[0].sum()),'experiment_identity':token,'scope':'2024 exploratory; training-filtered relations are not confirmed physical propagation'}
 (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
 for p in OUT.iterdir():
  if p.suffix in ['.csv','.json','.txt','.md'] and p.name not in ['previous_hashes.json','manifest.json']:shutil.copy2(p,RESULT/p.name)
 files=[p for p in OUT.iterdir() if p.suffix in ['.pt','.npz','.csv','.json','.txt','.md'] and p.name!='manifest.json']
 files += [ROOT/'风电运维Demo'/name for name in ['event_graph.py','research_events.py','verify_events.py','tests/test_event_propagation.py']]
 files += [ROOT/'研究笔记/事件传播研究方案_2026-11-01.md']
 report=ROOT/'09_事件传播研究_稳定性与预测对照_2026-11-01.pdf'
 if report.exists():files.append(report)
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(OUT/'manifest.json',RESULT/'manifest.json')
 print(json.dumps(verification,ensure_ascii=False));print(text)

if __name__=='__main__':run()

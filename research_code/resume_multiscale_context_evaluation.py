"""Preserve frozen predictions/scores; use declared numerical marginal equality for reordered CRPS."""
from evaluate_multiscale_context import *
from marginal_equivalence_core import check_equivalence

def evaluate():
 freeze();assert read(OUT/'prediction_complete.json')['status']=='COMPLETE';checks=[]
 for item in read(OUT/'inventory.json'):
  guard();name=item['fold'];family=item['model'];seed=item['seed'];method=item['method'];dest=Path(item['score_source']);assert sha(Path(item['source']))==item['source_sha256'] and sha(Path(item['ranks_source']))==item['ranks_sha256'];q=np.load(item['source'])[item['source_key']];ranks=np.load(item['ranks_source'])['ranks'];f=np.load(SRC/name/'data.npz');y=np.load(SRC/name/'test_truth.npz')['y']
  if dest.exists():s=dict(np.load(dest))
  elif family=='context1_graph':
   s=dict(np.load(ROLLING/name/f'scores_gated_graph_atoms_{seed}_{method}.npz'));np.savez_compressed(dest,**s)
  else:
   marginal=score_atoms(q,y);records=[]
   for i,(qi,yi) in enumerate(zip(q,y)):
    sc=couple_atoms(qi.astype(float),ranks);r=score_scenarios(sc,yi,train_scale=float(f['scale']));gp=[];gl=[]
    for offset in [0,24,48,66]:
     for group in [[0,1,2],[8,9,10],[13,15,16]]:
      obs=yi[offset:offset+6][:,group];valid=np.isfinite(obs).all();gp.append(float((sc[:,offset:offset+6][:,:,group]<=10).all((1,2)).mean()) if valid else np.nan);gl.append(float((obs<=10).all()) if valid else np.nan)
    r['group_probability']=np.array(gp);r['group_label']=np.array(gl);records.append(r)
    if i%80==0:print('SCORE',name,family,seed,method,i,flush=True);guard()
   s={k:np.stack([r[k] for r in records]) for k in records[0]}
   for key in ['crps','coverage','width','low_miss','high_miss','interval_score']:np.testing.assert_allclose(s[key],marginal[key],rtol=0,atol=3e-12,equal_nan=True)
   s.update({k:marginal[k] for k in ['probability','label','brier_hourly','mae']});assert len(s)==35;np.savez_compressed(dest,**s)
  if family in OLD:
   old=np.load(ROLLING/name/f'scores_{OLD[family]}_{seed}_{method}.npz');keys=s.keys() if family=='context1_graph' else ['crps','probability','label','brier_hourly','mae','coverage','width','low_miss','high_miss','interval_score']
   maximum=check_equivalence(s,old,keys,family!='context1_graph')
   checks.append(dict(fold=name,model=family,seed=seed,method=method,arrays=len(keys),common_graph_template=True,maximum_crps_roundoff=maximum))
  print('RESULT',name,family,seed,method,'CRPS49_72',np.nanmean(s['crps'][:,48:72]),'B6',np.nanmean(s['brier6']),flush=True)
 csv('old_marginal_equivalence.csv',checks)
 from multiscale_context_tables import tables,compare
 tables();compare();verify_parents();js(OUT/'evaluation_complete.json',dict(status='COMPLETE',flows=324,forecast_origins=71280,utc=datetime.now(timezone.utc).isoformat()));print('COMPLETE',flush=True)

if __name__=='__main__':
 guard();checkpoint=OUT/'scoring_resume_checkpoint.json'
 if checkpoint.exists():saved=read(checkpoint)
 else:
  saved={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and not p.name.startswith('scoring_resume_')};js(checkpoint,saved)
 for file,h in saved.items():assert sha(Path(file))==h,file
 files=[Path(__file__),Path(__file__).with_name('marginal_equivalence_core.py'),Path(__file__).parent/'tests/test_marginal_equivalence.py',checkpoint];identity={str(p):sha(p) for p in files};dest=OUT/'scoring_resume_identity.json'
 if dest.exists():assert read(dest)==identity
 else:js(dest,identity)
 evaluate()
 for file,h in saved.items():assert sha(Path(file))==h,file
 js(OUT/'scoring_resume_verification.json',dict(status='PASS',previous_files_unchanged=len(saved),identity_files=len(identity),correction='only changed-rank CRPS uses existing 3e-12 marginal tolerance; all other old marginals and same-rank scores stay exact; no prediction or score modified'))

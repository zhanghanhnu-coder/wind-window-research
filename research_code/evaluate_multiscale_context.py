"""Round32 scoring after all forecasts are frozen; common graph copula for every family."""
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from research_multiscale_context import OUT,SRC,COP,ROLLING,ART,FOLDS,SEEDS,OLD,NEW,STATS,FAMILIES,sha,read,js,csv,verify_parents
from research_crps_distribution import guard,METRICS as JOINT
from seasonal_probability_core import score_atoms
from crps_distribution_core import couple_atoms
from probability_core import score_scenarios
from lead_calibration_core import BINS
METRICS=JOINT+['brier_hourly','mae','crps_49_72']
COMPARE=['crps','crps_49_72','brier_hourly','brier6','group_brier','length_crps72','first_cdf_brier','spatial_variogram','temporal_variogram']
MAIN=[('context3_graph','raw','context1_graph','raw'),('context7_graph','raw','context1_graph','raw'),('context7_graph','raw','context7_own','raw'),('context7_graph','rolling28','context1_graph','rolling28'),('context7_graph','rolling28','climate28','raw'),('context7_graph','rolling28','analog112','raw')]
PAIRS=MAIN+[('context7_graph','raw','context7_mean','raw'),('context7_graph','rolling28','context7_own','rolling28'),('context1_graph','rolling28','climate28','raw'),('context1_graph','rolling28','analog112','raw'),('context7_graph','frozen','context1_graph','frozen'),('context7_graph','rolling28','context7_graph','raw'),('context7_graph','rolling28','context7_graph','frozen'),('climate28','raw','analog112','raw')]
def primary(scope,model,method,base,bmethod,metric):return scope=='ALL' and (model,method,base,bmethod) in MAIN and metric in ['crps_49_72','brier6']
def score_array(s,metric):return s['crps'][:,48:72] if metric=='crps_49_72' else s[metric]
def freeze():
 guard();OUT.mkdir(exist_ok=True);files=[Path(__file__),Path(__file__).with_name('multiscale_context_tables.py'),Path(__file__).with_name('probability_core.py'),Path(__file__).with_name('seasonal_probability_core.py')];identity={str(p):sha(p) for p in files};dest=OUT/'evaluation_identity.json'
 if dest.exists():assert read(dest)==identity
 else:js(dest,identity)
 js(OUT/'evaluation_recipe.json',dict(pairs=PAIRS,main_pairs=MAIN,metrics=COMPARE,primary_metrics=['crps_49_72','brier6'],bootstrap_seed=20261119,bootstrap_replicates=2000))
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
   for k in keys:np.testing.assert_array_equal(s[k],old[k])
   checks.append(dict(fold=name,model=family,seed=seed,method=method,arrays=len(keys),common_graph_template=True))
  print('RESULT',name,family,seed,method,'CRPS49_72',np.nanmean(s['crps'][:,48:72]),'B6',np.nanmean(s['brier6']),flush=True)
 csv('old_marginal_equivalence.csv',checks)
 from multiscale_context_tables import tables,compare
 tables();compare();verify_parents();js(OUT/'evaluation_complete.json',dict(status='COMPLETE',flows=324,forecast_origins=71280,utc=datetime.now(timezone.utc).isoformat()));print('COMPLETE',flush=True)
if __name__=='__main__':
 import sys
 freeze() if len(sys.argv)>1 and sys.argv[1]=='freeze' else evaluate()

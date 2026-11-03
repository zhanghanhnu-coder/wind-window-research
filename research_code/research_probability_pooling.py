"""Round29: prequential, globally weighted mixtures of complete joint forecasts."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
from probability_pooling_core import fit_at,weighted_paths
from probability_core import score_scenarios,quantile
from research_crps_distribution import guard,SEEDS,METRICS as JOINT
from seasonal_probability_core import FOLDS
from lead_calibration_core import BINS
OUT=ART/'probability_pooling_v1';SRC=ART/'seasonal_probability_v1';PARENT=ART/'seasonal_rolling_v1';PLAN=ART.parent.parent/'研究笔记/因果分布组合与完整联合情景方案_2026-11-02.md'
MODELS=['gated_graph_atoms','gated_temporal_atoms','gated_mean_atoms'];EXPERTS=['raw','frozen','rolling28','rolling56'];NEW=['equal_pool','online_pool28'];METRICS=JOINT+['brier_hourly','mae'];COMPARE=['crps','brier_hourly','brier6','group_brier','length_crps72','first_cdf_brier','spatial_variogram','temporal_variogram']
G=MODELS[0];PAIRS=[(G,'online_pool28',G,'equal_pool'),(G,'online_pool28',G,'rolling28'),(G,'online_pool28',MODELS[1],'online_pool28'),(G,'online_pool28',MODELS[2],'online_pool28'),(G,'equal_pool',G,'rolling28')]+[(G,m,G,b) for m in NEW[::-1] for b in ['raw','frozen','rolling56']]+[(MODELS[1],'online_pool28',MODELS[1],'rolling28')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def js(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def primary(scope,model,method,base,bmethod,metric):
 return scope=='ALL' and model==G and method=='online_pool28' and ((base==G and bmethod=='equal_pool' and metric in ['crps','brier6']) or (base==G and bmethod=='rolling28' and metric in ['crps','brier6','group_brier']) or (base==MODELS[1] and bmethod=='online_pool28' and metric in ['brier6','group_brier']))
def verify_parent():
 manifest=read(PARENT/'final_manifest.json')
 for p,h in manifest.items():assert sha(Path(p))==h,p
 return len(manifest)
def prepare():
 guard();verify_parent();OUT.mkdir(exist_ok=True);f=OUT/'protocol_frozen.md';protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0]
 if f.exists():assert f.read_text(encoding='utf-8')==protocol
 else:f.write_text(protocol,encoding='utf-8')
 files=[f,Path(__file__)]+[Path(__file__).with_name(n) for n in ['probability_pooling_core.py','probability_core.py','crps_distribution_core.py','seasonal_rolling_core.py','seasonal_probability_core.py','lead_calibration_core.py','research_crps_distribution.py','pooling_tables.py']]+[PARENT/'final_manifest.json',ART/'data.npz',ART/'metadata.json'];ident={str(p):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==ident
 else:js(OUT/'identity.json',ident)
 if not (OUT/'previous_hashes.json').exists():js(OUT/'previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 if not (OUT/'recipe.json').exists():js(OUT/'recipe.json',dict(models=MODELS,seeds=SEEDS,experts=EXPERTS,new_methods=NEW,folds=FOLDS,bootstrap_seed=20261116,bootstrap_replicates=2000,weight_regularizer=.001,optimizer='SLSQP maxiter200 ftol1e-10 FWgap1e-6',pairs=PAIRS,start_utc=datetime.now(timezone.utc).isoformat()))
def source_items(name,model,seed):
 inventory=read(PARENT/'inventory.json');return [next(r for r in inventory if r['fold']==name and r['model']==model and r['seed']==seed and r['method']==method) for method in EXPERTS]
def expert_arrays(items):
 values=[]
 for r in items:
  assert sha(Path(r['source']))==r['source_sha256'];values.append(np.load(r['source'])[r['source_key']])
 return np.stack(values)
def generate_weights():
 rows=[];inventory=[];failures=0;insufficient=0
 for name in FOLDS:
  path=OUT/name;path.mkdir(exist_ok=True);f=np.load(SRC/name/'data.npz');y=np.load(SRC/name/'test_truth.npz')['y'];ids=f['test_ids'];scale=float(f['scale']);schedule=read(PARENT/name/'schedule.json')
  for model in MODELS:
   for seed in SEEDS:
    guard();items=source_items(name,model,seed);q=expert_arrays(items);ranks=items[0]['ranks_source'];assert all(r['ranks_source']==ranks for r in items)
    for method in NEW:
     previous=np.full(4,.25);weights=np.empty((len(ids),4))
     for entry in schedule:
      guard();block=entry['block'];issue=entry['issue_id'];positions=np.asarray(entry['positions']);base=f'weights_{model}_{seed}_{method}_w{block}';npz=path/(base+'.npz');jp=path/(base+'.json')
      if method=='equal_pool' or block==0:
       w=np.full(4,.25);record=dict(audit=dict(issue_id=issue,initial=block==0,equal=method=='equal_pool',optimizer_called=False,insufficient_support=False,valid_origins=0,valid_labels=0,group_counts=[0]*4,source_positions=[],source_ids=[],earliest_label_id=None,latest_label_id=None),details=dict(success=None,fallback=False,reason='fixed equal weights; no fitting'))
       np.savez_compressed(npz,weights=w);js(jp,record)
      elif npz.exists() and jp.exists():w=np.load(npz)['weights'];record=read(jp)
      else:
       r=fit_at(q,y,ids,issue,previous,scale);w=r['weights'];record=dict(audit=dict(initial=False,equal=False,**r['audit']),details=r['details']);np.savez_compressed(npz,weights=w,mask=r['mask'],a=r['a'],b=r['b']);js(jp,record)
      audit=record['audit'];details=record['details'];assert audit['latest_label_id'] is None or audit['latest_label_id']<issue;weights[positions]=w;previous=w.copy();failures+=int(audit['optimizer_called'] and details['fallback']);insufficient+=int(audit['insufficient_support']);rows.append(dict(fold=name,model=model,seed=seed,method=method,block=block,update_id=issue,update_time=entry['issue_time'],applied_origins=len(positions),initial=audit['initial'],equal=audit['equal'],optimizer_called=audit['optimizer_called'],insufficient_support=audit['insufficient_support'],optimizer_failed=bool(audit['optimizer_called'] and details['fallback']),valid_origins=audit['valid_origins'],valid_labels=audit['valid_labels'],earliest_label_id=audit['earliest_label_id'],latest_label_id=audit['latest_label_id'],**{f'weight_{k}':float(w[j]) for j,k in enumerate(EXPERTS)},gap=details.get('candidate_gap'),objective=details.get('used_objective')));print('WEIGHTS',name,model,seed,method,block,w.tolist(),'failed',details['fallback'],flush=True)
     dest=path/f'weights_{model}_{seed}_{method}.npz';np.savez_compressed(dest,weights=weights);inventory.append(dict(fold=name,model=model,seed=seed,method=method,experts=items,weights_source=str(dest),weights_sha256=sha(dest),ranks_source=ranks,ranks_sha256=items[0]['ranks_sha256'],origins=len(ids),score_source=str(path/f'scores_{model}_{seed}_{method}.npz'),new=True))
 for r in read(PARENT/'inventory.json'):
  inventory.append(dict(**r,score_source=str(PARENT/r['fold']/f"scores_{r['model']}_{r['seed']}_{r['method']}.npz"),new=False))
 assert len(rows)==648 and len(inventory)==216;csv('weight_audit.csv',rows);js(OUT/'inventory.json',inventory);verify_parent();js(OUT/'weights_complete.json',dict(flows=216,new_flows=72,new_forecast_origins=sum(r['origins'] for r in inventory if r['new']),all_forecast_origins=sum(r['origins'] for r in inventory),weight_records=648,optimizer_calls=sum(r['optimizer_called'] for r in rows),support_fallbacks=insufficient,optimizer_failures=failures,utc=datetime.now(timezone.utc).isoformat()))
def evaluate():
 assert read(OUT/'weights_complete.json')['new_flows']==72
 for item in read(OUT/'inventory.json'):
  if not item['new']:continue
  guard();name=item['fold'];model=item['model'];seed=item['seed'];method=item['method'];dest=Path(item['score_source'])
  if dest.exists():continue
  f=np.load(SRC/name/'data.npz');y=np.load(SRC/name/'test_truth.npz')['y'];q=expert_arrays(item['experts']);w=np.load(item['weights_source'])['weights'];ranks=np.load(item['ranks_source'])['ranks'];records=[]
  for i,yi in enumerate(y):
   sc,sw=weighted_paths(q[:,i],ranks,w[i]);r=score_scenarios(sc,yi,sw,train_scale=float(f['scale']));valid=np.isfinite(yi);prob=np.sum(sw[:,None,None]*(sc<=10),axis=0);r.update(probability=np.where(valid,prob,np.nan),label=np.where(valid,yi<=10,np.nan),brier_hourly=np.where(valid,(prob-(yi<=10))**2,np.nan),mae=np.where(valid,abs(quantile(sc,.5,sw)-yi),np.nan));gp=[];gl=[]
   for offset in [0,24,48,66]:
    for group in [[0,1,2],[8,9,10],[13,15,16]]:
     obs=yi[offset:offset+6][:,group];ok=np.isfinite(obs).all();gp.append(float(sw@(sc[:,offset:offset+6][:,:,group]<=10).all((1,2))) if ok else np.nan);gl.append(float((obs<=10).all()) if ok else np.nan)
   r['group_probability']=np.array(gp);r['group_label']=np.array(gl);records.append(r)
   if i%80==0:guard();print('SCORE',name,model,seed,method,i,flush=True)
  s={k:np.stack([r[k] for r in records]) for k in records[0]};assert len(s)==35;np.savez_compressed(dest,**s);print('RESULT',name,model,seed,method,'CRPS',np.nanmean(s['crps']),'B6',np.nanmean(s['brier6']),flush=True)
 from pooling_tables import tables,compare
 tables();compare();verify_parent();js(OUT/'evaluation_complete.json',dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat()));print('COMPLETE',flush=True)
if __name__=='__main__':prepare();generate_weights();evaluate()

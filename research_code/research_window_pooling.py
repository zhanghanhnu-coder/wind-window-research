"""Round30: fixed composite scores and active-face optimization controls."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
from probability_pooling_core import weighted_paths
from window_pooling_core import fit_window_at
from probability_core import score_scenarios,quantile
from research_crps_distribution import guard,SEEDS,METRICS as JOINT
from seasonal_probability_core import FOLDS
from lead_calibration_core import BINS
OUT=ART/'window_pooling_v1';SRC=ART/'seasonal_probability_v1';ROLLING=ART/'seasonal_rolling_v1';PARENT=ART/'probability_pooling_v1';PLAN=ART.parent.parent/'研究笔记/连续窗口组合评分与精确权重方案_2026-11-02.md'
MODELS=['gated_graph_atoms','gated_temporal_atoms','gated_mean_atoms'];EXPERTS=['raw','frozen','rolling28','rolling56'];METHODS={'crps_exact':0,'window_exact1':1,'window_exact025':.25,'window_exact4':4};METRICS=JOINT+['brier_hourly','mae'];COMPARE=['crps','brier_hourly','brier6','group_brier','length_crps72','first_cdf_brier','spatial_variogram','temporal_variogram'];G=MODELS[0]
PAIRS=[(G,'window_exact1',G,m) for m in ['crps_exact','online_pool28','rolling28','equal_pool','raw','frozen','rolling56']]+[(G,'window_exact1',m,'window_exact1') for m in MODELS[1:]]+[(MODELS[1],'window_exact1',MODELS[1],'crps_exact')]+[(G,m,G,'window_exact1') for m in ['window_exact025','window_exact4']]+[(G,'crps_exact',G,'online_pool28'),(G,'window_exact4',G,'crps_exact')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def js(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def primary(scope,model,method,base,bmethod,metric):
 return scope=='ALL' and model==G and ((method=='window_exact1' and ((base==G and bmethod=='crps_exact' and metric in ['crps','brier6','group_brier']) or (base==G and bmethod=='rolling28' and metric in ['brier6','group_brier']) or (base==MODELS[1] and bmethod=='window_exact1' and metric in ['brier6','group_brier']))) or (method=='crps_exact' and base==G and bmethod=='online_pool28' and metric=='brier6'))
def verify_parents():
 counts={}
 for path in [PARENT,ROLLING]:
  manifest=read(path/'final_manifest.json')
  for p,h in manifest.items():assert sha(Path(p))==h,p
  counts[path.name]=len(manifest)
 return counts
def prepare():
 guard();verify_parents();OUT.mkdir(exist_ok=True);f=OUT/'protocol_frozen.md';protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0]
 if f.exists():assert f.read_text(encoding='utf-8')==protocol
 else:f.write_text(protocol,encoding='utf-8')
 files=[f,Path(__file__)]+[Path(__file__).with_name(n) for n in ['window_pooling_core.py','probability_pooling_core.py','probability_core.py','crps_distribution_core.py','seasonal_rolling_core.py','seasonal_probability_core.py','lead_calibration_core.py','research_crps_distribution.py','window_pooling_tables.py']]+[PARENT/'final_manifest.json',ROLLING/'final_manifest.json',ART/'data.npz',ART/'metadata.json'];ident={str(p):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==ident
 else:js(OUT/'identity.json',ident)
 if not (OUT/'previous_hashes.json').exists():js(OUT/'previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 if not (OUT/'recipe.json').exists():js(OUT/'recipe.json',dict(models=MODELS,seeds=SEEDS,experts=EXPERTS,new_methods=METHODS,folds=FOLDS,bootstrap_seed=20261117,bootstrap_replicates=2000,weight_regularizer=.001,optimizer='15 active faces; candidate FWgap1e-8 before objective tie1e-12',pairs=PAIRS,start_utc=datetime.now(timezone.utc).isoformat()))
def source_items(name,model,seed):
 inventory=read(ROLLING/'inventory.json');return [next(r for r in inventory if r['fold']==name and r['model']==model and r['seed']==seed and r['method']==method) for method in EXPERTS]
def expert_arrays(items):
 values=[]
 for r in items:
  assert sha(Path(r['source']))==r['source_sha256'];values.append(np.load(r['source'])[r['source_key']])
 return np.stack(values)
def generate_weights():
 rows=[];inventory=[];failures=0;insufficient=0
 for name in FOLDS:
  path=OUT/name;path.mkdir(exist_ok=True);f=np.load(SRC/name/'data.npz');ids=f['test_ids'];schedule=read(ROLLING/name/'schedule.json')
  for model in MODELS:
   for seed in SEEDS:
    guard();items=source_items(name,model,seed);old=[dict(np.load(ROLLING/name/f'scores_{model}_{seed}_{m}.npz')) for m in EXPERTS];sp=np.stack([z['prob6'] for z in old]);sl=old[0]['label6'];gp=np.stack([z['group_probability'] for z in old]);gl=old[0]['group_label']
    for z in old:np.testing.assert_array_equal(z['label6'],sl);np.testing.assert_array_equal(z['group_label'],gl)
    for method,lam in METHODS.items():
     if model!=G and method in ['window_exact025','window_exact4']:continue
     previous=np.full(4,.25);weights=np.empty((len(ids),4))
     for entry in schedule:
      guard();block=entry['block'];issue=entry['issue_id'];positions=np.asarray(entry['positions']);base=f'weights_{model}_{seed}_{method}_w{block}';npz=path/(base+'.npz');jp=path/(base+'.json')
      if block==0:
       w=np.full(4,.25);record=dict(audit=dict(issue_id=issue,lambda_value=lam,initial=True,optimizer_called=False,insufficient_support=False,hour_origins=0,hour_labels=0,hour_group_counts=[0]*4,single=dict(valid_events=0),group=dict(valid_events=0)),details=dict(success=None,fallback=False,reason='initial equal weights; no fitting'));np.savez_compressed(npz,weights=w);js(jp,record)
      elif npz.exists() and jp.exists():w=np.load(npz)['weights'];record=read(jp)
      else:
       source=PARENT/name/f'weights_{model}_{seed}_online_pool28_w{block}.npz';hour=np.load(source);r=fit_window_at(hour['a'],hour['b'],hour['mask'],ids,issue,previous,sp,sl,gp,gl,lam);w=r['weights'];record=dict(audit=dict(initial=False,**r['audit']),details=r['details'],hour_moments_source=str(source),hour_moments_sha256=sha(source));np.savez_compressed(npz,weights=w,a=r['a'],b=r['b'],single_mask=r['single']['mask'],single_C=r['single']['C'],single_d=r['single']['d'],single_e=r['single']['e'],group_mask=r['group']['mask'],group_C=r['group']['C'],group_d=r['group']['d'],group_e=r['group']['e']);js(jp,record)
      au=record['audit'];detail=record['details'];weights[positions]=w;previous=w.copy();failures+=int(au['optimizer_called'] and detail['fallback']);insufficient+=int(au['insufficient_support']);rows.append(dict(fold=name,model=model,seed=seed,method=method,lambda_value=lam,block=block,update_id=issue,update_time=entry['issue_time'],applied_origins=len(positions),initial=au['initial'],optimizer_called=au['optimizer_called'],insufficient_support=au['insufficient_support'],optimizer_failed=bool(au['optimizer_called'] and detail['fallback']),hour_origins=au['hour_origins'],hour_labels=au['hour_labels'],single_events=au['single']['valid_events'],group_events=au['group']['valid_events'],**{f'weight_{k}':float(w[j]) for j,k in enumerate(EXPERTS)},gap=detail.get('candidate_gap'),objective=detail.get('full_used_objective')));print('WEIGHTS',name,model,seed,method,block,w.tolist(),'failed',detail['fallback'],flush=True)
     dest=path/f'weights_{model}_{seed}_{method}.npz';np.savez_compressed(dest,weights=weights);inventory.append(dict(fold=name,model=model,seed=seed,method=method,lambda_value=lam,experts=items,weights_source=str(dest),weights_sha256=sha(dest),ranks_source=items[0]['ranks_source'],ranks_sha256=items[0]['ranks_sha256'],origins=len(ids),score_source=str(path/f'scores_{model}_{seed}_{method}.npz'),new=True))
 for item in read(PARENT/'inventory.json'):
  r=dict(item);r['new']=False;inventory.append(r)
 assert len(rows)==864 and len(inventory)==312;csv('weight_audit.csv',rows);js(OUT/'inventory.json',inventory);verify_parents();js(OUT/'weights_complete.json',dict(flows=312,new_flows=96,new_forecast_origins=sum(r['origins'] for r in inventory if r['new']),all_forecast_origins=sum(r['origins'] for r in inventory),weight_records=864,optimizer_calls=sum(r['optimizer_called'] for r in rows),support_fallbacks=insufficient,optimizer_failures=failures,utc=datetime.now(timezone.utc).isoformat()))
def evaluate():
 assert read(OUT/'weights_complete.json')['new_flows']==96
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
 from window_pooling_tables import tables,compare
 tables();compare();verify_parents();js(OUT/'evaluation_complete.json',dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat()));print('COMPLETE',flush=True)
if __name__=='__main__':prepare();generate_weights();evaluate()

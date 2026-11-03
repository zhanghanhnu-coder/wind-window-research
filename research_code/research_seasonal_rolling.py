"""Round27: weekly matured-label calibration, with fixed networks and copulas."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import pandas as pd
import torch
from model import ART
from crps_distribution_core import AtomNet,predict_atoms,couple_atoms
from lead_calibration_core import apply_groups,BINS
from seasonal_probability_core import FOLDS,score_atoms
from seasonal_rolling_core import archive_indices,weekly_schedule,calibrate_at
from probability_core import score_scenarios
from research_crps_distribution import guard,SEEDS,METRICS as JOINT_METRICS
OUT=ART/'seasonal_rolling_v1';SRC=ART/'seasonal_probability_v1';COP=ART/'partial_copula_v1';PLAN=ART.parent.parent/'研究笔记/跨月份变化的因果滚动校准方案_2026-11-02.md';MODELS=['gated_graph_atoms','gated_temporal_atoms','gated_mean_atoms'];METHODS=['raw','frozen','rolling28','rolling56'];METRICS=JOINT_METRICS+['brier_hourly','mae'];COMPARE=['crps','brier_hourly','brier6','group_brier','length_crps72','first_cdf_brier','spatial_variogram','temporal_variogram']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def js(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def verify_parents():
 counts={}
 for parent in [SRC,COP]:
  manifest=read(parent/'final_manifest.json')
  for p,h in manifest.items():assert sha(Path(p))==h,p
  counts[parent.name]=len(manifest)
 return counts

def prepare():
 guard();verify_parents();OUT.mkdir(exist_ok=True);frozen=OUT/'protocol_frozen.md';protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0]
 if frozen.exists():assert frozen.read_text(encoding='utf-8')==protocol
 else:frozen.write_text(protocol,encoding='utf-8')
 files=[frozen,Path(__file__)]+[Path(__file__).with_name(n) for n in ['seasonal_rolling_core.py','model.py','crps_distribution_core.py','direct_window_core.py','lead_calibration_core.py','seasonal_probability_core.py','probability_core.py','research_crps_distribution.py']]+[SRC/'final_manifest.json',COP/'final_manifest.json',ART/'data.npz',ART/'metadata.json'];identity={str(p):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==identity
 else:js(OUT/'identity.json',identity)
 if not (OUT/'previous_hashes.json').exists():js(OUT/'previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 if not (OUT/'recipe.json').exists():js(OUT/'recipe.json',dict(models=MODELS,seeds=SEEDS,methods=METHODS,folds=FOLDS,initial_week='exact frozen25 parameters',lookbacks=[28,56],weekly_anchor='test period first date at 00:00; update at first eligible issue per block',strict_maturity='source issue + lead < update issue',neural_training=False,copula='frozen round26 partial05',bootstrap_seed=20261115,bootstrap_replicates=2000,start_utc=datetime.now(timezone.utc).isoformat()))

def build_archives():
 d=dict(np.load(ART/'data.npz'));times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');assert (np.diff(times.asi8)==3600000000000).all();rows=[]
 for name,cfg in FOLDS.items():
  path=OUT/name;path.mkdir(exist_ok=True);f=np.load(SRC/name/'data.npz');ids=archive_indices(f['x'],times,cfg['cal_start'],int(max(f['test_ids'])));positions={int(i):j for j,i in enumerate(ids)};test_positions=np.array([positions[int(i)] for i in f['test_ids']]);truth=np.stack([d['raw'][i+1:i+73] for i in ids]);schedule=weekly_schedule(times,f['test_ids'],cfg['test_start']);assert [r['block'] for r in schedule]==list(range(9));blocks=np.empty(len(f['test_ids']),int)
  for r in schedule:blocks[r['positions']]=r['block']
  kinds=np.full(len(ids),'new',dtype='<U4');known={}
  for split in ['cal','test']:
   for j,i in enumerate(f[split+'_ids']):known[int(i)]=(split,j);kinds[positions[int(i)]]=split
  np.savez_compressed(path/'archive_data.npz',ids=ids,truth=truth,test_ids=f['test_ids'],test_positions=test_positions,blocks=blocks,update_ids=np.array([r['issue_id'] for r in schedule]),source_kind=kinds);js(path/'schedule.json',[dict(block=r['block'],issue_id=r['issue_id'],positions=r['positions'].tolist(),issue_time=str(times[r['issue_id']])) for r in schedule])
  for model in MODELS:
   for seed in SEEDS:
    guard();dest=path/f'archive_predictions_{model}_{seed}.npz';source=dict(np.load(SRC/name/f'predictions_{model}_{seed}_budget40.npz'));new_positions=np.flatnonzero(kinds=='new');ckpath=SRC/name/f'{model}_{seed}_budget40.pt'
    if not dest.exists():
     q=np.empty((len(ids),72,17,19),np.float32)
     for i,(split,j) in known.items():q[positions[i]]=source[split][j]
     ck=torch.load(ckpath,map_location='cpu',weights_only=False);net=AtomNet(ck['adj'],ck['scale'],hidden=24,gated=True);net.load_state_dict(ck['state'])
     if len(new_positions):q[new_positions]=predict_atoms(net,f['x'],ids[new_positions])
     np.savez_compressed(dest,q=q)
    q=np.load(dest)['q'];assert q.shape==(len(ids),72,17,19) and np.isfinite(q).all()
    for split in ['cal','test']:np.testing.assert_array_equal(q[[positions[int(i)] for i in f[split+'_ids']]],source[split])
    rows.append(dict(fold=name,model=model,seed=seed,archive_origins=len(ids),new_origins=int(len(new_positions)),cal_reused=int(sum(kinds=='cal')),test_reused=int(sum(kinds=='test')),first_issue=str(times[ids[0]]),last_issue=str(times[ids[-1]]),checkpoint_sha256=sha(ckpath),archive_sha256=sha(dest)));print('ARCHIVE',name,model,seed,len(ids),'new',len(new_positions),flush=True)
 csv('archive_summary.csv',rows);assert len(rows)==36;js(OUT/'archive_complete.json',dict(models=36,utc=datetime.now(timezone.utc).isoformat()))

def generate_predictions():
 assert read(OUT/'archive_complete.json')['models']==36;audits=[];details=[];inventory=[];reused=[];fitted=0;support_fallback=0;optimizer_failures=0
 for name in FOLDS:
  path=OUT/name;f=np.load(SRC/name/'data.npz');a=np.load(path/'archive_data.npz');schedule=read(path/'schedule.json');ids=a['ids'];truth=a['truth'];scale=float(f['scale'])
  for model in MODELS:
   for seed in SEEDS:
    guard();archive=np.load(path/f'archive_predictions_{model}_{seed}.npz')['q'];raw=np.load(SRC/name/f'predictions_{model}_{seed}_budget40.npz')['test'];frozen=np.load(SRC/name/f'calibrated_{model}_{seed}.npz')['q'];theta0=np.load(SRC/name/f'calibration_{model}_{seed}.npz')['theta'];initial_details=read(SRC/name/f'calibration_{model}_{seed}.json');np.testing.assert_array_equal(apply_groups(raw,theta0,scale,True),frozen)
    for method,days in [('rolling28',28),('rolling56',56)]:
     previous=theta0.copy();pred=np.empty(frozen.shape,float)
     for entry in schedule:
      guard();block=entry['block'];issue=entry['issue_id'];positions=np.asarray(entry['positions']);base=f'update_{model}_{seed}_{method}_w{block}';npzpath=path/(base+'.npz');jsonpath=path/(base+'.json')
      if block==0:
       theta=theta0.copy();audit=dict(issue_id=issue,lookback_days=days,initial_frozen=True,reused_previous=False,fitted_groups=0,optimizer_failures=0,valid_origins=int(np.isfinite(np.load(SRC/name/'cal_truth.npz')['y']).any((1,2)).sum()),valid_labels=int(np.isfinite(np.load(SRC/name/'cal_truth.npz')['y']).sum()),group_counts=[int(r['valid_targets']) for r in initial_details],earliest_label_id=int(min(f['cal_ids'])+1),latest_label_id=int(max(f['cal_ids']+72)));record=dict(audit=audit,details=initial_details,source_calibration=str(SRC/name/f'calibration_{model}_{seed}.npz'));np.savez_compressed(npzpath,theta=theta);js(jsonpath,record);pred[positions]=frozen[positions]
      else:
       if jsonpath.exists() and npzpath.exists():record=read(jsonpath);theta=np.load(npzpath)['theta']
       else:
        r=calibrate_at(archive,truth,ids,issue,days,scale,previous);theta=r['theta'];record=dict(audit=dict(initial_frozen=False,**r['audit']),details=r['details']);np.savez_compressed(npzpath,theta=theta,mask=r['mask']);js(jsonpath,record)
       audit=record['audit'];assert audit['latest_label_id'] is None or audit['latest_label_id']<issue;fitted+=audit['fitted_groups'];support_fallback+=audit['reused_previous'];optimizer_failures+=audit['optimizer_failures'];pred[positions]=apply_groups(raw[positions],theta,scale,True)
       for info in record['details']:details.append(dict(fold=name,model=model,seed=seed,method=method,block=block,update_id=issue,reused_previous=audit['reused_previous'],**info))
      previous=theta.copy();audit=record['audit'];audits.append(dict(fold=name,model=model,seed=seed,method=method,block=block,update_id=issue,update_time=entry['issue_time'],applied_origins=len(positions),initial_frozen=audit['initial_frozen'],reused_previous=audit['reused_previous'],fitted_groups=audit['fitted_groups'],optimizer_failures=audit['optimizer_failures'],valid_origins=audit['valid_origins'],valid_labels=audit['valid_labels'],earliest_label_id=audit['earliest_label_id'],latest_label_id=audit['latest_label_id'],oldest_observation_age_hours=issue-audit['earliest_label_id'] if audit['earliest_label_id'] is not None else None));print('UPDATE',name,model,seed,method,block,'labels',audit['valid_labels'],'failed',audit['optimizer_failures'],flush=True)
     assert np.isfinite(pred).all() and (np.diff(pred,axis=-1)>=0).all();np.testing.assert_array_equal(pred[a['blocks']==0],frozen[a['blocks']==0]);np.savez_compressed(path/f'predictions_{model}_{seed}_{method}.npz',q=pred);reused.append(dict(fold=name,model=model,seed=seed,method=method,initial_origins=int(sum(a['blocks']==0)),initial_exact=True))
    for method in METHODS:
     source=SRC/name/f'predictions_{model}_{seed}_budget40.npz' if method=='raw' else SRC/name/f'calibrated_{model}_{seed}.npz' if method=='frozen' else path/f'predictions_{model}_{seed}_{method}.npz';inventory.append(dict(fold=name,model=model,seed=seed,method=method,source=str(source),source_sha256=sha(source),source_key='test' if method=='raw' else 'q',ranks_source=str(COP/name/f'ranks_{model}_{seed}_partial05.npz'),ranks_sha256=sha(COP/name/f'ranks_{model}_{seed}_partial05.npz'),origins=len(raw)))
 assert len(inventory)==144 and len(audits)==648 and len(details)==2304;csv('update_audit.csv',audits);csv('calibration.csv',details);csv('initial_week_equivalence.csv',reused);js(OUT/'inventory.json',inventory);verify_parents();js(OUT/'prediction_complete.json',dict(flows=144,forecast_origins=sum(r['origins'] for r in inventory),initial_frozen_updates=72,weekly_update_calls=576,fitted_groups=fitted,support_fallbacks=int(support_fallback),optimizer_failures=int(optimizer_failures),utc=datetime.now(timezone.utc).isoformat()))

def evaluate():
 assert read(OUT/'prediction_complete.json')['flows']==144;d=np.load(ART/'data.npz');times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');rows=[];months=[];weeks=[];leads=[];rel=[];high=[];checks=[]
 for item in read(OUT/'inventory.json'):
  guard();name=item['fold'];model=item['model'];seed=item['seed'];method=item['method'];path=OUT/name;f=np.load(SRC/name/'data.npz');assert sha(Path(item['source']))==item['source_sha256'] and sha(Path(item['ranks_source']))==item['ranks_sha256'];q=np.load(item['source'])[item['source_key']];ranks=np.load(item['ranks_source'])['ranks'];y=np.load(SRC/name/'test_truth.npz')['y'];dest=path/f'scores_{model}_{seed}_{method}.npz';blocks=np.load(path/'archive_data.npz')['blocks'];ts=times[f['test_ids']]
  if dest.exists():s=dict(np.load(dest))
  else:
   marginal=score_atoms(q,y)
   if method=='frozen':s=dict(np.load(COP/name/f'scores_{model}_{seed}_partial05.npz'))
   else:
    records=[]
    for i,(qi,yi) in enumerate(zip(q,y)):
     sc=couple_atoms(qi,ranks);np.testing.assert_array_equal(np.sort(sc,axis=0),np.repeat(qi,3,axis=-1).transpose(2,0,1));r=score_scenarios(sc,yi,train_scale=float(f['scale']));gp=[];gl=[]
     for offset in [0,24,48,66]:
      for group in [[0,1,2],[8,9,10],[13,15,16]]:
       obs=yi[offset:offset+6][:,group];ok=np.isfinite(obs).all();gp.append(float((sc[:,offset:offset+6][:,:,group]<=10).all((1,2)).mean()) if ok else np.nan);gl.append(float((obs<=10).all()) if ok else np.nan)
     r['group_probability']=np.array(gp);r['group_label']=np.array(gl);records.append(r)
     if i%80==0:guard();print('SCORE',name,model,seed,method,i,flush=True)
    s={k:np.stack([r[k] for r in records]) for k in records[0]}
   for metric in ['crps','coverage','width','low_miss','high_miss','interval_score']:np.testing.assert_allclose(s[metric],marginal[metric],atol=3e-12,rtol=0,equal_nan=True)
   s.update({k:marginal[k] for k in ['probability','label','brier_hourly','mae']});np.savez_compressed(dest,**s)
  if method=='frozen':
   old=np.load(COP/name/f'scores_{model}_{seed}_partial05.npz');previous=np.load(SRC/name/f'scores_{model}_{seed}_smooth_affine.npz')
   for key in old.files:np.testing.assert_array_equal(s[key],old[key])
   for key in ['probability','label','brier_hourly','mae']:np.testing.assert_array_equal(s[key],previous[key])
   checks.append(dict(fold=name,model=model,seed=seed,joint_exact=True,hourly_exact=True))
  common=dict(fold=name,model=model,seed=seed,method=method)
  for metric in METRICS:
   z=s[metric];rows.append(dict(**common,metric=metric,value=float(np.nanmean(z)),n=int(np.isfinite(z).sum())))
   for month in sorted(set(ts.month)):
    a=z[ts.month==month];months.append(dict(**common,issue_month=int(month),metric=metric,value=float(np.nanmean(a)),n=int(np.isfinite(a).sum())))
   for block in sorted(set(blocks)):
    a=z[blocks==block];weeks.append(dict(**common,week_block=int(block),metric=metric,value=float(np.nanmean(a)),n=int(np.isfinite(a).sum())))
  for a,b in BINS:
   for metric in ['crps','brier_hourly','mae','coverage','width','interval_score','low_miss','high_miss']:
    z=s[metric][:,a:b];leads.append(dict(**common,lead_start=a+1,lead_end=b,metric=metric,value=float(np.nanmean(z)),n=int(np.isfinite(z).sum())))
  for target,pk,yk in [('hourly','probability','label'),('single_6h','prob6','label6'),('group_6h','group_probability','group_label')]:
   p=s[pk].ravel();label=s[yk].ravel();valid=np.isfinite(label);p=p[valid];label=label[valid];select=p>=.8;high.append(dict(**common,target=target,n=int(select.sum()),total=len(p),mean_probability=float(p[select].mean()) if select.any() else np.nan,observed_rate=float(label[select].mean()) if select.any() else np.nan))
   for b in range(10):
    select=(p>=b/10)&(p<(b+1)/10 if b<9 else p<=1);rel.append(dict(**common,target=target,bin=b,n=int(select.sum()),mean_probability=float(p[select].mean()) if select.any() else np.nan,observed_rate=float(label[select].mean()) if select.any() else np.nan))
  print('RESULT',name,model,seed,method,'CRPS',np.nanmean(s['crps']),'hourly',np.nanmean(s['brier_hourly']),'window',np.nanmean(s['brier6']),flush=True)
 df=pd.DataFrame(rows)
 for key,g in df.groupby(['model','seed','method','metric'],sort=False):rows.append(dict(fold='ALL',**dict(zip(['model','seed','method','metric'],key)),value=float(np.sum(g.value*g.n)/g.n.sum()),n=int(g.n.sum())))
 csv('metrics.csv',rows);csv('month_metrics.csv',months);csv('weekly_metrics.csv',weeks);csv('lead_metrics.csv',leads);csv('high_confidence.csv',high);csv('reliability.csv',rel);csv('frozen_baseline_equivalence.csv',checks)

def compare():
 d=np.load(ART/'data.npz');times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');rng=np.random.default_rng(20261115);weeks={};draws={};sizes={}
 for name,cfg in FOLDS.items():
  ids=np.load(SRC/name/'data.npz')['test_ids'];weeks[name]=((times[ids]-pd.Timestamp(cfg['test_start'],tz=times.tz)).days//7).to_numpy();k=int(weeks[name].max())+1;draws[name]=rng.integers(0,k,(2000,k));sizes[name]=len(ids)
 graph='gated_graph_atoms';pairs=[(graph,'rolling28',graph,'frozen'),(graph,'rolling28','gated_temporal_atoms','rolling28'),(graph,'rolling28','gated_mean_atoms','rolling28'),(graph,'rolling56',graph,'rolling28'),(graph,'rolling28',graph,'raw'),(graph,'frozen',graph,'raw')]+[(model,'rolling28',model,'frozen') for model in MODELS[1:]];rows=[]
 for model,method,base,bmethod in pairs:
  for metric in COMPARE:
   sums={};counts={}
   for name in FOLDS:
    diff=[]
    for seed in SEEDS:
     a=np.load(OUT/name/f'scores_{model}_{seed}_{method}.npz')[metric];b=np.load(OUT/name/f'scores_{base}_{seed}_{bmethod}.npz')[metric];np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b));diff.append(a-b)
    z=np.mean(diff,0);sums[name]=np.array([np.nansum(z[weeks[name]==j]) for j in range(draws[name].shape[1])]);counts[name]=np.array([np.isfinite(z[weeks[name]==j]).sum() for j in range(draws[name].shape[1])])
   for scope in ['ALL']+list(FOLDS):
    chosen=list(FOLDS) if scope=='ALL' else [scope];num=np.zeros(2000);den=np.zeros(2000);total=0.;n=0;blocks=[]
    for name in chosen:
     num+=sums[name][draws[name]].sum(1);den+=counts[name][draws[name]].sum(1);total+=sums[name].sum();n+=counts[name].sum();blocks.append(int((counts[name]>0).sum()))
    valid=den>0;supported=min(blocks)>=4 and valid.sum()>=1900;lo,hi=np.quantile(num[valid]/den[valid],[.025,.975]) if supported else [np.nan,np.nan];primary=scope=='ALL' and model==graph and method=='rolling28' and ((base==graph and bmethod=='frozen' and metric in ['crps','brier_hourly','brier6','group_brier']) or (base=='gated_temporal_atoms' and bmethod=='rolling28' and metric in ['brier6','group_brier']));rows.append(dict(scope=scope,candidate=model,method=method,baseline=base,baseline_method=bmethod,metric=metric,primary=primary,difference=float(total/n),low95=float(lo),high95=float(hi),n=int(n),origins=sum(sizes[x] for x in chosen),nonempty_blocks=sum(blocks),minimum_blocks_in_fold=min(blocks),valid_draws=int(valid.sum()),interval_supported=bool(supported)))
 assert len(rows)==320 and sum(r['primary'] for r in rows)==6;csv('paired_comparisons.csv',rows)

if __name__=='__main__':
 prepare();build_archives();generate_predictions();evaluate();compare();verify_parents();print('COMPLETE',flush=True)

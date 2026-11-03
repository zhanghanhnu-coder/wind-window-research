"""Independent all-weight, all-score, all-table and block-bootstrap audit for round29."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
from pooling_reference import moments_reference,simplex_reference,marginal_reference,reconstruct_mixture,joint_reference,mixture_probability_reference
from research_crps_distribution import guard,SEEDS,METRICS as JOINT
OUT=ART/'probability_pooling_v1';SRC=ART/'seasonal_probability_v1';PARENT=ART/'seasonal_rolling_v1'
FOLDS={'M05':'2024-05-01','M07':'2024-07-01','M09':'2024-09-01','M11':'2024-11-01'};MODELS=['gated_graph_atoms','gated_temporal_atoms','gated_mean_atoms'];EXPERTS=['raw','frozen','rolling28','rolling56'];NEW=['equal_pool','online_pool28'];BINS=[(0,6),(6,24),(24,48),(48,72)];METRICS=JOINT+['brier_hourly','mae'];COMPARE=['crps','brier_hourly','brier6','group_brier','length_crps72','first_cdf_brier','spatial_variogram','temporal_variogram']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=3e-11):np.testing.assert_allclose(a,b,atol=tol,rtol=0,equal_nan=True)
def compare_table(name,rows,keys):
 a=pd.read_csv(OUT/name);b=pd.DataFrame(rows);assert len(a)==len(b),(name,len(a),len(b));a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
 for k in b:
  if b[k].dtype.kind in 'biufc':close(a[k],b[k])
  else:assert a[k].fillna('').tolist()==b[k].fillna('').tolist(),(name,k)
 return len(a)
def audit_weights(times):
 inventory=read(OUT/'inventory.json');parent_inventory=read(PARENT/'inventory.json');rows=[];fits=[];counts=dict(weight_records=0,maturity_masks=0,independent_moments=0,active_face_optima=0,direct_moment_cells=0,future_perturbations=0,applied_weight_flows=0);seen=set()
 for name,start in FOLDS.items():
  f=np.load(SRC/name/'data.npz');ids=f['test_ids'];y=np.load(SRC/name/'test_truth.npz')['y'];scale=float(f['scale']);blocks=((times[ids]-pd.Timestamp(start,tz=times.tz)).days//7).to_numpy();schedule=read(PARENT/name/'schedule.json');target=ids[:,None]+np.arange(1,73)
  assert len(schedule)==9
  for entry in schedule:
   positions=np.flatnonzero(blocks==entry['block']);np.testing.assert_array_equal(positions,entry['positions']);assert entry['issue_id']==ids[positions[0]]
  for model in MODELS:
   for seed in SEEDS:
    guard();items=[next(r for r in parent_inventory if r['fold']==name and r['model']==model and r['seed']==seed and r['method']==m) for m in EXPERTS];q=np.stack([np.load(r['source'])[r['source_key']] for r in items])
    for method in NEW:
     inv=next(r for r in inventory if (r['fold'],r['model'],r['seed'],r['method'])==(name,model,seed,method));assert inv['experts']==items and inv['new'];assert inv['origins']==len(ids);assert inv['ranks_source']==items[0]['ranks_source'] and inv['ranks_sha256']==items[0]['ranks_sha256'];assert sha(Path(inv['weights_source']))==inv['weights_sha256'];assert sha(Path(inv['ranks_source']))==inv['ranks_sha256'];applied=np.load(inv['weights_source'])['weights'];previous=np.full(4,.25)
     for entry in schedule:
      block=entry['block'];issue=entry['issue_id'];pos=np.array(entry['positions']);base=f'weights_{model}_{seed}_{method}_w{block}';z=np.load(OUT/name/(base+'.npz'));record=read(OUT/name/(base+'.json'));w=z['weights'];au=record['audit'];detail=record['details'];assert w.shape==(4,) and (w>=0).all();close(w.sum(),1,1e-14);assert au['issue_id']==issue and au['initial']==(block==0) and au['equal']==(method=='equal_pool')
      if method=='equal_pool' or block==0:
       np.testing.assert_array_equal(w,np.full(4,.25));assert not au['optimizer_called'] and not au['insufficient_support'];assert au['source_positions']==[] and au['source_ids']==[] and au['valid_labels']==0 and au['valid_origins']==0 and au['group_counts']==[0]*4;assert au['earliest_label_id'] is None and au['latest_label_id'] is None;assert detail['success'] is None and not detail['fallback']
      else:
       mask=np.isfinite(y)&((ids[:,None]<issue)&(target<issue)&(target>=issue-672))[:,:,None];np.testing.assert_array_equal(mask,z['mask']);selected=np.flatnonzero(mask.any((1,2)));groupcounts=[int(mask[:,lo:hi].sum()) for lo,hi in BINS];known=target[mask.any(-1)];support=len(selected)>=20 and min(groupcounts)>=100;assert au['source_positions']==selected.tolist() and au['source_ids']==ids[selected].tolist();assert au['valid_labels']==mask.sum() and au['valid_origins']==len(selected) and au['group_counts']==groupcounts;assert au['earliest_label_id']==(int(known.min()) if len(known) else None) and au['latest_label_id']==(int(known.max()) if len(known) else None);assert au['insufficient_support']==(not support) and au['optimizer_called']==support;counts['maturity_masks']+=1
       if support:
        aa,bb=moments_reference(q[:,mask],y[mask],scale);close(aa,z['a'],2e-12);close(bb,z['b'],2e-12);counts['independent_moments']+=1;best,bestw=simplex_reference(aa,bb);candidate=np.asarray(detail['candidate_weights']);a=z['a'];b=z['b'];value=float(a@candidate-.5*candidate@b@candidate+.001*np.sum((candidate-.25)**2));gradient=a-b@candidate+.002*(candidate-.25);gap=float(gradient@candidate-gradient.min());tangent=np.vstack([np.eye(3),-np.ones(3)]);mineig=float(np.linalg.eigvalsh(tangent.T@(-b+.002*np.eye(4))@tangent).min());close(value,detail['candidate_objective']);close(gradient,detail['candidate_gradient']);close(gap,detail['candidate_gap']);close(mineig,detail['tangent_min_eigenvalue']);used=float(aa@w-.5*w@bb@w+.001*np.sum((w-.25)**2));close(used,detail['used_objective']);assert detail['fallback']==(not detail['success'])
        if detail['success']:
         assert detail['status']==0 and detail['feasible'] and mineig>=-1e-9 and gap<=1e-6;np.testing.assert_array_equal(w,candidate);assert used-best<=1.01e-6 and used-best>=-1e-10
        else:np.testing.assert_array_equal(w,previous)
        fits.append(dict(fold=name,model=model,seed=seed,block=block,used_objective=used,global_objective=best,objective_gap=used-best,frank_wolfe_gap=gap,optimizer_success=detail['success'],weight_distance=float(np.max(abs(w-bestw)))));counts['active_face_optima']+=1
        # Direct pair enumeration on fixed spread of 32 valid coordinates, all 16 pairs.
        chosen=np.linspace(0,mask.sum()-1,min(32,int(mask.sum())),dtype=int);samples=q[:,mask][:,chosen]/scale
        for i in range(4):
         for j in range(4):
          direct=abs(samples[i,:,:,None]-samples[j,:,None,:]).mean((1,2));from probability_pooling_core import cross_distance
          close(cross_distance(samples[i],samples[j]),direct,2e-13)
        counts['direct_moment_cells']+=len(chosen)
       else:np.testing.assert_array_equal(w,previous);assert detail['fallback'];assert np.isnan(z['a']).all() and np.isnan(z['b']).all()
       if model==MODELS[0] and seed==SEEDS[0] and block in [1,8]:
        from probability_pooling_core import fit_at
        yy=y.copy();qq=q.copy();future=np.broadcast_to((target>=issue)[:,:,None],y.shape);yy[future]=12345;qq[:,future]=-12345;changed=fit_at(qq,yy,ids,issue,previous,scale);np.testing.assert_array_equal(changed['weights'],w);np.testing.assert_array_equal(changed['mask'],mask);assert changed['audit']=={k:v for k,v in au.items() if k not in ['initial','equal']};counts['future_perturbations']+=1
      np.testing.assert_array_equal(applied[pos],np.broadcast_to(w,(len(pos),4)));previous=w.copy();counts['weight_records']+=1;rows.append(dict(fold=name,model=model,seed=seed,method=method,block=block,update_id=issue,update_time=str(times[issue]),applied_origins=len(pos),initial=au['initial'],equal=au['equal'],optimizer_called=au['optimizer_called'],insufficient_support=au['insufficient_support'],optimizer_failed=bool(au['optimizer_called'] and detail['fallback']),valid_origins=au['valid_origins'],valid_labels=au['valid_labels'],earliest_label_id=au['earliest_label_id'],latest_label_id=au['latest_label_id'],**{f'weight_{k}':float(w[j]) for j,k in enumerate(EXPERTS)},gap=detail.get('candidate_gap'),objective=detail.get('used_objective')))
     seen.add((name,model,seed,method));counts['applied_weight_flows']+=1
    print('WEIGHT_AUDIT',name,model,seed,flush=True)
 assert len(seen)==72 and counts['weight_records']==648 and counts['maturity_masks']==288;counts['weight_table_rows']=compare_table('weight_audit.csv',rows,['fold','model','seed','method','block']);pd.DataFrame(fits).to_csv(OUT/'independent_weight_optima.csv',index=False,encoding='utf-8-sig');return counts

def scores_and_tables(times):
 inventory=read(OUT/'inventory.json');assert len(inventory)==216 and sum(r['new'] for r in inventory)==72;rows=[];months=[];weeks=[];leads=[];rel=[];high=[];cache={};counts=dict(new_forecast_cases=0,new_score_arrays=0,baseline_arrays_unchanged=0,mixture_probability_conservations=0)
 for item in inventory:
  guard();name=item['fold'];model=item['model'];seed=item['seed'];method=item['method'];key=name,model,seed,method;s=dict(np.load(item['score_source']));f=np.load(SRC/name/'data.npz');y=np.load(SRC/name/'test_truth.npz')['y'];ts=times[f['test_ids']];blocks=((ts-pd.Timestamp(FOLDS[name],tz=times.tz)).days//7).to_numpy();assert len(s)==35
  if item['new']:
   q=np.stack([np.load(r['source'])[r['source_key']] for r in item['experts']]);w=np.load(item['weights_source'])['weights'];ranks=np.load(item['ranks_source'])['ranks'];old=[dict(np.load(PARENT/name/f'scores_{model}_{seed}_{method0}.npz')) for method0 in EXPERTS]
   for start in range(0,len(y),4):
    stop=min(start+4,len(y));sc=reconstruct_mixture(q[:,start:stop],ranks);ref={**marginal_reference(q[:,start:stop],y[start:stop],w[start:stop]),**joint_reference(sc,y[start:stop],w[start:stop],float(f['scale']))};assert set(ref)==set(s)
    for k,value in ref.items():close(s[k][start:stop],value,3e-11)
    counts['new_forecast_cases']+=stop-start
   for metric in ['probability','prob3','prob6','prob12','group_probability','any6_prob24','any6_prob48','any6_prob72']:
    valid=np.isfinite(y) if metric=='probability' else np.isfinite(y[:,:int(metric.removeprefix('any6_prob'))]).all(1) if metric.startswith('any6_prob') else np.isfinite(s['group_label']) if metric=='group_probability' else np.isfinite(s['label'+metric.removeprefix('prob')]);weighted=mixture_probability_reference(np.stack([z[metric] for z in old]),w,valid);close(weighted,s[metric],3e-13);counts['mixture_probability_conservations']+=1
   counts['new_score_arrays']+=len(s)
  else:
   assert Path(item['score_source'])==PARENT/name/f'scores_{model}_{seed}_{method}.npz';parent_hash=read(PARENT/'final_manifest.json')[str(Path(item['score_source']))];assert sha(Path(item['score_source']))==parent_hash;counts['baseline_arrays_unchanged']+=len(s)
  cache[key]={k:s[k] for k in COMPARE};common=dict(fold=name,model=model,seed=seed,method=method)
  for metric in METRICS:
   z=s[metric];rows.append(dict(**common,metric=metric,value=float(np.nanmean(z)),n=int(np.isfinite(z).sum())))
   for month in sorted(set(ts.month)):
    v=z[ts.month==month];months.append(dict(**common,issue_month=int(month),metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
   for week in sorted(set(blocks)):
    v=z[blocks==week];weeks.append(dict(**common,week_block=int(week),metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
  for lo,hi in BINS:
   for metric in ['crps','brier_hourly','mae','coverage','width','interval_score','low_miss','high_miss']:
    v=s[metric][:,lo:hi];leads.append(dict(**common,lead_start=lo+1,lead_end=hi,metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
  for target,pk,yk in [('hourly','probability','label'),('single_6h','prob6','label6'),('group_6h','group_probability','group_label')]:
   ok=np.isfinite(s[yk]);p=s[pk][ok];labels=s[yk][ok];select=p>=.8;high.append(dict(**common,target=target,n=int(select.sum()),total=len(p),mean_probability=float(p[select].mean()) if select.any() else np.nan,observed_rate=float(labels[select].mean()) if select.any() else np.nan))
   for b in range(10):
    select=(p>=b/10)&(p<(b+1)/10 if b<9 else p<=1);rel.append(dict(**common,target=target,bin=b,n=int(select.sum()),mean_probability=float(p[select].mean()) if select.any() else np.nan,observed_rate=float(labels[select].mean()) if select.any() else np.nan))
  print('SCORES_AUDIT',name,model,seed,method,flush=True)
 df=pd.DataFrame(rows)
 for key,g in df.groupby(['model','seed','method','metric'],sort=False):rows.append(dict(fold='ALL',**dict(zip(['model','seed','method','metric'],key)),value=float(np.dot(g.value,g.n)/g.n.sum()),n=int(g.n.sum())))
 tablecounts={};keys=['fold','model','seed','method']
 for file,values,extra in [('metrics.csv',rows,['metric']),('month_metrics.csv',months,['metric','issue_month']),('weekly_metrics.csv',weeks,['metric','week_block']),('lead_metrics.csv',leads,['metric','lead_start']),('high_confidence.csv',high,['target']),('reliability.csv',rel,['target','bin'])]:tablecounts[file]=compare_table(file,values,keys+extra)
 counts['table_rows']=tablecounts;return cache,counts

def paired(cache,times):
 rng=np.random.default_rng(20261116);weights={};weeks={};sizes={}
 for name,start in FOLDS.items():
  ids=np.load(SRC/name/'data.npz')['test_ids'];w=((times[ids]-pd.Timestamp(start,tz=times.tz)).days//7).to_numpy();weeks[name]=w;k=int(w.max())+1;draw=rng.integers(0,k,(2000,k));weights[name]=np.stack([np.bincount(row,minlength=k)[w] for row in draw]);sizes[name]=len(ids)
 df=pd.read_csv(OUT/'paired_comparisons.csv');assert len(df)==480 and df.primary.sum()==7;g=MODELS[0];expected=[(g,'online_pool28',g,'equal_pool'),(g,'online_pool28',g,'rolling28'),(g,'online_pool28',MODELS[1],'online_pool28'),(g,'online_pool28',MODELS[2],'online_pool28'),(g,'equal_pool',g,'rolling28')]+[(g,m,g,b) for m in ['online_pool28','equal_pool'] for b in ['raw','frozen','rolling56']]+[(MODELS[1],'online_pool28',MODELS[1],'rolling28')];assert set(map(tuple,df[['candidate','method','baseline','baseline_method']].drop_duplicates().values))==set(expected)
 for r in df.itertuples():
  selected=list(FOLDS) if r.scope=='ALL' else [r.scope];num=np.zeros(2000);den=np.zeros(2000);ss=0.;nn=0;blocks=[]
  for name in selected:
   a=np.stack([cache[(name,r.candidate,seed,r.method)][r.metric] for seed in SEEDS]);b=np.stack([cache[(name,r.baseline,seed,r.baseline_method)][r.metric] for seed in SEEDS]);np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b));diff=(a-b).mean(0);flat=diff.reshape(len(diff),-1);sums=np.nansum(flat,1);counts=np.isfinite(flat).sum(1);num+=weights[name]@sums;den+=weights[name]@counts;ss+=sums.sum();nn+=counts.sum();blocks.append(len(np.unique(weeks[name][counts>0])))
  valid=den>0;supported=min(blocks)>=4 and valid.sum()>=1900;lo,hi=np.quantile(num[valid]/den[valid],[.025,.975]) if supported else [np.nan,np.nan];close(r.difference,ss/nn);close(r.low95,lo);close(r.high95,hi);assert r.n==nn and r.origins==sum(sizes[n] for n in selected) and r.nonempty_blocks==sum(blocks) and r.minimum_blocks_in_fold==min(blocks) and r.valid_draws==valid.sum() and r.interval_supported==supported;main=r.scope=='ALL' and r.candidate==g and r.method=='online_pool28' and ((r.baseline==g and r.baseline_method=='equal_pool' and r.metric in ['crps','brier6']) or (r.baseline==g and r.baseline_method=='rolling28' and r.metric in ['crps','brier6','group_brier']) or (r.baseline==MODELS[1] and r.baseline_method=='online_pool28' and r.metric in ['brier6','group_brier']));assert r.primary==main
 return len(df)
def weight_phase():
 d=np.load(ART/'data.npz');times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');counts=audit_weights(times)
 files=[Path(__file__),Path(__file__).with_name('pooling_reference.py'),OUT/'identity.json',OUT/'inventory.json',OUT/'weight_audit.csv',OUT/'weights_complete.json',OUT/'independent_weight_optima.csv']+[p for p in OUT.rglob('weights_*') if p.is_file()]
 record=dict(status='PASS',counts=counts,hashes={str(p):sha(p) for p in files});(OUT/'weight_verification_checkpoint.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print('WEIGHT_PHASE_PASS',json.dumps(counts),flush=True);return counts
def run():
 guard();assert read(OUT/'evaluation_complete.json')['status']=='COMPLETE';previous=read(OUT/'previous_hashes.json');identity=read(OUT/'identity.json');parent=read(PARENT/'final_manifest.json')
 for file,h in previous.items():assert sha(ART/file)==h,file
 for mapping in [identity,parent]:
  for file,h in mapping.items():assert sha(Path(file))==h,file
 d=np.load(ART/'data.npz');times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai')
 if (OUT/'weight_verification_checkpoint.json').exists():
  checkpoint=read(OUT/'weight_verification_checkpoint.json');assert checkpoint['status']=='PASS'
  for file,h in checkpoint['hashes'].items():assert sha(Path(file))==h,file
  counts=dict(checkpoint['counts'])
 else:counts=weight_phase()
 cache,sc=scores_and_tables(times);counts.update(sc);phase=read(OUT/'weights_complete.json');audit=pd.read_csv(OUT/'weight_audit.csv');assert phase['new_forecast_origins']==counts['new_forecast_cases']==15840;assert phase['all_forecast_origins']==47520;assert phase['optimizer_calls']==audit.optimizer_called.sum() and phase['optimizer_failures']==audit.optimizer_failed.sum() and phase['support_fallbacks']==audit.insufficient_support.sum();counts.update(status='PASS',previous_unchanged=len(previous),frozen_identities=len(identity),parent_manifest_files=len(parent),paired_comparisons=paired(cache,times));(OUT/'verification.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(json.dumps(counts,indent=2),flush=True)
if __name__=='__main__':
 import sys
 weight_phase() if sys.argv[1:]==['weights'] else run()

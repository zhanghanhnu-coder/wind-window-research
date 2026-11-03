"""Independent dynamic-rank scores, complete bins and all fixed contrasts."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
from rolling_copula_audit import audit_templates
from verify_seasonal_probability import reference_scores
from verify_partial_copula import windows_reference,longest_direct
from research_crps_distribution import guard,SEEDS,METRICS as JOINT
OUT=ART/'rolling_copula_v1';SRC=ART/'seasonal_probability_v1';COP=ART/'partial_copula_v1';ROLLING=ART/'seasonal_rolling_v1';FOLDS={'M05':'2024-05-01','M07':'2024-07-01','M09':'2024-09-01','M11':'2024-11-01'};CASES={'graph_raw':('gated_graph_atoms','raw',['frozen','rolling28']),'graph_rolling28':('gated_graph_atoms','rolling28',['frozen','rolling28','rolling56','blend28','time_broken','space_broken']),'own_rolling28':('gated_temporal_atoms','rolling28',['frozen','rolling28'])};BINS=[(0,6),(6,24),(24,48),(48,72)];METRICS=JOINT+['brier_hourly','mae'];COMPARE=['brier6','length_crps72','first_cdf_brier','group_brier','temporal_variogram','spatial_variogram']
PAIRS=[('graph_rolling28','rolling28','graph_rolling28','frozen'),('graph_raw','rolling28','graph_raw','frozen'),('graph_rolling28','rolling28','own_rolling28','rolling28'),('graph_rolling28','rolling28','graph_rolling28','time_broken'),('graph_rolling28','rolling28','graph_rolling28','space_broken'),('graph_rolling28','rolling56','graph_rolling28','frozen'),('graph_rolling28','blend28','graph_rolling28','frozen'),('graph_rolling28','rolling28','graph_raw','rolling28')]
def primary(scope,pair,metric):return scope=='ALL' and ((pair==PAIRS[0] and metric in ['brier6','length_crps72','first_cdf_brier','group_brier']) or (pair in PAIRS[1:3] and metric in ['brier6','group_brier']) or (pair==PAIRS[3] and metric=='brier6') or (pair==PAIRS[4] and metric=='group_brier'))
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=3e-11):np.testing.assert_allclose(a,b,atol=tol,rtol=0,equal_nan=True)
def table(name,rows,keys):
 a=pd.read_csv(OUT/name).sort_values(keys).reset_index(drop=True);b=pd.DataFrame(rows).sort_values(keys).reset_index(drop=True);assert len(a)==len(b)
 for k in b:
  if b[k].dtype.kind in 'biufc':close(a[k],b[k])
  else:assert a[k].fillna('').tolist()==b[k].fillna('').tolist(),(name,k)
 return len(a)
def reconstruct(q,ranks):return np.stack([v[np.arange(72)[None,:,None],np.arange(17)[None,None,:],r//3] for v,r in zip(q,ranks)]).astype(float)

def scores(times):
 inv=read(OUT/'inventory.json');expected={(f,c,s,m) for f in FOLDS for c,(_,_,ms) in CASES.items() for s in SEEDS for m in ms};assert len(inv)==120 and {(r['fold'],r['model'],r['seed'],r['method']) for r in inv}==expected;rows=[];months=[];weeks=[];leads=[];high=[];rel=[];bins=[];equiv=[];cache={};counts=dict(forecast_cases=0,new_score_arrays=0,baseline_arrays_unchanged=0,independent_run_length_cases=0,first_week_arrays=0,marginal_arrays=0)
 for item in inv:
  guard();name=item['fold'];case=item['model'];seed=item['seed'];method=item['method'];net,margin,_=CASES[case];assert item['network']==net and item['marginal_method']==margin and item['new']==(method!='frozen');assert sha(Path(item['source']))==item['source_sha256'];baseline=ROLLING/name/f'scores_{net}_{seed}_{margin}.npz';assert item['baseline_score_source']==str(baseline) and sha(baseline)==item['baseline_score_sha256'];old=dict(np.load(baseline));q=np.load(item['source'])[item['source_key']];y=np.load(SRC/name/'test_truth.npz')['y'];f=np.load(SRC/name/'data.npz');ts=times[f['test_ids']];blocks=((ts-pd.Timestamp(FOLDS[name],tz=times.tz)).days//7).to_numpy();s=dict(np.load(item['score_source']));assert len(s)==35 and len(y)==item['origins'];schedule=read(ROLLING/name/'schedule.json');ranks=np.empty((len(y),57,72,17),np.int16)
  for r,entry in zip(item['ranks'],schedule):
   assert r['block']==entry['block'] and r['issue_id']==entry['issue_id'] and r['positions']==entry['positions'];rp=OUT/name/f'ranks_{net}_{seed}_{method}_w{r["block"]}.npz';assert r['source']==str(rp) and sha(rp)==r['sha256'];ranks[r['positions']]=np.load(rp)['ranks']
  assert len(item['ranks'])==len(schedule)==9
  if not item['new']:
   assert item['score_source']==str(baseline);counts['baseline_arrays_unchanged']+=len(s)
  else:
   marginal=reference_scores(q,y)
   for k,v in marginal.items():close(s[k],v,3e-12)
   for start in range(0,len(q),8):
    stop=min(start+8,len(q));sc=reconstruct(q[start:stop],ranks[start:stop]);np.testing.assert_array_equal(np.sort(sc,axis=1),np.repeat(q[start:stop],3,axis=-1).transpose(0,3,1,2));ref=windows_reference(sc,y[start:stop],float(f['scale']))
    for k,v in ref.items():close(s[k][start:stop],v,3e-12)
   counts['new_score_arrays']+=len(set(marginal)|set(ref))
   for pos in [0,len(q)//2,len(q)-1]:
    sc=reconstruct(q[pos:pos+1],ranks[pos:pos+1])[0]
    for node in [0,8,16]:
     obs=y[pos,:,node];length=np.array([longest_direct(v<=10) for v in sc[:,:,node]]);actual=longest_direct(obs<=10);score=np.mean(abs(length-actual))-.5*np.mean(abs(length[:,None]-length[None,:]));close(s['length_crps72'][pos,node],score if np.isfinite(obs).all() else np.nan,3e-12);counts['independent_run_length_cases']+=1
   maximum=0.
   for key in ['crps','coverage','width','interval_score','low_miss','high_miss','probability','label','brier_hourly','mae']:close(s[key],old[key],3e-12);maximum=max(maximum,float(np.nanmax(abs(s[key]-old[key]))));counts['marginal_arrays']+=1
   if method=='space_broken':
    full=np.load(OUT/name/f'scores_{case}_{seed}_rolling28.npz')
    for key in ['brier3','brier6','brier12','length_crps24','length_crps48','length_crps72','any6_brier24','any6_brier48','any6_brier72','first_cdf_brier']:close(s[key],full[key],3e-12)
   if method in ['rolling28','rolling56','blend28']:
    for key in s:np.testing.assert_array_equal(s[key][schedule[0]['positions']],old[key][schedule[0]['positions']]);counts['first_week_arrays']+=1
   equiv.append(dict(fold=name,model=case,seed=seed,method=method,max_marginal_difference=maximum,space_single_node_conserved=method=='space_broken',first_week_conserved=method in ['rolling28','rolling56','blend28']))
  counts['forecast_cases']+=len(y);cache[name,case,seed,method]={k:s[k] for k in COMPARE};common=dict(fold=name,model=case,seed=seed,method=method)
  for metric in METRICS:
   z=s[metric];rows.append(dict(**common,metric=metric,value=float(np.nanmean(z)),n=int(np.isfinite(z).sum())))
   for month in sorted(set(ts.month)):
    v=z[ts.month==month];months.append(dict(**common,issue_month=int(month),metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
   for block in sorted(set(blocks)):
    v=z[blocks==block];weeks.append(dict(**common,week_block=int(block),metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
  for lo,hi in BINS:
   for metric in ['crps','brier_hourly','mae','coverage','width','interval_score','low_miss','high_miss']:
    v=s[metric][:,lo:hi];leads.append(dict(**common,lead_start=lo+1,lead_end=hi,metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
  for target,pk,yk in [('hourly','probability','label'),('single_6h','prob6','label6'),('group_6h','group_probability','group_label')]:
   valid=np.isfinite(s[yk]);p=s[pk][valid];label=s[yk][valid];assert np.isfinite(p).all() and (p>=-1e-12).all() and (p<=1+1e-12).all() and np.isin(label,[0,1]).all();select=p>=.8;high.append(dict(**common,target=target,n=int(select.sum()),total=len(p),mean_probability=float(p[select].mean()) if select.any() else np.nan,observed_rate=float(label[select].mean()) if select.any() else np.nan));index=np.searchsorted(np.arange(1,10)/10,p,side='right');n=np.bincount(index,minlength=10);ps=np.bincount(index,weights=p,minlength=10);ys=np.bincount(index,weights=label,minlength=10)
   for b in range(10):rel.append(dict(**common,target=target,bin=b,n=int(n[b]),mean_probability=ps[b]/n[b] if n[b] else np.nan,observed_rate=ys[b]/n[b] if n[b] else np.nan))
   outside=(p<0)|(p>1);oldn=len(p)-int(outside.sum());bins.append(dict(**common,target=target,valid_targets=len(p),new_binned=int(n.sum()),old_binned=oldn,old_unbinned=int(outside.sum()),roundoff_outside_unit_interval=int(outside.sum()),maximum_unit_interval_excess=float(max(0,p.max()-1,-p.min())) if len(p) else 0.))
  print('SCORES_AUDIT',name,case,seed,method,flush=True)
 df=pd.DataFrame(rows)
 for key,g in df.groupby(['model','seed','method','metric'],sort=False):rows.append(dict(fold='ALL',**dict(zip(['model','seed','method','metric'],key)),value=float(np.dot(g.value,g.n)/g.n.sum()),n=int(g.n.sum())))
 keys=['fold','model','seed','method'];counts['table_rows']={}
 for file,values,extra in [('metrics.csv',rows,['metric']),('month_metrics.csv',months,['metric','issue_month']),('weekly_metrics.csv',weeks,['metric','week_block']),('lead_metrics.csv',leads,['metric','lead_start']),('high_confidence.csv',high,['target']),('reliability.csv',rel,['target','bin']),('reliability_bin_coverage.csv',bins,['target']),('marginal_conservation.csv',equiv,[])]:counts['table_rows'][file]=table(file,values,keys+extra)
 return cache,counts

def paired(cache,times):
 rng=np.random.default_rng(20261123);weights={};weeks={};sizes={}
 for name,start in FOLDS.items():
  ids=np.load(SRC/name/'data.npz')['test_ids'];w=((times[ids]-pd.Timestamp(start,tz=times.tz)).days//7).to_numpy();weeks[name]=w;k=int(w.max())+1;draw=rng.integers(0,k,(2000,k));weights[name]=np.stack([np.bincount(row,minlength=k)[w] for row in draw]);sizes[name]=len(ids)
 df=pd.read_csv(OUT/'paired_comparisons.csv');assert len(df)==240 and df.primary.sum()==10;assert set(map(tuple,df[['candidate','method','baseline','baseline_method']].drop_duplicates().values))==set(PAIRS)
 for r in df.itertuples():
  selected=list(FOLDS) if r.scope=='ALL' else [r.scope];num=np.zeros(2000);den=np.zeros(2000);ss=0.;nn=0;blocks=[]
  for name in selected:
   a=np.stack([cache[name,r.candidate,seed,r.method][r.metric] for seed in SEEDS]);b=np.stack([cache[name,r.baseline,seed,r.baseline_method][r.metric] for seed in SEEDS]);np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b));diff=(a-b).mean(0).reshape(len(a[0]),-1);sums=np.nansum(diff,1);counts=np.isfinite(diff).sum(1);num+=weights[name]@sums;den+=weights[name]@counts;ss+=sums.sum();nn+=counts.sum();blocks.append(len(np.unique(weeks[name][counts>0])))
  valid=den>0;supported=min(blocks)>=4 and valid.sum()>=1900;lo,hi=np.quantile(num[valid]/den[valid],[.025,.975]) if supported else [np.nan,np.nan];close(r.difference,ss/nn);close(r.low95,lo);close(r.high95,hi);assert r.n==nn and r.origins==sum(sizes[f] for f in selected) and r.nonempty_blocks==sum(blocks) and r.minimum_blocks_in_fold==min(blocks) and r.valid_draws==valid.sum() and r.interval_supported==supported;assert r.primary==primary(r.scope,(r.candidate,r.method,r.baseline,r.baseline_method),r.metric)
 return len(df)

def run():
 guard();assert read(OUT/'evaluation_complete.json')['status']=='COMPLETE';parents={}
 for root in [SRC,COP,ROLLING]:
  manifest=read(root/'final_manifest.json')
  for file,h in manifest.items():assert sha(Path(file))==h,file
  parents[root.name]=len(manifest)
 for file,h in read(OUT/'identity.json').items():assert sha(Path(file))==h,file
 for file,h in read(OUT/'previous_hashes.json').items():assert sha(ART/file)==h,file
 for check in ['source_adapter_identity.json','source_adapter_checkpoint.json']:
  for file,h in read(OUT/check).items():assert sha(Path(file))==h,file
 assert read(OUT/'source_adapter_verification.json')['status']=='PASS';counts=audit_templates();times=pd.DatetimeIndex(pd.to_datetime(np.load(ART/'data.npz')['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');cache,sc=scores(times);counts.update(sc);counts['paired_comparisons']=paired(cache,times);phase=read(OUT/'prediction_complete.json');assert counts['forecast_cases']==26400 and counts['new_score_arrays']==2940 and counts['baseline_arrays_unchanged']==1260;assert phase['support_fallbacks']==counts['support_fallbacks'];counts.update(status='PASS',parent_manifest_files=parents,previous_unchanged=len(read(OUT/'previous_hashes.json')),frozen_identities=len(read(OUT/'identity.json')),source_adapter_checkpoint_files=len(read(OUT/'source_adapter_checkpoint.json')));(OUT/'verification.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(json.dumps(counts,indent=2),flush=True)

if __name__=='__main__':run()

"""Full round32 audit: prediction provenance, all scores and prespecified contrasts."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
from multiscale_forecast_audit import audit_forecasts
from verify_seasonal_probability import reference_scores
from verify_partial_copula import windows_reference,reconstruct,longest_direct
from research_crps_distribution import guard,SEEDS,METRICS as JOINT
OUT=ART/'multiscale_context_v1';SRC=ART/'seasonal_probability_v1';COP=ART/'partial_copula_v1';ROLLING=ART/'seasonal_rolling_v1'
FOLDS={'M05':'2024-05-01','M07':'2024-07-01','M09':'2024-09-01','M11':'2024-11-01'};OLD={'context1_graph':'gated_graph_atoms','context1_own':'gated_temporal_atoms','context1_mean':'gated_mean_atoms'};FAMILIES=list(OLD)+['context3_graph','context7_graph','context7_own','context7_mean','climate28','analog112'];BINS=[(0,6),(6,24),(24,48),(48,72)];METRICS=JOINT+['brier_hourly','mae','crps_49_72'];COMPARE=['crps','crps_49_72','brier_hourly','brier6','group_brier','length_crps72','first_cdf_brier','spatial_variogram','temporal_variogram']
MAIN=[('context3_graph','raw','context1_graph','raw'),('context7_graph','raw','context1_graph','raw'),('context7_graph','raw','context7_own','raw'),('context7_graph','rolling28','context1_graph','rolling28'),('context7_graph','rolling28','climate28','raw'),('context7_graph','rolling28','analog112','raw')]
PAIRS=MAIN+[('context7_graph','raw','context7_mean','raw'),('context7_graph','rolling28','context7_own','rolling28'),('context1_graph','rolling28','climate28','raw'),('context1_graph','rolling28','analog112','raw'),('context7_graph','frozen','context1_graph','frozen'),('context7_graph','rolling28','context7_graph','raw'),('context7_graph','rolling28','context7_graph','frozen'),('climate28','raw','analog112','raw')]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=3e-11):np.testing.assert_allclose(a,b,atol=tol,rtol=0,equal_nan=True)
def value(s,m):return s['crps'][:,48:72] if m=='crps_49_72' else s[m]
def compare_table(name,rows,keys):
 a=pd.read_csv(OUT/name).sort_values(keys).reset_index(drop=True);b=pd.DataFrame(rows).sort_values(keys).reset_index(drop=True);assert len(a)==len(b),(name,len(a),len(b))
 for k in b:
  if b[k].dtype.kind in 'biufc':close(a[k],b[k])
  else:assert a[k].fillna('').tolist()==b[k].fillna('').tolist(),(name,k)
 return len(a)
def scores_and_tables(times):
 inventory=read(OUT/'inventory.json');expected={(f,m,s,method) for f in FOLDS for m in FAMILIES for s in SEEDS for method in ['raw','frozen','rolling28']};assert {(r['fold'],r['model'],r['seed'],r['method']) for r in inventory}==expected and len(inventory)==324;cache={};rows=[];months=[];weeks=[];leads=[];high=[];rel=[];equiv=[];counts=dict(forecast_cases=0,score_arrays=0,independent_run_length_cases=0)
 for item in inventory:
  guard();name=item['fold'];model=item['model'];seed=item['seed'];method=item['method'];assert sha(Path(item['source']))==item['source_sha256'] and sha(Path(item['ranks_source']))==item['ranks_sha256'];q=np.load(item['source'])[item['source_key']];ranks=np.load(item['ranks_source'])['ranks'];np.testing.assert_array_equal(ranks,np.load(COP/name/f'ranks_gated_graph_atoms_{seed}_partial05.npz')['ranks']);y=np.load(SRC/name/'test_truth.npz')['y'];s=dict(np.load(item['score_source']));f=np.load(SRC/name/'data.npz');ts=times[f['test_ids']];blocks=((ts-pd.Timestamp(FOLDS[name],tz=times.tz)).days//7).to_numpy();assert len(q)==item['origins'] and len(s)==35;marginal=reference_scores(q,y)
  for k,v in marginal.items():close(s[k],v,3e-12)
  for start in range(0,len(q),8):
   stop=min(start+8,len(q));sc=reconstruct(q[start:stop].astype(float),ranks);np.testing.assert_array_equal(np.sort(sc,axis=1),np.repeat(q[start:stop],3,axis=-1).transpose(0,3,1,2));ref=windows_reference(sc,y[start:stop],float(f['scale']))
   for k,v in ref.items():close(s[k][start:stop],v,3e-12)
   counts['forecast_cases']+=stop-start
  counts['score_arrays']+=len(set(marginal)|set(ref))
  for pos in [0,len(q)//2,len(q)-1]:
   sc=reconstruct(q[pos:pos+1],ranks)[0]
   for node in [0,8,16]:
    obs=y[pos,:,node];lengths=np.array([longest_direct(v<=10) for v in sc[:,:,node]]);actual=longest_direct(obs<=10);score=np.mean(abs(lengths-actual))-.5*np.mean(abs(lengths[:,None]-lengths[None,:]));close(s['length_crps72'][pos,node],score if np.isfinite(obs).all() else np.nan,3e-12);counts['independent_run_length_cases']+=1
  if model in OLD:
   previous=np.load(ROLLING/name/f'scores_{OLD[model]}_{seed}_{method}.npz');keys=s.keys() if model=='context1_graph' else ['crps','probability','label','brier_hourly','mae','coverage','width','low_miss','high_miss','interval_score']
   maximum=0.
   for k in keys:
    if k=='crps' and model!='context1_graph':close(s[k],previous[k],3e-12);maximum=float(np.nanmax(abs(s[k]-previous[k])))
    else:np.testing.assert_array_equal(s[k],previous[k])
   equiv.append(dict(fold=name,model=model,seed=seed,method=method,arrays=len(keys),common_graph_template=True,maximum_crps_roundoff=maximum))
  if model in ['climate28','analog112']:
   assert item['producer_seed']==0
   for otherseed in SEEDS:
    other=next(r for r in inventory if (r['fold'],r['model'],r['seed'],r['method'])==(name,model,otherseed,method));assert other['source']==item['source'] and other['source_sha256']==item['source_sha256']
  cache[name,model,seed,method]={k:value(s,k) for k in COMPARE};common=dict(fold=name,model=model,seed=seed,method=method)
  for metric in METRICS:
   z=value(s,metric);rows.append(dict(**common,metric=metric,value=float(np.nanmean(z)),n=int(np.isfinite(z).sum())))
   for month in sorted(set(ts.month)):
    v=z[ts.month==month];months.append(dict(**common,issue_month=int(month),metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
   for week in sorted(set(blocks)):
    v=z[blocks==week];weeks.append(dict(**common,week_block=int(week),metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
  for lo,hi in BINS:
   for metric in ['crps','brier_hourly','mae','coverage','width','interval_score','low_miss','high_miss']:
    v=s[metric][:,lo:hi];leads.append(dict(**common,lead_start=lo+1,lead_end=hi,metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
  for target,pk,yk in [('hourly','probability','label'),('single_6h','prob6','label6'),('group_6h','group_probability','group_label')]:
   valid=np.isfinite(s[yk]);p=s[pk][valid];label=s[yk][valid];select=p>=.8;high.append(dict(**common,target=target,n=int(select.sum()),total=len(p),mean_probability=float(p[select].mean()) if select.any() else np.nan,observed_rate=float(label[select].mean()) if select.any() else np.nan))
   for b in range(10):
    select=(p>=b/10)&(p<(b+1)/10 if b<9 else p<=1);rel.append(dict(**common,target=target,bin=b,n=int(select.sum()),mean_probability=float(p[select].mean()) if select.any() else np.nan,observed_rate=float(label[select].mean()) if select.any() else np.nan))
  print('SCORES_AUDIT',name,model,seed,method,flush=True)
 df=pd.DataFrame(rows)
 for key,g in df.groupby(['model','seed','method','metric'],sort=False):rows.append(dict(fold='ALL',**dict(zip(['model','seed','method','metric'],key)),value=float(np.dot(g.value,g.n)/g.n.sum()),n=int(g.n.sum())))
 keys=['fold','model','seed','method'];counts['table_rows']={}
 for file,values,extra in [('metrics.csv',rows,['metric']),('month_metrics.csv',months,['metric','issue_month']),('weekly_metrics.csv',weeks,['metric','week_block']),('lead_metrics.csv',leads,['metric','lead_start']),('high_confidence.csv',high,['target']),('reliability.csv',rel,['target','bin']),('old_marginal_equivalence.csv',equiv,[])]:counts['table_rows'][file]=compare_table(file,values,keys+extra)
 return cache,counts
def paired(cache,times):
 rng=np.random.default_rng(20261119);weights={};weeks={};sizes={}
 for name,start in FOLDS.items():
  ids=np.load(SRC/name/'data.npz')['test_ids'];w=((times[ids]-pd.Timestamp(start,tz=times.tz)).days//7).to_numpy();weeks[name]=w;k=int(w.max())+1;draw=rng.integers(0,k,(2000,k));weights[name]=np.stack([np.bincount(row,minlength=k)[w] for row in draw]);sizes[name]=len(ids)
 df=pd.read_csv(OUT/'paired_comparisons.csv');assert len(df)==630 and df.primary.sum()==12;assert set(map(tuple,df[['candidate','method','baseline','baseline_method']].drop_duplicates().values))==set(PAIRS)
 for r in df.itertuples():
  selected=list(FOLDS) if r.scope=='ALL' else [r.scope];num=np.zeros(2000);den=np.zeros(2000);ss=0.;nn=0;blocks=[]
  for name in selected:
   a=np.stack([cache[name,r.candidate,seed,r.method][r.metric] for seed in SEEDS]);b=np.stack([cache[name,r.baseline,seed,r.baseline_method][r.metric] for seed in SEEDS]);np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b));diff=(a-b).mean(0).reshape(len(a[0]),-1);sums=np.nansum(diff,1);counts=np.isfinite(diff).sum(1);num+=weights[name]@sums;den+=weights[name]@counts;ss+=sums.sum();nn+=counts.sum();blocks.append(len(np.unique(weeks[name][counts>0])))
  valid=den>0;supported=min(blocks)>=4 and valid.sum()>=1900;lo,hi=np.quantile(num[valid]/den[valid],[.025,.975]) if supported else [np.nan,np.nan];close(r.difference,ss/nn);close(r.low95,lo);close(r.high95,hi);assert r.n==nn and r.origins==sum(sizes[f] for f in selected) and r.nonempty_blocks==sum(blocks) and r.minimum_blocks_in_fold==min(blocks) and r.valid_draws==valid.sum() and r.interval_supported==supported;assert r.primary==(r.scope=='ALL' and (r.candidate,r.method,r.baseline,r.baseline_method) in MAIN and r.metric in ['crps_49_72','brier6'])
 return len(df)
def run():
 guard();assert read(OUT/'evaluation_complete.json')['status']=='COMPLETE';parents={}
 for root in [SRC,COP,ROLLING]:
  manifest=read(root/'final_manifest.json')
  for file,h in manifest.items():assert sha(Path(file))==h,file
  parents[root.name]=len(manifest)
 for file,h in read(OUT/'previous_hashes.json').items():assert sha(ART/file)==h,file
 for identity in ['identity.json','evaluation_identity.json']:
  for file,h in read(OUT/identity).items():assert sha(Path(file))==h,file
 for check in ['scoring_resume_identity.json','scoring_resume_checkpoint.json']:
  for file,h in read(OUT/check).items():assert sha(Path(file))==h,file
 assert read(OUT/'scoring_resume_verification.json')['status']=='PASS'
 counts=audit_forecasts();data=np.load(ART/'data.npz');times=pd.DatetimeIndex(pd.to_datetime(data['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');cache,sc=scores_and_tables(times);counts.update(sc);counts['paired_comparisons']=paired(cache,times);assert counts['forecast_cases']==71280;phase=read(OUT/'prediction_complete.json');assert phase['flows']==324 and phase['unique_producers']==92 and phase['new_producers']==56;counts.update(status='PASS',parent_manifest_files=parents,previous_unchanged=len(read(OUT/'previous_hashes.json')),prediction_identities=len(read(OUT/'identity.json')),evaluation_identities=len(read(OUT/'evaluation_identity.json')),scoring_resume_checkpoint_files=len(read(OUT/'scoring_resume_checkpoint.json')));(OUT/'verification.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(json.dumps(counts,indent=2),flush=True)
if __name__=='__main__':run()

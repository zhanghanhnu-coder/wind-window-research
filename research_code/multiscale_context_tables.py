"""Descriptive and prespecified block comparisons; no tuning on these scores."""
import numpy as np
import pandas as pd
from evaluate_multiscale_context import OUT,SRC,ROLLING,ART,FOLDS,SEEDS,METRICS,COMPARE,PAIRS,BINS,read,csv,primary,score_array
def times_load():return pd.DatetimeIndex(pd.to_datetime(np.load(ART/'data.npz')['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai')
def tables():
 times=times_load();rows=[];months=[];weeks=[];leads=[];high=[];rel=[]
 for item in read(OUT/'inventory.json'):
  name=item['fold'];f=np.load(SRC/name/'data.npz');ts=times[f['test_ids']];blocks=np.load(ROLLING/name/'archive_data.npz')['blocks'];s=dict(np.load(item['score_source']));common={k:item[k] for k in ['fold','model','seed','method']}
  for metric in METRICS:
   z=score_array(s,metric);rows.append(dict(**common,metric=metric,value=float(np.nanmean(z)),n=int(np.isfinite(z).sum())))
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
 df=pd.DataFrame(rows)
 for key,g in df.groupby(['model','seed','method','metric'],sort=False):rows.append(dict(fold='ALL',**dict(zip(['model','seed','method','metric'],key)),value=float(np.sum(g.value*g.n)/g.n.sum()),n=int(g.n.sum())))
 for name,values in [('metrics.csv',rows),('month_metrics.csv',months),('weekly_metrics.csv',weeks),('lead_metrics.csv',leads),('high_confidence.csv',high),('reliability.csv',rel)]:csv(name,values)
def compare():
 times=times_load();rng=np.random.default_rng(20261119);weeks={};draws={};sizes={};lookup={(r['fold'],r['model'],r['seed'],r['method']):r['score_source'] for r in read(OUT/'inventory.json')};cache={}
 for name,cfg in FOLDS.items():
  ids=np.load(SRC/name/'data.npz')['test_ids'];weeks[name]=((times[ids]-pd.Timestamp(cfg['test_start'],tz=times.tz)).days//7).to_numpy();k=int(weeks[name].max())+1;draws[name]=rng.integers(0,k,(2000,k));sizes[name]=len(ids)
 for key,path in lookup.items():
  z=np.load(path);cache[key]={m:score_array(z,m) for m in COMPARE}
 rows=[]
 for model,method,base,bmethod in PAIRS:
  for metric in COMPARE:
   sums={};counts={}
   for name in FOLDS:
    diff=[]
    for seed in SEEDS:
     a=cache[name,model,seed,method][metric];b=cache[name,base,seed,bmethod][metric];np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b));diff.append(a-b)
    z=np.mean(diff,0);sums[name]=np.array([np.nansum(z[weeks[name]==j]) for j in range(draws[name].shape[1])]);counts[name]=np.array([np.isfinite(z[weeks[name]==j]).sum() for j in range(draws[name].shape[1])])
   for scope in ['ALL']+list(FOLDS):
    chosen=list(FOLDS) if scope=='ALL' else [scope];num=np.zeros(2000);den=np.zeros(2000);total=0.;n=0;blocks=[]
    for name in chosen:
     num+=sums[name][draws[name]].sum(1);den+=counts[name][draws[name]].sum(1);total+=sums[name].sum();n+=counts[name].sum();blocks.append(int((counts[name]>0).sum()))
    valid=den>0;supported=min(blocks)>=4 and valid.sum()>=1900;lo,hi=np.quantile(num[valid]/den[valid],[.025,.975]) if supported else [np.nan,np.nan];rows.append(dict(scope=scope,candidate=model,method=method,baseline=base,baseline_method=bmethod,metric=metric,primary=primary(scope,model,method,base,bmethod,metric),difference=float(total/n),low95=float(lo),high95=float(hi),n=int(n),origins=sum(sizes[x] for x in chosen),nonempty_blocks=sum(blocks),minimum_blocks_in_fold=min(blocks),valid_draws=int(valid.sum()),interval_supported=bool(supported)))
 assert len(rows)==630 and sum(r['primary'] for r in rows)==12;csv('paired_comparisons.csv',rows)

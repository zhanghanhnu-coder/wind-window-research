"""Round44 tables adapted from frozen31 shared-event bounds; new evidence only."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from model import ART
from seasonal_probability_core import FOLDS
from research_crps_distribution import guard,SEEDS
from missing_window_bounds_core import coefficients,sharp_bounds,bootstrap_bounds
from reliability_boundaries_core import reliability_rows
OUT=ART/'mature_window_calibration_v2';PARENT=ART/'missing_window_bounds_v1';CASES=['base','calibrated'];PAIRS=[('calibrated','base')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def parents():
 for file,h in read(PARENT/'final_manifest.json').items():assert sha(file)==h,file
def analyze():
 rng=np.random.default_rng(20261129);pops={};draws={};probs={};high=[]
 for name in FOLDS:
  path=OUT/name;pops[name]=dict(np.load(path/'population.npz'));k=int(pops[name]['blocks'].max())+1;draws[name]=rng.integers(0,k,(2000,k));np.savez_compressed(path/'bootstrap_draws.npz',draws=draws[name])
  for case in CASES:
   packs=[np.load(path/f'probabilities_{case}_{seed}.npz') for seed in SEEDS]
   for kind in ['single','group']:
    a=np.stack([p[kind] for p in packs]);probs[name,case,kind]=a;label=np.broadcast_to(pops[name][kind+'_labels'],a.shape);select=a>=.8;n=int(select.sum());success=int((select&(label==1)).sum());unknown=int((select&np.isnan(label)).sum());high.append(dict(fold=name,case=case,metric=kind,n=n,total=a.size,known_successes=success,known_failures=int((select&(label==0)).sum()),unknown=unknown,probability_sum=float(a[select].sum()),mean_probability=float(a[select].mean()) if n else np.nan,observed_lower=success/n if n else np.nan,observed_upper=(success+unknown)/n if n else np.nan))
 for case in CASES:
  for kind in ['single','group']:
   r=[r for r in high if r['case']==case and r['metric']==kind and r['fold']!='ALL'];n=sum(x['n'] for x in r);success=sum(x['known_successes'] for x in r);unknown=sum(x['unknown'] for x in r);ps=sum(x['probability_sum'] for x in r);high.append(dict(fold='ALL',case=case,metric=kind,n=n,total=sum(x['total'] for x in r),known_successes=success,known_failures=sum(x['known_failures'] for x in r),unknown=unknown,probability_sum=ps,mean_probability=ps/n if n else np.nan,observed_lower=success/n if n else np.nan,observed_upper=(success+unknown)/n if n else np.nan))
 csv('high_confidence_bounds.csv',high);absolute=[];paired=[]
 for candidate,baseline in [(c,None) for c in CASES]+PAIRS:
  for kind in ['single','group']:
   values={};samples={};complete_sums={};complete_counts={}
   for name in FOLDS:
    guard();pop=pops[name];pa=probs[name,candidate,kind];pb=probs[name,baseline,kind] if baseline else None;c,d=coefficients(pa,pb);label=pop[kind+'_labels'];inv=pop[kind+'_inverse'];value=sharp_bounds(c,d,label,inv);sample=bootstrap_bounds(c,d,label,inv,pop['blocks'],draws[name]);values[name]=value;samples[name]=sample;complete=pop[kind+'_complete'];complete_sums[name]=float((c+d*np.where(np.isfinite(label),label,0))[complete].sum());complete_counts[name]=int(complete.sum());prefix=f'{candidate}__{baseline or "absolute"}__{kind}';np.savez_compressed(OUT/name/f'bounds_{prefix}.npz',c=c,d=d,witness_lower=value['witness_lower'],witness_upper=value['witness_upper'],bootstrap_lower=sample['lower'],bootstrap_upper=sample['upper'],bootstrap_n=sample['n'])
    for side in ['lower','upper']:np.testing.assert_allclose(np.mean(c+d*value['witness_'+side][inv]),value[side],rtol=0,atol=3e-13)
    assert value['pointwise_lower']<=value['lower']+1e-13 and value['upper']<=value['pointwise_upper']+1e-13
   for scope in ['ALL']+list(FOLDS):
    selected=list(FOLDS) if scope=='ALL' else [scope];n=sum(values[f]['n'] for f in selected);lower=sum(values[f]['lower']*values[f]['n'] for f in selected)/n;upper=sum(values[f]['upper']*values[f]['n'] for f in selected)/n;den=sum(samples[f]['n'] for f in selected);lb=sum(samples[f]['lower']*samples[f]['n'] for f in selected)/den;ub=sum(samples[f]['upper']*samples[f]['n'] for f in selected)/den;valid=np.isfinite(lb)&np.isfinite(ub);blocks=[len(np.unique(pops[f]['blocks'])) for f in selected];supported=min(blocks)>=4 and valid.sum()>=1900;lq=np.quantile(lb[valid],[.025,.975]) if supported else [np.nan,np.nan];uq=np.quantile(ub[valid],[.025,.975]) if supported else [np.nan,np.nan];cn=sum(complete_counts[f] for f in selected);row=dict(scope=scope,candidate=candidate,baseline=baseline or 'absolute',metric=kind,primary=bool(baseline and scope=='ALL'),lower=lower,upper=upper,pointwise_lower=sum(values[f]['pointwise_lower']*values[f]['n'] for f in selected)/n,pointwise_upper=sum(values[f]['pointwise_upper']*values[f]['n'] for f in selected)/n,n=int(n),known_targets=int(sum(values[f]['known_targets'] for f in selected)),unknown_targets=int(sum(values[f]['unknown_targets'] for f in selected)),physical_events=sum(values[f]['physical_events'] for f in selected),unknown_physical_events=sum(values[f]['unknown_physical_events'] for f in selected),complete_case_value=sum(complete_sums[f] for f in selected)/cn if cn else np.nan,complete_case_n=cn,lower_percentile_low=float(lq[0]),lower_percentile_high=float(lq[1]),upper_percentile_low=float(uq[0]),upper_percentile_high=float(uq[1]),outer_low=float(lq[0]),outer_high=float(uq[1]),valid_draws=int(valid.sum()),minimum_blocks_in_fold=min(blocks),resampling_supported=supported)
    (paired if baseline else absolute).append(row)
   print('BOUNDS',candidate,baseline,kind,flush=True)
 assert len(absolute)==20 and len(paired)==10 and sum(r['primary'] for r in paired)==2;csv('absolute_bounds.csv',absolute);csv('paired_bounds.csv',paired);parents();js(OUT/'evaluation_complete.json',dict(status='COMPLETE',absolute_rows=20,paired_rows=10,primary_comparisons=2,utc=datetime.now(timezone.utc).isoformat()))

def reliability():
 rows=[]
 for scope in ['ALL',*FOLDS]:
  folds=list(FOLDS) if scope=='ALL' else [scope]
  for case in CASES:
   for kind in ['single','group']:
    probs=[];labels=[]
    for fold in folds:
     pop=np.load(OUT/fold/'population.npz');a=np.stack([np.load(OUT/fold/f'probabilities_{case}_{seed}.npz')[kind] for seed in SEEDS]);probs.append(a.ravel());labels.append(np.broadcast_to(pop[kind+'_labels'],a.shape).ravel())
    bins,audit=reliability_rows(np.concatenate(probs),np.concatenate(labels))
    for row in bins:rows.append(dict(scope=scope,case=case,metric=kind,**row))
 assert len(rows)==200;csv('reliability.csv',rows)

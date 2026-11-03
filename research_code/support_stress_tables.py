"""Fixed round39 summaries; missing-weather costs remain bounds."""
import numpy as np
import pandas as pd
from seasonal_probability_core import FOLDS
from support_stress_core import empirical_cvar

MAIN='stress_e010'
BASELINES=['base_tv025','base_tv000']


def make_tables(df,times):
 summaries=[]
 for scope in ['ALL']+list(FOLDS):
  selected=df if scope=='ALL' else df[df.fold==scope]
  for case,g in selected.groupby('case',sort=False):
   summaries.append(dict(scope=scope,case=case,n=len(g),unique_origins=g.issue_id.nunique(),complete=int(g.complete.sum()),complete_unique_origins=g[g.complete].issue_id.nunique(),**{k:float(g[k].mean()) for k in ['cost_lower','cost_upper','exact_cost','exact_completed','booking_cost','objective','nominal_total','no_booking']}))
 rng=np.random.default_rng(20261125);draws={};blocks={};weights={};paired=[]
 for fold,cfg in FOLDS.items():
  ids=np.sort(df[df.fold==fold].issue_id.unique());blocks[fold]=((times[ids]-pd.Timestamp(cfg['test_start'],tz=times.tz)).days//7).to_numpy();k=int(blocks[fold].max())+1
  draws[fold]=rng.integers(0,k,(2000,k));weights[fold]=np.stack([np.bincount(r,minlength=k)[blocks[fold]] for r in draws[fold]])
 for baseline in BASELINES:
  for metric in ['all_cost_bounds','exact_cost','exact_completed']:
   values={}
   for fold in FOLDS:
    a=df[(df.fold==fold)&(df.case==MAIN)].sort_values(['seed','issue_id']);b=df[(df.fold==fold)&(df.case==baseline)].sort_values(['seed','issue_id'])
    assert len(a)==len(b)==36;np.testing.assert_array_equal(a[['seed','issue_id']],b[['seed','issue_id']])
    if metric=='all_cost_bounds':lo=(a.cost_lower.to_numpy()-b.cost_upper.to_numpy()).reshape(3,12).mean(0);hi=(a.cost_upper.to_numpy()-b.cost_lower.to_numpy()).reshape(3,12).mean(0)
    else:lo=hi=(a[metric].to_numpy()-b[metric].to_numpy()).reshape(3,12).mean(0)
    values[fold]=lo,hi
   for scope in ['ALL']+list(FOLDS):
    fs=list(FOLDS) if scope=='ALL' else [scope];numlo=np.zeros(2000);numhi=np.zeros(2000);den=np.zeros(2000);sumlo=sumhi=0.;n=0;ks=[]
    for fold in fs:
     lo,hi=values[fold];valid=np.isfinite(lo)&np.isfinite(hi);numlo+=weights[fold]@np.where(valid,lo,0);numhi+=weights[fold]@np.where(valid,hi,0);den+=weights[fold]@valid.astype(int);sumlo+=lo[valid].sum();sumhi+=hi[valid].sum();n+=int(valid.sum());ks.append(len(np.unique(blocks[fold][valid])))
    ok=den>0;supported=min(ks)>=4 and ok.sum()>=1900;ql=np.quantile(numlo[ok]/den[ok],[.025,.975]) if supported else [np.nan,np.nan];qu=np.quantile(numhi[ok]/den[ok],[.025,.975]) if supported else [np.nan,np.nan]
    paired.append(dict(scope=scope,candidate=MAIN,baseline=baseline,metric=metric,primary=scope=='ALL' and metric=='all_cost_bounds',difference_lower=sumlo/n if n else np.nan,difference_upper=sumhi/n if n else np.nan,outer_low=float(ql[0]),outer_high=float(qu[1]),lower_percentile_high=float(ql[1]),upper_percentile_low=float(qu[0]),n=n,minimum_blocks_in_fold=min(ks),valid_draws=int(ok.sum()),resampling_supported=supported))
 tails=[];tail_pairs=[];first=int(df.seed.min())
 for seed_scope,cases in [('three_seed_mean',[MAIN,*BASELINES]),('first_seed',['stress_e001','stress_e050',*BASELINES])]:
  subset=df if seed_scope=='three_seed_mean' else df[df.seed==first]
  for case in cases:
   g=subset[subset.case==case].groupby(['fold','issue_id'],sort=True)[['cost_lower','cost_upper']].mean();assert len(g)==48
   tails.append(dict(case=case,seed_scope=seed_scope,n_origins=48,alpha=.9,cvar_lower=empirical_cvar(g.cost_lower),cvar_upper=empirical_cvar(g.cost_upper)))
 for case,seed_scope in [(MAIN,'three_seed_mean'),('stress_e001','first_seed'),('stress_e050','first_seed')]:
  a=next(r for r in tails if r['case']==case and r['seed_scope']==seed_scope)
  for baseline in BASELINES:
   b=next(r for r in tails if r['case']==baseline and r['seed_scope']==seed_scope);tail_pairs.append(dict(candidate=case,baseline=baseline,seed_scope=seed_scope,difference_lower=a['cvar_lower']-b['cvar_upper'],difference_upper=a['cvar_upper']-b['cvar_lower'],scope='fixed 48 origins; descriptive tail cost; no population guarantee'))
 assert len(summaries)==25 and len(paired)==30 and sum(r['primary'] for r in paired)==2 and len(tails)==7 and len(tail_pairs)==6
 return dict(metrics=summaries,paired_comparisons=paired,tail_risk=tails,tail_comparisons=tail_pairs),draws

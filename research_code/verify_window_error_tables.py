"""Independent table-level audit supplement for round48."""
from datetime import datetime,timezone
import numpy as np,pandas as pd
from verify_window_error_attribution import OUT,ART,FOLDS,SEEDS,OFFSETS,BINS,read,sha,close,reference_bound

def verify():
 assert read(OUT/'verification.json')['status']=='PASS'
 diag=pd.read_csv(OUT/'diagnosis.csv');count=0;boot=0
 for row in diag.itertuples():
  path=OUT/row.fold;pop=np.load(path/'population.npz');z=[np.load(path/f'diagnostic_B_{s}.npz') for s in SEEDS];kind=row.kind;p=np.mean([a[kind] for a in z],0);lo=np.mean([a[kind+'_lower'] for a in z],0);hi=np.mean([a[kind+'_upper'] for a in z],0);ind=np.mean([a[kind+'_independent'] for a in z],0);eb=np.mean([a[kind+'_expected_bad'] for a in z],0);y=pop[kind+'_labels'];m=np.load(path/'fixed_cohorts.npz')[kind] if row.cohort=='baseline_high' else np.ones_like(y,bool)
  if row.offset!=-1:m&=(np.repeat(OFFSETS,17 if kind=='single' else 3)==row.offset)[None,:]
  n=int(m.sum());assert n==row.n;obs=np.load(path/'observation_counts.npz');assert int((m&(y==0)).sum())==row.known_failure and int((m&(y==1)).sum())==row.known_success and int((m&np.isnan(y)).sum())==row.unknown
  if n:close([p[m].mean(),lo[m].mean(),hi[m].mean(),ind[m].mean(),(p-lo)[m].mean(),eb[m].mean(),obs[kind+'_bad_lower'][m].mean(),obs[kind+'_bad_upper'][m].mean()],[row.mean_probability,row.mean_lower,row.mean_upper,row.mean_independent,row.dependence_room,row.expected_bad,row.observed_bad_lower,row.observed_bad_upper])
  saturated=m&(lo>=1-1e-12);assert saturated.sum()==row.saturated_n and (saturated&(y==0)).sum()==row.saturated_known_failures
  if row.offset==-1:
   ll,uu,bl,bu,bn,unique=reference_bound(lo,-np.ones_like(lo),y,pop[kind+'_inverse'],m,pop['blocks'],np.load(path/'draws.npz')['draws']);close([ll,uu],[row.marginal_gap_lower,row.marginal_gap_upper]);saved=np.load(path/f'diagnostic_bounds_{kind}_{row.cohort}.npz');close(saved['lower'],bl);close(saved['upper'],bu);close(saved['n'],bn,0);valid=np.isfinite(bl)&np.isfinite(bu)
   if valid.sum()>=1900:close([np.quantile(bl[valid],.025),np.quantile(bu[valid],.975)],[row.outer_low,row.outer_high])
   boot+=len(bl)
  count+=1
 # Rebuild per-seed and aggregate observed-hour tables from verified loss arrays.
 seed_rows=0
 for r in pd.read_csv(OUT/'marginal_metrics.csv').itertuples():
  z=np.load(OUT/r.fold/f'hourly_scores_{r.case}_{r.seed}.npz')[r.metric][:,r.lead_start-1:r.lead_end];assert np.isfinite(z).sum()==r.n;close(np.nanmean(z),r.value);seed_rows+=1
 aggregate_rows=0
 for r in pd.read_csv(OUT/'marginal_paired.csv').itertuples():
  fs=FOLDS if r.scope=='ALL' else [r.scope];sums={'B':[],'M':[]};counts=[];blocks=[];draw=[]
  for i,fold in enumerate(fs):
   for case in ['B','M']:
    z=np.mean([np.load(OUT/fold/f'hourly_scores_{case}_{s}.npz')[r.metric][:,r.lead_start-1:r.lead_end] for s in SEEDS],0);sums[case].extend(np.nansum(z,(1,2)))
   counts.extend(np.isfinite(z).sum((1,2)));blocks.extend(np.load(OUT/fold/'population.npz')['blocks']+9*i);draw.append(np.load(OUT/fold/'draws.npz')['draws']+9*i)
  n=np.array(counts);bb=np.array(sums['B']);mm=np.array(sums['M']);block=np.array(blocks);dd=np.concatenate(draw,1);num=np.array([(mm-bb)[block==k].sum() for k in range(block.max()+1)]);den=np.array([n[block==k].sum() for k in range(block.max()+1)]);samples=num[dd].sum(1)/den[dd].sum(1);close([bb.sum()/n.sum(),mm.sum()/n.sum(),(mm-bb).sum()/n.sum(),np.quantile(samples,.025),np.quantile(samples,.975)],[r.base,r.candidate,r.difference,r.low,r.high]);assert n.sum()==r.n;aggregate_rows+=1
 result=dict(status='PASS',diagnostic_rows=count,diagnostic_bootstrap_draws=boot,marginal_seed_rows=seed_rows,marginal_paired_rows=aggregate_rows,utc=datetime.now(timezone.utc).isoformat());(OUT/'table_verification.json').write_text(__import__('json').dumps(result,indent=2),encoding='utf-8');print(result,flush=True)

if __name__=='__main__':verify()

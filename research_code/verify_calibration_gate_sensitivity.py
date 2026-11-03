"""Independent all-draw verification for the fixed sensitivity designs."""
import numpy as np
import pandas as pd
from datetime import datetime
from research_calibration_gate import OUT,ART,SEEDS,FOLDS,read,sha,js
from calibration_gate_tables import population
from verify_window_error_attribution import reference_bound,close

def run():
 assert read(OUT/'sensitivity_complete.json')['status']=='COMPLETE'
 for p,h in read(OUT/'sensitivity_identity.json').items():assert sha(p)==h,p
 times=pd.to_datetime(np.load(ART/'data.npz')['timestamps'].tolist(),utc=True).asi8//3600000000000
 rng=np.random.default_rng(20261210)
 for days in [3,7,14]:
  for shift in [0,.5]:
   all_blocks=[];all_draws=[];shift_blocks=0
   for f,cfg in FOLDS.items():
    ids=np.load(OUT/f/'population.npz')['ids'];anchor=datetime.fromisoformat(cfg['test_start']+'T00:00:00+08:00').timestamp()/3600-days*shift*24
    original=np.floor((times[ids]-anchor)/(days*24)).astype(int);_,bb=np.unique(original,return_inverse=True);k=len(set(bb));all_blocks.append(bb+shift_blocks);all_draws.append(rng.integers(0,k,(2000,k))+shift_blocks);shift_blocks+=k
   z=np.load(OUT/f'sensitivity_design_d{days}_s{int(shift*2)}.npz');np.testing.assert_array_equal(z['blocks'],np.concatenate(all_blocks));np.testing.assert_array_equal(z['draws'],np.concatenate(all_draws,1))
 table=pd.read_csv(OUT/'block_sensitivity.csv');pops={k:population('ALL',k) for k in ['single','group']};count=0
 for row in table.itertuples():
  name=f'd{row.days}_s{int(row.anchor_shift_fraction*2)}';pair=f'{row.candidate}_{row.baseline}';design=np.load(OUT/f'sensitivity_design_{name}.npz');blocks=design['blocks'];draws=design['draws']
  if row.metric=='crps_all_observed':
   z={c:np.concatenate([np.mean([np.load(OUT/f/f'hourly_scores_{c}_{s}.npz')['crps'] for s in SEEDS],0) for f in FOLDS]) for c in [row.candidate,row.baseline]}
   numerator=np.array([np.nansum((z[row.candidate]-z[row.baseline])[blocks==k]) for k in range(int(blocks.max())+1)]);denominator=np.array([np.isfinite(z[row.baseline][blocks==k]).sum() for k in range(int(blocks.max())+1)])
   dden=denominator[draws].sum(1);vals=numerator[draws].sum(1)/dden;valid=np.isfinite(vals);close(np.load(OUT/f'sensitivity_{pair}_{name}_crps.npz')['difference'],vals)
   close([row.point_lower,row.point_upper],[numerator.sum()/denominator.sum()]*2);count+=len(vals);mask=np.isfinite(z[row.baseline]).any((1,2))[:,None]
   lo=hi=vals
  else:
   kind='single' if row.metric.startswith('single') else 'group';pop=pops[kind];pa=pop['probabilities'][row.candidate];pb=pop['probabilities'][row.baseline];c=np.mean(pa*pa-pb*pb,0);d=2*np.mean(pb-pa,0);mask=np.ones_like(pop['labels'],bool) if row.metric=='single_all' else pop['high'];pl,pu,lo,hi,n,unique=reference_bound(c,d,pop['labels'],pop['inverse'],mask,blocks,draws);saved=np.load(OUT/f'sensitivity_{pair}_{name}_{row.metric}.npz');close(saved['lower'],lo);close(saved['upper'],hi);close(saved['n'],n,0);close([pl,pu],[row.point_lower,row.point_upper]);valid=np.isfinite(lo)&np.isfinite(hi);count+=len(lo)
  nonempty=[];start=0
  for f in FOLDS:
   length=len(np.load(OUT/f/'population.npz')['ids']);bb=blocks[start:start+length];mm=mask[start:start+length];nonempty.append(sum(mm[bb==k].any() for k in np.unique(bb)));start+=length
  support=bool(min(nonempty)>=4 and valid.sum()>=1900);assert support==row.supported and min(nonempty)==row.minimum_nonempty_blocks and valid.sum()==row.valid_draws
  if support:close([np.quantile(lo[valid],.025),np.quantile(hi[valid],.975)],[row.outer_low,row.outer_high])
 js(OUT/'sensitivity_verification.json',dict(status='PASS',rows=len(table),draws_recomputed=count,supported_rows=int(table.supported.sum()),independent_shared_label_bounds=True));print('PASS sensitivity',len(table),count,flush=True)
if __name__=='__main__':run()

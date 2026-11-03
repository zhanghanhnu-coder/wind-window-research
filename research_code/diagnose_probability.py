"""Descriptive diagnostics after the frozen experiment; no model selection here."""
import json
import numpy as np
import pandas as pd
from model import ART
from probability_core import spread,longest,first_start,make_bank,crps
from research_probability import OUT,SEEDS,METHODS,csv

def main():
 d=np.load(OUT/'predictions.npz');meta=json.loads((ART/'metadata.json').read_text(encoding='utf-8'));raw=np.load(ART/'data.npz');times=pd.DatetimeIndex(pd.to_datetime(raw['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');testtime=times[d['test_ids']]
 rows=[];inner=[]
 for s in SEEDS:
  pieces=[('Oct_fit',d['cal_truth'][d['inner_fit']],d[f'cal_q_{s}'][d['inner_fit']]),('Oct_validation',d['cal_truth'][d['inner_val']],d[f'cal_q_{s}'][d['inner_val']])]
  for month in [11,12]:
   mask=testtime.month==month;pieces.append((str(month),d['test_truth'][mask],d[f'test_q_{s}'][mask]))
  for period,y,q in pieces:
   z=(y-q[...,1])/spread(q);r=y-q[...,1];mask=np.isfinite(y)
   rows.append(dict(seed=s,period=period,n=int(mask.sum()),observed_mean=float(np.nanmean(y)),observed_q90=float(np.nanquantile(y,.9)),median_error=float(np.nanmedian(r)),normalized_abs_q80=float(np.nanquantile(abs(z),.8)),normalized_median=float(np.nanmedian(z)),predicted_half_width=float(np.mean(spread(q)))))
  bank=make_bank(d[f'cal_q_{s}'][d['inner_fit']],d['cal_truth'][d['inner_fit']],d['cal_context'][d['inner_fit']]);loss=[]
  for q,y in zip(d[f'cal_q_{s}'][d['inner_val']],d['cal_truth'][d['inner_val']]):loss.append(crps(np.maximum(0,q[...,1]+bank['residual']),y))
  inner.append(dict(seed=s,method='joint_all',crps=float(np.nanmean(loss))))
 csv(rows,'period_diagnostics.csv');csv(inner,'inner_baseline_diagnostic.csv')
 sc=np.load(OUT/'first_origin_scenarios.npz');lengthrows=[];firstrows=[];winrows=[]
 for m in METHODS:
  x=sc[m];w=sc[m+'_weights'];lr=longest(x<=10);first=first_start(x<=10)
  for n,node in enumerate(meta['nodes']):
   for length in range(73):lengthrows.append(dict(method=m,node=node,hours=length,probability=float(w@(lr[:,n]==length))))
   for start in range(67):firstrows.append(dict(method=m,node=node,start_offset=start,probability=float(w@(first[:,n]==start))))
   firstrows.append(dict(method=m,node=node,start_offset='no_window',probability=float(w@np.isinf(first[:,n]))))
   valid=np.isfinite(sc['truth'][:,n]).all();obs=longest((sc['truth']<=10)[None])[0,n] if valid else np.nan
   winrows.append(dict(method=m,node=node,issue=str(testtime[0]),p_any6=float(w@(lr[:,n]>=6)),p_no6=float(w@(lr[:,n]<6)),expected_longest=float(w@lr[:,n]),observed_longest=obs,truth_complete=valid))
 csv(lengthrows,'first_origin_length_pmf.csv');csv(firstrows,'first_origin_start_pmf.csv');csv(winrows,'first_origin_window_summary.csv')
 # Collapsed reliability bins mix repeated seeds; counts explicitly represent forecast instances.
 r=pd.read_csv(OUT/'reliability_bins.csv');r=r[r.month==0];merged=[]
 for (m,b),g in r.groupby(['method','bin']):
  n=g.n.sum();merged.append(dict(method=m,bin=b,forecast_instances=int(n),mean_probability=float((g.mean_probability*g.n).sum()/n) if n else np.nan,observed_rate=float((g.observed_rate*g.n).sum()/n) if n else np.nan))
 csv(merged,'reliability_three_seed_instances.csv')
 print('diagnostics and first-origin joint-window distributions saved')

if __name__=='__main__':main()

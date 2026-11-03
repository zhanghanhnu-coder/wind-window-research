"""Post-hoc descriptions only: no selection or changes to the frozen replay."""
import numpy as np
import pandas as pd
from model import ART
from research_rolling import OUT,SEEDS,METHODS,csv

def main():
 data=np.load(OUT/'data.npz');raw=np.load(ART/'data.npz');times=pd.DatetimeIndex(pd.to_datetime(raw['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');testtimes=times[data['ids'][data['test_positions']]];week=((testtimes-pd.Timestamp('2024-11-01',tz=testtimes.tz)).days//7).to_numpy()
 rows=[];support=[]
 for s in SEEDS:
  for m in METHODS:
   with np.load(OUT/f'scores_{m}_{s}.npz') as p:
    for b in np.unique(week):
     mask=week==b;pr=p['prob6'][mask];label=p['label6'][mask];rows.append(dict(seed=s,method=m,week=int(b),start=str(testtimes[mask][0]),end=str(testtimes[mask][-1]),n=int(np.isfinite(label).sum()),mean_probability=float(np.nanmean(pr)),observed_rate=float(np.nanmean(label)),brier=float(np.nanmean(p['brier6'][mask])),crps=float(np.nanmean(p['crps'][mask])) if 'crps' in p else np.nan))
  a=np.load(OUT/f'scores_rolling28_{s}.npz');b=np.load(OUT/f'scores_tilted28_{s}.npz');c=np.load(OUT/f'scores_direct28_{s}.npz')
  for month in [0,11,12]:
   mask=np.ones(len(testtimes),bool) if month==0 else testtimes.month==month
   pp=a['prob6'][mask];yy=a['label6'][mask];valid=np.isfinite(yy);zero=valid&(pp==0);one=valid&(pp==1);interior=valid&(pp>0)&(pp<1)
   for group,sel in [('support_zero',zero),('support_one',one),('interior',interior)]:
    support.append(dict(seed=s,month=month,group=group,n=int(sel.sum()),observed_rate=float(np.mean(yy[sel])) if sel.any() else np.nan,raw_probability=float(np.mean(pp[sel])) if sel.any() else np.nan,tilted_probability=float(np.mean(b['prob6'][mask][sel])) if sel.any() else np.nan,direct_probability=float(np.mean(c['prob6'][mask][sel])) if sel.any() else np.nan,raw_brier=float(np.mean(a['brier6'][mask][sel])) if sel.any() else np.nan,tilted_brier=float(np.mean(b['brier6'][mask][sel])) if sel.any() else np.nan,direct_brier=float(np.mean(c['brier6'][mask][sel])) if sel.any() else np.nan))
 csv('weekly_diagnostics.csv',rows);csv('support_diagnostics.csv',support)
 bank=pd.read_csv(OUT/'bank_audit.csv');audit=pd.read_csv(OUT/'calibration_audit.csv');records=[]
 for name,g in bank.groupby('bank'):
  records.append(dict(item='bank_'+str(name),n=len(g),minimum=g.n.min(),median=g.n.median(),maximum=g.n.max(),mean=g.n.mean(),fallbacks=int(g.fallback.sum())))
 for name in ['map_a','map_b','map_n','map_origins','tilt_ess','tilt_gap_before','tilt_gap_after','tilt_impossible_fraction']:
  v=audit[name];records.append(dict(item=name,n=len(v),minimum=v.min(),median=v.median(),maximum=v.max(),mean=v.mean(),fallbacks=int(audit.map_fallback.sum()) if name=='map_a' else np.nan))
 csv('audit_summary.csv',records)
 print(pd.DataFrame(records).round(5).to_string(index=False));print('descriptive diagnostics saved')

if __name__=='__main__':main()

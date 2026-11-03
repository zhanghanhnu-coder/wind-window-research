"""Complete diagnostic bins without altering a stored probability or score."""
import numpy as np
def reliability_rows(probability,label):
 p=np.asarray(probability,float);y=np.asarray(label,float)
 if p.shape!=y.shape:raise ValueError('aligned probabilities and binary observations required')
 valid=np.isfinite(y);p=p[valid];y=y[valid]
 if not np.isfinite(p).all() or not np.isin(y,[0,1]).all() or (p< -1e-12).any() or (p>1+1e-12).any():raise ValueError('finite probabilities in numerical unit interval and binary labels required')
 rows=[];assignment=np.zeros(len(p),int);oldcount=0
 for b in range(10):
  old=(p>=b/10)&(p<(b+1)/10 if b<9 else p<=1);oldcount+=int(old.sum());select=(p>=b/10 if b else np.ones(len(p),bool))&(p<(b+1)/10 if b<9 else np.ones(len(p),bool));assignment+=select;rows.append(dict(bin=b,n=int(select.sum()),mean_probability=float(p[select].mean()) if select.any() else np.nan,observed_rate=float(y[select].mean()) if select.any() else np.nan))
 assert (assignment==1).all();return rows,dict(valid_targets=len(p),new_binned=sum(r['n'] for r in rows),old_binned=oldcount,old_unbinned=len(p)-oldcount,roundoff_outside_unit_interval=int(((p<0)|(p>1)).sum()),maximum_unit_interval_excess=float(max(0,p.max()-1,-p.min())) if len(p) else 0.)

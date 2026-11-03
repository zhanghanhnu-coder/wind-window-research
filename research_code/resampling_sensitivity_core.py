"""Fixed-loss calendar-block sensitivity, with explicit support limits."""
import numpy as np

def origin_totals(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float)
 if a.shape!=b.shape or a.ndim<2 or not a.shape[0] or not a.shape[1] or np.isinf(a).any() or np.isinf(b).any():raise ValueError('aligned finite-or-missing seed/origin losses required')
 mask=np.isfinite(a)
 if not np.array_equal(mask,np.isfinite(b)) or not np.array_equal(mask,np.broadcast_to(mask[0],mask.shape)):raise ValueError('candidate, reference and all seeds require identical target masks')
 diff=(a-b).mean(0).reshape(a.shape[1],-1);return np.nansum(diff,axis=1),np.isfinite(diff).sum(1)

def calendar_blocks(elapsed_hours,days,offset_days):
 h=np.asarray(elapsed_hours,float)
 if isinstance(days,(bool,np.bool_)) or isinstance(offset_days,(bool,np.bool_)) or not isinstance(days,(int,np.integer)) or days<=0 or not isinstance(offset_days,(int,np.integer)) or not 0<=offset_days<days or h.ndim!=1 or not len(h) or not np.isfinite(h).all() or (h<0).any():raise ValueError('nonnegative elapsed hours, positive integer days and in-block integer offset required')
 return np.floor((h+24*offset_days)/(24*days)).astype(int)

def bootstrap_totals(folds,min_blocks=4,min_valid=1900):
 if not folds or min_blocks<1 or min_valid<1:raise ValueError('nonempty folds and positive support thresholds required')
 numerator=None;denominator=None;total=0.;count=0;nonempty=[]
 for f in folds:
  s=np.asarray(f['sums'],float);c=np.asarray(f['counts']);b=np.asarray(f['blocks']);draw=np.asarray(f['draws'])
  if s.ndim!=1 or not len(s) or c.shape!=s.shape or b.shape!=s.shape or not np.isfinite(s).all() or not np.issubdtype(c.dtype,np.integer) or (c<0).any() or not np.issubdtype(b.dtype,np.integer) or (b<0).any() or ((c==0)&(s!=0)).any():raise ValueError('aligned finite totals, nonnegative integer counts and calendar indices required')
  k=int(b.max())+1
  if draw.ndim!=2 or draw.shape[1]!=k or not len(draw) or not np.issubdtype(draw.dtype,np.integer) or (draw<0).any() or (draw>=k).any():raise ValueError('each replicate must draw exactly the original number of calendar blocks')
  if numerator is None:numerator=np.zeros(len(draw));denominator=np.zeros(len(draw),np.int64)
  if len(draw)!=len(numerator):raise ValueError('all folds require the same replicate count')
  block_sum=np.bincount(b,weights=s,minlength=k);block_count=np.zeros(k,np.int64);np.add.at(block_count,b,c);numerator+=block_sum[draw].sum(1);denominator+=block_count[draw].sum(1);total+=s.sum();count+=int(c.sum());nonempty.append(int((block_count>0).sum()))
 valid=denominator>0;replicates=np.full(len(numerator),np.nan);replicates[valid]=numerator[valid]/denominator[valid];supported=min(nonempty)>=min_blocks and int(valid.sum())>=min_valid;lo,hi=np.quantile(replicates[valid],[.025,.975]) if supported else [np.nan,np.nan]
 return dict(difference=total/count if count else np.nan,n=count,minimum_blocks=min(nonempty),nonempty_blocks=sum(nonempty),valid_draws=int(valid.sum()),supported=bool(supported),low95=float(lo),high95=float(hi),replicates=replicates)

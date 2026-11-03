"""Sharp finite-population Brier bounds under shared, unknown binary window labels."""
import numpy as np
OFFSETS=np.array([0,24,48,66]);GROUPS=[[0,1,2],[8,9,10],[13,15,16]]

def classify_window(observations,axis=-1):
 x=np.asarray(observations);valid=np.isfinite(x);bad=(valid&(x>10)).any(axis=axis);complete=valid.all(axis=axis)
 return np.where(bad,0.,np.where(complete,1.,np.nan))

def physical_mapping(ids,kind):
 ids=np.asarray(ids)
 if ids.ndim!=1 or not len(ids) or not np.issubdtype(ids.dtype,np.integer) or np.any(np.diff(ids)<=0) or np.any((ids-ids[0])%6):raise ValueError('sorted common six-hour issue grid required')
 if kind not in ['single','group']:raise ValueError('unknown event kind')
 count=17 if kind=='single' else 3;starts=(ids[:,None]+OFFSETS[None,:]+1).ravel();unique_starts=np.unique(starts)
 if len(unique_starts)>1 and np.diff(unique_starts).min()<6:raise ValueError('physical six-hour blocks overlap')
 assert len(set(sum(GROUPS,[])))==sum(map(len,GROUPS))
 pairs=np.column_stack([np.repeat(starts,count),np.tile(np.arange(count),len(starts))]);keys,inverse=np.unique(pairs,axis=0,return_inverse=True)
 return keys,inverse.reshape(len(ids),4*count)

def window_labels(raw,ids):
 raw=np.asarray(raw);ids=np.asarray(ids);physical_mapping(ids,'single')
 if raw.ndim!=2 or raw.shape[1]!=17 or ids.min()<0 or ids.max()+72>=len(raw):raise ValueError('complete target time range and 17 nodes required')
 single=[];group=[]
 for offset in OFFSETS:
  x=raw[ids[:,None]+np.arange(offset+1,offset+7)[None,:]];single.append(classify_window(x,axis=1))
  for nodes in GROUPS:group.append(classify_window(x[:,:,nodes],axis=(1,2)))
 return np.concatenate(single,axis=1),np.stack(group,axis=1)

def collapse_labels(labels,inverse):
 labels=np.asarray(labels);inverse=np.asarray(inverse)
 if labels.shape!=inverse.shape or not np.issubdtype(inverse.dtype,np.integer) or inverse.size==0 or inverse.min()<0:raise ValueError('invalid shared-label map')
 y=labels.ravel();ix=inverse.ravel();valid=np.isfinite(y)
 if not np.isin(y[valid],[0,1]).all():raise ValueError('binary labels or missing values required')
 total=np.bincount(ix);known=np.bincount(ix,weights=valid,minlength=len(total));sums=np.bincount(ix,weights=np.where(valid,y,0),minlength=len(total))
 if (total==0).any() or not ((known==0)|(known==total)).all() or not ((sums==0)|(sums==known)).all():raise ValueError('same physical window has inconsistent labels')
 out=np.full(len(total),np.nan);np.divide(sums,known,out=out,where=known>0);return out

def coefficients(pa,pb=None):
 pa=np.asarray(pa,float)
 if pa.ndim!=3 or min(pa.shape)<1 or not np.isfinite(pa).all() or pa.min()< -1e-12 or pa.max()>1+1e-12:raise ValueError('complete seed-by-origin-by-event probabilities required')
 if pb is None:return np.mean(pa*pa,axis=0),1-2*np.mean(pa,axis=0)
 pb=np.asarray(pb,float)
 if pb.shape!=pa.shape or not np.isfinite(pb).all() or pb.min()< -1e-12 or pb.max()>1+1e-12:raise ValueError('aligned complete comparator probabilities required')
 return np.mean(pa*pa-pb*pb,axis=0),-2*np.mean(pa-pb,axis=0)

def aggregate_bounds(constant,group_coefficient,unique_labels,total,witness=False):
 d=np.asarray(group_coefficient);y=np.asarray(unique_labels);known=np.isfinite(y);fixed=np.asarray(constant)+np.sum(d[...,known]*y[known],axis=-1);lower=(fixed+np.minimum(d[...,~known],0).sum(axis=-1))/total;upper=(fixed+np.maximum(d[...,~known],0).sum(axis=-1))/total;out=dict(lower=lower,upper=upper,n=np.asarray(total))
 if witness:
  out['witness_lower']=np.where(known,y,(d<0).astype(float));out['witness_upper']=np.where(known,y,(d>0).astype(float))
 return out

def sharp_bounds(c,d,labels,inverse,origin_weights=None):
 c=np.asarray(c,float);d=np.asarray(d,float);labels=np.asarray(labels);inverse=np.asarray(inverse);unique=collapse_labels(labels,inverse)
 if c.shape!=labels.shape or d.shape!=labels.shape or c.ndim!=2 or not np.isfinite(c).all() or not np.isfinite(d).all():raise ValueError('aligned finite loss coefficients required')
 weights=np.ones(len(c)) if origin_weights is None else np.asarray(origin_weights,float)
 if weights.shape!=(len(c),) or not np.isfinite(weights).all() or weights.min()<0 or weights.sum()<=0:raise ValueError('nonnegative nonempty origin multiplicities required')
 cc=c*weights[:,None];dd=d*weights[:,None];total=float(weights.sum()*c.shape[1]);group=np.bincount(inverse.ravel(),weights=dd.ravel(),minlength=len(unique));out=aggregate_bounds(cc.sum(),group,unique,total,True);valid=np.isfinite(labels);out.update(pointwise_lower=float(np.where(valid,cc+dd*labels,cc+np.minimum(dd,0)).sum()/total),pointwise_upper=float(np.where(valid,cc+dd*labels,cc+np.maximum(dd,0)).sum()/total),known_targets=float(np.sum(weights[:,None]*valid)),unknown_targets=float(np.sum(weights[:,None]*~valid)),physical_events=len(unique),unknown_physical_events=int(np.isnan(unique).sum()))
 out['lower']=float(out['lower']);out['upper']=float(out['upper']);out['n']=total
 return out

def bootstrap_bounds(c,d,labels,inverse,blocks,draws):
 c=np.asarray(c);d=np.asarray(d);blocks=np.asarray(blocks);draws=np.asarray(draws);unique=collapse_labels(labels,inverse)
 if blocks.shape!=(len(c),) or blocks.min()<0 or not np.issubdtype(blocks.dtype,np.integer) or not np.issubdtype(draws.dtype,np.integer):raise ValueError('integer block identifiers required')
 k=int(blocks.max())+1
 if draws.ndim!=2 or draws.shape[1]!=k or draws.min()<0 or draws.max()>=k:raise ValueError('invalid block resampling draws')
 const=[];groups=[];counts=[]
 for block in range(k):
  use=blocks==block;const.append(float(c[use].sum()));groups.append(np.bincount(inverse[use].ravel(),weights=d[use].ravel(),minlength=len(unique)));counts.append(int(use.sum()*c.shape[1]))
 multiplicities=np.stack([np.bincount(row,minlength=k) for row in draws]);total=multiplicities@np.asarray(counts)
 if (total<=0).any():raise ValueError('empty bootstrap population')
 return aggregate_bounds(multiplicities@np.asarray(const),multiplicities@np.stack(groups),unique,total,False)

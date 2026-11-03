"""Fixed-marginal intersection limits and common-cohort missing-label scores."""
import numpy as np
from crps_distribution_core import couple_atoms
from missing_window_bounds_core import OFFSETS,GROUPS,collapse_labels,aggregate_bounds

def intersection_bounds(p):
 p=np.asarray(p,float)
 if p.ndim<1 or p.shape[-1]<1 or not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('nonempty valid marginal probabilities required')
 return np.maximum(0,p.sum(-1)-(p.shape[-1]-1)),p.min(-1)

def summarize(q,ranks):
 q=np.asarray(q,float);ranks=np.asarray(ranks)
 if q.shape!=(72,17,19) or not np.isfinite(q).all() or (np.diff(q,axis=-1)<0).any():raise ValueError('ordered 72x17x19 atoms required')
 if ranks.shape!=(57,72,17) or not np.issubdtype(ranks.dtype,np.integer) or not np.array_equal(np.sort(ranks,axis=0),np.broadcast_to(np.arange(57)[:,None,None],ranks.shape)):raise ValueError('57 rank permutations required')
 ph=(q<=10).mean(-1);safe=couple_atoms(q,ranks)<=10;out={'hourly':ph};fields={k:[] for k in ['single','group']}
 for k in ['single','group']:
  for suffix in ['lower','upper','independent','expected_bad']:fields[k+'_'+suffix]=[]
 for off in OFFSETS:
  p=ph[off:off+6].T;l,u=intersection_bounds(p)
  for key,val in [('single',safe[:,off:off+6].all(1).mean(0)),('single_lower',l),('single_upper',u),('single_independent',p.prod(-1)),('single_expected_bad',(1-p).sum(-1))]:fields[key].extend(val)
  for nodes in GROUPS:
   p=ph[off:off+6][:,nodes].ravel();l,u=intersection_bounds(p)
   for key,val in [('group',safe[:,off:off+6][:,:,nodes].all((1,2)).mean()),('group_lower',l),('group_upper',u),('group_independent',p.prod()),('group_expected_bad',(1-p).sum())]:fields[key].append(val)
 out.update({k:np.array(v) for k,v in fields.items()});return out

def observation_events(y):
 y=np.asarray(y,float)
 if y.shape!=(72,17):raise ValueError('72x17 observations required')
 out={k+'_'+side:[] for k in ['single','group'] for side in ['bad_lower','bad_upper']}
 for off in OFFSETS:
  z=y[off:off+6];bad=(np.isfinite(z)&(z>10)).sum(0);unknown=(~np.isfinite(z)).sum(0)
  out['single_bad_lower'].extend(bad);out['single_bad_upper'].extend(bad+unknown)
  for nodes in GROUPS:out['group_bad_lower'].append(bad[nodes].sum());out['group_bad_upper'].append((bad+unknown)[nodes].sum())
 return {k:np.array(v) for k,v in out.items()}

def fixed_cohort(probabilities,threshold=.8):
 p=np.asarray(probabilities,float)
 if p.ndim!=3 or not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('seed x origin x event probabilities required')
 return p.mean(0)>=threshold-1e-12

def selected_bounds(c,d,labels,inverse,mask,weights=None):
 c=np.asarray(c,float);d=np.asarray(d,float);mask=np.asarray(mask,bool);labels=np.asarray(labels);inverse=np.asarray(inverse);unique=collapse_labels(labels,inverse)
 if c.shape!=d.shape or c.shape!=labels.shape or mask.shape!=c.shape or not np.isfinite(c).all() or not np.isfinite(d).all():raise ValueError('aligned finite coefficients and selection required')
 w=np.ones(len(c)) if weights is None else np.asarray(weights,float)
 if w.shape!=(len(c),) or not np.isfinite(w).all() or (w<0).any():raise ValueError('valid origin multiplicities required')
 wm=mask*w[:,None];total=float(wm.sum())
 if total==0:return dict(lower=float('nan'),upper=float('nan'),n=0)
 group=np.bincount(inverse.ravel(),weights=(d*wm).ravel(),minlength=len(unique))
 result=aggregate_bounds((c*wm).sum(),group,unique,total,True)
 result.update(lower=float(result['lower']),upper=float(result['upper']),n=total);return result

def selected_bootstrap(c,d,labels,inverse,mask,blocks,draws):
 c=np.asarray(c,float);d=np.asarray(d,float);mask=np.asarray(mask,bool);blocks=np.asarray(blocks);draws=np.asarray(draws);unique=collapse_labels(labels,inverse);k=int(blocks.max())+1
 if draws.ndim!=2 or draws.shape[1]!=k or draws.min()<0 or draws.max()>=k:raise ValueError('invalid block draws')
 const=[];groups=[];counts=[]
 for block in range(k):
  m=mask*(blocks==block)[:,None];const.append(float((c*m).sum()));counts.append(int(m.sum()));groups.append(np.bincount(inverse.ravel(),weights=(d*m).ravel(),minlength=len(unique)))
 mult=np.stack([np.bincount(row,minlength=k) for row in draws]);total=mult@np.asarray(counts);valid=total>0
 lower=np.full(len(draws),np.nan);upper=lower.copy()
 if valid.any():
  result=aggregate_bounds(mult[valid]@np.asarray(const),mult[valid]@np.stack(groups),unique,total[valid]);lower[valid]=result['lower'];upper[valid]=result['upper']
 return dict(lower=lower,upper=upper,n=total)

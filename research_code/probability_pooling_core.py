"""Causal simplex pooling of four complete empirical predictive distributions."""
import numpy as np
from scipy.optimize import minimize
from crps_distribution_core import couple_atoms
from seasonal_rolling_core import matured_labels
from lead_calibration_core import BINS

REG=.001

def checked_weights(w):
 w=np.asarray(w,dtype=float)
 if w.shape!=(4,) or not np.isfinite(w).all() or np.any(w<0) or abs(w.sum()-1)>1e-9:raise ValueError('four nonnegative unit-sum weights required')
 return w.copy()

def cross_distance(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float)
 if x.shape[:-1]!=y.shape[:-1] or not x.shape[-1] or not y.shape[-1]:raise ValueError('unaligned empirical distributions')
 k=x.shape[-1];l=y.shape[-1];z=np.concatenate([x,y],axis=-1);order=np.argsort(z,axis=-1,kind='stable');z=np.take_along_axis(z,order,axis=-1)
 fx=np.cumsum(order<k,axis=-1)/k;fy=np.cumsum(order>=k,axis=-1)/l
 return np.sum(np.diff(z,axis=-1)*(fx+fy-2*fx*fy)[...,:-1],axis=-1)

def moments(q,y,scale=1.,chunk=2048):
 q=np.asarray(q);y=np.asarray(y)
 if q.ndim!=3 or q.shape[:2]!=(4,len(y)) or q.shape[-1]!=19 or not len(y) or y.ndim!=1 or not np.isfinite(q).all() or not np.isfinite(y).all() or not np.isfinite(scale) or scale<=0:raise ValueError('finite aligned four-expert atoms and labels required')
 a=np.zeros(4);b=np.zeros((4,4))
 for start in range(0,len(y),chunk):
  x=np.asarray(q[:,start:start+chunk],float)/scale;t=np.asarray(y[start:start+chunk],float)/scale
  a+=np.sum(np.mean(abs(x-t[None,:,None]),axis=-1),axis=1)
  for m in range(4):
   for n in range(m,4):b[m,n]+=np.sum(cross_distance(x[m],x[n]))
 b=b+np.triu(b,1).T
 return a/len(y),b/len(y)

def pool_objective(w,a,b):
 w=np.asarray(w,float);a=np.asarray(a,float);b=np.asarray(b,float)
 return float(a@w-.5*w@b@w+REG*np.sum((w-.25)**2)),a-b@w+2*REG*(w-.25)

def solve_weights(a,b,previous):
 previous=checked_weights(previous);a=np.asarray(a,float);b=np.asarray(b,float)
 if a.shape!=(4,) or b.shape!=(4,4) or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(b,b.T,rtol=0,atol=1e-12):raise ValueError('invalid CRPS moments')
 tangent=np.vstack([np.eye(3),-np.ones(3)]);mineig=float(np.linalg.eigvalsh(tangent.T@(-b+2*REG*np.eye(4))@tangent).min())
 result=minimize(lambda w:pool_objective(w,a,b),previous,method='SLSQP',jac=True,bounds=[(0.,1.)]*4,constraints=[dict(type='eq',fun=lambda w:w.sum()-1,jac=lambda w:np.ones(4))],options=dict(maxiter=200,ftol=1e-10))
 candidate=np.asarray(result.x,float);feasible=bool(candidate.shape==(4,) and np.isfinite(candidate).all() and candidate.min()>=-1e-9 and abs(candidate.sum()-1)<=1e-9)
 if feasible:candidate=np.maximum(candidate,0);candidate/=candidate.sum()
 value,gradient=pool_objective(candidate,a,b) if candidate.shape==(4,) else (float('nan'),np.full(4,np.nan))
 gap=float(gradient@candidate-gradient.min()) if candidate.shape==(4,) else float('inf')
 good=bool(result.success and feasible and np.isfinite(value) and mineig>=-1e-9 and gap<=1e-6)
 used=candidate.copy() if good else previous.copy()
 reasons=[]
 if not result.success:reasons.append(str(result.message))
 if not feasible:reasons.append('infeasible candidate')
 if not np.isfinite(value):reasons.append('nonfinite objective')
 if mineig< -1e-9:reasons.append('nonconvex tangent Hessian')
 if not np.isfinite(gap) or gap>1e-6:reasons.append('Frank-Wolfe gap exceeds 1e-6')
 return used,dict(success=good,fallback=not good,status=int(result.status),iterations=int(result.nit),reason='; '.join(reasons) if reasons else str(result.message),candidate_weights=candidate.tolist(),candidate_objective=value,used_objective=pool_objective(used,a,b)[0],candidate_gradient=gradient.tolist(),candidate_gap=gap,tangent_min_eigenvalue=mineig,feasible=feasible)

def fit_at(q,truth,source_ids,issue,previous,scale):
 q=np.asarray(q);truth=np.asarray(truth);ids=np.asarray(source_ids);previous=checked_weights(previous)
 if q.shape!=(4,*truth.shape,19):raise ValueError('unaligned expert forecast archive')
 y=matured_labels(truth,ids,issue,28);mask=np.isfinite(y);positions=np.flatnonzero(mask.any((1,2)));counts=[int(mask[:,a:b].sum()) for a,b in BINS];target=ids[:,None]+np.arange(1,73)[None,:];known=target[mask.any(-1)];enough=len(positions)>=20 and min(counts)>=100
 audit=dict(issue_id=int(issue),lookback_days=28,valid_origins=int(len(positions)),valid_labels=int(mask.sum()),group_counts=counts,source_positions=positions.tolist(),source_ids=ids[positions].tolist(),earliest_label_id=int(known.min()) if len(known) else None,latest_label_id=int(known.max()) if len(known) else None,insufficient_support=not enough,optimizer_called=enough)
 if enough:
  a,b=moments(q[:,mask],y[mask],scale);w,details=solve_weights(a,b,previous)
 else:
  a=np.full(4,np.nan);b=np.full((4,4),np.nan);w=previous.copy();details=dict(success=False,fallback=True,reason='insufficient mature support; retain previous weights')
 return dict(weights=w,mask=mask,a=a,b=b,audit=audit,details=details)

def weighted_paths(q,ranks,w):
 q=np.asarray(q);w=checked_weights(w)
 if q.ndim!=4 or q.shape[0]!=4 or q.shape[1]!=72 or q.shape[-1]!=19 or not np.isfinite(q).all():raise ValueError('four complete 72-hour atom distributions required')
 sc=np.concatenate([couple_atoms(x,ranks).astype(float,copy=False) for x in q],axis=0)
 return sc,np.repeat(w/len(ranks),len(ranks))

def weights_for_issue(updates,issue):
 eligible=[row for row in updates if row['issue_id']<=issue]
 if not eligible:raise ValueError('no deployed weights at this issue')
 return checked_weights(max(eligible,key=lambda row:row['issue_id'])['weights'])

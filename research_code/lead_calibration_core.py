"""Four October-fitted lead groups and fixed interpolation; no test-time fitting."""
import numpy as np
from scipy.optimize import minimize_scalar
from crps_distribution_core import fit_affine,affine_objective
BINS=((0,6),(6,24),(24,48),(48,72))
ANCHORS=np.array([3.5,15.5,36.5,60.5])

def lead_parameters(theta,smooth=False):
 theta=np.asarray(theta,float)
 if theta.shape!=(4,2) or not np.isfinite(theta).all() or np.any(theta[:,1]<=0):raise ValueError('four finite positive-scale parameter pairs required')
 if smooth:return np.stack([np.interp(np.arange(1,73),ANCHORS,theta[:,j]) for j in range(2)],axis=1)
 return np.concatenate([np.repeat(theta[j:j+1],b-a,axis=0) for j,(a,b) in enumerate(BINS)])

def apply_groups(q,theta,scale,smooth=False):
 q=np.asarray(q,float)
 if q.ndim not in [3,4] or q.shape[-3]!=72:raise ValueError('requires [origins,] 72 x nodes x atoms')
 p=lead_parameters(theta,smooth);shape=(1,)*(q.ndim-3)+(72,1,1);delta=p[:,0].reshape(shape);gamma=p[:,1].reshape(shape);med=q[...,q.shape[-1]//2,None]
 return np.maximum(0,med+delta*scale+gamma*(q-med))

def fit_groups(cal_q,cal_y,scale,shift_only=False):
 cal_q=np.asarray(cal_q);cal_y=np.asarray(cal_y)
 if cal_q.shape[:-1]!=cal_y.shape or cal_y.ndim!=3 or cal_y.shape[1]!=72:raise ValueError('calibration arrays do not align')
 params=[];infos=[]
 for a,b in BINS:
  q=cal_q[:,a:b];y=cal_y[:,a:b]
  if shift_only:
   valid=np.isfinite(y);qq=q[valid].astype(float)/scale;yy=y[valid].astype(float)/scale
   fun=lambda delta:affine_objective([delta,1.],qq,yy)[0]
   result=minimize_scalar(fun,method='bounded',bounds=(-2,2),options={'xatol':1e-6,'maxiter':500});success=bool(result.success and np.isfinite(result.fun));theta=np.array([result.x,1.]) if success else np.array([0.,1.]);info=dict(success=success,fallback=not success,reason=str(result.message),iterations=int(result.nfev),objective_candidate=float(result.fun),objective_used=float(fun(theta[0])),valid_targets=int(valid.sum()),delta=float(theta[0]),gamma=1.)
  else:theta,info=fit_affine(q,y,scale)
  params.append(theta);infos.append(dict(lead_start=a+1,lead_end=b,**info))
 return np.array(params),infos

def jump_summary(sc):
 sc=np.asarray(sc,float);diff=abs(np.diff(sc,axis=1));positions=np.array([5,23,47]);remaining=np.setdiff1d(np.arange(71),positions)
 def mean(v,axis=None):
  den=np.isfinite(v).sum(axis=axis);return np.divide(np.nansum(v,axis=axis),den,out=np.full(np.shape(den),np.nan),where=den>0)
 return dict(boundaries=mean(diff[:,positions],axis=(0,2)),other_mean=float(mean(diff[:,remaining])),hourly=mean(diff,axis=(0,2)))

"""Restricted separable Gaussian rank coupling from partially observed archives."""
import numpy as np
from scipy.stats import rankdata,norm

def assert_matured(ids,issue):
 ids=np.asarray(ids)
 if ids.ndim!=1 or not len(ids) or not np.issubdtype(ids.dtype,np.integer) or np.any(ids+72>=issue):raise ValueError('all calibration labels must precede the first evaluated issue')

def normal_scores(residual,min_count=20):
 residual=np.asarray(residual,float)
 if residual.ndim!=3:raise ValueError('requires calibration origins x hours x nodes')
 output=np.full(residual.shape,np.nan)
 for h in range(residual.shape[1]):
  for node in range(residual.shape[2]):
   values=residual[:,h,node];valid=np.isfinite(values);n=int(valid.sum())
   if n<min_count:raise ValueError('insufficient observed calibration residuals in a coordinate')
   output[valid,h,node]=norm.ppf((rankdata(values[valid],method='average')-.5)/n)
 return output

def pairwise_correlation(values,min_pairs=100):
 values=np.asarray(values,float)
 if values.ndim!=2 or values.shape[1]<1:raise ValueError('requires rows x variables')
 mask=np.isfinite(values).astype(float);x=np.where(np.isfinite(values),values,0.);counts=mask.T@mask
 if np.any(counts<min_pairs):raise ValueError('insufficient jointly observed pairs')
 sums=x.T@mask;squares=(x*x).T@mask;cross=x.T@x;variance=squares-sums*sums/counts
 if np.any(variance<=0):raise ValueError('zero variance in a jointly observed pair')
 corr=(cross-sums*sums.T/counts)/np.sqrt(variance*variance.T);corr=np.clip((corr+corr.T)/2,-1,1);np.fill_diagonal(corr,1.)
 return corr,counts.astype(np.int64)

def repair_correlation(correlation,shrink=.05):
 r=np.asarray(correlation,float)
 if r.ndim!=2 or r.shape[0]!=r.shape[1] or not np.isfinite(r).all() or not 0<=shrink<=1:raise ValueError('invalid correlation matrix or shrinkage')
 r=(r+r.T)/2;values,vectors=np.linalg.eigh(r);fixed=(vectors*np.maximum(values,1e-6))@vectors.T;sd=np.sqrt(np.diag(fixed));fixed=fixed/sd[:,None]/sd[None,:];fixed=(fixed+fixed.T)/2;np.fill_diagonal(fixed,1.);out=(1-shrink)*fixed+shrink*np.eye(len(r));np.linalg.cholesky(out)
 return out,dict(shrink=float(shrink),minimum_eigenvalue_before=float(values.min()),minimum_eigenvalue_repaired=float(np.linalg.eigvalsh(fixed).min()),minimum_eigenvalue_final=float(np.linalg.eigvalsh(out).min()),repair_frobenius=float(np.linalg.norm(fixed-r)),total_change_frobenius=float(np.linalg.norm(out-r)))

def correlated_noise(base,time_correlation,space_correlation):
 base=np.asarray(base,float);t=np.asarray(time_correlation,float);s=np.asarray(space_correlation,float)
 if base.ndim!=3 or t.shape!=(base.shape[1],base.shape[1]) or s.shape!=(base.shape[2],base.shape[2]) or not np.isfinite(base).all():raise ValueError('noise and correlation dimensions do not align')
 return np.linalg.cholesky(t)@base@np.linalg.cholesky(s).T

def make_ranks(base,time_correlation,space_correlation):
 z=correlated_noise(base,time_correlation,space_correlation);return np.argsort(np.argsort(z,axis=0,kind='stable'),axis=0,kind='stable').astype(np.int16)

def fit_dependence(cal_q,cal_y,scale,shrink=.05):
 q=np.asarray(cal_q,float);y=np.asarray(cal_y,float)
 if q.shape[:-1]!=y.shape or y.ndim!=3 or y.shape[1]!=72 or q.shape[-1]!=19 or not np.isfinite(q).all() or np.any(np.diff(q,axis=-1)<0) or not np.isfinite(scale) or scale<=0:raise ValueError('invalid calibration distribution/truth/scale')
 residual=(y-q[...,9])/np.maximum(q[...,17]-q[...,1],.1*scale);z=normal_scores(residual);space,space_counts=pairwise_correlation(z.reshape(-1,y.shape[-1]));time,time_counts=pairwise_correlation(z.transpose(0,2,1).reshape(-1,72));sr,si=repair_correlation(space,shrink);tr,ti=repair_correlation(time,shrink)
 return dict(normal_scores=z,residual=residual,space_raw=space,time_raw=time,space_counts=space_counts,time_counts=time_counts,space_correlation=sr,time_correlation=tr,space_info=si,time_info=ti)

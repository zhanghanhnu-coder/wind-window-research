"""Controlled past-only single-sensor outages; target-only forecast products."""
import numpy as np
import torch
from probability_core import crps,quantile,longest,first_start
from lead_calibration_core import apply_groups
from crps_distribution_core import couple_atoms

def mask_history(histories,targets,length):
 x=np.asarray(histories);targets=np.asarray(targets)
 if x.ndim!=4 or x.shape[1]!=24 or x.shape[-1]!=3 or not np.isfinite(x).all():raise ValueError('finite batch/24h/node/3 input required')
 if isinstance(length,(bool,np.bool_)) or not isinstance(length,(int,np.integer)) or not 0<=length<=24:raise ValueError('invalid outage hours')
 if targets.shape!=(len(x),) or not np.issubdtype(targets.dtype,np.integer) or (targets<0).any() or (targets>=x.shape[2]).any():raise ValueError('invalid target nodes')
 z=x.copy()
 if length:z[np.arange(len(x))[:,None],np.arange(24-length,24)[None,:],targets[:,None],:]=0
 return z

def eligible_cases(histories):
 x=np.asarray(histories);valid=x[...,2];full=(valid==1).all(axis=1);remaining=(valid.sum(axis=(1,2))[:,None]-valid.sum(axis=1))/(24*x.shape[2]);return full&(remaining>=.65)

def target_predict(net,histories,targets,length,batch=64):
 x=mask_history(histories,targets,length);targets=np.asarray(targets);out=[];net.eval()
 with torch.no_grad():
  for a in range(0,len(x),batch):
   q=net(torch.from_numpy(x[a:a+batch]));j=torch.as_tensor(targets[a:a+batch],dtype=torch.long);out.append(q[torch.arange(len(q)),:,j,:].numpy())
 return np.concatenate(out)

def target_paths(q,bank,node,ordinary):
 if ordinary:return np.maximum(0,q[:,1]+bank['residual'][:,:,node])
 atoms=apply_groups(q[:,None,:],bank['theta'],bank['scale'],True);return couple_atoms(atoms,bank['ranks'][:,:,node:node+1])[...,0]

def node_scores(scenarios,truth):
 sc=np.asarray(scenarios);y=np.asarray(truth)
 if sc.ndim!=2 or sc.shape[1]!=72 or y.shape!=(72,) or not np.isfinite(sc).all():raise ValueError('72-hour target paths required')
 valid=np.isfinite(y);lo=quantile(sc,.1);hi=quantile(sc,.9);r=dict(crps=crps(sc,y),coverage=np.where(valid,(y>=lo)&(y<=hi),np.nan),width=np.where(valid,hi-lo,np.nan),low_miss=np.where(valid,y<lo,np.nan),high_miss=np.where(valid,y>hi,np.nan));probs=[];labels=[]
 for off in [0,24,48,66]:
  yy=y[off:off+6];ok=np.isfinite(yy).all();probs.append((sc[:,off:off+6]<=10).all(1).mean() if ok else np.nan);labels.append(float((yy<=10).all()) if ok else np.nan)
 r['prob6']=np.array(probs);r['label6']=np.array(labels);r['brier6']=(r['prob6']-r['label6'])**2;length=longest((sc<=10)[...,None])[:,0];actual=longest((y<=10)[None,:,None])[0,0];r['length_crps72']=np.asarray(crps(length,actual) if valid.all() else np.nan);fs=first_start((sc<=10)[...,None])[:,0];ys=first_start((y<=10)[None,:,None])[0,0];cdf=(fs[:,None]<=np.arange(67)).mean(0);r['first_cdf_brier']=np.asarray(np.mean((cdf-(ys<=np.arange(67)))**2) if valid.all() else np.nan);return r

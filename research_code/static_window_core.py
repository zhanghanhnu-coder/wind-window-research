"""Frozen October calibration for four direct-window probabilities only."""
import numpy as np
from scipy.special import expit
from rolling_core import fit_map
OFFSETS=np.array([0,24,48,66])

def select_primary(p,offsets):
 offsets=np.asarray(offsets);positions=[]
 for a in OFFSETS:
  found=np.flatnonzero(offsets==a)
  if len(found)!=1:raise ValueError('each primary offset must exist exactly once')
  positions.append(found[0])
 return np.asarray(p)[:,positions]

def matched_positions(ids,selected):
 ids=np.asarray(ids);selected=np.asarray(selected)
 if len(np.unique(ids))!=len(ids) or len(np.unique(selected))!=len(selected):raise ValueError('duplicate origin')
 lookup={int(v):i for i,v in enumerate(ids)}
 if not all(int(v) in lookup for v in selected):raise ValueError('origin absent')
 return np.array([lookup[int(v)] for v in selected])

def logit(p):
 p=np.clip(np.asarray(p,float),1e-6,1-1e-6);return np.log(p)-np.log1p(-p)

def fit_calibration(cal_probability,cal_labels,by_offset=False):
 p=np.asarray(cal_probability);y=np.asarray(cal_labels)
 if p.shape!=y.shape or p.ndim!=3 or p.shape[1]!=4:raise ValueError('require origin x four offsets x nodes')
 if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('invalid predicted probability')
 theta=[];infos=[]
 for idx in ([slice(j,j+1) for j in range(4)] if by_offset else [slice(None)]):
  labels=y[:,idx];valid=np.isfinite(labels);x=logit(p[:,idx]);norig=int(valid.any(axis=(1,2)).sum())
  t,info=fit_map(x[valid],labels[valid],norig);theta.append(t);infos.append(info)
 return np.array(theta),infos

def apply_calibration(probability,theta):
 p=np.asarray(probability);theta=np.asarray(theta)
 if p.ndim!=3 or p.shape[1]!=4 or theta.shape not in [(1,2),(4,2)]:raise ValueError('only four primary offsets supported')
 return expit(logit(p)*theta[None,:,0,None]+theta[None,:,1,None])

def frequency_baseline(labels):
 y=np.asarray(labels);return (np.nansum(y,axis=0)+1)/(np.isfinite(y).sum(axis=0)+2)

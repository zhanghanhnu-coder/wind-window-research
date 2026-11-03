"""Choose a scalar quantile-pooling weight from strictly matured forecast errors."""
import numpy as np
from residual_shape_core import BINS

GRID=np.array([0.,.25,.5,.75,1.])

def update_blocks(ids,hours):
 ids=np.asarray(ids)
 if ids.ndim!=1 or not len(ids) or not np.issubdtype(ids.dtype,np.integer) or (np.diff(ids)<=0).any() or hours<1:raise ValueError('chronological integer origins and positive period required')
 return (ids-ids[0])//hours

def crps(q,y):
 k=q.shape[-1];coef=(2*np.arange(1,k+1)-k-1)/(k*k)
 return np.mean(abs(q-y[...,None]),axis=-1)-np.sum(q*coef,axis=-1)

def blend(raw,calibrated,weights):
 raw=np.asarray(raw,float);cal=np.asarray(calibrated,float);w=np.asarray(weights,float)
 if raw.ndim!=4 or raw.shape!=cal.shape or raw.shape[1]!=72 or raw.shape[-1]!=19 or w.shape!=(4,) or not np.isfinite(w).all() or (w<0).any() or (w>1).any() or not np.isfinite(raw).all() or not np.isfinite(cal).all() or (np.diff(raw,axis=-1)<0).any() or (np.diff(cal,axis=-1)<0).any():raise ValueError('aligned finite ordered atoms and four unit weights required')
 out=raw.copy()
 for j,(a,b) in enumerate(BINS):out[:,a:b]=(1-w[j])*raw[:,a:b]+w[j]*cal[:,a:b]
 return out

def choose_weight(raw,calibrated,y,ids,issue,lookback,minimum_origins=20,minimum_labels=100):
 raw=np.asarray(raw,float);cal=np.asarray(calibrated,float);y=np.asarray(y,float);ids=np.asarray(ids)
 if raw.ndim!=4 or raw.shape[1]!=72 or raw.shape[-1]!=19 or cal.shape!=raw.shape or y.shape!=raw.shape[:-1] or ids.shape!=(len(raw),) or not np.issubdtype(ids.dtype,np.integer) or (np.diff(ids)<=0).any() or not isinstance(issue,(int,np.integer)) or lookback<1 or minimum_origins<1 or minimum_labels<1:raise ValueError('aligned forecasts/labels and chronological integer times required')
 t=ids[:,None,None]+np.arange(1,73)[None,:,None];eligible=(t<issue)&(t>=issue-lookback)
 weights=[];losses=[];audit=[]
 for a,b in BINS:
  mask=np.isfinite(y[:,a:b])&eligible[:,a:b];counts=mask.sum((1,2));n=int((counts>0).sum());labels=int(mask.sum());fallback=n<minimum_origins or labels<minimum_labels
  ts=np.broadcast_to(t[:,a:b],mask.shape)[mask]
  row=dict(lead_start=a+1,lead_end=b,origins=n,labels=labels,fallback=bool(fallback),earliest_label=int(ts.min()) if labels else None,latest_label=int(ts.max()) if labels else None,origin_weight_sum=1. if n else 0.)
  if fallback:w=.5;scores=np.full(5,np.nan)
  else:
   r=raw[:,a:b][mask];c=cal[:,a:b][mask];yy=y[:,a:b][mask]
   if not np.isfinite(r).all() or not np.isfinite(c).all() or (np.diff(r,axis=-1)<0).any() or (np.diff(c,axis=-1)<0).any():raise ValueError('mature forecasts must be finite ordered atoms')
   origin_weight=np.divide(1.,n*counts,out=np.zeros(len(counts)),where=counts>0)
   ww=np.broadcast_to(origin_weight[:,None,None],mask.shape)[mask]
   scores=np.array([np.sum(ww*crps((1-g)*r+g*c,yy)) for g in GRID]);ties=np.flatnonzero(scores<=scores.min()+1e-12)
   chosen=min(ties,key=lambda i:(abs(GRID[i]-.5),GRID[i]));w=float(GRID[chosen])
  weights.append(w);losses.append(scores);row['weight']=w;row.update({f'loss_{i}':float(v) if np.isfinite(v) else None for i,v in enumerate(scores)});audit.append(row)
 return dict(weights=np.array(weights),losses=np.stack(losses),audit=audit)

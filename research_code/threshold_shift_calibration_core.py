"""Past-only, origin-balanced threshold Brier shifts of ordered wind atoms."""
import numpy as np
BINS=[(0,6),(6,24),(24,48),(48,72)]
GRID=np.arange(-2,4.0001,.25)

def fit_shift(q,y,ids,issue,scale,minimum_origins=20,minimum_labels=100,lookback=672):
 q=np.asarray(q,float);y=np.asarray(y,float);ids=np.asarray(ids)
 if q.ndim!=4 or q.shape[:-1]!=y.shape or q.shape[1]!=72 or q.shape[-1]!=19 or ids.shape!=(len(q),) or not np.issubdtype(ids.dtype,np.integer) or (np.diff(ids)<=0).any() or not np.isfinite(scale) or scale<=0:raise ValueError('aligned forecasts, chronological IDs and positive scale required')
 if minimum_origins<1 or minimum_labels<1 or lookback<1:raise ValueError('positive support requirements needed')
 label_ids=ids[:,None,None]+np.arange(1,73)[None,:,None];time_ok=(label_ids<issue)&(label_ids>=issue-lookback)
 deltas=[];objectives=[];audits=[]
 for a,b in BINS:
  valid=np.isfinite(y[:,a:b])&time_ok[:,a:b];counts=valid.sum((1,2));use=counts>0;n=int(use.sum());nlabel=int(valid.sum());fallback=n<minimum_origins or nlabel<minimum_labels
  record=dict(lead_start=a+1,lead_end=b,origins=n,labels=nlabel,fallback=bool(fallback),earliest_label=int(np.broadcast_to(label_ids[:,a:b],valid.shape)[valid].min()) if nlabel else None,latest_label=int(np.broadcast_to(label_ids[:,a:b],valid.shape)[valid].max()) if nlabel else None)
  if fallback:delta=0.;loss=np.full(len(GRID),np.nan)
  else:
   qq=q[:,a:b][valid];yy=(y[:,a:b][valid]<=10).astype(float)
   if not np.isfinite(qq).all() or (np.diff(qq,axis=-1)<0).any():raise ValueError('mature training forecasts must be finite and ordered')
   perorigin=np.zeros(len(q));perorigin[use]=1/(n*counts[use]);weight=np.broadcast_to(perorigin[:,None,None],valid.shape)[valid];assert abs(weight.sum()-1)<1e-12
   loss=np.array([np.dot(weight,((qq<=10-d).mean(-1)-yy)**2)+.002*(d/scale)**2 for d in GRID]);best=np.flatnonzero(loss<=loss.min()+1e-12);index=min(best,key=lambda i:(abs(GRID[i]),GRID[i]));delta=float(GRID[index])
  deltas.append(delta);objectives.append(loss);audits.append(record)
 return dict(delta=np.array(deltas),grid=GRID.copy(),objectives=np.stack(objectives),audit=audits)

def apply_shift(q,delta):
 q=np.asarray(q,float);delta=np.asarray(delta,float)
 if q.ndim!=4 or q.shape[1]!=72 or q.shape[-1]!=19 or delta.shape!=(4,) or not np.isfinite(q).all() or not np.isfinite(delta).all():raise ValueError('finite 72h atoms and four shifts required')
 out=q.copy()
 for (a,b),d in zip(BINS,delta):out[:,a:b]=np.maximum(0,q[:,a:b]+d)
 return out

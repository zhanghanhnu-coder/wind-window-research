"""Past-observed wind state and causal nearest-origin calibration selection."""
import numpy as np
from residual_shape_core import BINS
from calibration_gate_core import GRID,crps

def state_features(raw,ids,r,w,center,scale):
 raw=np.asarray(raw,float);ids=np.asarray(ids);r=np.asarray(r,float);w=np.asarray(w,float)
 if raw.ndim!=2 or ids.ndim!=1 or (ids<23).any() or (ids>=len(raw)).any() or r.shape!=w.shape or r.shape!=(len(ids),72,raw.shape[1],19) or scale<=0:raise ValueError('aligned forecast, raw timeline, and valid origins required')
 out=np.empty((len(ids),4,6),float)
 for i,t in enumerate(ids):
  recent=raw[t-5:t+1];previous=raw[t-11:t-5];day=raw[t-23:t+1]
  mean=lambda a:float(np.mean(a[np.isfinite(a)])) if np.isfinite(a).any() else float(center)
  obs=[(mean(recent)-center)/scale,(mean(recent)-mean(previous))/scale,float(np.std(day[np.isfinite(day)]))/scale if np.isfinite(day).any() else 0.,float(np.isfinite(recent).mean())]
  for j,(a,b) in enumerate(BINS):
   rr=r[i,a:b];ww=w[i,a:b];out[i,j,:4]=obs;out[i,j,4]=np.mean(rr[...,9]-ww[...,9])/scale
   out[i,j,5]=np.log((np.mean(ww[...,17]-ww[...,1])+.1*scale)/(np.mean(rr[...,17]-rr[...,1])+.1*scale))
 if not np.isfinite(out).all():raise ValueError('finite causal features required')
 return out

def select_state_weight(r,w,y,ids,features,current,issue,dimensions,neighbors=40,minimum_origins=20,minimum_labels=100):
 r=np.asarray(r,float);w=np.asarray(w,float);y=np.asarray(y,float);ids=np.asarray(ids);features=np.asarray(features,float);current=np.asarray(current,float)
 if dimensions not in (4,6) or r.shape!=w.shape or r.shape[:-1]!=y.shape or r.ndim!=4 or r.shape[1]!=72 or r.shape[-1]!=19 or ids.shape!=(len(r),) or (np.diff(ids)<=0).any() or features.shape!=(len(r),4,6) or current.shape!=(4,6) or not np.isfinite(current).all() or neighbors<minimum_origins:raise ValueError('aligned histories, supported dimensions and current features required')
 target=ids[:,None,None]+np.arange(1,73)[None,:,None];eligible=(target<issue)&(target>=issue-672)
 weights=[];losses=[];selected=[];audits=[]
 for j,(a,b) in enumerate(BINS):
  mask=np.isfinite(y[:,a:b])&eligible[:,a:b];counts=mask.sum((1,2));valid=np.flatnonzero(counts>0);fallback=len(valid)<minimum_origins or counts.sum()<minimum_labels
  chosen=np.array([],int);risk=np.full(5,np.nan);iqr=np.full(dimensions,np.nan)
  if not fallback:
   f=features[valid,j,:dimensions]
   if not np.isfinite(f).all():raise ValueError('eligible historical features must be finite')
   iqr=np.maximum(np.quantile(f,.75,axis=0)-np.quantile(f,.25,axis=0),.1)
   distances=np.square(np.clip((f-current[j,:dimensions])/iqr,-3,3)).sum(1)
   chosen=valid[np.lexsort((-ids[valid],distances))[:neighbors]]
   fallback=counts[chosen].sum()<minimum_labels
  if not fallback:
   m=mask[chosen];rr=r[chosen,a:b][m];ww=w[chosen,a:b][m];yy=y[chosen,a:b][m]
   if not np.isfinite(rr).all() or not np.isfinite(ww).all() or (np.diff(rr,axis=-1)<0).any() or (np.diff(ww,axis=-1)<0).any():raise ValueError('mature forecasts must be ordered finite atoms')
   ow=np.broadcast_to((1/(len(chosen)*counts[chosen]))[:,None,None],m.shape)[m]
   risk=np.array([np.sum(ow*crps((1-g)*rr+g*ww,yy)) for g in GRID]);ties=np.flatnonzero(risk<=risk.min()+1e-12);weight=float(GRID[min(ties,key=lambda k:(abs(GRID[k]-.5),GRID[k]))])
  else:weight=.5
  times=np.broadcast_to(target[:,a:b],mask.shape)[chosen][mask[chosen]] if len(chosen) else np.array([],int)
  weights.append(weight);losses.append(risk);selected.append(chosen)
  audits.append(dict(lead_start=a+1,lead_end=b,eligible_origins=len(valid),selected_origins=len(chosen),selected_labels=int(counts[chosen].sum()),fallback=bool(fallback),earliest_label=int(times.min()) if len(times) else None,latest_label=int(times.max()) if len(times) else None,weight=weight,estimated_harm=float(risk[4]-risk[2]) if not fallback else None))
 return dict(weights=np.array(weights),losses=np.stack(losses),selected=selected,audit=audits)

def decision_counts(p,y,threshold):
 p=np.asarray(p,float);y=np.asarray(y,float)
 if p.shape!=y.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.isin(y[np.isfinite(y)],[0,1]).all():raise ValueError('aligned probabilities and binary/missing labels required')
 accept=p>=threshold-1e-12;known=np.isfinite(y);tp=int((accept&(y==1)).sum());fp=int((accept&(y==0)).sum());fn=int((~accept&(y==1)).sum());tn=int((~accept&(y==0)).sum());ua=int((accept&~known).sum());ur=int((~accept&~known).sum());n=p.size;na=int(accept.sum())
 div=lambda x,z:float(x/z) if z else float('nan')
 return dict(n=n,accepted=na,tp=tp,fp=fp,fn=fn,tn=tn,unknown_accepted=ua,unknown_rejected=ur,coverage=div(na,n),precision_lower=div(tp,na),precision_upper=div(tp+ua,na),missed_success_lower=div(fn,tp+fn+ua),missed_success_upper=div(fn+ur,tp+fn+ur),fp_fraction_lower=div(fp,n),fp_fraction_upper=div(fp+ua,n),fn_fraction_lower=div(fn,n),fn_fraction_upper=div(fn+ur,n))

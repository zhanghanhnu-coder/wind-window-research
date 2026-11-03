"""Four-weight composite distribution/window score, using exact active-face solves."""
import numpy as np
from probability_pooling_core import checked_weights,pool_objective
BINS=[(0,6),(6,24),(24,48),(48,72)]

def event_mask(labels,ids,issue,kind):
 labels=np.asarray(labels);ids=np.asarray(ids)
 if kind=='single' and labels.shape==(len(ids),4,17):ends=ids[:,None,None]+np.array([6,30,54,72])[None,:,None];source=ids[:,None,None]
 elif kind=='group' and labels.shape==(len(ids),12):ends=ids[:,None]+np.repeat([6,30,54,72],3)[None,:];source=ids[:,None]
 else:raise ValueError('invalid event layout')
 if not np.issubdtype(ids.dtype,np.integer):raise ValueError('integer issue indices required')
 return np.isfinite(labels)&(source<issue)&(ends<issue)&(ends>=issue-672)

def event_statistics(prob,labels,ids,issue,kind):
 prob=np.asarray(prob);labels=np.asarray(labels);ids=np.asarray(ids);mask=event_mask(labels,ids,issue,kind)
 if prob.shape!=(4,*labels.shape):raise ValueError('four aligned event forecasts required')
 p=prob[:,mask].T.astype(float);y=labels[mask].astype(float)
 if not np.isfinite(p).all() or np.any(p< -1e-12) or np.any(p>1+1e-12) or not np.isin(y,[0,1]).all():raise ValueError('invalid mature event probabilities or labels')
 # Existing exact weighted probabilities can be a few ulps outside [0,1]; preserve values.
 if kind=='single':ends=np.broadcast_to(ids[:,None,None]+np.array([6,30,54,72])[None,:,None],labels.shape)
 else:ends=ids[:,None]+np.repeat([6,30,54,72],3)[None,:]
 n=len(y);source_positions=np.flatnonzero(mask.reshape(len(ids),-1).any(1));known=ends[mask]
 return dict(C=p.T@p/n if n else np.zeros((4,4)),d=p.T@y/n if n else np.zeros(4),e=float(np.mean(y*y)) if n else 0.,mask=mask,audit=dict(valid_events=n,valid_origins=len(source_positions),source_positions=source_positions.tolist(),source_ids=ids[source_positions].tolist(),earliest_end_id=int(known.min()) if n else None,latest_end_id=int(known.max()) if n else None))

def combined_moments(a,b,single,group,lam):
 if lam not in [0,.25,1,4]:raise ValueError('undeclared composite coefficient')
 return np.asarray(a)-lam*(single['d']+group['d']),np.asarray(b)-lam*(single['C']+group['C']),float(lam/2*(single['e']+group['e']))

def solve_simplex_exact(a,b,previous):
 a=np.asarray(a,float);b=np.asarray(b,float);previous=checked_weights(previous)
 if a.shape!=(4,) or b.shape!=(4,4) or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(b,b.T,rtol=0,atol=1e-12):raise ValueError('finite symmetric four-weight quadratic required')
 h=-b+.002*np.eye(4);c=a-.0005;tangent=np.vstack([np.eye(3),-np.ones(3)]);mineig=float(np.linalg.eigvalsh(tangent.T@h@tangent).min());faces=[];best=None
 if mineig>=-1e-9:
  for face in range(1,16):
   ix=np.flatnonzero([face&(1<<j) for j in range(4)]);n=len(ix);matrix=np.zeros((n+1,n+1));matrix[:n,:n]=h[np.ix_(ix,ix)];matrix[-1,:n]=1;matrix[:n,-1]=1
   try:solution=np.linalg.solve(matrix,np.r_[-c[ix],1])
   except np.linalg.LinAlgError:faces.append(dict(face=face,solved=False,feasible=False));continue
   w=np.zeros(4);w[ix]=solution[:n];feasible=bool(np.isfinite(w).all() and w.min()>=-1e-10 and abs(w.sum()-1)<=1e-10)
   if not feasible:faces.append(dict(face=face,solved=True,feasible=False));continue
   w=np.maximum(w,0);w/=w.sum();value,gradient=pool_objective(w,a,b);gap=float(gradient@w-gradient.min());faces.append(dict(face=face,solved=True,feasible=True,weights=w.tolist(),objective=value,gap=gap))
   # Near-equal objectives need not have equal stationarity residuals at a boundary.
   # Only globally certified faces may participate in the deterministic tie rule.
   if np.isfinite(value) and gap<=1e-8 and (best is None or value<best[0]-1e-12):best=value,w,gradient,gap,face
 success=bool(best is not None and np.isfinite(best[0]) and best[3]<=1e-8);used=best[1].copy() if success else previous.copy();reason='enumerated all nonempty active faces' if success else 'nonconvex tangent Hessian' if mineig< -1e-9 else 'no certified finite feasible face'
 return used,dict(success=success,fallback=not success,reason=reason,faces=faces,tangent_min_eigenvalue=mineig,candidate_weights=best[1].tolist() if best else None,candidate_gradient=best[2].tolist() if best else None,candidate_gap=best[3] if best else None,candidate_objective=best[0] if best else None,selected_face=best[4] if best else None,used_objective=pool_objective(used,a,b)[0])

def fit_window_at(a,b,hour_mask,ids,issue,previous,single_prob,single_labels,group_prob,group_labels,lam):
 previous=checked_weights(previous);ids=np.asarray(ids);hm=np.asarray(hour_mask)
 if hm.shape!=(len(ids),72,17) or hm.dtype!=bool:raise ValueError('aligned mature hourly mask required')
 target=ids[:,None]+np.arange(1,73);legal=(ids[:,None]<issue)&(target<issue)&(target>=issue-672)
 if (hm&~legal[:,:,None]).any():raise ValueError('hourly mask contains immature or expired labels')
 single=event_statistics(single_prob,single_labels,ids,issue,'single');group=event_statistics(group_prob,group_labels,ids,issue,'group');hour_origins=int(hm.any((1,2)).sum());hour_counts=[int(hm[:,lo:hi].sum()) for lo,hi in BINS];hours_enough=hour_origins>=20 and min(hour_counts)>=100;events_enough=single['audit']['valid_events']>=100 and group['audit']['valid_events']>=50;support=hours_enough and (lam==0 or events_enough)
 audit=dict(issue_id=int(issue),lambda_value=lam,hour_origins=hour_origins,hour_labels=int(hm.sum()),hour_group_counts=hour_counts,single=single['audit'],group=group['audit'],hour_support=hours_enough,event_support=events_enough,insufficient_support=not support,optimizer_called=support)
 aa,bb,constant=combined_moments(a,b,single,group,lam)
 if support:w,details=solve_simplex_exact(aa,bb,previous)
 else:w=previous.copy();details=dict(success=False,fallback=True,reason='insufficient mature support; retain previous weights',used_objective=None)
 details['constant']=constant;details['full_used_objective']=details['used_objective']+constant if details['used_objective'] is not None else None
 return dict(weights=w,a=aa,b=bb,single=single,group=group,audit=audit,details=details)

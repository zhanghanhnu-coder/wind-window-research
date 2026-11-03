"""Sequential calibration with explicit label availability; no model training."""
import numpy as np
from scipy.special import expit,softmax
from scipy.optimize import minimize

OFFSETS=np.array([0,24,48,66])

def mature_paths(ids,complete,issue,days=None,min_count=8):
 ids=np.asarray(ids);end=ids+72;eligible=(end<=issue-1)&np.asarray(complete)
 chosen=eligible.copy()
 if days is not None:chosen&=end>=issue-days*24
 if chosen.sum()<min_count:chosen=eligible
 return np.flatnonzero(chosen)

def event_matrix(sc):
 return np.stack([(sc[:,a:a+6]<=10).all(1) for a in OFFSETS],axis=1).reshape(len(sc),-1).astype(float)

def logits(p,sizes):
 p=np.asarray(p,float);epsilon=.5/(np.asarray(sizes,float)+1)
 q=np.clip(p,epsilon,1-epsilon);return np.log(q)-np.log1p(-q)

def map_probability(p,size,theta):return expit(theta[0]*logits(p,size)+theta[1])

def mature_events(ids,probs,labels,sizes,issue,min_issue,days=28):
 ids=np.asarray(ids);probs=np.asarray(probs);labels=np.asarray(labels);sizes=np.asarray(sizes)
 if len(ids)==0:return np.array([]),np.array([]),0,None
 ends=ids[:,None,None]+OFFSETS[None,:,None]+6
 mask=(ends<=issue-1)&(ids[:,None,None]>=max(min_issue,issue-days*24))&(ids[:,None,None]<issue)
 mask=np.broadcast_to(mask,labels.shape)&np.isfinite(labels)
 x=logits(probs,sizes[:,None,None]);norig=int(mask.any(axis=(1,2)).sum())
 latest=int(np.broadcast_to(ends,labels.shape)[mask].max()) if mask.any() else None
 return x[mask],labels[mask],norig,latest

def fit_map(x,y,norig):
 x=np.asarray(x);y=np.asarray(y)
 if len(y)<256 or norig<12:return np.array([1.,0.]),dict(success=True,fallback=True,reason='insufficient_history',n=len(y),origins=norig)
 def objective(theta):
  a,b=theta;z=a*x+b;res=expit(z)-y
  f=np.mean(np.logaddexp(0,z)-y*z)+.01*((a-1)**2+b*b)
  grad=np.array([np.mean(res*x)+.02*(a-1),np.mean(res)+.02*b])
  return f,grad
 result=minimize(objective,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(.25,4),(-3,3)],options={'maxiter':200,'ftol':1e-12,'gtol':1e-9})
 success=bool(result.success and np.isfinite(result.fun));theta=result.x if success else np.array([1.,0.])
 return theta,dict(success=success,fallback=not success,reason=str(result.message),n=len(y),origins=norig,objective=float(result.fun),iterations=int(result.nit))

def tilt_objective(z,E,target):
 q=softmax(z);u=np.full(len(q),1/len(q));w=.25*u+.75*q;err=E.T@w-target
 loss=.5*np.mean(err*err)+.01*np.sum(w*np.log(w/u))
 derivative=E@err/len(target)+.01*(np.log(w/u)+1)
 grad=.75*q*(derivative-np.dot(derivative,q))
 return loss,grad

def tilt_weights(E,target):
 E=np.asarray(E,float);target=np.asarray(target,float).reshape(-1);u=np.full(len(E),1/len(E))
 result=minimize(tilt_objective,np.zeros(len(E)),args=(E,target),jac=True,method='L-BFGS-B',bounds=[(-10,10)]*len(E),options={'maxiter':200,'ftol':1e-12,'gtol':1e-9})
 success=bool(result.success and np.isfinite(result.fun));w=.25*u+.75*softmax(result.x) if success else u
 return w,dict(success=success,iterations=int(result.nit),objective=float(result.fun),gap_before=float(np.sqrt(np.mean((E.T@u-target)**2))),gap_after=float(np.sqrt(np.mean((E.T@w-target)**2))),ess=float(1/(w@w)),impossible_fraction=float(np.mean((E.min(0)==E.max(0))&(abs(E[0]-target)>1e-8))))

"""Ordered finite distributions trained by exact empirical CRPS, plus rank coupling."""
import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize
from direct_window_core import WindowNet

class AtomNet(WindowNet):
 def __init__(self,adj,scale,hidden=24,gated=True,atoms=19):
  super().__init__(adj,hidden,gated);self.atoms=atoms;self.register_buffer('scale',torch.tensor(float(scale),dtype=torch.float32));self.head=nn.Linear(hidden,72*atoms)
 def forward(self,x):
  b,t,n,c=x.shape;h=self.encode(x);z=self.head(h[...,-1]).reshape(b,n,72,self.atoms).transpose(1,2)
  return torch.cumsum(nn.functional.softplus(z),dim=-1)*(.25*self.scale)

def atom_crps(q,y):
 k=q.shape[-1];coef=(2*torch.arange(1,k+1,dtype=q.dtype,device=q.device)-k-1)/(k*k)
 return torch.mean(abs(q-y[...,None]),dim=-1)-torch.sum(coef*q,dim=-1)

def masked_crps(q,y):
 valid=torch.isfinite(y);loss=atom_crps(q,torch.nan_to_num(y));return (loss*valid).sum()/valid.sum().clamp_min(1)

def predict_atoms(net,x,ids,batch=32):
 net.eval();out=[]
 with torch.no_grad():
  for start in range(0,len(ids),batch):out.append(net(torch.from_numpy(np.stack([x[i-23:i+1] for i in ids[start:start+batch]]))).numpy())
 return np.concatenate(out)

def transform_atoms(q,theta,scale):
 q=np.asarray(q,float);med=q[...,q.shape[-1]//2,None];return np.maximum(0,med+theta[0]*scale+theta[1]*(q-med))

def affine_objective(theta,q,y):
 """q/y are dimensionless, valid rows only; clipping derivative zero at its kink."""
 q=np.asarray(q,float);y=np.asarray(y,float);k=q.shape[-1];med=q[:,k//2,None];delta=q-med;z=med+theta[0]+theta[1]*delta;v=np.maximum(0,z);coef=(2*np.arange(1,k+1)-k-1)/(k*k)
 value=np.mean(np.mean(abs(v-y[:,None]),axis=-1)-np.sum(coef*v,axis=-1))+.001*(theta[0]**2+(theta[1]-1)**2)
 dv=(np.sign(v-y[:,None])/k-coef)*(z>0)/len(y);grad=np.array([dv.sum(),np.sum(dv*delta)])+.002*np.array([theta[0],theta[1]-1])
 return value,grad

def fit_affine(cal_atoms,cal_truth,scale):
 valid=np.isfinite(cal_truth);q=np.asarray(cal_atoms,float)[valid]/scale;y=np.asarray(cal_truth,float)[valid]/scale
 result=minimize(affine_objective,[0.,1.],args=(q,y),jac=True,method='L-BFGS-B',bounds=[(-2,2),(.25,4)],options={'maxiter':200,'ftol':1e-10,'gtol':1e-7})
 success=bool(result.success and np.isfinite(result.fun));theta=result.x if success else np.array([0.,1.]);deployed=affine_objective(theta,q,y)[0]
 return theta,dict(success=success,fallback=not success,reason=str(result.message),iterations=int(result.nit),objective_candidate=float(result.fun),objective_used=float(deployed),valid_targets=int(valid.sum()),delta=float(theta[0]),gamma=float(theta[1]))

def rank_template(residual):
 residual=np.asarray(residual)
 if not np.isfinite(residual).all():raise ValueError('complete finite residual paths required')
 return np.argsort(np.argsort(residual,axis=0,kind='stable'),axis=0,kind='stable')

def couple_atoms(q,ranks):
 q=np.asarray(q);ranks=np.asarray(ranks);k=q.shape[-1];s=len(ranks)
 if s%k or ranks.shape[1:]!=q.shape[:-1]:raise ValueError('scenario count must be a multiple of atom count; coordinates must align')
 values=np.repeat(q,s//k,axis=-1).transpose(2,0,1);return np.take_along_axis(values,ranks,axis=0)

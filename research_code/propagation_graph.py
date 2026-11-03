"""Exploratory lead/lag associations, never a claim of physical causation.

Positive pair lag: x leads y. Tensor convention: [lag, receiver, source].
Missing timestamps stay in place. All fitted graphs use Jan-Aug only.
"""
import numpy as np
import torch
from model import Net

def difference(x):
 return np.concatenate([np.full((1,)+x.shape[1:],np.nan),np.diff(x,axis=0)],axis=0)

def circular_mean(degrees):
 v=np.asarray(degrees);v=v[np.isfinite(v)]
 if not len(v):return np.nan,0.
 z=np.exp(1j*np.deg2rad(v)).mean()
 return float(np.rad2deg(np.angle(z))%360),float(abs(z))

def lag_profile(x,y,max_lag=6,method='corr',minimum=100):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
 if x.shape!=y.shape or x.ndim!=1:raise ValueError('aligned one-dimensional series required')
 lags=np.arange(-max_lag,max_lag+1)
 if len(x)<=2*max_lag:return lags,np.zeros(len(lags)),0
 ix=np.arange(max_lag,len(x)-max_lag);ys=np.stack([y[ix+l] for l in lags])
 valid=np.isfinite(x[ix])&np.isfinite(ys).all(axis=0)
 a=x[ix][valid];b=ys[:,valid];n=len(a);scores=np.zeros(len(lags))
 if n<minimum or np.std(a)<1e-12:return lags,scores,n
 if method=='corr':
  aa=a-a.mean();bb=b-b.mean(axis=1,keepdims=True);den=np.sqrt((aa*aa).sum()*(bb*bb).sum(axis=1))
  scores=np.divide(bb@aa,den,out=scores,where=den>1e-12)
 elif method=='mi':
  # Eight training-window quantile bins. Common samples for every searched lag.
  ex=np.unique(np.quantile(x[np.isfinite(x)],np.arange(1,8)/8))
  ey=np.unique(np.quantile(y[np.isfinite(y)],np.arange(1,8)/8))
  ax=np.searchsorted(ex,a);by=np.searchsorted(ey,b);nx=len(ex)+1;ny=len(ey)+1
  for k in range(len(lags)):
   p=np.bincount(ax*ny+by[k],minlength=nx*ny).reshape(nx,ny)/n
   px=p.sum(1);py=p.sum(0);expected=px[:,None]*py[None,:];nz=p>0
   hx=-np.sum(px[px>0]*np.log(px[px>0]));hy=-np.sum(py[py>0]*np.log(py[py>0]))
   if hx*hy>1e-12:scores[k]=np.sum(p[nz]*np.log(p[nz]/expected[nz]))/np.sqrt(hx*hy)
 else:raise ValueError('unknown association method')
 return lags,scores,n

def best_lag(x,y,max_lag=6,method='corr',minimum=100):
 lags,scores,n=lag_profile(x,y,max_lag,method,minimum)
 best=float(max(0,np.max(scores)))
 # Ties within numerical tolerance prefer synchronization, then shorter delays.
 tied=np.flatnonzero(scores>=best-1e-12)
 if not len(tied):return 0,0.,n
 k=min(tied,key=lambda i:(abs(lags[i]),lags[i]))
 return int(lags[k]),best,n

def dtw(x,y,band=6):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
 if not np.isfinite(x).all() or not np.isfinite(y).all():return None
 n=len(x);m=len(y)
 if not n or not m or abs(n-m)>band:return None
 d=np.full((n+1,m+1),np.inf);d[0,0]=0.;parent=np.zeros((n+1,m+1),np.int8)
 for i in range(1,n+1):
  for j in range(max(1,i-band),min(m,i+band)+1):
   values=(d[i-1,j-1],d[i-1,j],d[i,j-1]);k=int(np.argmin(values))
   d[i,j]=(x[i-1]-y[j-1])**2+values[k];parent[i,j]=k
 if not np.isfinite(d[n,m]):return None
 i,j=n,m;path=[]
 while i>0 and j>0:
  path.append((i-1,j-1));k=parent[i,j]
  if k==0:i-=1;j-=1
  elif k==1:i-=1
  else:j-=1
 path.reverse();lag=float(np.median([b-a for a,b in path]))
 return float(np.sqrt(d[n,m]/len(path))),lag,path

def lag_adjacency(n,edges,max_lag=6,top_k=4):
 """Edges are already directed tuples (source, receiver, nonnegative lag, strength)."""
 a=np.zeros((max_lag+1,n,n),dtype=np.float32)
 for source,receiver,lag,weight in edges:
  if source==receiver or weight<=0:continue
  if not 0<=lag<=max_lag:raise ValueError('lag outside declared horizon')
  a[int(lag),receiver,source]+=weight
 for receiver in range(n):
  strengths=a[:,receiver,:].sum(0);keep=np.argsort(-strengths,kind='stable')[:top_k]
  discard=np.ones(n,bool);discard[keep]=False;a[:,receiver,discard]=0
  total=a[:,receiver,:].sum()
  if total>0:a[:,receiver,:]/=total
 return a

def directed_pair_edges(i,j,lag,weight):
 if lag>0:return [(i,j,int(lag),weight)]
 if lag<0:return [(j,i,int(-lag),weight)]
 return [(i,j,0,weight),(j,i,0,weight)]

def aggregate_lags(x,lag_adj):
 out=torch.zeros_like(x)
 for lag,a in enumerate(lag_adj):
  if lag>=x.shape[1]:break
  # A leading pad is unavailable normalized input; never circularly wrap the end.
  if lag==0:out=out+torch.einsum('ij,btjf->btif',a,x)
  else:out[:,lag:]=out[:,lag:]+torch.einsum('ij,btjf->btif',a,x[:,:-lag])
 return out

class LagNet(Net):
 def __init__(self,lag_adj,hidden=24,dilation=5):
  lag_adj=np.asarray(lag_adj,dtype=np.float32)
  super().__init__(lag_adj.sum(0),hidden,dilation)
  self.register_buffer('lag_adj',torch.as_tensor(lag_adj))
 def forward(self,x):
  h=torch.relu(self.self_in(x)+self.neighbor_in(aggregate_lags(x,self.lag_adj)))
  b,t,n,c=h.shape;h=h.permute(0,2,3,1).reshape(b*n,c,t)
  h=torch.relu(self.temporal[0](h)[...,:t]);h=torch.relu(self.temporal[2](h)[...,:t])
  z=self.head(h[...,-1]).reshape(b,n,72,3).permute(0,2,1,3)
  median=z[...,0]
  return torch.stack([median-torch.nn.functional.softplus(z[...,1]),median,median+torch.nn.functional.softplus(z[...,2])],-1)

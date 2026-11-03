"""Small, auditable temporal graph quantile model. No coordinates are assumed."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'研究笔记/基线实验_2026-11-01/严格完整小时风速.pkl'
ART=ROOT/'artifacts'; ART.mkdir(exist_ok=True)
torch.set_num_threads(4)

def load_data():
 with DATA.open('rb') as f:d=pickle.load(f)
 return d['mean'],d['max']

def inputs(mean,maximum,center,scale):
 a=np.stack([mean.to_numpy(),maximum.to_numpy()],axis=-1).astype('float32')
 mask=np.isfinite(a).all(-1,keepdims=True).astype('float32')
 return np.concatenate([np.nan_to_num((a-center)/scale,nan=0),mask],-1)

class Net(nn.Module):
 def __init__(self,adj,hidden=24,dilation=1):
  super().__init__(); self.register_buffer('adj',torch.as_tensor(adj,dtype=torch.float32))
  self.self_in=nn.Linear(3,hidden);self.neighbor_in=nn.Linear(3,hidden,bias=False)
  self.temporal=nn.Sequential(nn.Conv1d(hidden,hidden,5,padding=4),nn.ReLU(),nn.Conv1d(hidden,hidden,5,padding=4*dilation,dilation=dilation),nn.ReLU())
  self.head=nn.Linear(hidden,72*3)
 def forward(self,x):
  # x: batch,time,node,feature. Each convolution is cropped causally.
  h=torch.relu(self.self_in(x)+self.neighbor_in(torch.einsum('ij,btjf->btif',self.adj,x)))
  b,t,n,c=h.shape;h=h.permute(0,2,3,1).reshape(b*n,c,t)
  h=torch.relu(self.temporal[0](h)[...,:t]);h=torch.relu(self.temporal[2](h)[...,:t])
  z=self.head(h[...,-1]).reshape(b,n,72,3).permute(0,2,1,3)
  median=z[...,0];return torch.stack([median-torch.nn.functional.softplus(z[...,1]),median,median+torch.nn.functional.softplus(z[...,2])],-1)

def pinball(q,y):
 mask=torch.isfinite(y);e=torch.nan_to_num(y)[...,None]-q
 levels=torch.tensor([.1,.5,.9],device=q.device)
 return (torch.maximum(levels*e,(levels-1)*e)*mask[...,None]).sum()/(mask.sum()*3).clamp_min(1)

def origins(index,start,end,stride=6):
 # All outcomes in the declared period, including the 72-hour last outcome.
 a=pd.Timestamp(start,tz=index.tz);b=pd.Timestamp(end,tz=index.tz)
 return np.array([i for i in range(23,len(index)-72,stride) if index[i]>=a and index[i+72]<b],dtype=int)

def batches(x,y,ids,batch=32,shuffle=False,seed=0):
 ids=ids.copy()
 if shuffle:np.random.default_rng(seed).shuffle(ids)
 for a in range(0,len(ids),batch):
  ix=ids[a:a+batch]
  xx=np.stack([x[i-23:i+1] for i in ix]); yy=np.stack([y[i+1:i+73] for i in ix])
  yield torch.from_numpy(xx),torch.from_numpy(yy)

def predict(net,x,ids,batch=64):
 out=[];net.eval()
 with torch.no_grad():
  for a in range(0,len(ids),batch):
   xx=np.stack([x[i-23:i+1] for i in ids[a:a+batch]])
   out.append(net(torch.from_numpy(xx)).numpy())
 return np.concatenate(out)

"""Direct window classifiers; outputs do not define a joint weather distribution."""
import numpy as np
import torch
from torch import nn

PRIMARY=np.array([0,24,48,66])

def window_labels(y):
 y=np.asarray(y);return np.stack([np.where(np.isfinite(y[:,a:a+6]).all(1),(y[:,a:a+6]<=10).all(1),np.nan) for a in range(67)],axis=1).astype('float32')

def aggregate(adj,x):return torch.einsum('ij,btjf->btif',adj,x)

class WindowNet(nn.Module):
 def __init__(self,adj,hidden=24,gated=True):
  super().__init__();self.register_buffer('adj',torch.as_tensor(adj,dtype=torch.float32));self.gated=gated
  self.self_in=nn.Linear(3,hidden);self.neighbor_in=nn.Linear(3,hidden,bias=False)
  self.filters=nn.ModuleList([nn.Conv1d(hidden,hidden,5,padding=4),nn.Conv1d(hidden,hidden,5,padding=20,dilation=5)])
  if gated:self.gates=nn.ModuleList([nn.Conv1d(hidden,hidden,5,padding=4),nn.Conv1d(hidden,hidden,5,padding=20,dilation=5)])
  self.head=nn.Linear(hidden,67)
 def encode(self,x):
  h=torch.relu(self.self_in(x)+self.neighbor_in(aggregate(self.adj,x)));b,t,n,c=h.shape;h=h.permute(0,2,3,1).reshape(b*n,c,t)
  for i,layer in enumerate(self.filters):
   f=layer(h)[...,:t]
   h=torch.tanh(f)*torch.sigmoid(self.gates[i](h)[...,:t]) if self.gated else torch.relu(f)
  return h
 def forward(self,x):
  b,t,n,c=x.shape;h=self.encode(x);return self.head(h[...,-1]).reshape(b,n,67).transpose(1,2)

def masked_loss(logits,y,kind='bce'):
 valid=torch.isfinite(y);safe=torch.nan_to_num(y)
 if kind=='bce':loss=nn.functional.binary_cross_entropy_with_logits(logits,safe,reduction='none')
 elif kind=='brier':loss=(torch.sigmoid(logits)-safe)**2
 else:raise ValueError(kind)
 return (loss*valid).sum()/valid.sum().clamp_min(1)

def predict_windows(net,x,ids,batch=64):
 net.eval();out=[]
 with torch.no_grad():
  for a in range(0,len(ids),batch):
   xx=torch.from_numpy(np.stack([x[i-23:i+1] for i in ids[a:a+batch]]));out.append(torch.sigmoid(net(xx)).numpy())
 return np.concatenate(out)

def linear_features(x,ids):
 xs=np.stack([x[i-23:i+1] for i in ids]);b,t,n,f=xs.shape;valid=xs[...,2];den=valid.sum(2)
 mean=np.sum(xs[...,:2]*valid[...,None],axis=2)/np.maximum(den[...,None],1)
 global_x=np.concatenate([mean,(den/n)[...,None]],axis=-1);own=xs.transpose(0,2,1,3).reshape(b,n,t*f);global_x=np.broadcast_to(global_x.reshape(b,1,t*f),(b,n,t*f));onehot=np.broadcast_to(np.eye(n),(b,n,n))
 return np.concatenate([own,global_x,onehot],axis=-1).reshape(b*n,-1).astype('float32')

def persistence_prob(raw,ids,prior):
 values=[]
 for i in ids:
  history=raw[i-23:i+1];p=np.array(prior,dtype=float).copy()
  for j in range(raw.shape[1]):
   good=np.flatnonzero(np.isfinite(history[:,j]))
   if len(good):p[j]=float(history[good[-1],j]<=10)
  values.append(np.broadcast_to(p,(67,len(prior))).copy())
 return np.stack(values)

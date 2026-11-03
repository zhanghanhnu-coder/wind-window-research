"""Shared-encoder multiscale histories and two causal empirical baselines."""
import numpy as np
import torch
from torch import nn
from crps_distribution_core import AtomNet
LEVELS=(np.arange(19)+.5)/19

class MultiScaleAtomNet(AtomNet):
 def __init__(self,adj,scale,span,hidden=24):
  if span not in [1,3,7]:raise ValueError('undeclared temporal scale')
  super().__init__(adj,scale,hidden=hidden,gated=True);self.span=span
 def forward(self,x,coarse=None):
  if self.span==1:return super().forward(x)
  if coarse is None or coarse.shape!=x.shape or x.shape[1]!=24:raise ValueError('aligned 24-step fine and coarse histories required')
  b,t,n,c=x.shape;h=.5*(self.encode(x)[...,-1]+self.encode(coarse)[...,-1]);z=self.head(h).reshape(b,n,72,self.atoms).transpose(1,2)
  return torch.cumsum(nn.functional.softplus(z),dim=-1)*(.25*self.scale)

def multiscale_inputs(x,ids,span):
 x=np.asarray(x);ids=np.asarray(ids)
 if span not in [1,3,7] or x.ndim!=3 or x.shape[-1]!=3 or ids.ndim!=1 or not len(ids) or not np.issubdtype(ids.dtype,np.integer) or ids.min()<24*span-1 or ids.max()>=len(x):raise ValueError('insufficient or invalid causal input history')
 fine=np.stack([x[i-23:i+1] for i in ids]).astype(np.float32,copy=False)
 if span==1:return fine,fine.copy()
 history=np.stack([x[i-24*span+1:i+1] for i in ids]).reshape(len(ids),24,span,x.shape[1],3).astype(float);valid=history[...,2];count=valid.sum(2);values=np.sum(history[...,:2]*valid[...,None],axis=2)/np.maximum(count[...,None],1);coarse=np.concatenate([values,(count/span)[...,None]],axis=-1)
 return fine,coarse.astype(np.float32)

def predict_multiscale(net,x,ids,batch=32):
 net.eval();output=[]
 with torch.no_grad():
  for start in range(0,len(ids),batch):
   fine,coarse=multiscale_inputs(x,ids[start:start+batch],net.span);output.append(net(torch.from_numpy(fine),torch.from_numpy(coarse)).numpy())
 return np.concatenate(output)

def context_features(raw):
 raw=np.asarray(raw,float)
 if raw.ndim!=2:raise ValueError('hour-by-node observations required')
 padded=np.pad(raw,((23,0),(0,0)),constant_values=np.nan);windows=np.lib.stride_tricks.sliding_window_view(padded,24,axis=0);valid=np.isfinite(windows);counts=valid.sum(-1);mean=np.where(valid,windows,0).sum(-1)/np.maximum(counts,1);recent=windows[...,-6:];before=windows[...,-12:-6];rvalid=np.isfinite(recent);bvalid=np.isfinite(before);rcount=rvalid.sum(-1);bcount=bvalid.sum(-1);rmean=np.where(rvalid,recent,0).sum(-1)/np.maximum(rcount,1);bmean=np.where(bvalid,before,0).sum(-1)/np.maximum(bcount,1);std=np.sqrt(np.where(valid,(windows-mean[...,None])**2,0).sum(-1)/np.maximum(counts,1));features=np.stack([mean,rmean,rmean-bmean,std],axis=-1);qualified=(counts>=18)&(rcount>=4)&(bcount>=4)
 return features,qualified

def climate_forecast(raw,issue,prior):
 raw=np.asarray(raw);prior=np.asarray(prior,float)
 if raw.ndim!=2 or not 0<issue<len(raw) or prior.shape!=(raw.shape[1],19) or not np.isfinite(prior).all():raise ValueError('valid causal observations and training prior required')
 q=prior.copy();counts=[];fallback=[]
 for node in range(raw.shape[1]):
  bank=raw[max(0,issue-672):issue,node];bank=bank[np.isfinite(bank)];counts.append(len(bank))
  if len(bank)>=24:q[node]=np.quantile(bank,LEVELS,method='inverted_cdf')
  else:fallback.append(node)
 return np.broadcast_to(q,(72,*q.shape)).copy(),dict(issue_id=int(issue),counts=counts,fallback_nodes=fallback)

def analog_forecast(raw,issue,prior,scale,feature_cache=None):
 raw=np.asarray(raw);base,climate=climate_forecast(raw,issue,prior)
 if not np.isfinite(scale) or scale<=0:raise ValueError('positive training-only scale required')
 features,qualified=context_features(raw) if feature_cache is None else feature_cache
 if features.shape!=(len(raw),raw.shape[1],4) or qualified.shape!=raw.shape:raise ValueError('unaligned causal feature cache')
 candidates=np.arange(23,max(23,issue-72),6,dtype=int);candidates=candidates[candidates>=issue-112*24];q=base.copy();sources=[];fallback=[];valid_counts=[]
 for node in range(raw.shape[1]):
  selected=np.empty(0,int)
  if qualified[issue,node]:
   eligible=candidates[qualified[candidates,node]];distance=np.sum(((features[eligible,node]-features[issue,node])/scale)**2,axis=-1);selected=eligible[np.argsort(distance,kind='stable')[:32]]
  sources.append(selected.tolist());bank=raw[selected[:,None]+np.arange(1,73)[None,:],node];counts=np.isfinite(bank).sum(0);valid_counts.append(counts.tolist());fallback.append(int((counts<8).sum()))
  for h in range(72):
   if counts[h]>=8:q[h,node]=np.quantile(bank[np.isfinite(bank[:,h]),h],LEVELS,method='inverted_cdf')
 return q,dict(issue_id=int(issue),source_ids=sources,valid_targets=valid_counts,fallback_targets=fallback,current_qualified=qualified[issue].tolist(),climate=climate)

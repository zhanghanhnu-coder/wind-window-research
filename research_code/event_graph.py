"""Causal event detection and minute-scale graph messages. No physical causal claim."""
import numpy as np
import pandas as pd
import torch
from model import Net

KINDS=['none','up','down','turn']

def angle_delta(a,b):return (np.asarray(a)-np.asarray(b)+180)%360-180

def event_scores(wind,direction):
 wind=np.asarray(wind,dtype=float);direction=np.asarray(direction,dtype=float)
 w=pd.DataFrame(wind);smooth=w.rolling(6,min_periods=6).mean();complete=w.rolling(18,min_periods=18).count()==18
 delta=(smooth-smooth.shift(12)).where(complete);change=delta.median(axis=1).where(delta.count(axis=1)>=12).to_numpy()
 rad=pd.DataFrame(np.deg2rad(direction));sn=np.sin(rad).rolling(6,min_periods=6).mean();cs=np.cos(rad).rolling(6,min_periods=6).mean()
 concentration=np.sqrt(sn**2+cs**2);angle=(np.rad2deg(np.arctan2(sn,cs))%360).where((concentration>=.5)&(smooth>=3))
 dc=angle_delta(angle,angle.shift(12));ok=pd.DataFrame(np.isfinite(direction)&(wind>=3)).rolling(18,min_periods=18).sum()==18
 dc=pd.DataFrame(dc).where(ok);turn=dc.median(axis=1).where(dc.count(axis=1)>=12).to_numpy()
 return {'wind_change':change,'turn':turn,'smoothed_wind':smooth.to_numpy(),'smoothed_direction':angle.to_numpy(),'wind_signal':smooth.diff().to_numpy(),'turn_signal':angle_delta(angle,angle.shift(1))}

def crossings(values,threshold,cooldown=72):
 v=np.asarray(values);out=[];last=-cooldown
 for i in range(1,len(v)):
  if np.isfinite(v[i-1]) and np.isfinite(v[i]) and v[i-1]<threshold<=v[i] and i-last>=cooldown:out.append(i);last=i
 return out

def state_at(events,cutoff,lookback=72):
 priority={'up':1,'down':2,'turn':3}
 eligible=[e for e in events if 0<cutoff-e['index']<=lookback]
 if not eligible:return 'none'
 return max(eligible,key=lambda e:(e['index'],priority[e['kind']]))['kind']

def stable_rule(lags,months,gains,chosen):
 lags=np.asarray(lags);months=np.asarray(months);gains=np.asarray(gains)
 if chosen==0 or len(lags)<12 or len(np.unique(months))<3:return False
 if np.mean(np.sign(lags)==np.sign(chosen))<.7 or np.mean(abs(lags-chosen)<=1)<.6 or np.mean(gains)<.02:return False
 for half in [months<=4,months>=5]:
  if half.sum()<4 or np.mean(np.sign(lags[half])==np.sign(chosen))<.6:return False
 return True

def shifted_hour_features(wind,cutoffs,lag,center,scale):
 """cutoff is exclusive row index; lag counts five-minute steps into the past."""
 w=pd.DataFrame(wind);mean=w.rolling(12,min_periods=12).mean().to_numpy();maximum=w.rolling(12,min_periods=12).max().to_numpy()
 ends=np.asarray(cutoffs,dtype=int)-int(lag)-1;valid=(ends>=11)&(ends<len(w));out=np.zeros((len(ends),w.shape[1],3),np.float32)
 vals=np.stack([mean[ends[valid]],maximum[ends[valid]]],-1);mask=np.isfinite(vals).all(-1)
 out[valid,:,:2]=np.where(mask[...,None],(vals-center)/scale,0);out[valid,:,2]=mask
 return out

def require_identity(record,expected):
 if record.get('identity')!=expected:raise ValueError('Cached experiment identity differs. Use a new version directory; old results were not reused.')

class EventNet(Net):
 def __init__(self,hidden=24,dilation=5):super().__init__(np.eye(1),hidden,dilation)
 def forward(self,x):
  h=torch.relu(self.self_in(x[...,:3])+self.neighbor_in(x[...,3:]))
  b,t,n,c=h.shape;h=h.permute(0,2,3,1).reshape(b*n,c,t)
  h=torch.relu(self.temporal[0](h)[...,:t]);h=torch.relu(self.temporal[2](h)[...,:t]);z=self.head(h[...,-1]).reshape(b,n,72,3).permute(0,2,1,3)
  median=z[...,0];return torch.stack([median-torch.nn.functional.softplus(z[...,1]),median,median+torch.nn.functional.softplus(z[...,2])],-1)

"""Chronological, missing-label-aware marginal probability study (round 25)."""
import numpy as np
import pandas as pd
from model import inputs,origins
from lead_calibration_core import BINS

FOLDS={
 'M05':dict(train_end='2024-03-01',cal_start='2024-04-01',test_start='2024-05-01',test_end='2024-07-01'),
 'M07':dict(train_end='2024-05-01',cal_start='2024-06-01',test_start='2024-07-01',test_end='2024-09-01'),
 'M09':dict(train_end='2024-07-01',cal_start='2024-08-01',test_start='2024-09-01',test_end='2024-11-01'),
 'M11':dict(train_end='2024-09-01',cal_start='2024-10-01',test_start='2024-11-01',test_end='2025-01-01')}

def build_fold(data,name):
 cfg=FOLDS[name];times=pd.DatetimeIndex(pd.to_datetime(data['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');raw=data['raw'];mean=pd.DataFrame(data['mean'],index=times);maximum=pd.DataFrame(raw,index=times)
 fit=times<pd.Timestamp(cfg['train_end'],tz=times.tz)
 center=float(np.nanmean(np.stack([data['mean'][fit],raw[fit]])));scale=float(np.nanstd(np.stack([data['mean'][fit],raw[fit]])))
 if not np.isfinite(center) or not np.isfinite(scale) or scale<=0:raise ValueError('invalid training-only normalization')
 x=inputs(mean,maximum,center,scale);corr=mean.loc[fit].corr(min_periods=100).fillna(0).to_numpy();np.fill_diagonal(corr,0);adj=np.zeros_like(corr)
 for i in range(len(adj)):
  keep=np.argsort(corr[i])[-4:];adj[i,keep]=np.maximum(corr[i,keep],0)
 adj/=np.maximum(adj.sum(1,keepdims=True),1e-6)
 result=dict(x=x,adj=adj,center=center,scale=scale)
 for split,start,end,stride in [('train','2024-01-12',cfg['train_end'],3),('val',cfg['train_end'],cfg['cal_start'],6),('cal',cfg['cal_start'],cfg['test_start'],6),('test',cfg['test_start'],cfg['test_end'],6)]:
  ids=origins(times,start,end,stride);ids=np.array([i for i in ids if x[i-23:i+1,:,-1].mean()>=.65],int);result[split+'_ids']=ids
 return result

def check_calibration(y):
 y=np.asarray(y)
 if y.ndim!=3 or y.shape[1]!=72:raise ValueError('requires origins x 72 hours x nodes')
 if np.isfinite(y).any((1,2)).sum()<20:raise ValueError('fewer than 20 calibration origins with observed outcomes')
 counts=[int(np.isfinite(y[:,a:b]).sum()) for a,b in BINS]
 if min(counts)<100:raise ValueError('fewer than 100 valid labels in a lead group')
 return counts

def score_atoms(q,y):
 q=np.asarray(q,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
 if q.shape[:-1]!=y.shape or q.shape[-1]!=19 or not np.isfinite(q).all() or np.any(np.diff(q,axis=-1)<0):raise ValueError('requires finite ordered 19-atom predictions aligned with truth')
 valid=np.isfinite(y);k=19;coef=(2*np.arange(1,k+1)-k-1)/(k*k);probability=(q<=10).mean(-1);label=np.where(valid,(y<=10).astype(float),np.nan);lo=q[...,1];hi=q[...,17]
 values=dict(probability=probability,label=label,crps=np.mean(abs(q-y[...,None]),-1)-np.sum(q*coef,-1),brier_hourly=(probability-label)**2,mae=abs(q[...,9]-y),coverage=((y>=lo)&(y<=hi)).astype(float),width=hi-lo,low_miss=(y<lo).astype(float),high_miss=(y>hi).astype(float),interval_score=hi-lo+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0))
 for key in values:
  if key not in ['probability']:values[key]=np.where(valid,values[key],np.nan)
 return values

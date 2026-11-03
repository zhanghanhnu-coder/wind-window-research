"""Matured-label-only weekly updates of fixed seasonal marginal forecasters."""
import numpy as np
import pandas as pd
from lead_calibration_core import fit_groups,BINS

def matured_labels(truth,source_ids,issue,days):
 y=np.asarray(truth);ids=np.asarray(source_ids)
 if y.ndim!=3 or y.shape[1]!=72 or ids.shape!=(len(y),) or not np.issubdtype(ids.dtype,np.integer) or days not in [28,56]:raise ValueError('invalid archive or lookback')
 target=ids[:,None]+np.arange(1,73)[None,:];keep=(ids[:,None]<issue)&(target>=issue-days*24)&(target<issue);return np.where(keep[:,:,None]&np.isfinite(y),y,np.nan)

def weekly_schedule(times,ids,test_start):
 times=pd.DatetimeIndex(times);ids=np.asarray(ids)
 if ids.ndim!=1 or not len(ids) or not np.issubdtype(ids.dtype,np.integer) or np.any(np.diff(ids)<=0):raise ValueError('sorted unique eligible forecast indices required')
 start=pd.Timestamp(test_start,tz=times.tz);blocks=((times[ids]-start).days//7).to_numpy()
 if blocks.min()<0:raise ValueError('test issue before declared period')
 return [dict(block=int(block),issue_id=int(ids[np.flatnonzero(blocks==block)[0]]),positions=np.flatnonzero(blocks==block)) for block in np.unique(blocks)]

def latest_parameters(updates,issue):
 eligible=[row for row in updates if row['issue_id']<=issue]
 if not eligible:raise ValueError('no available calibration at this issue')
 return np.asarray(max(eligible,key=lambda row:row['issue_id'])['theta'],float).copy()

def archive_indices(x,times,start,last_id):
 x=np.asarray(x);times=pd.DatetimeIndex(times)
 if len(x)!=len(times) or x.ndim!=3 or not 23<=last_id<len(x):raise ValueError('invalid hourly feature archive')
 boundary=pd.Timestamp(start,tz=times.tz)
 return np.array([i for i in range(23,int(last_id)+1,6) if times[i]>=boundary and x[i-23:i+1,:,-1].mean()>=.65],dtype=int)

def calibrate_at(q,truth,source_ids,issue,days,scale,previous):
 q=np.asarray(q);ids=np.asarray(source_ids);previous=np.asarray(previous,float)
 if q.shape[:-1]!=np.shape(truth) or q.shape[-1]!=19 or previous.shape!=(4,2) or not np.isfinite(q).all() or not np.isfinite(previous).all():raise ValueError('unaligned marginal distribution or previous parameters')
 y=matured_labels(truth,ids,issue,days);mask=np.isfinite(y);positions=np.flatnonzero(mask.any((1,2)));counts=[int(mask[:,a:b].sum()) for a,b in BINS];target=ids[:,None]+np.arange(1,73)[None];known=target[mask.any(-1)];enough=len(positions)>=20 and min(counts)>=100
 audit=dict(issue_id=int(issue),lookback_days=int(days),valid_origins=int(len(positions)),valid_labels=int(mask.sum()),group_counts=counts,source_positions=positions.tolist(),source_ids=ids[positions].tolist(),earliest_label_id=int(known.min()) if len(known) else None,latest_label_id=int(known.max()) if len(known) else None,reused_previous=not enough,fitted_groups=4 if enough else 0)
 if enough:theta,details=fit_groups(q[positions],y[positions],scale)
 else:
  theta=previous.copy();details=[dict(lead_start=a+1,lead_end=b,success=False,fallback=True,reason='insufficient mature support; retain previous deployed group',iterations=0,objective_candidate=None,objective_used=None,valid_targets=counts[j],delta=float(theta[j,0]),gamma=float(theta[j,1])) for j,(a,b) in enumerate(BINS)]
 audit['optimizer_failures']=sum(bool(row['fallback']) for row in details) if enough else 0
 return dict(theta=theta,details=details,audit=audit,mask=mask)

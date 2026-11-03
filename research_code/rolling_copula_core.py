"""Causal, coordinate-matured updates of a separable rank copula."""
import numpy as np
from partial_copula_core import fit_dependence

def maturity_mask(y,ids,issue,days):
 y=np.asarray(y);ids=np.asarray(ids)
 if y.ndim!=3 or y.shape[1]!=72 or ids.shape!=(len(y),) or not np.issubdtype(ids.dtype,np.integer) or not isinstance(issue,(int,np.integer)) or not np.isfinite(days) or days<=0:raise ValueError('invalid history shape, integer times or lookback')
 targets=ids[:,None]+np.arange(1,73)
 return np.isfinite(y)&((ids[:,None]<issue)&(targets>=issue-24*days)&(targets<issue))[:,:,None]

def update_dependence(q,y,ids,issue,days,scale,previous):
 q=np.asarray(q,float);y=np.asarray(y,float);mask=maturity_mask(y,ids,issue,days);masked=np.where(mask,y,np.nan);targets=np.asarray(ids)[:,None]+np.arange(1,73);known=targets[mask.any(-1)];positions=np.flatnonzero(mask.any((1,2)));audit=dict(issue_id=int(issue),days=int(days),valid_labels=int(mask.sum()),valid_origins=len(positions),source_positions=positions.tolist(),source_ids=np.asarray(ids)[positions].tolist(),earliest_label_id=int(known.min()) if len(known) else None,latest_label_id=int(known.max()) if len(known) else None,min_coordinate=int(mask.sum(0).min()),reused_previous=False,reason='')
 try:r=fit_dependence(q,masked,scale,.05)
 except ValueError as error:
  if str(error) not in ['insufficient observed calibration residuals in a coordinate','insufficient jointly observed pairs','zero variance in a jointly observed pair']:raise
  audit.update(reused_previous=True,reason=str(error));r={k:np.asarray(previous[k],float).copy() for k in ['time_correlation','space_correlation']}
 return dict(**r,mask=mask,audit=audit)

def blend_correlations(frozen,current):
 return {key:.5*(np.asarray(frozen[key],float)+np.asarray(current[key],float)) for key in ['time_correlation','space_correlation']}

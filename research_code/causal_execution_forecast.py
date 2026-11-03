"""Forecast updates from past-only slices using frozen research networks."""
import hashlib
import numpy as np
import torch
from lead_calibration_core import apply_groups
from crps_distribution_core import couple_atoms
MODELS=['old57','gated_graph57','gated_temporal57']

def forecast_batch(f,issue_ids,model,seed):
 ids=np.asarray(issue_ids)
 if ids.ndim!=1 or not len(ids) or not np.issubdtype(ids.dtype,np.integer) or (ids<23).any() or (ids>=len(f.x)).any():raise ValueError('invalid forecast issue indices')
 if model not in MODELS or seed not in [20241101,20241102,20241103]:raise ValueError('unknown frozen model')
 if max(f.cal_ids+72)>=min(ids):raise ValueError('calibration labels are not mature')
 histories=np.stack([f.x[i-23:i+1] for i in ids]);valid=histories[...,2].mean((1,2));quality=valid>=.65;q=np.full((len(ids),72,len(f.nodes),3 if model=='old57' else 19),np.nan,dtype=np.float32);key=('old' if model=='old57' else model,seed);net=f.models[key];net.eval();chosen=np.flatnonzero(quality)
 with torch.no_grad():
  for a in range(0,len(chosen),64):
   idx=chosen[a:a+64];v=net(torch.from_numpy(histories[idx])).numpy()
   if model=='old57':v=v*f.meta['scale']+f.meta['center']
   q[idx]=v
 return dict(q=q,quality=quality,valid_fraction=valid,valid_hours=(histories[...,2]>0).sum(1),history_sha256=np.array([hashlib.sha256(np.ascontiguousarray(h).tobytes()).hexdigest() for h in histories]))

def scenarios_from_quantiles(f,q,model,seed):
 q=np.asarray(q)
 if model not in MODELS or q.shape!=(72,len(f.nodes),3 if model=='old57' else 19) or not np.isfinite(q).all():raise ValueError('invalid forecast quantiles')
 if model=='old57':return np.maximum(0,q[...,1][None]+f.residual[seed][f.selected57])
 key=(model,seed);return couple_atoms(apply_groups(q,f.params[key],f.meta['scale'],True),f.ranks[key])

def completion_probabilities(sc,config,threshold=10.):
 sc=np.asarray(sc)
 if sc.ndim!=3 or not len(sc) or not np.isfinite(sc).all() or not np.isfinite(threshold):raise ValueError('invalid future scenarios')
 if any(t['duration']>sc.shape[1] or t['node']>=sc.shape[2] for t in config['tasks']):raise ValueError('task outside forecast')
 return np.array([(sc[:,:t['duration'],t['node']]<=threshold).all(1).mean() for t in config['tasks']])

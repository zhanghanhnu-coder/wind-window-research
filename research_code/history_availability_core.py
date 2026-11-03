"""Strictly mature, jointly complete historical paths; no interpolation."""
import numpy as np

def _integer(v,minimum=1):
 if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v<minimum:raise ValueError('integer outside permitted range')
 return int(v)

def complete_history_paths(valid,issue,horizon,lookback_hours,nodes):
 m=np.asarray(valid)
 if m.ndim!=2 or m.dtype.kind!='b' or m.shape[1]<1:raise ValueError('boolean time by node validity mask required')
 issue=_integer(issue,0);horizon=_integer(horizon);lookback_hours=_integer(lookback_hours)
 if issue>len(m):raise ValueError('issue beyond mask')
 nodes=list(nodes)
 if not nodes:raise ValueError('nonempty distinct node set required')
 nodes=[_integer(v,0) for v in nodes]
 if len(set(nodes))!=len(nodes) or max(nodes)>=m.shape[1]:raise ValueError('invalid node set')
 ids=np.arange(23,max(23,issue-horizon),6,dtype=int);ids=ids[ids+1>=issue-lookback_hours]
 if not len(ids):return ids
 bad=(~m[:issue,nodes].all(1)).astype(np.int64);prefix=np.r_[0,np.cumsum(bad)]
 return ids[(prefix[ids+horizon+1]-prefix[ids+1])==0]

def max_disjoint(ids,horizon):
 horizon=_integer(horizon);a=np.asarray(ids)
 if a.ndim!=1 or (len(a) and (a.dtype.kind not in 'iu' or (a<0).any() or (np.diff(a)<=0).any())):raise ValueError('strictly increasing nonnegative integer starts required')
 count=0;next_start=0
 for j in a:
  if j>=next_start:count+=1;next_start=int(j)+horizon
 return count

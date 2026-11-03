"""Authoritative original-precision truth, separate from float32 model tensors.

Historical experiments are immutable. New experiments import this module.
The hourly index denotes the end of the observed interval.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'研究笔记/基线实验_2026-11-01/严格完整小时风速.pkl'


def load_truth():
    frame=pd.read_pickle(SOURCE)['max']
    if frame.to_numpy().dtype!=np.float64:
        raise ValueError('authoritative truth must retain original float64 precision')
    times=pd.DatetimeIndex(frame.index)
    if times.tz is None or not times.is_monotonic_increasing or not times.is_unique:
        raise ValueError('truth requires a unique chronological timezone-aware index')
    if len(times)>1 and not np.all(np.diff(times.asi8)==3600*10**9):
        raise ValueError('truth must have an explicit regular hourly grid')
    nodes=[str(n) for n in frame.columns]
    if len(set(nodes))!=len(nodes):raise ValueError('duplicate node identity')
    expected=json.loads((ROOT/'artifacts/metadata.json').read_text(encoding='utf-8'))['nodes']
    if nodes!=[str(n) for n in expected]:raise ValueError('frozen model node order differs')
    with np.load(ROOT/'artifacts/data.npz') as archived:
        archive_times=pd.DatetimeIndex(pd.to_datetime(archived['timestamps'].tolist(),utc=True))
        if not np.array_equal(times.asi8,archive_times.asi8):raise ValueError('archive time index differs')
        raw=frame.to_numpy(copy=True)
        if not np.array_equal(raw.astype('float32'),archived['raw'],equal_nan=True):
            raise ValueError('original truth does not reproduce frozen float32 archive')
    return times,nodes,raw


def window_labels(raw,origins,offset=0,duration=6,threshold=10.,node_indices=None,joint=False):
    raw=np.asarray(raw)
    if raw.dtype!=np.float64 or raw.ndim!=2:
        raise ValueError('use original float64 [hour,node] truth; casting lost precision back is not a repair')
    for name,value,minimum in [('offset',offset,0),('duration',duration,1)]:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<minimum:
            raise ValueError(f'{name} is outside its integer domain')
    if isinstance(threshold,(bool,np.bool_)) or not np.isfinite(threshold):raise ValueError('finite threshold required')
    ix=np.asarray(origins)
    if ix.ndim!=1 or ix.dtype.kind not in 'iu' or len(ix)==0:
        raise ValueError('nonempty integer origin indices required')
    if (ix<0).any() or (ix+offset+duration>=len(raw)).any():
        raise ValueError('window crosses the truth timeline')
    nodes=np.arange(raw.shape[1]) if node_indices is None else np.asarray(node_indices)
    if nodes.ndim!=1 or nodes.dtype.kind not in 'iu' or len(nodes)==0 or len(np.unique(nodes))!=len(nodes) or (nodes<0).any() or (nodes>=raw.shape[1]).any():
        raise ValueError('unique in-range integer node indices required')
    values=raw[ix[:,None]+np.arange(offset+1,offset+duration+1)[None,:]][:,:,nodes]
    known=np.isfinite(values)
    failure=(known&(values>threshold)).any(axis=1)
    success=known.all(axis=1)&~failure
    labels=np.where(failure,0.,np.where(success,1.,np.nan))
    if joint:
        return np.where((labels==0).any(axis=1),0.,np.where((labels==1).all(axis=1),1.,np.nan))
    return labels

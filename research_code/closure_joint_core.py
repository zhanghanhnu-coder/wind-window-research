"""Finite-scenario dependence interventions; frozen marginal atoms are untouched."""
import numpy as np
from review_baselines_core import OFFSETS,GROUPS

def shuffle_indices(shape,kind,seed):
    s,h,n=shape;rng=np.random.default_rng(seed)
    if kind=='time':
        return np.broadcast_to(np.stack([rng.permutation(s) for _ in range(h)],1)[:,:,None],shape).copy()
    if kind=='space':
        return np.broadcast_to(np.stack([rng.permutation(s) for _ in range(n)],1)[:,None,:],shape).copy()
    if kind=='both':
        return np.stack([rng.permutation(s) for _ in range(h*n)],1).reshape(shape)
    if kind=='global':return np.broadcast_to(rng.permutation(s)[:,None,None],shape).copy()
    raise ValueError(kind)

def intervene(ranks,kind,seed):
    ranks=np.asarray(ranks);ix=shuffle_indices(ranks.shape,kind,seed)
    out=np.take_along_axis(ranks,ix,axis=0)
    # Exact integer certificate at EVERY lead,node. This proves equality of the
    # scenario-value multiset for every possible marginal q, not only samples.
    if not np.array_equal(np.sort(out,axis=0),np.sort(ranks,axis=0)):
        raise AssertionError('marginal scenario multiset changed')
    if kind=='time':assert (ix==ix[:,:,:1]).all()
    if kind=='space':assert (ix==ix[:,:1,:]).all()
    return out,ix

def materialize(q,ranks):
    q=np.asarray(q,dtype=np.float64)
    return np.take_along_axis(np.repeat(q,3,axis=-1).transpose(0,3,1,2),np.broadcast_to(ranks,(len(q),)+ranks.shape),axis=1)

def probabilities(q,ranks,threshold):
    safe=materialize(q,ranks)<=np.float64(threshold)
    single=[];group=[]
    for off in OFFSETS:
        single.append(safe[:,:,off:off+6].all(2).mean(1))
        for nodes in GROUPS:group.append(safe[:,:,off:off+6,list(nodes)].all((2,3)).mean(1))
    return np.concatenate(single,1),np.stack(group,1)

def direct_probabilities(q,ranks,threshold):
    """Independent scalar-loop audit, no materialize/probabilities calls."""
    counts=np.zeros((4,17),int);groups=np.zeros((4,3),int)
    for s in range(len(ranks)):
        for j,off in enumerate(OFFSETS):
            ok=np.array([all(q[h,n,int(ranks[s,h,n])//3]<=threshold for h in range(off,off+6)) for n in range(17)])
            counts[j]+=ok
            for g,nodes in enumerate(GROUPS):groups[j,g]+=ok[list(nodes)].all()
    return counts.ravel()/len(ranks),groups.ravel()/len(ranks)

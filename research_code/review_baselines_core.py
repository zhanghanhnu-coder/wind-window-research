"""Small auditable helpers for the post-review frozen-model benchmark."""
import numpy as np

OFFSETS=(0,24,48,66)
GROUPS=((0,1,2),(8,9,10),(13,15,16))

def training_masks(valid,train_ids,test_ids,seed):
    valid=np.asarray(valid);tr=np.asarray(train_ids);te=np.asarray(test_ids)
    if valid.ndim!=2 or valid.dtype!=bool or tr.ndim!=1 or te.ndim!=1 or not len(tr) or not len(te) or tr.min()<167 or tr.max()>=te.min():
        raise ValueError('168-hour masks must come entirely from training-period origins before test')
    selected=np.random.default_rng(seed).choice(tr,size=len(te),replace=True)
    return np.stack([valid[t-167:t+1] for t in selected]),selected

def masked_histories(x,ids,mask):
    x=np.asarray(x);ids=np.asarray(ids);mask=np.asarray(mask)
    if x.ndim!=3 or x.shape[-1]!=3 or mask.shape!=(len(ids),168,x.shape[1]) or mask.dtype!=bool or ids.min()<167:
        raise ValueError('aligned empirical masks and complete causal input range required')
    h=np.stack([x[i-167:i+1] for i in ids]).copy()
    keep=(h[...,2]>0)&mask
    return np.where(keep[...,None],h,0).astype(np.float32)

def coarse_inputs(history,span):
    h=np.asarray(history)
    if h.ndim!=4 or h.shape[1]!=168 or h.shape[-1]!=3 or span not in [1,3,7]:raise ValueError('168-hour histories and a supported span required')
    fine=h[:,-24:].astype(np.float32)
    if span==1:return fine,fine.copy()
    a=h[:,-24*span:].reshape(len(h),24,span,h.shape[2],3).astype(float)
    valid=a[...,2];count=valid.sum(2)
    values=(a[...,:2]*valid[...,None]).sum(2)/np.maximum(count[...,None],1)
    return fine,np.concatenate([values,(count/span)[...,None]],axis=-1).astype(np.float32)

def window_probabilities(q,ranks):
    q=np.asarray(q);ranks=np.asarray(ranks)
    if q.ndim!=4 or q.shape[1:]!=(72,17,19) or ranks.shape!=(57,72,17) or not np.isfinite(q).all() or (np.diff(q,axis=-1)<0).any():raise ValueError('finite ordered 19-atom forecasts and 57 ranks required')
    safe=np.take_along_axis(np.repeat(q,3,axis=-1).transpose(0,3,1,2),np.broadcast_to(ranks,(len(q),)+ranks.shape),axis=1)<=10
    single=[];group=[]
    for off in OFFSETS:
        single.append(safe[:,:,off:off+6].all(2).mean(1))
        for nodes in GROUPS:group.append(safe[:,:,off:off+6,list(nodes)].all((2,3)).mean(1))
    return np.concatenate(single,axis=1),np.stack(group,axis=1)

def loss_event_totals(pa,pb,labels,inverse):
    pa=np.asarray(pa,float);y=np.asarray(labels);ix=np.asarray(inverse)
    if pb is not None:pb=np.asarray(pb,float)
    if pa.ndim!=3 or pa.shape[1:]!=y.shape or ix.shape!=y.shape or not np.isfinite(pa).all() or (pb is not None and (pa.shape!=pb.shape or not np.isfinite(pb).all())):raise ValueError('aligned seed probabilities and event labels required')
    n=int(ix.max())+1
    c=np.mean(pa*pa,axis=0) if pb is None else np.mean(pa*pa-pb*pb,axis=0)
    d=1-2*np.mean(pa,axis=0) if pb is None else -2*np.mean(pa-pb,axis=0)
    ec=np.bincount(ix.ravel(),weights=c.ravel(),minlength=n);ed=np.bincount(ix.ravel(),weights=d.ravel(),minlength=n);count=np.bincount(ix.ravel(),minlength=n)
    unique=np.full(n,np.nan);known=np.isfinite(y)
    unique[ix[known]]=y[known]
    if not np.array_equal(np.isnan(unique[ix]),np.isnan(y)) or not np.array_equal(unique[ix][known],y[known]):raise ValueError('same physical event has conflicting truth')
    fixed=np.isfinite(unique);lo=ec.copy();hi=ec.copy()
    lo[fixed]+=ed[fixed]*unique[fixed];hi[fixed]+=ed[fixed]*unique[fixed]
    lo[~fixed]+=np.minimum(ed[~fixed],0);hi[~fixed]+=np.maximum(ed[~fixed],0)
    return lo,hi,count

"""Past-matured, origin-weighted monotone event calibration; no trajectory claims."""
import numpy as np
ENDS=np.array([6,30,54,72]);GROUPS=[[0,1,2],[8,9,10],[13,15,16]]

def weighted_pava(x,y,w):
 x=np.asarray(x,float);y=np.asarray(y,float);w=np.asarray(w,float)
 if x.ndim!=1 or not len(x) or y.shape!=x.shape or w.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all() or (w<=0).any() or (y<0).any() or (y>1).any():raise ValueError('finite aligned values, labels in [0,1], positive weights required')
 unique,inverse=np.unique(x,return_inverse=True);weight=np.bincount(inverse,weights=w);total=np.bincount(inverse,weights=w*y);blocks=[]
 for i in range(len(unique)):
  blocks.append([i,i,float(weight[i]),float(total[i])])
  while len(blocks)>1 and blocks[-2][3]/blocks[-2][2]>blocks[-1][3]/blocks[-1][2]:
   right=blocks.pop();left=blocks.pop();blocks.append([left[0],right[1],left[2]+right[2],left[3]+right[3]])
 fitted=np.empty(len(unique))
 for left,right,mass,value in blocks:fitted[left:right+1]=value/mass
 return unique,fitted

def fit_mature_map(single,group,labels_single,labels_group,ids,update,lookback=672,min_origins=20,prior=20.):
 ids=np.asarray(ids);ps=np.asarray(single,float);pg=np.asarray(group,float);ys=np.asarray(labels_single,float);yg=np.asarray(labels_group,float)
 if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or np.any(np.diff(ids)<=0) or ps.shape!=(len(ids),68) or pg.shape!=(len(ids),12) or ys.shape!=ps.shape or yg.shape!=pg.shape:raise ValueError('sorted origin ids and fixed 6h single/group shapes required')
 if isinstance(update,bool) or not isinstance(update,(int,np.integer)) or isinstance(min_origins,bool) or not isinstance(min_origins,(int,np.integer)) or min_origins<1 or not np.isfinite(prior) or prior<=0 or not np.isfinite(lookback) or lookback<=0:raise ValueError('positive support/prior/lookback and integer update required')
 matrices=[(ps,ys,np.repeat(ENDS,17)),(pg,yg,np.repeat(ENDS,3))];records=[]
 for i in np.flatnonzero(ids<update):
  chosen=[]
  for kind,(prob,label,ends) in enumerate(matrices):
   targets=ids[i]+ends;use=(targets<update)&(targets>=update-lookback)&np.isfinite(label[i]);indices=np.flatnonzero(use)
   if len(indices):chosen.append((kind,prob,label,targets,indices))
  for kind,prob,label,targets,indices in chosen:
   pv=prob[i,indices];yv=label[i,indices]
   if not np.isfinite(pv).all() or (pv< -1e-12).any() or (pv>1+1e-12).any() or not np.isin(yv,[0,1]).all():raise ValueError('invalid selected historical probability or binary label')
   weight=1/(len(chosen)*len(indices))
   for j in indices:records.append([i,kind,int(j),int(targets[j]),float(np.clip(prob[i,j],0,1)),float(label[i,j]),weight])
 train=np.asarray(records,float).reshape(-1,7);origins=len(np.unique(train[:,0]));fallback=origins<min_origins
 if fallback:knots=np.array([0.,1.]);values=knots.copy()
 else:
  anchors=np.linspace(0,1,21);knots,values=weighted_pava(np.r_[train[:,4],anchors],np.r_[train[:,5],anchors],np.r_[train[:,6],np.full(21,prior/21)])
 return dict(knots=knots,values=values,training=train,audit=dict(update=int(update),valid_origins=origins,valid_labels=len(train),training_weight=float(train[:,6].sum()),earliest_end=int(train[:,3].min()) if len(train) else None,latest_end=int(train[:,3].max()) if len(train) else None,fallback=bool(fallback),prior_weight=float(prior),lookback=float(lookback),minimum_origins=int(min_origins)))

def apply_window_map(single,group,mapping):
 ps=np.asarray(single,float);pg=np.asarray(group,float);x=np.asarray(mapping['knots'],float);y=np.asarray(mapping['values'],float)
 if ps.ndim!=2 or ps.shape[1]!=68 or pg.shape!=(len(ps),12) or not np.isfinite(ps).all() or not np.isfinite(pg).all() or min(ps.min(),pg.min())< -1e-12 or max(ps.max(),pg.max())>1+1e-12:raise ValueError('valid fixed window probability shapes required')
 if x.ndim!=1 or len(x)<2 or x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or (np.diff(x)<=0).any() or (np.diff(y)<-1e-14).any() or x[0]!=0 or x[-1]!=1 or y.min()<0 or y.max()>1:raise ValueError('monotone anchored map required')
 a=np.interp(np.clip(ps,0,1),x,y);raw=np.interp(np.clip(pg,0,1),x,y);view=a.reshape(-1,4,17);lower=np.stack([np.maximum(0,view[:,:,nodes].sum(2)-2) for nodes in GROUPS],axis=2).reshape(-1,12);upper=np.stack([view[:,:,nodes].min(2) for nodes in GROUPS],axis=2).reshape(-1,12);b=np.clip(raw,lower,upper);delta=abs(b-raw)
 return a,b,dict(projected=int((delta>1e-12).sum()),maximum_shift=float(delta.max()),absolute_shift_sum=float(delta.sum()))

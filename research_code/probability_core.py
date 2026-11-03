"""Frozen probability study primitives: scenario axis first, original wind units."""
import numpy as np

BINS=((0,6),(6,24),(24,48),(48,72))

def weights(x,w=None):
 if w is None:return np.full(len(x),1/len(x),dtype=float)
 w=np.asarray(w,dtype=float)
 if w.shape!=(len(x),) or np.any(w<0) or not np.isfinite(w).all() or w.sum()<=0:raise ValueError('invalid weights')
 return w/w.sum()

def crps(x,y,w=None):
 x=np.asarray(x,dtype=float);w=weights(x,w);expand=(slice(None),)+(None,)*(x.ndim-1)
 order=np.argsort(x,axis=0,kind='stable');sx=np.take_along_axis(x,order,axis=0)
 sw=np.broadcast_to(w[expand],x.shape);sw=np.take_along_axis(sw,order,axis=0)
 return np.sum(w[expand]*abs(x-y),axis=0)-np.sum(sw*sx*(2*np.cumsum(sw,axis=0)-sw-1),axis=0)

def quantile(x,q,w=None):
 x=np.asarray(x);w=weights(x,w);order=np.argsort(x,axis=0,kind='stable');sx=np.take_along_axis(x,order,axis=0)
 sw=np.take_along_axis(np.broadcast_to(w.reshape((-1,)+(1,)*(x.ndim-1)),x.shape),order,axis=0)
 ix=np.argmax(np.cumsum(sw,axis=0)>=q-1e-12,axis=0)
 return np.take_along_axis(sx,ix[None],axis=0)[0]

def longest(good):
 good=np.asarray(good,dtype=bool);cur=np.zeros((len(good),good.shape[2]),int);best=cur.copy()
 for h in range(good.shape[1]):
  cur=np.where(good[:,h],cur+1,0);best=np.maximum(best,cur)
 return best

def first_start(good,duration=6):
 good=np.asarray(good,dtype=bool);out=np.full((len(good),good.shape[2]),np.inf)
 for start in range(good.shape[1]-duration+1):
  ok=good[:,start:start+duration].all(1);out=np.where(ok&np.isinf(out),start,out)
 return out

def window_prob(x,duration,offset,w=None,threshold=10):
 return np.sum(weights(x,w)[:,None]*(x[:,offset:offset+duration]<=threshold).all(1),axis=0)

def break_dependence(x,kind):
 rng=np.random.default_rng(20261101);out=x.copy()
 if kind=='time':
  for h in range(x.shape[1]):out[:,h]=x[rng.permutation(len(x)),h]
 elif kind=='space':
  for n in range(x.shape[2]):out[:,:,n]=x[rng.permutation(len(x)),:,n]
 else:raise ValueError(kind)
 return out

def spread(q):return np.maximum((q[...,2]-q[...,0])/2,.5)

def residual_pools(z):
 return [z[:,a:b][np.isfinite(z[:,a:b])] for a,b in BINS]

def ecc_template(z,pools,gamma=1.):
 """Historical standardized-residual ranks; mid-probabilities, linear pool quantiles."""
 out=np.empty_like(z,dtype=float);levels=(np.arange(len(z))+.5)/len(z)
 for (a,b),pool in zip(BINS,pools):
  values=np.quantile(pool,levels);mid=np.median(pool);values=mid+gamma*(values-mid)
  ranks=np.argsort(np.argsort(z[:,a:b],axis=0,kind='stable'),axis=0,kind='stable')
  out[:,a:b]=values[ranks]
 return out

def calibration_split(ids,horizon,boundary):
 ids=np.asarray(ids);return ids+horizon<boundary,ids>=boundary

def analog_weights(bank,current,k):
 bank=np.asarray(bank,float);current=np.asarray(current,float)
 median=np.nanmedian(bank,axis=0);bank=np.where(np.isfinite(bank),bank,median);current=np.where(np.isfinite(current),current,median)
 scale=np.maximum(bank.std(axis=0),1e-6);distance=np.sum(((bank-current)/scale)**2,axis=1)
 selected=np.argsort(distance,kind='stable')[:min(k,len(bank))]
 w=np.full(len(bank),.5/len(bank));w[selected]+=.5/len(selected);return w

def context(raw,ids):
 # Past-only farm wind maximum and six-hour change; missing inputs never use future labels.
 return np.array([[np.nanmean(raw[i]),np.nanmean(raw[i])-np.nanmean(raw[i-6])] for i in ids])

def make_bank(q,truth,contexts):
 complete=np.isfinite(truth).all(axis=(1,2));r=truth-q[...,1];z=r/spread(q)
 if not complete.any():raise ValueError('no complete calibration trajectories')
 return dict(residual=r[complete],standard=z[complete],pools=residual_pools(z),contexts=contexts[complete],complete=complete)

def scenarios(bank,q,ctx,gamma=1.,k=16):
 r=bank['residual'];n=len(r);med=q[...,1];s=spread(q)
 selected=np.random.default_rng(20241101).choice(n,32,replace=n<32)
 raw=np.maximum(0,med+r);scaled=np.maximum(0,med+s*bank['standard'])
 plain=np.maximum(0,med+s*ecc_template(bank['standard'],bank['pools']))
 tuned=np.maximum(0,med+s*ecc_template(bank['standard'],bank['pools'],gamma))
 return {'joint32':(raw[selected],None),'joint_all':(raw,None),'scaled_joint':(scaled,None),'pooled_ecc':(plain,None),'tuned_ecc':(tuned,None),'analog_joint':(raw,analog_weights(bank['contexts'],ctx,k)),'time_broken':(break_dependence(tuned,'time'),None),'space_broken':(break_dependence(tuned,'space'),None)}

def score_scenarios(sc,truth,w=None,train_scale=4.014734326870501):
 """Return per-target scores with NaN labels kept out. No complete-case filtering of forecasts."""
 w=weights(sc,w);valid=np.isfinite(truth);lo=quantile(sc,.1,w);hi=quantile(sc,.9,w)
 out={'crps':crps(sc,truth,w),'coverage':np.where(valid,(truth>=lo)&(truth<=hi),np.nan),'width':np.where(valid,hi-lo,np.nan),'interval_score':np.where(valid,hi-lo+10*np.maximum(lo-truth,0)+10*np.maximum(truth-hi,0),np.nan),'low_miss':np.where(valid,truth<lo,np.nan),'high_miss':np.where(valid,truth>hi,np.nan)}
 for duration in [3,6,12]:
  probs=[];labels=[];errors=[]
  for a in [0,24,48,72-duration]:
   yy=truth[a:a+duration];vv=np.isfinite(yy).all(0);label=(yy<=10).all(0);prob=window_prob(sc,duration,a,w)
   probs.append(np.where(vv,prob,np.nan));labels.append(np.where(vv,label,np.nan));errors.append(np.where(vv,(prob-label)**2,np.nan))
  out[f'brier{duration}']=np.array(errors);out[f'prob{duration}']=np.array(probs);out[f'label{duration}']=np.array(labels)
 for horizon in [24,48,72]:
  yy=truth[:horizon];vv=np.isfinite(yy).all(0);obs=longest((yy<=10)[None])[0];pred=longest(sc[:,:horizon]<=10)
  out[f'length_crps{horizon}']=np.where(vv,crps(pred,obs,w),np.nan)
  pp=np.sum(w[:,None]*(pred>=6),axis=0);label=obs>=6
  out[f'any6_brier{horizon}']=np.where(vv,(pp-label)**2,np.nan)
  out[f'any6_prob{horizon}']=np.where(vv,pp,np.nan)
 vv=valid.all(0);fs=first_start(sc<=10);ys=first_start((truth<=10)[None])[0]
 pp=np.sum(w[:,None,None]*(fs[:,None,:]<=np.arange(67)[None,:,None]),axis=0)
 labels=ys[None,:]<=np.arange(67)[:,None]
 out['first_cdf_brier']=np.where(vv,np.mean((pp-labels)**2,axis=0),np.nan)
 groups=[[0,1,2],[8,9,10],[13,15,16]];gb=[]
 for a in [0,24,48,66]:
  for g in groups:
   yy=truth[a:a+6][:,g];good=(sc[:,a:a+6][:,:,g]<=10).all(axis=(1,2));pr=w@good
   gb.append((pr-(yy<=10).all())**2 if np.isfinite(yy).all() else np.nan)
 out['group_brier']=np.array(gb)
 spatial=[];temporal=[]
 for h in [5,23,47,71]:
  for i in range(truth.shape[1]):
   for j in range(i+1,truth.shape[1]):
    term=np.sum(w*np.sqrt(abs(sc[:,h,i]-sc[:,h,j])/train_scale))
    spatial.append((np.sqrt(abs(truth[h,i]-truth[h,j])/train_scale)-term)**2)
 hh=[0,5,11,23,47,71]
 for pos,h in enumerate(hh):
  for h2 in hh[pos+1:]:
   term=np.sum(w[:,None]*np.sqrt(abs(sc[:,h]-sc[:,h2])/train_scale),axis=0)
   temporal.extend((np.sqrt(abs(truth[h]-truth[h2])/train_scale)-term)**2)
 out['spatial_variogram']=np.array(spatial);out['temporal_variogram']=np.array(temporal)
 out['ess']=np.array([1/np.sum(w*w)])
 return out

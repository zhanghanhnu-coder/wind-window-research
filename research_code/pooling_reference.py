"""Independent weighted CDF integrals and vectorized joint-event audit formulas."""
import numpy as np

def mixture_probability_reference(experts,w,valid):
 experts=np.asarray(experts);w=np.asarray(w);valid=np.asarray(valid)
 if experts.shape[0]!=4 or experts.shape[1:]!=valid.shape or w.shape!=(len(valid),4):raise ValueError('unaligned mixture probability check')
 mass=w.T.reshape((4,len(valid))+(1,)*(valid.ndim-1));return np.where(valid,np.sum(experts*mass,axis=0),np.nan)

def moments_reference(q,y,scale):
 a=np.zeros(4);b=np.zeros((4,4));count=len(y)
 for start in range(0,count,2048):
  z=np.asarray(q[:,start:start+2048],float)/scale;t=np.asarray(y[start:start+2048],float)/scale;a+=abs(z-t[None,:,None]).sum((1,2))/19
  for i in range(4):
   for j in range(i,4):
    values=np.concatenate([z[i],z[j]],axis=-1);order=np.argsort(values,axis=-1,kind='stable');v=np.take_along_axis(values,order,axis=-1);left=order<19;cl=np.cumsum(left,axis=-1);cr=np.cumsum(~left,axis=-1);sl=np.cumsum(np.where(left,v,0),axis=-1);sr=np.cumsum(np.where(left,0,v),axis=-1)
    b[i,j]+=np.where(left,v*cr-sr,v*cl-sl).sum()/361
 b+=np.triu(b,1).T
 return a/count,b/count

def simplex_reference(a,b):
 h=-b+.002*np.eye(4);c=a-.0005;best=None
 for subset in range(1,16):
  ix=np.flatnonzero([subset&(1<<j) for j in range(4)]);n=len(ix);matrix=np.zeros((n+1,n+1));matrix[:n,:n]=h[np.ix_(ix,ix)];matrix[-1,:n]=1;matrix[:n,-1]=1;sol=np.linalg.solve(matrix,np.r_[-c[ix],1]);w=np.zeros(4);w[ix]=sol[:n]
  if w.min()< -1e-9:continue
  value=float(a@w-.5*w@b@w+.001*np.sum((w-.25)**2))
  if best is None or value<best[0]:best=(value,w)
 assert best is not None
 return best

def cdf_crps(values,y,weights):
 # Last axis holds arbitrary finite weighted atoms. Integrate (F-I)^2 exactly.
 values=np.asarray(values,float);y=np.asarray(y,float);weights=np.broadcast_to(weights,values.shape);z=np.concatenate([values,np.where(np.isfinite(y),y,0)[...,None]],axis=-1);mass=np.concatenate([weights,np.zeros(y.shape+(1,))],axis=-1);order=np.argsort(z,axis=-1,kind='stable');z=np.take_along_axis(z,order,axis=-1);mass=np.take_along_axis(mass,order,axis=-1);cdf=np.cumsum(mass,axis=-1);out=np.sum(np.diff(z,axis=-1)*(cdf[...,:-1]-(z[...,:-1]>=y[...,None]))**2,axis=-1)
 return np.where(np.isfinite(y),out,np.nan)

def marginal_reference(q,y,w):
 # q = experts,batch,hour,node,atom. CDF mixture has 76 atoms before path replication.
 values=q.transpose(1,2,3,0,4).reshape(len(y),72,17,76).astype(float);mass=np.broadcast_to(np.repeat(w/19,19,axis=1)[:,None,None,:],values.shape);order=np.argsort(values,axis=-1,kind='stable');z=np.take_along_axis(values,order,axis=-1);sw=np.take_along_axis(mass,order,axis=-1);cdf=np.cumsum(sw,axis=-1);points=[]
 for p in [.1,.5,.9]:
  ix=(cdf>=p-1e-12).argmax(-1);points.append(np.take_along_axis(z,ix[...,None],axis=-1)[...,0])
 lo,med,hi=points;valid=np.isfinite(y);p=np.sum(mass*(values<=10),axis=-1);label=np.where(valid,y<=10,np.nan)
 return dict(crps=cdf_crps(values,y,mass),coverage=np.where(valid,(y>=lo)&(y<=hi),np.nan),width=np.where(valid,hi-lo,np.nan),low_miss=np.where(valid,y<lo,np.nan),high_miss=np.where(valid,y>hi,np.nan),interval_score=np.where(valid,hi-lo+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0),np.nan),probability=np.where(valid,p,np.nan),label=label,brier_hourly=(p-label)**2,mae=np.where(valid,abs(med-y),np.nan))

def reconstruct_mixture(q,ranks):
 return np.concatenate([np.take_along_axis(x,(ranks//3).transpose(1,2,0)[None],axis=-1).transpose(0,3,1,2) for x in q],axis=1).astype(float)

def joint_reference(sc,y,w,scale):
 mass=np.repeat(w/57,57,axis=1);good=sc<=10;out={};valid=np.isfinite(y)
 for duration in [3,6,12]:
  probs=[];labels=[]
  for start in [0,24,48,72-duration]:
   mask=valid[:,start:start+duration].all(1);p=np.einsum('bs,bsn->bn',mass,good[:,:,start:start+duration].all(2));label=(y[:,start:start+duration]<=10).all(1);probs.append(np.where(mask,p,np.nan));labels.append(np.where(mask,label,np.nan))
  out[f'prob{duration}']=np.stack(probs,1);out[f'label{duration}']=np.stack(labels,1);out[f'brier{duration}']=(out[f'prob{duration}']-out[f'label{duration}'])**2
 for h in [24,48,72]:
  positions=np.arange(1,h+1);last=np.maximum.accumulate(np.where(~good[:,:,:h],positions[None,None,:,None],0),axis=2);lengths=(positions[None,None,:,None]-last).max(2);observed=(positions[None,:,None]-np.maximum.accumulate(np.where(~(y[:,:h]<=10),positions[None,:,None],0),axis=1)).max(1);mask=valid[:,:h].all(1);out[f'length_crps{h}']=np.where(mask,cdf_crps(lengths.transpose(0,2,1),observed,mass[:,None,:]),np.nan);p=np.einsum('bs,bsn->bn',mass,lengths>=6);out[f'any6_prob{h}']=np.where(mask,p,np.nan);out[f'any6_brier{h}']=np.where(mask,(p-(observed>=6))**2,np.nan)
 has=np.lib.stride_tricks.sliding_window_view(good,6,axis=2).all(-1);cdf=np.einsum('bs,bshn->bhn',mass,np.maximum.accumulate(has,axis=2));observed=np.lib.stride_tricks.sliding_window_view(y<=10,6,axis=1).all(-1);labelcdf=np.maximum.accumulate(observed,axis=1);out['first_cdf_brier']=np.where(valid.all(1),((cdf-labelcdf)**2).mean(1),np.nan);gp=[];gl=[]
 for offset in [0,24,48,66]:
  for nodes in [[0,1,2],[8,9,10],[13,15,16]]:
   mask=valid[:,offset:offset+6][:,:,nodes].all((1,2));p=np.einsum('bs,bs->b',mass,good[:,:,offset:offset+6][:,:,:,nodes].all((2,3)));label=(y[:,offset:offset+6][:,:,nodes]<=10).all((1,2));gp.append(np.where(mask,p,np.nan));gl.append(np.where(mask,label,np.nan))
 out['group_probability']=np.stack(gp,1);out['group_label']=np.stack(gl,1);out['group_brier']=(out['group_probability']-out['group_label'])**2
 pairs=[(h,i,h,j) for h in [5,23,47,71] for i in range(17) for j in range(i+1,17)];hours=[0,5,11,23,47,71];tpairs=[(h,n,h2,n) for pos,h in enumerate(hours) for h2 in hours[pos+1:] for n in range(17)]
 for key,items in [('spatial_variogram',pairs),('temporal_variogram',tpairs)]:
  a,i,b,j=np.array(items).T;pred=np.einsum('bs,bsk->bk',mass,np.sqrt(abs(sc[:,:,a,i]-sc[:,:,b,j])/scale));obs32=y.astype(np.float32);actual=np.sqrt(abs(obs32[:,a,i]-obs32[:,b,j])/scale);out[key]=(pred-actual)**2
 out['ess']=(1/np.sum(mass*mass,axis=1))[:,None];return out

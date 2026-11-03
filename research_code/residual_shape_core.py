"""Past-only standardized-error calibration; no claim of future coverage."""
import numpy as np

BINS=((0,6),(6,24),(24,48),(48,72))
TAU=(np.arange(19)+.5)/19

def weighted_quantile(values,weights,probabilities):
    x=np.asarray(values,float);w=np.asarray(weights,float);p=np.asarray(probabilities,float)
    if x.ndim!=1 or not x.size or w.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(w).all() or (w<0).any() or not (w>0).any() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('finite values, nonnegative nonempty mass and probabilities required')
    active=w>0;x=x[active];w=w[active];order=np.argsort(x,kind='stable');x=x[order];w=w[order]
    cum=np.cumsum(w.astype(np.longdouble));target=p.astype(np.longdouble)*cum[-1]
    # One float64 ulp at a CDF jump counts as the same boundary, not a new bin.
    target=np.maximum(0,target-2*np.finfo(float).eps*cum[-1])
    index=np.minimum(np.searchsorted(cum,target,side='left'),len(x)-1)
    return x[index]

def _validate(q,scale):
    q=np.asarray(q,float)
    if q.ndim!=4 or q.shape[1]!=72 or q.shape[2]<1 or q.shape[-1]!=19 or not np.isfinite(scale) or scale<=0:
        raise ValueError('origin x 72 x nodes x 19 forecasts and positive finite training scale required')
    return q

def spread(q,scale):
    return np.maximum((q[...,17]-q[...,1])/2,.1*scale)

def fit_residual(q,y,ids,issue,scale,minimum_origins=20,minimum_labels=100,lookback=672):
    q=_validate(q,scale);y=np.asarray(y,float);ids=np.asarray(ids)
    if y.shape!=q.shape[:-1] or ids.shape!=(len(q),) or not np.issubdtype(ids.dtype,np.integer) or (np.diff(ids)<=0).any() or not isinstance(issue,(int,np.integer)):
        raise ValueError('aligned truth and strictly chronological integer issue ids required')
    if minimum_origins<1 or minimum_labels<1 or lookback<1:raise ValueError('positive history support required')
    label_ids=ids[:,None,None]+np.arange(1,73)[None,:,None]
    time_ok=(label_ids<issue)&(label_ids>=issue-lookback)
    zq=[];fallbacks=[];audit=[]
    for a,b in BINS:
        valid=np.isfinite(y[:,a:b])&time_ok[:,a:b];count=valid.sum((1,2));use=count>0
        n=int(use.sum());labels=int(valid.sum());fallback=n<minimum_origins or labels<minimum_labels
        record=dict(lead_start=a+1,lead_end=b,origins=n,labels=labels,fallback=bool(fallback),earliest_label=int(np.broadcast_to(label_ids[:,a:b],valid.shape)[valid].min()) if labels else None,latest_label=int(np.broadcast_to(label_ids[:,a:b],valid.shape)[valid].max()) if labels else None)
        if fallback:z=np.full(19,np.nan)
        else:
            qq=q[:,a:b][valid]
            if not np.isfinite(qq).all() or (np.diff(qq,axis=-1)<0).any():raise ValueError('mature forecasts must be finite and ordered')
            residual=(y[:,a:b][valid]-qq[:,9])/spread(qq,scale)
            perorigin=np.zeros(len(q));perorigin[use]=1/(n*count[use]);weight=np.broadcast_to(perorigin[:,None,None],valid.shape)[valid]
            z=weighted_quantile(residual,weight,TAU)
            record.update(weight_sum=float(weight.sum()),minimum_scale=float(spread(qq,scale).min()),floor_labels=int(((qq[:,17]-qq[:,1])/2<.1*scale).sum()))
        zq.append(z);fallbacks.append(fallback);audit.append(record)
    return dict(z_quantiles=np.stack(zq),fallback=np.array(fallbacks,bool),audit=audit)

def apply_residual(q,fit,scale,method):
    q=_validate(q,scale);z=np.asarray(fit['z_quantiles'],float);fallback=np.asarray(fit['fallback'])
    if not np.isfinite(q).all() or (np.diff(q,axis=-1)<0).any() or z.shape!=(4,19) or fallback.shape!=(4,) or fallback.dtype!=bool or method not in ['L','S','T']:
        raise ValueError('finite ordered forecasts, four parameter groups and L/S/T required')
    if not np.isfinite(z[~fallback]).all() or (np.diff(z[~fallback],axis=-1)<0).any():raise ValueError('fitted residual quantiles must be finite and ordered')
    out=q.copy()
    for j,(a,b) in enumerate(BINS):
        if fallback[j]:continue
        m=q[:,a:b,...,9,None];s=spread(q[:,a:b],scale)[...,None]
        if method=='L':changed=q[:,a:b]+s*z[j,9]
        elif method=='S':changed=m+s*(z[j,9]+(z[j]-z[j,::-1])/2)
        else:changed=m+s*z[j]
        out[:,a:b]=np.maximum(0,changed)
    return out

def high_hour_mask(high):
    high=np.asarray(high)
    if high.ndim!=2 or high.shape[1]!=68 or high.dtype!=bool:raise ValueError('four offsets by 17 nodes boolean mask required')
    out=np.zeros((len(high),72,17),bool)
    for j,off in enumerate([0,24,48,66]):out[:,off:off+6]=high[:,j*17:(j+1)*17][:,None,:]
    return out

def diagnose_marginals(q,y,selection,scale):
    q=_validate(q,scale);y=np.asarray(y,float);selection=np.asarray(selection)
    if y.shape!=q.shape[:-1] or selection.shape!=y.shape or selection.dtype!=bool or not np.isfinite(q).all():raise ValueError('aligned finite predictions, truth and boolean mask required')
    valid=selection&np.isfinite(y);sat=selection&(q[...,-1]<=10);satn=int(sat.sum());satfail=int((sat&np.isfinite(y)&(y>10)).sum());satunknown=int((sat&~np.isfinite(y)).sum());observed=int(valid.sum())
    mean=lambda a:float(np.mean(a)) if len(a) else float('nan')
    error=(y-q[...,9])[valid];z=(y-q[...,9])[valid]/spread(q,scale)[valid]
    result=dict(records=int(selection.sum()),observed=observed,unknown=int((selection&~np.isfinite(y)).sum()),mean_error=mean(error),median_error=float(np.median(error)) if observed else float('nan'),mae=mean(abs(error)),mean_width=mean((q[...,17]-q[...,1])[selection]),below_second_fraction=mean((y<q[...,1])[valid]),above_eighteenth_fraction=mean((y>q[...,17])[valid]),above_largest_fraction=mean((y>q[...,18])[valid]),central_interval_coverage=mean(((y>=q[...,1])&(y<=q[...,17]))[valid]),mean_predicted_safe=mean((q<=10).mean(-1)[selection]),observed_safe=mean((y<=10)[valid]),saturated_records=satn,saturated_known_failures=satfail,saturated_unknown=satunknown,saturated_failure_lower=satfail/satn if satn else float('nan'),saturated_failure_upper=(satfail+satunknown)/satn if satn else float('nan'),zero_atom_fraction=mean((q==0).mean(-1)[selection]))
    for prob,name in [(.05,'z05'),(.5,'z50'),(.95,'z95')]:result[name]=float(np.quantile(z,prob)) if observed else float('nan')
    return result

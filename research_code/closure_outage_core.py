"""Causal whole-farm outage stress and independent shared-event bounds."""
import numpy as np
from review_baselines_core import masked_histories,loss_event_totals

def outage_history(x,ids,hours):
    if hours not in [0,6,12,24]:raise ValueError('predeclared outage length required')
    mask=np.ones((len(ids),168,x.shape[1]),bool)
    if hours:mask[:,-hours:]=False
    return masked_histories(x,ids,mask)

def paired_bounds(a,b,y,inverse):
    """Pair seed losses; independently verify eventwise endpoint formula."""
    lo,hi,count=loss_event_totals(a,b,y,inverse)
    zero=np.bincount(inverse.ravel(),weights=np.mean(a*a-b*b,axis=0).ravel(),minlength=len(count))
    one=np.bincount(inverse.ravel(),weights=np.mean((a-1)**2-(b-1)**2,axis=0).ravel(),minlength=len(count))
    label=np.full(len(count),np.nan);known=np.isfinite(y);label[inverse[known]]=y[known]
    reflo=np.where(label==0,zero,np.where(label==1,one,np.minimum(zero,one)))
    refhi=np.where(label==0,zero,np.where(label==1,one,np.maximum(zero,one)))
    np.testing.assert_allclose(lo,reflo,atol=1e-12,rtol=0)
    np.testing.assert_allclose(hi,refhi,atol=1e-12,rtol=0)
    return float(lo.sum()/count.sum()),float(hi.sum()/count.sum())

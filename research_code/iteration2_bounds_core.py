"""Exact bounds conditional on frozen actions and unrestricted shared binary missing hours.

No weather completion is exposed to forecasting, calibration, or dispatch.
"""
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

def window_term(raw, issue, start, end, node, threshold=10.):
    """False=known failure, True=known success, tuple=unknown physical hours."""
    if np.asarray(raw).dtype != np.float64:raise ValueError('original raw64 required')
    hours=np.arange(issue+start+1,issue+end+1)
    if not len(hours) or hours.min()<0 or hours.max()>=len(raw):raise ValueError('invalid window')
    values=raw[hours,node];known=np.isfinite(values)
    if np.any(values[known]>threshold):return False
    if known.all():return True
    return tuple((int(hour),int(node)) for hour in hours[~known])

def solve_bounds(constant, terms, time_limit=120.):
    """Terms map tuples of (absolute_hour,node) to signed AND coefficients."""
    combined=defaultdict(float)
    for ks,coefficient in terms.items():
        key=tuple(sorted(set(ks)))
        if not key:constant+=coefficient
        else:combined[key]+=coefficient
    combined={k:v for k,v in combined.items() if abs(v)>1e-12}
    hours=sorted(set().union(*(set(k) for k in combined))) if combined else []
    hindex={key:i for i,key in enumerate(hours)};windows=list(combined)
    n=len(hours)+len(windows);rr=[];cc=[];vv=[];upper=[]
    def add(items,bound):
        row=len(upper);upper.append(bound)
        for col,value in items:rr.append(row);cc.append(col);vv.append(value)
    for i,ks in enumerate(windows):
        col=len(hours)+i
        for k in ks:add([(col,1),(hindex[k],-1)],0)
        add([(col,-1)]+[(hindex[k],1) for k in ks],len(ks)-1)
    objective=np.r_[np.zeros(len(hours)),[combined[k] for k in windows]]
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(upper),n)).tocsc()
    result=dict(constant=float(constant),unknown_hours=len(hours),window_terms=len(windows),constraints=len(upper))
    for side,sign in [('lower',1),('upper',-1)]:
        if not n:
            result[side]=float(constant);result[side+'_witness']=[]
            result[side+'_certificate']=dict(status=0,dual_bound=float(constant),primal=float(constant),absolute_gap=0.,mip_gap=0.,witness_recomputed=float(constant))
            continue
        opt=milp(sign*objective,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=LinearConstraint(matrix,np.full(len(upper),-np.inf),np.array(upper)),options=dict(presolve=True,time_limit=time_limit,mip_rel_gap=0.))
        if opt.status!=0 or opt.x is None:raise RuntimeError(f'Uncertified bound: {opt.message}')
        integer=np.rint(opt.x)
        if np.max(np.abs(integer-opt.x))>1e-6 or np.max(matrix@integer-np.array(upper))>1e-7:raise RuntimeError('invalid feasible witness')
        assign={k:int(integer[i]) for i,k in enumerate(hours)}
        value=float(constant+sum(coef*all(assign[k] for k in ks) for ks,coef in combined.items()))
        primal=float(constant+sign*opt.fun);dual=float(constant+sign*opt.mip_dual_bound)
        gap=abs(primal-dual)
        if abs(value-primal)>1e-7 or gap>1e-7 or opt.mip_gap>1e-9:raise RuntimeError('objective/dual certificate mismatch')
        result[side]=value
        result[side+'_witness']=[[h,node,assign[h,node]] for h,node in hours]
        result[side+'_certificate']=dict(status=int(opt.status),dual_bound=dual,primal=primal,absolute_gap=gap,mip_gap=float(opt.mip_gap),witness_recomputed=value)
    return result

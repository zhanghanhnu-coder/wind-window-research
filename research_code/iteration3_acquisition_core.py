"""Exact nonadaptive acquisition residual width. No unobserved truth is used."""
from collections import defaultdict
import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

def canonical(constant, terms):
    combined=defaultdict(float)
    for key,value in terms.items():
        ks=tuple(sorted(set(key)))
        if not ks:constant+=value
        else:combined[ks]+=value
    return float(constant),{k:v for k,v in sorted(combined.items()) if abs(v)>1e-12}

def evaluate(constant,terms,z):
    return float(constant+sum(v*all(z.get(h,0) for h in ks) for ks,v in terms.items()))

def worst_width(constant, terms, query, time_limit=120.):
    constant,terms=canonical(constant,terms)
    hours=sorted(set().union(*map(set,terms))) if terms else []
    index={h:i for i,h in enumerate(hours)};windows=list(terms);block=len(hours)+len(windows)
    q=sorted(set(query)&set(hours));rr=[];cc=[];vv=[];ub=[]
    def row(items,b):
        r=len(ub);ub.append(b)
        for c,v in items:rr.append(r);cc.append(c);vv.append(v)
    for copy in [0,1]:
        offset=copy*block
        for j,ks in enumerate(windows):
            col=offset+len(hours)+j
            for h in ks:row([(col,1),(offset+index[h],-1)],0)
            row([(col,-1)]+[(offset+index[h],1) for h in ks],len(ks)-1)
    for h in q:
        row([(index[h],1),(block+index[h],-1)],0)
        row([(index[h],-1),(block+index[h],1)],0)
    if not block:return dict(certificate='PASS',worst_remaining_width=0.,status=0,primal=0.,dual_upper=0.,absolute_gap=0.,mip_gap=0.,witness_plus=[],witness_minus=[],query=[],seconds=0.)
    objective=np.r_[np.zeros(len(hours)),list(terms.values()),np.zeros(len(hours)),-np.array(list(terms.values()))]
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(ub),2*block)).tocsc();upper=np.array(ub)
    started=time.perf_counter()
    opt=milp(-objective,integrality=np.ones(2*block),bounds=Bounds(np.zeros(2*block),np.ones(2*block)),constraints=LinearConstraint(matrix,np.full(len(ub),-np.inf),upper),options=dict(presolve=True,time_limit=time_limit,mip_rel_gap=0.))
    result=dict(certificate='UNAVAILABLE',status=int(opt.status),message=str(opt.message),seconds=time.perf_counter()-started,query=[list(h) for h in q])
    if opt.x is None:return result
    integer=np.rint(opt.x)
    if max(abs(opt.x-integer))>1e-6 or max(matrix@integer-upper,default=0)>1e-7:raise AssertionError('infeasible witness')
    plus={h:int(integer[i]) for i,h in enumerate(hours)};minus={h:int(integer[block+i]) for i,h in enumerate(hours)}
    assert all(plus[h]==minus[h] for h in q)
    dp=evaluate(constant,terms,plus);dm=evaluate(constant,terms,minus);value=dp-dm
    primal=float(-opt.fun);dual=float(-opt.mip_dual_bound);gap=abs(primal-dual)
    assert abs(value-primal)<1e-7
    result.update(primal=primal,dual_upper=dual,absolute_gap=gap,mip_gap=float(opt.mip_gap),direct_symbolic_plus=dp,direct_symbolic_minus=dm,witness_plus=[[h,n,plus[h,n]] for h,n in hours],witness_minus=[[h,n,minus[h,n]] for h,n in hours])
    if opt.status==0 and gap<=1e-7 and opt.mip_gap<=1e-9:result.update(certificate='PASS',worst_remaining_width=float(value))
    return result

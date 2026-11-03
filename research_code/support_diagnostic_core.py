"""Fixed-resource finite-support diagnostics; no policy fitting or tuning."""
import numpy as np


def cost_support(booking, recourse, actual_lower, actual_upper, complete, tolerance=1e-6):
    values=np.asarray(recourse,float)
    scalars=np.asarray([booking,actual_lower,actual_upper,tolerance],float)
    if (values.ndim!=1 or not len(values) or not np.isfinite(values).all()
        or (values<0).any() or not np.isfinite(scalars).all() or booking<0 or tolerance<0
        or actual_lower>actual_upper or actual_lower<booking-tolerance
        or not isinstance(complete,(bool,np.bool_))
        or (complete and actual_upper-actual_lower>tolerance)):
        raise ValueError('valid fixed-resource costs and complete/missing bounds required')
    low=float(booking+values.min());high=float(booking+values.max())
    definite=bool(actual_lower>high+tolerance);possible=bool(actual_upper>high+tolerance)
    return dict(support_min=low,support_max=high,definite_excess=definite,
                possible_excess=possible,uncertain_excess=possible and not definite,
                definite_within_support_max=not possible,
                excess_lower=float(max(0.,actual_lower-high)),
                excess_upper=float(max(0.,actual_upper-high)),
                complete_excess=definite if complete else None)


def task_windows(scenarios, truth, tasks, step=6, threshold=10.):
    sc=np.asarray(scenarios,float);y=np.asarray(truth,float)
    if (sc.ndim!=3 or not len(sc) or min(sc.shape[1:])<1 or y.shape!=sc.shape[1:]
        or not np.isfinite(sc).all() or np.isinf(y).any()
        or isinstance(step,(bool,np.bool_)) or not isinstance(step,(int,np.integer))
        or step<1 or not np.isfinite(threshold) or threshold<0 or not tasks):
        raise ValueError('complete scenario cube, aligned partial truth and positive integer step required')
    horizon,nodes=sc.shape[1:];seen=set();rows=[]
    for index,t in enumerate(tasks):
        ident=t['id'];node=t['node'];duration=t['duration']
        if (not isinstance(ident,str) or not ident or ident in seen
            or isinstance(node,(bool,np.bool_)) or not isinstance(node,(int,np.integer))
            or not 0<=node<nodes or isinstance(duration,(bool,np.bool_))
            or not isinstance(duration,(int,np.integer)) or not 1<=duration<=horizon):
            raise ValueError('unique tasks and valid integer node/duration required')
        seen.add(ident)
        for start in range(0,horizon-duration+1,step):
            end=start+duration;actual=y[start:end,node];known=np.isfinite(actual)
            label=0 if np.any(actual[known]>threshold) else 1 if known.all() else -1
            count=int(np.all(sc[:,start:end,node]<=threshold,axis=1).sum())
            rows.append(dict(task_index=index,task_id=ident,node=int(node),start=start,end=end,
                             scenario_count=len(sc),scenario_successes=count,label=label))
    return rows

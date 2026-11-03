"""Explicit artificial closed-weather support; no probability calibration claim."""
import numpy as np
from distributional_schedule_core import checked_probability
from two_stage_core import make_problem,booking_cost


def augment_closed(scenarios,probability,epsilon,threshold=10.):
 sc=np.asarray(scenarios,float)
 if sc.ndim!=3 or min(sc.shape)<1 or not np.isfinite(sc).all():raise ValueError('finite nonempty scenario/time/node array required')
 if isinstance(epsilon,(bool,np.bool_)) or not np.isscalar(epsilon) or not np.isfinite(epsilon) or not 0<epsilon<1:raise ValueError('epsilon must be strictly between zero and one')
 if isinstance(threshold,(bool,np.bool_)) or not np.isscalar(threshold) or not np.isfinite(threshold) or threshold<=0 or not np.isfinite(threshold+1.) or threshold+1.<=threshold:raise ValueError('positive finite threshold with representable closing value required')
 p=checked_probability(probability,len(sc));closed=np.full((1,*sc.shape[1:]),float(threshold)+1.)
 return np.concatenate([sc,closed]),np.r_[(1-float(epsilon))*p,float(epsilon)]


def empirical_cvar(values,alpha=.9):
 y=np.asarray(values,float)
 if y.ndim!=1 or len(y)<1 or not np.isfinite(y).all():raise ValueError('finite nonempty one-dimensional losses required')
 if isinstance(alpha,(bool,np.bool_)) or not np.isscalar(alpha) or not np.isfinite(alpha) or not 0<=alpha<1:raise ValueError('alpha must lie in [0,1)')
 mass=len(y)*(1-float(alpha));weights=np.clip(mass-np.arange(len(y)),0,1)
 return float(np.sort(y)[::-1]@weights/mass)


def closed_support_bound(config,resources):
 p=make_problem(config);book=booking_cost(p,resources);defer=float(sum(t['penalty'] for t in p.config['tasks']))
 return dict(all_defer=defer,booking_cost=book,upper_total=book+defer,strictly_positive_booking=bool((p.resource_cost>0).all()),scope='All tasks may be deferred; this support bound is algebraic, not a weather coverage guarantee.')

"""Recover known equal-ensemble success counts without changing stored forecasts."""
import numpy as np
def canonical_equal_probability(probability,scenario_count):
 if isinstance(scenario_count,bool) or not isinstance(scenario_count,(int,np.integer)) or scenario_count<1:raise ValueError('positive integer equal-ensemble size required')
 p=np.asarray(probability,float)
 if not np.isfinite(p).all() or (p< -1e-12).any() or (p>1+1e-12).any():raise ValueError('finite numerical-unit probabilities required')
 count=np.rint(p*scenario_count);canonical=count/scenario_count
 if (count<0).any() or (count>scenario_count).any() or (abs(canonical-p)>1e-12).any():raise ValueError('probability not on declared equal-ensemble count grid')
 return canonical

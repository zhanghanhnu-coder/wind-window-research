"""Availability audit for cases whose task-node observations are complete."""
from two_stage_core import weather_options
import numpy as np
def completed_task_weather_options(p,weather,threshold=10.):
 w=np.asarray(weather,float);nodes=sorted({t['node'] for t in p.config['tasks']})
 if w.ndim!=3 or w.shape[1]!=p.H or w.shape[2]<=max(nodes) or not np.isfinite(w[:,:,nodes]).all():raise ValueError('all task-node observations must be complete for this audit')
 # Unused-node values cannot affect any of p.choices; no task label is filled.
 return weather_options(p,w,threshold,unknown='pessimistic')

"""Exact removal of zero-mass groups dominated by retained harder conditions."""
import copy,time
import numpy as np
from scenario_equivalence_core import group_conditions,solve_equivalent,lift_probability
from distributional_schedule_core import worst_tv
from regret_core import solve_recourse
from two_stage_core import booking_cost

def reduce_zero_dominated(available,probability):
 groups=group_conditions(available,probability);a=groups['available'];p=groups['probability'];kept=list(np.flatnonzero(p>0));witness=np.arange(len(p));removed=[]
 for i in sorted(np.flatnonzero(p==0),key=lambda j:(int(a[j].sum()),int(j))):
  hard=next((j for j in kept if np.all(~a[j]|a[i])),None)
  if hard is None:kept.append(int(i))
  else:witness[i]=hard;removed.append(int(i))
 kept=np.array(kept,int);removed=np.array(sorted(removed),int)
 return dict(groups=groups,kept_classes=kept,removed_classes=removed,witness=witness,available=a[kept].copy(),probability=p[kept].copy())

def solve_dominance(problem,available,probability,rho,algorithm='ccg',time_limit=60.,guard=lambda:None):
 start=time.monotonic();guard();reduction=reduce_zero_dominated(available,probability);groups=reduction['groups'];keep=reduction['kept_classes'];removed=reduction['removed_classes'];inner=solve_equivalent(problem,reduction['available'],reduction['probability'],rho,algorithm,False,time_limit,guard);guard();solutions=[None]*len(groups['available'])
 for k,g in enumerate(keep):solutions[g]=copy.deepcopy(inner['scenario_solutions'][k])
 extra_start=time.monotonic()
 for g in removed:guard();solutions[g]=solve_recourse(problem,groups['available'][g],inner['x'],time_limit)
 extra_seconds=time.monotonic()-extra_start;book=booking_cost(problem,inner['x']);costs=book+np.array([s['recourse_cost'] for s in solutions]);inverse=groups['inverse'];original=groups['original_probability'];expanded_costs=costs[inverse]
 for g in removed:
  if costs[g]>costs[reduction['witness'][g]]+1e-6:raise RuntimeError('dominated recourse cost violates monotonicity')
 # Lift a reduced optimal adversary through zero padding then exact equivalence.
 qgroup=np.zeros(len(costs));qgroup[keep]=inner['adversary_weights'];q=lift_probability(qgroup,groups);worst=worst_tv(expanded_costs,original,rho);objective=float(q@expanded_costs)
 if abs(objective-inner['objective'])>1e-5 or abs(worst['objective']-objective)>1e-5:raise RuntimeError('full original-support objective disagrees with reduced certificate')
 if .5*abs(q-original).sum()>rho+1e-10:raise RuntimeError('expanded adversary exceeds TV radius')
 guard();return dict(algorithm=algorithm,scenario_count=len(original),equivalence_classes=len(costs),optimization_scenarios=len(keep),zero_mass_classes=int((groups['probability']==0).sum()),kept_zero_mass_classes=int((reduction['probability']==0).sum()),class_inverse=inverse.tolist(),class_probability=groups['probability'].tolist(),kept_classes=keep.tolist(),removed_classes=removed.tolist(),dominance_witness=reduction['witness'].tolist(),optimization_probability=reduction['probability'].tolist(),x=copy.deepcopy(inner['x']),scenario_solutions=[copy.deepcopy(solutions[int(g)]) for g in inverse],scenario_costs=expanded_costs.tolist(),scenario_weights=original.tolist(),adversary_weights=q.tolist(),adversary_tv=float(.5*abs(q-original).sum()),objective=objective,nominal_total=float(original@expanded_costs),booking_cost=book,lower_bound=inner['lower_bound'],upper_bound=inner['upper_bound'],gap=inner['gap'],iterations=inner['iterations'],inner=inner,stage_seconds=dict(inner=inner['seconds'],removed_recourse=extra_seconds),seconds=time.monotonic()-start,certificate=dict(inner['certificate'],original_support_difference=float(worst['objective']-inner['objective']),zero_mass_only=True,all_dominance_witnesses_checked=True),scope='Exact under weather-independent costs and monotone task feasibility; zero nominal mass and retained harder witness required; every original scenario restored.')

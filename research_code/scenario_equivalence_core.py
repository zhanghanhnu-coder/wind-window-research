"""Exact finite-TV aggregation for identical integer-recourse feasibility vectors."""
import copy,time
import numpy as np
from distributional_schedule_core import checked_probability,checked_radius,worst_tv,solve_distribution_ccg,solve_distribution_master
from two_stage_core import booking_cost
from regret_core import solve_recourse

def group_conditions(available,probability):
 a=np.asarray(available)
 if a.ndim!=2 or a.shape[0]<1 or a.shape[1]<1 or a.dtype.kind not in 'biuf' or not np.isfinite(a).all() or not np.isin(a,[0,1]).all():raise ValueError('nonempty binary scenario-by-choice matrix required')
 a=a.astype(bool);p=checked_probability(probability,len(a));lookup={};rows=[];inverse=[]
 for row in a:
  key=row.tobytes()
  if key not in lookup:lookup[key]=len(rows);rows.append(row.copy())
  inverse.append(lookup[key])
 inverse=np.asarray(inverse,int);mass=np.bincount(inverse,weights=p,minlength=len(rows))
 return dict(available=np.stack(rows),probability=mass,inverse=inverse,original_probability=p.copy())

def lift_probability(group_probability,grouping):
 gp=np.asarray(grouping['probability'],float);original=np.asarray(grouping['original_probability'],float);inverse=np.asarray(grouping['inverse']);k=len(gp)
 if k<1 or inverse.ndim!=1 or inverse.dtype.kind not in 'iu' or len(inverse)!=len(original) or (inverse<0).any() or (inverse>=k).any() or (np.bincount(inverse,minlength=k)==0).any():raise ValueError('invalid equivalence mapping')
 original=checked_probability(original,len(inverse));expected=np.bincount(inverse,weights=original,minlength=k)
 if gp.shape!=(k,) or not np.isfinite(gp).all() or not np.allclose(gp,expected,rtol=0,atol=1e-12):raise ValueError('group probabilities do not match original masses')
 q=checked_probability(group_probability,k);lifted=np.zeros(len(inverse))
 for g in range(k):
  members=np.flatnonzero(inverse==g);lifted[members]=q[g]*original[members]/expected[g] if expected[g]>0 else q[g]/len(members)
 return lifted

def solve_equivalent(problem,available,probability,rho,algorithm='ccg',compressed=True,time_limit=60.,guard=lambda:None):
 if algorithm not in ['ccg','full'] or not isinstance(compressed,bool):raise ValueError('known algorithm and boolean compression flag required')
 begin=time.monotonic();guard();groups=group_conditions(available,probability);original=groups['original_probability'];a=np.asarray(available,bool);rho=checked_radius(rho);S=len(a)
 if compressed:work=groups['available'];prob=groups['probability'];inverse=groups['inverse'];mapping=groups
 else:work=a;prob=original;inverse=np.arange(S);mapping=dict(probability=original,inverse=inverse,original_probability=original)
 stages=dict(grouping_seconds=time.monotonic()-begin);guard();start=time.monotonic()
 if algorithm=='ccg':
  plan=solve_distribution_ccg(problem,work,prob,rho,time_limit=time_limit,guard=guard);stages['primary_seconds']=time.monotonic()-start;guard();start=time.monotonic();full=solve_distribution_master(problem,work,prob,rho,time_limit=time_limit);stages['reference_seconds']=time.monotonic()-start;x=plan['x'];solutions=plan['scenario_solutions'];difference=plan['objective']-full['objective'];lower=plan['lower_bound'];upper=plan['upper_bound'];iterations=plan['iterations']
 else:
  full=solve_distribution_master(problem,work,prob,rho,time_limit=time_limit);stages['primary_seconds']=time.monotonic()-start;stages['reference_seconds']=0.;x=full['x'];solutions=[];guard();start=time.monotonic()
  for row in work:guard();solutions.append(solve_recourse(problem,row,x,time_limit))
  stages['recourse_seconds']=time.monotonic()-start;plan=None;difference=0.;lower=full['dual_bound'];upper=full['objective'];iterations=None
 if abs(difference)>1e-5 or full['solver_status']!=0 or full['mip_gap']>1e-9:raise RuntimeError('equivalent-model solver certificate failed')
 guard();book=booking_cost(problem,x);costs=book+np.asarray([s['recourse_cost'] for s in solutions]);adversary=worst_tv(costs,prob,rho);weights=lift_probability(adversary['weights'],mapping);expanded_costs=costs[inverse];expanded=[copy.deepcopy(solutions[int(i)]) for i in inverse];objective=float(weights@expanded_costs)
 if abs(objective-full['objective'])>1e-5:raise RuntimeError('expanded original-support objective failed accounting')
 if abs(.5*abs(weights-original).sum()-adversary['tv'])>1e-10:raise RuntimeError('probability lift changed total variation')
 if abs(worst_tv(expanded_costs,original,rho)['objective']-objective)>1e-6:raise RuntimeError('original-support adversary mismatch')
 return dict(algorithm=algorithm,compressed=compressed,scenario_count=S,optimization_scenarios=len(work),equivalence_classes=len(groups['available']),zero_mass_classes=int((groups['probability']==0).sum()),class_inverse=groups['inverse'].tolist(),class_probability=groups['probability'].tolist(),optimization_inverse=inverse.tolist(),optimization_probability=prob.tolist(),x=copy.deepcopy(x),scenario_solutions=expanded,scenario_costs=expanded_costs.tolist(),scenario_weights=original.tolist(),adversary_weights=weights.tolist(),adversary_tv=float(.5*abs(weights-original).sum()),objective=objective,nominal_total=float(original@expanded_costs),booking_cost=book,lower_bound=float(lower),upper_bound=float(upper),gap=float(max(0,upper-lower)),iterations=iterations,full_reference=full,ccg_plan=plan,reduced_adversary=adversary,stage_seconds=stages,seconds=time.monotonic()-begin,certificate=dict(passed=True,full_status=full['solver_status'],full_gap=full['mip_gap'],algorithm_difference=float(difference),expanded_objective_difference=float(objective-full['objective'])),scope='Exact only when scenario-dependent recourse enters solely through the entire feasibility vector; original support and zero-mass classes retained.')

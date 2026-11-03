"""Exact nonnegative-cost upper bounds; separate from frozen 33/41/43 cores."""
import time,copy
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from two_stage_core import checked_resources,booking_cost,audit_schedule,SolverFailure
from regret_core import unique_resource_groups,unique_node_groups,solve_recourse

from distributional_schedule_core import checked_probability,checked_radius,worst_tv

def solve_distribution_master(problem,available,probability,rho,cuts=None,time_limit=60.):
 begin=time.monotonic();p=problem;available=np.asarray(available)
 if available.ndim!=2 or len(available)<1 or available.shape[1]!=len(p.choices) or not np.isfinite(available).all() or not np.isin(available,[0,1]).all():raise ValueError('binary scenario feasibility required')
 available=available.astype(bool);S,M=available.shape;prob=checked_probability(probability,S);rho=checked_radius(rho);F=float(sum(t['penalty'] for t in p.config['tasks']));original_dual=cuts is None
 if cuts is None and rho==0:cuts=[prob]
 dual=cuts is None;J=len(p.config['tasks']);R=len(p.resource_cost);extra=S+2 if dual else 1;offset=R+extra;nv=offset+S*(M+J);cost=np.zeros(nv);lower=np.zeros(nv);upper=np.ones(nv);integer=np.ones(nv);upper[:R]=p.resource_max;cost[:R]=p.resource_cost;integer[R:offset]=0;upper[R:offset]=np.inf
 if dual:
  eta=R;lam=R+1;u=np.arange(R+2,offset);lower[eta]=-np.inf;lower[u]=-np.inf;cost[eta]=1.;cost[lam]=2*rho;cost[u]=prob
  L=F/(2*rho);lower[eta]=-L;upper[eta]=F+L;upper[lam]=L;lower[u]=-L;upper[u]=L
 else:
  L=None
  if not len(cuts):raise ValueError('at least one probability cut required')
  cuts=np.stack([checked_probability(q,S) for q in cuts])
  if (.5*np.abs(cuts-prob).sum(1)>rho+1e-12).any():raise ValueError('cut distribution outside ambiguity set')
  cost[R]=1.
 recost=np.array([o['cost'] for o in p.choices]+[t['penalty'] for t in p.config['tasks']]);rr=[];cc=[];vv=[];lb=[];ub=[]
 def add(cols,vals,lo=-np.inf,hi=0.):
  row=len(lb)
  for col,value in zip(cols,vals):
   if value:rr.append(row);cc.append(int(col));vv.append(float(value))
  lb.append(lo);ub.append(hi)
 for s in range(S):
  start=offset+s*(M+J);upper[start:start+M]=available[s]
  for j,ix in enumerate(p.task_groups):add([start+i for i in ix]+[start+M+j],[1.]*(len(ix)+1),1.,1.)
  for resource,ix,coef in unique_resource_groups(p):add([start+i for i in ix]+[resource],[*coef,-1.])
  for ix in unique_node_groups(p):add([start+i for i in ix],[1.]*len(ix),hi=1.)
  add(list(range(start,start+M+J)),recost,hi=F)
  if dual:
   add(list(range(start,start+M+J))+[eta,u[s]],[*recost,-1.,-1.]);add([u[s],lam],[1.,-1.]);add([u[s],lam],[-1.,-1.])
 if not dual:
  for q in cuts:
   columns=[R];values=[-1.]
   for s in range(S):
    start=offset+s*(M+J);columns.extend(range(start,start+M+J));values.extend(q[s]*recost)
   add(columns,values)
 add(list(np.flatnonzero(cost)),cost[cost!=0],hi=F)
 matrix=coo_matrix((vv,(rr,cc)),shape=(len(lb),nv)).tocsc();result=milp(cost,integrality=integer,bounds=Bounds(lower,upper),constraints=LinearConstraint(matrix,np.array(lb),np.array(ub)),options=dict(presolve=True,time_limit=float(time_limit),mip_rel_gap=0.))
 if result.status!=0:raise SolverFailure(result.status,result.message,getattr(result,'fun',None),getattr(result,'mip_gap',None))
 if result.x is None or not np.isfinite(result.x).all():raise SolverFailure(4,'optimal status without finite incumbent')
 imask=integer>0
 if np.max(abs(result.x[imask]-np.rint(result.x[imask])))>1e-5:raise SolverFailure(4,'nonintegral incumbent')
 x=np.rint(result.x[:R]).astype(int);solutions=[]
 for s in range(S):
  start=offset+s*(M+J);selected=np.flatnonzero(result.x[start:start+M]>.5).tolist();a=audit_schedule(p,x,available[s],selected);unserved=np.array([j in a['deferred'] for j in range(J)],float);np.testing.assert_allclose(result.x[start+M:start+M+J],unserved,rtol=0,atol=1e-5);solutions.append(a)
 values=np.array([r['recourse_cost'] for r in solutions]);book=booking_cost(p,x);support=worst_tv(values,prob,rho)['objective'] if dual else float(np.max(cuts@values));rebuilt=book+support
 if not np.isclose(rebuilt,result.fun,atol=1e-6,rtol=1e-8):raise SolverFailure(4,'objective failed explicit integer-schedule accounting',float(result.fun))
 return dict(bounds_certificate=dict(active=True,all_defer_cost=F,lambda_upper=L,zero_radius_expected_form=bool(original_dual and rho==0)),x=x.tolist(),objective=float(result.fun),booking_cost=book,scenario_solutions=solutions,recourse_costs=values.tolist(),formulation='probability_LP_dual_integer_recourse' if dual else 'probability_cut_master',probability=prob.tolist(),radius=rho,dual_variables=dict(eta=float(result.x[eta]),lambda_value=float(result.x[lam]),u=result.x[u].tolist()) if dual else None,epigraph=None if dual else float(result.x[R]),solver_status=0,mip_gap=float(result.mip_gap),dual_bound=float(result.mip_dual_bound),seconds=time.monotonic()-begin,variables=nv,constraints=len(lb))
def solve_distribution_ccg(problem,available,probability,rho,time_limit=60.,guard=None):
 p=problem;available=np.asarray(available);prob=checked_probability(probability,len(available));rho=checked_radius(rho);begin=time.monotonic();cuts=[prob];cache={};upper=float('inf');best=None;trace=[];oracle_solves=0;limit=int(np.prod(p.resource_max+1))+1
 for iteration in range(1,limit+1):
  if guard:guard()
  master=solve_distribution_master(p,available,prob,rho,cuts,time_limit);lower=master['objective'];x=master['x'];key=tuple(x)
  if key not in cache:
   solutions=[]
   for a in available:
    if guard:guard()
    solutions.append(solve_recourse(p,a,x,time_limit));oracle_solves+=1
   cache[key]=solutions
  solutions=cache[key];values=np.array([r['recourse_cost'] for r in solutions]);adversary=worst_tv(values,prob,rho);candidate=booking_cost(p,x)+adversary['objective']
  if candidate<upper:upper=candidate;best=dict(x=x.copy(),solutions=solutions,values=values.copy(),adversary=adversary)
  tolerance=1e-7*(1+abs(upper))
  if lower>upper+tolerance:raise SolverFailure(4,'constraint-generation lower bound exceeds upper bound',lower)
  trace.append(dict(iteration=iteration,cuts=[q.tolist() for q in cuts],resources=x.copy(),lower_bound=float(lower),upper_bound=float(upper),candidate_upper=float(candidate),worst_weights=adversary['weights'],tv=adversary['tv'],recourse_costs=values.tolist(),master=master,oracle_solves=oracle_solves))
  if upper-lower<=tolerance:return dict(x=best['x'],objective=float(upper),booking_cost=booking_cost(p,best['x']),lower_bound=float(lower),upper_bound=float(upper),gap=max(0.,upper-lower),iterations=len(trace),trace=trace,oracle_solves=oracle_solves,recourse_costs=best['values'].tolist(),scenario_solutions=best['solutions'],adversary=best['adversary'],nominal_total=booking_cost(p,best['x'])+float(prob@best['values']),radius=rho,probability=prob.tolist(),seconds=time.monotonic()-begin,solver_status=0,scope='finite support; full-weather integer recourse; simulated costs; no confidence guarantee')
  q=np.asarray(adversary['weights'])
  if any(np.max(abs(q-old))<=1e-12 for old in cuts):raise SolverFailure(4,'unclosed gap with repeated worst distribution')
  cuts.append(q)
 raise SolverFailure(4,'finite first-stage resource iteration bound exceeded')

from scenario_equivalence_core import group_conditions,lift_probability
from support_dominance_core import reduce_zero_dominated

def solve_bounded_equivalent(problem,available,probability,rho,algorithm='ccg',compressed=True,time_limit=60.,guard=lambda:None):
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

def solve_bounded_dominance(problem,available,probability,rho,algorithm='ccg',time_limit=60.,guard=lambda:None):
 start=time.monotonic();guard();reduction=reduce_zero_dominated(available,probability);groups=reduction['groups'];keep=reduction['kept_classes'];removed=reduction['removed_classes'];inner=solve_bounded_equivalent(problem,reduction['available'],reduction['probability'],rho,algorithm,False,time_limit,guard);guard();solutions=[None]*len(groups['available'])
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

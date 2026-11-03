"""Round18 finite scenario offsets for minimax regret, isolated from frozen round17.

Inputs offsets are nonnegative perfect-information total costs. This solver cannot
certify their provenance; use perfect_information and retain its certificates.
"""
import time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from two_stage_core import checked_resources,booking_cost,audit_schedule,SolverFailure

def checked_offsets(offsets,S):
 v=np.zeros(S) if offsets is None else np.asarray(offsets,float)
 if v.shape!=(S,) or not np.isfinite(v).all() or (v<0).any():raise ValueError('invalid nonnegative scenario offsets')
 return v

def unique_resource_groups(p):
 seen=set()
 for resource,ix,coef in p.resource_groups:
  key=(resource,tuple(ix),tuple(coef))
  if key not in seen:seen.add(key);yield resource,ix,coef

def unique_node_groups(p):
 seen=set()
 for ix in p.node_groups:
  key=tuple(ix)
  if key not in seen:seen.add(key);yield ix

def perfect_information(p,available,time_limit=60.,guard=None):
 available=np.asarray(available)
 if available.ndim!=2 or len(available)<1:raise ValueError('invalid scenarios')
 results=[]
 for a in available:
  if guard:guard()
  results.append(solve_master(p,a[None],'stochastic',time_limit=time_limit))
 return dict(values=[r['objective'] for r in results],solutions=results)

def solve_master(p,available,mode='robust',fixed=None,time_limit=60.,offsets=None):
 begin=time.monotonic();available=np.asarray(available)
 if available.ndim!=2 or available.shape[1]!=len(p.choices) or len(available)<1 or not np.isfinite(available).all() or not np.isin(available,[0,1]).all():raise ValueError('invalid scenario feasibility array')
 if mode not in ['robust','stochastic']:raise ValueError('unknown optimization mode')
 available=available.astype(bool);S,M=available.shape;shift=checked_offsets(offsets,S);
 if mode=='stochastic' and np.any(shift):raise ValueError('scenario offsets require robust mode')
 if fixed is not None and np.any(shift):raise ValueError('offsets require full first-stage cost')
 J=len(p.config['tasks']);R=len(p.resource_cost);robust=mode=='robust';offset=R+int(robust);nv=offset+S*(M+J);cost=np.zeros(nv);lower=np.zeros(nv);upper=np.ones(nv);integer=np.ones(nv);upper[:R]=p.resource_max
 if fixed is None:cost[:R]=p.resource_cost
 else:lower[:R]=upper[:R]=checked_resources(p,fixed)
 if robust:lower[R]=-float(shift.max());upper[R]=np.inf;integer[R]=0;cost[R]=1
 recost=np.array([o['cost'] for o in p.choices]+[t['penalty'] for t in p.config['tasks']]);rr=[];cc=[];vv=[];lb=[];ub=[]
 def add(cols,vals,lo=-np.inf,hi=0.):
  row=len(lb)
  for col,value in zip(cols,vals):
   if value!=0:rr.append(row);cc.append(int(col));vv.append(float(value))
  lb.append(lo);ub.append(hi)
 for s in range(S):
  start=offset+s*(M+J);upper[start:start+M]=available[s]
  if not robust:cost[start:start+M+J]=recost/S
  for j,ix in enumerate(p.task_groups):add([start+i for i in ix]+[start+M+j],[1.]*(len(ix)+1),1.,1.)
  for resource,ix,coef in unique_resource_groups(p):add([start+i for i in ix]+[resource],[*coef,-1.])
  for ix in unique_node_groups(p):add([start+i for i in ix],[1.]*len(ix),hi=1.)
  if robust:add(list(range(start,start+M+J))+[R],[*recost,-1.],hi=float(shift[s]))
 matrix=coo_matrix((vv,(rr,cc)),shape=(len(lb),nv)).tocsc();result=milp(cost,integrality=integer,bounds=Bounds(lower,upper),constraints=LinearConstraint(matrix,np.array(lb),np.array(ub)),options={'presolve':True,'time_limit':float(time_limit),'mip_rel_gap':0.})
 if result.status!=0:raise SolverFailure(result.status,result.message,getattr(result,'fun',None),getattr(result,'mip_gap',None))
 if result.x is None or not np.isfinite(result.x).all():raise SolverFailure(4,'optimal status without finite incumbent')
 imask=integer>0
 if np.max(abs(result.x[imask]-np.rint(result.x[imask])))>1e-5:raise SolverFailure(4,'nonintegral incumbent')
 x=np.rint(result.x[:R]).astype(int);solutions=[]
 for s in range(S):
  start=offset+s*(M+J);selected=np.flatnonzero(result.x[start:start+M]>.5).tolist();a=audit_schedule(p,x,available[s],selected);unserved=np.array([j in a['deferred'] for j in range(J)],float);np.testing.assert_allclose(result.x[start+M:start+M+J],unserved,atol=1e-5,rtol=0);solutions.append(a)
 values=np.array([r['recourse_cost'] for r in solutions]);book=booking_cost(p,x) if fixed is None else 0.;rebuilt=book+((values-shift).max() if robust else values.mean())
 if not np.isclose(rebuilt,result.fun,atol=1e-6,rtol=1e-8):raise SolverFailure(4,'objective failed independent schedule accounting',float(result.fun))
 return dict(x=x.tolist(),objective=float(result.fun),booking_cost=float(book),eta=float(result.x[R]) if robust else None,scenario_solutions=solutions,solver_status=0,mip_gap=float(result.mip_gap) if getattr(result,'mip_gap',None) is not None else None,dual_bound=float(result.mip_dual_bound) if getattr(result,'mip_dual_bound',None) is not None else None,seconds=time.monotonic()-begin,variables=nv,constraints=len(lb))

def solve_recourse(p,allowed,x,time_limit=60.):
 result=solve_master(p,np.asarray(allowed)[None],mode='stochastic',fixed=x,time_limit=time_limit);answer=result['scenario_solutions'][0];return dict(**answer,solver_status=result['solver_status'],mip_gap=result['mip_gap'],seconds=result['seconds'])

def solve_ccg(p,available,time_limit=60.,guard=None,progress=None,offsets=None):
 available=np.asarray(available);S=len(available)
 if S<1:raise ValueError('empty scenario set')
 shift=checked_offsets(offsets,S)
 begin=time.monotonic();active=[0];cache={};upper=float('inf');best=None;trace=[];oracle_solves=0
 for iteration in range(1,S+2):
  if guard:guard()
  master=solve_master(p,available[active],'robust',time_limit=time_limit,offsets=shift[active]);lower=master['objective'];x=master['x'];key=tuple(x)
  if key not in cache:
   sols=[]
   for s in range(S):
    if guard:guard()
    sols.append(solve_recourse(p,available[s],x,time_limit));oracle_solves+=1
   cache[key]=sols
  sols=cache[key];q=np.array([r['recourse_cost'] for r in sols]);worst=int(np.argmax(q-shift));candidate=booking_cost(p,x)+float(q[worst]-shift[worst])
  if candidate<upper:upper=candidate;best=dict(x=x.copy(),solutions=sols,costs=q.copy())
  tolerance=1e-7*(1+abs(upper))
  if lower>upper+tolerance:raise SolverFailure(4,'C&CG lower bound exceeds upper bound',lower)
  row=dict(iteration=iteration,active_scenarios=active.copy(),resources=x.copy(),lower_bound=float(lower),upper_bound=float(upper),candidate_upper=float(candidate),worst_scenario=worst,oracle_solves=oracle_solves,master_seconds=master['seconds'],master_gap=master['mip_gap']);trace.append(row)
  if progress:progress(row)
  if upper-lower<=tolerance:
   return dict(x=best['x'],objective=float(upper),booking_cost=booking_cost(p,best['x']),lower_bound=float(lower),upper_bound=float(upper),gap=max(0.,upper-lower),trace=trace,iterations=len(trace),active_scenarios=active,oracle_solves=oracle_solves,recourse_costs=best['costs'].tolist(),offsets=shift.tolist(),scenario_solutions=best['solutions'],seconds=time.monotonic()-begin,solver_status=0,scope='finite scenario set, fully revealed stage-two weather, simulated resources/costs')
  if worst in active:raise SolverFailure(4,'unclosed gap despite already-active worst scenario')
  active.append(worst)
 raise SolverFailure(4,'finite scenario iteration bound exceeded')

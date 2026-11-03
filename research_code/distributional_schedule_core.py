"""Finite-support total-variation ambiguity with explicit integer recourse."""
import time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from two_stage_core import checked_resources,booking_cost,audit_schedule,SolverFailure
from regret_core import unique_resource_groups,unique_node_groups,solve_recourse

def checked_probability(p,n):
 p=np.asarray(p,float)
 if p.shape!=(n,) or n<1 or not np.isfinite(p).all() or (p<0).any() or abs(p.sum()-1)>1e-12:raise ValueError('finite nonnegative unit-sum reference probabilities required')
 return p/p.sum()
def checked_radius(rho):
 if not np.isscalar(rho) or not np.isfinite(rho) or not 0<=rho<=1:raise ValueError('total variation radius must lie in [0,1]')
 return float(rho)
def worst_tv(values,probability,rho):
 values=np.asarray(values,float)
 if values.ndim!=1 or not len(values) or not np.isfinite(values).all():raise ValueError('finite scenario losses required')
 p=checked_probability(probability,len(values));rho=checked_radius(rho);q=p.copy();top=int(np.argmax(values));remaining=min(rho,1-p[top])
 for low in np.argsort(values,kind='stable'):
  if remaining<=0 or values[low]>=values[top]:break
  moved=min(float(q[low]),remaining);q[low]-=moved;q[top]+=moved;remaining-=moved
 tv=float(.5*np.abs(q-p).sum());assert q.min()>=-1e-14 and abs(q.sum()-1)<=1e-12 and tv<=rho+1e-12
 return dict(weights=q.tolist(),objective=float(q@values),nominal_objective=float(p@values),tv=tv,radius=rho)
def observed_options(problem,weather,threshold=10.):
 w=np.asarray(weather,float)
 if w.ndim!=2 or w.shape[0]!=problem.H or w.shape[1]<=max(t['node'] for t in problem.config['tasks']) or not np.isfinite(threshold):raise ValueError('invalid observed weather')
 optimistic=[];pessimistic=[];nodes=sorted({t['node'] for t in problem.config['tasks']})
 for o in problem.choices:
  z=w[o['start']:o['end'],o['node']];known=np.isfinite(z);good=not np.any(z[known]>threshold);optimistic.append(good);pessimistic.append(good and known.all())
 return np.array(optimistic),np.array(pessimistic),bool(np.isfinite(w[:,nodes]).all())
def solve_distribution_master(problem,available,probability,rho,cuts=None,time_limit=60.):
 begin=time.monotonic();p=problem;available=np.asarray(available)
 if available.ndim!=2 or len(available)<1 or available.shape[1]!=len(p.choices) or not np.isfinite(available).all() or not np.isin(available,[0,1]).all():raise ValueError('binary scenario feasibility required')
 available=available.astype(bool);S,M=available.shape;prob=checked_probability(probability,S);rho=checked_radius(rho);dual=cuts is None;J=len(p.config['tasks']);R=len(p.resource_cost);extra=S+2 if dual else 1;offset=R+extra;nv=offset+S*(M+J);cost=np.zeros(nv);lower=np.zeros(nv);upper=np.ones(nv);integer=np.ones(nv);upper[:R]=p.resource_max;cost[:R]=p.resource_cost;integer[R:offset]=0;upper[R:offset]=np.inf
 if dual:
  eta=R;lam=R+1;u=np.arange(R+2,offset);lower[eta]=-np.inf;lower[u]=-np.inf;cost[eta]=1.;cost[lam]=2*rho;cost[u]=prob
 else:
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
  if dual:
   add(list(range(start,start+M+J))+[eta,u[s]],[*recost,-1.,-1.]);add([u[s],lam],[1.,-1.]);add([u[s],lam],[-1.,-1.])
 if not dual:
  for q in cuts:
   columns=[R];values=[-1.]
   for s in range(S):
    start=offset+s*(M+J);columns.extend(range(start,start+M+J));values.extend(q[s]*recost)
   add(columns,values)
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
 return dict(x=x.tolist(),objective=float(result.fun),booking_cost=book,scenario_solutions=solutions,recourse_costs=values.tolist(),formulation='probability_LP_dual_integer_recourse' if dual else 'probability_cut_master',probability=prob.tolist(),radius=rho,dual_variables=dict(eta=float(result.x[eta]),lambda_value=float(result.x[lam]),u=result.x[u].tolist()) if dual else None,epigraph=None if dual else float(result.x[R]),solver_status=0,mip_gap=float(result.mip_gap),dual_bound=float(result.mip_dual_bound),seconds=time.monotonic()-begin,variables=nv,constraints=len(lb))
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

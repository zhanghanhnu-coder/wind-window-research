"""Finite-scenario resource booking with integer recourse and exact-oracle C&CG.

Stage-two weather is assumed fully revealed. Costs/resources are explicitly simulated.
"""
from dataclasses import dataclass
import copy,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix

class SolverFailure(RuntimeError):
 def __init__(self,status,message,objective=None,gap=None):
  self.details=dict(status=int(status),message=str(message),objective=objective,gap=gap);super().__init__(str(self.details))

@dataclass
class Problem:
 config:dict
 choices:list
 resource_cost:np.ndarray
 resource_max:np.ndarray
 task_groups:list
 resource_groups:list
 node_groups:list
 B:int
 C:int
 K:int
 H:int

def default_config():
 return dict(horizon=72,step=6,tasks=[dict(id=f'T{j+1}',node=n,duration=h,skill=s,spare=k,quantity=1,penalty=p) for j,(n,h,s,k,p) in enumerate(zip([0,3,6,9,12,16],[6,6,12,6,12,6],['A','B','A','B','A','B'],[0,1,0,1,0,1],[120.,150.,220.,100.,180.,140.]))],boat_booking=[18.,28.],boat_hourly=[1.,1.3],crew_skills=[['A'],['B'],['A','B']],crew_booking=[8.,8.,14.],crew_hourly=[.4,.4,.65],spare_cost=[5.,7.],spare_max=[3,3],late_fraction=.05)

def make_problem(config):
 c=copy.deepcopy(config)
 def integer(v,minimum):
  if isinstance(v,(bool,np.bool_)) or not np.isfinite(v) or int(v)!=v or v<minimum:raise ValueError('invalid integer configuration')
  return int(v)
 H=integer(c['horizon'],1);step=integer(c['step'],1);B=len(c['boat_booking']);C=len(c['crew_booking']);K=len(c['spare_cost'])
 c['horizon']=H;c['step']=step
 if len(c['boat_hourly'])!=B or len(c['crew_hourly'])!=C or len(c['crew_skills'])!=C or len(c['spare_max'])!=K:raise ValueError('resource dimensions differ')
 for key in ['boat_booking','boat_hourly','crew_booking','crew_hourly','spare_cost']:
  v=np.asarray(c[key],float)
  if v.ndim!=1 or not np.isfinite(v).all() or (v<0).any():raise ValueError('resource costs must be finite and nonnegative')
  c[key]=v.tolist()
 if not np.isfinite(c['late_fraction']) or c['late_fraction']<0:raise ValueError('invalid lateness cost')
 c['spare_max']=[integer(v,0) for v in c['spare_max']];seen=set();choices=[]
 if not c['tasks']:raise ValueError('at least one task required')
 for j,t in enumerate(c['tasks']):
  if not isinstance(t['id'],str) or not t['id'] or t['id'] in seen:raise ValueError('task ids must be unique strings')
  seen.add(t['id']);t['node']=integer(t['node'],0);t['duration']=integer(t['duration'],1);t['spare']=integer(t['spare'],0);t['quantity']=integer(t['quantity'],1)
  if t['spare']>=K or not isinstance(t['skill'],str) or not np.isfinite(t['penalty']) or t['penalty']<0:raise ValueError('invalid task requirements/cost')
  for start in range(0,H-t['duration']+1,step):
   for b in range(B):
    for cr in range(C):
     if t['skill'] not in c['crew_skills'][cr]:continue
     cost=(c['boat_hourly'][b]+c['crew_hourly'][cr])*t['duration']+c['late_fraction']*t['penalty']*start/H
     choices.append(dict(task=j,node=t['node'],start=start,end=start+t['duration'],boat=b,crew=cr,spare=t['spare'],quantity=t['quantity'],cost=float(cost)))
 resource_cost=np.array(c['boat_booking']+c['crew_booking']+c['spare_cost'],float);resource_max=np.array([1]*(B+C)+c['spare_max'],int);task_groups=[[i for i,o in enumerate(choices) if o['task']==j] for j in range(len(c['tasks']))];rg=[];ng=[]
 for b in range(B):
  for h in range(H):
   ix=[i for i,o in enumerate(choices) if o['boat']==b and o['start']<=h<o['end']]
   if ix:rg.append((b,ix,np.ones(len(ix))))
 for cr in range(C):
  for h in range(H):
   ix=[i for i,o in enumerate(choices) if o['crew']==cr and o['start']<=h<o['end']]
   if ix:rg.append((B+cr,ix,np.ones(len(ix))))
 for k in range(K):
  ix=[i for i,o in enumerate(choices) if o['spare']==k]
  if ix:rg.append((B+C+k,ix,np.array([choices[i]['quantity'] for i in ix],float)))
 for n in sorted({t['node'] for t in c['tasks']}):
  for h in range(H):
   ix=[i for i,o in enumerate(choices) if o['node']==n and o['start']<=h<o['end']]
   if ix:ng.append(ix)
 return Problem(c,choices,resource_cost,resource_max,task_groups,rg,ng,B,C,K,H)

def weather_options(problem,weather,threshold=10.,unknown='error'):
 w=np.asarray(weather,float)
 if w.ndim!=3 or w.shape[1]!=problem.H or w.shape[2]<=max(t['node'] for t in problem.config['tasks']) or len(w)<1:raise ValueError('weather dimensions do not match')
 if not np.isfinite(threshold) or unknown not in ['error','optimistic','pessimistic']:raise ValueError('invalid weather policy')
 missing=~np.isfinite(w)
 if unknown=='error' and missing.any():raise ValueError('forecast contains unknown weather')
 good=np.where(missing,unknown=='optimistic',w<=threshold);out=np.empty((len(w),len(problem.choices)),bool)
 for i,o in enumerate(problem.choices):out[:,i]=good[:,o['start']:o['end'],o['node']].all(1)
 return out

def checked_resources(p,x):
 x=np.asarray(x,float)
 if x.shape!=p.resource_max.shape or not np.isfinite(x).all() or np.any(abs(x-np.rint(x))>1e-7) or np.any(x<0) or np.any(x>p.resource_max):raise ValueError('invalid first-stage resources')
 return np.rint(x).astype(int)

def booking_cost(p,x):return float(p.resource_cost@checked_resources(p,x))

def audit_schedule(p,x,allowed,selected):
 x=checked_resources(p,x);allowed=np.asarray(allowed,bool);selected=list(selected);M=len(p.choices)
 if allowed.shape!=(M,) or len(selected)!=len(set(selected)):raise ValueError('invalid or repeated choice')
 boat=np.zeros((p.B,p.H),int);crew=np.zeros((p.C,p.H),int);node={t['node']:np.zeros(p.H,int) for t in p.config['tasks']};spares=np.zeros(p.K,int);seen=set();scheduled=[];opcost=0.
 for i in selected:
  if not isinstance(i,(int,np.integer)) or not 0<=i<M or not allowed[i]:raise ValueError('unavailable choice selected')
  o=p.choices[i];t=p.config['tasks'][o['task']]
  if o['task'] in seen or t['skill'] not in p.config['crew_skills'][o['crew']] or not 0<=o['start']<o['end']<=p.H:raise ValueError('task/skill/time violated')
  seen.add(o['task']);boat[o['boat'],o['start']:o['end']]+=1;crew[o['crew'],o['start']:o['end']]+=1;node[o['node']][o['start']:o['end']]+=1;spares[o['spare']]+=o['quantity'];opcost+=o['cost'];scheduled.append(dict(choice=int(i),id=t['id'],**o))
 if (boat>x[:p.B,None]).any() or (crew>x[p.B:p.B+p.C,None]).any() or (spares>x[p.B+p.C:]).any() or any((v>1).any() for v in node.values()):raise ValueError('resource or same-node overlap violated')
 deferred=[j for j in range(len(p.config['tasks'])) if j not in seen];penalty=sum(p.config['tasks'][j]['penalty'] for j in deferred)
 return dict(recourse_cost=float(opcost+penalty),operating_cost=float(opcost),unserved_penalty=float(penalty),completed=len(seen),deferred=deferred,schedule=sorted(scheduled,key=lambda o:(o['start'],o['id'])),spares_used=spares.tolist())

def solve_master(p,available,mode='robust',fixed=None,time_limit=60.):
 begin=time.monotonic();available=np.asarray(available)
 if available.ndim!=2 or available.shape[1]!=len(p.choices) or len(available)<1 or not np.isfinite(available).all() or not np.isin(available,[0,1]).all():raise ValueError('invalid scenario feasibility array')
 if mode not in ['robust','stochastic']:raise ValueError('unknown optimization mode')
 available=available.astype(bool);S,M=available.shape;J=len(p.config['tasks']);R=len(p.resource_cost);robust=mode=='robust';offset=R+int(robust);nv=offset+S*(M+J);cost=np.zeros(nv);lower=np.zeros(nv);upper=np.ones(nv);integer=np.ones(nv);upper[:R]=p.resource_max
 if fixed is None:cost[:R]=p.resource_cost
 else:lower[:R]=upper[:R]=checked_resources(p,fixed)
 if robust:upper[R]=np.inf;integer[R]=0;cost[R]=1
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
  for resource,ix,coef in p.resource_groups:add([start+i for i in ix]+[resource],[*coef,-1.])
  for ix in p.node_groups:add([start+i for i in ix],[1.]*len(ix),hi=1.)
  if robust:add(list(range(start,start+M+J))+[R],[*recost,-1.])
 matrix=coo_matrix((vv,(rr,cc)),shape=(len(lb),nv)).tocsc();result=milp(cost,integrality=integer,bounds=Bounds(lower,upper),constraints=LinearConstraint(matrix,np.array(lb),np.array(ub)),options={'presolve':True,'time_limit':float(time_limit),'mip_rel_gap':0.})
 if result.status!=0:raise SolverFailure(result.status,result.message,getattr(result,'fun',None),getattr(result,'mip_gap',None))
 if result.x is None or not np.isfinite(result.x).all():raise SolverFailure(4,'optimal status without finite incumbent')
 imask=integer>0
 if np.max(abs(result.x[imask]-np.rint(result.x[imask])))>1e-5:raise SolverFailure(4,'nonintegral incumbent')
 x=np.rint(result.x[:R]).astype(int);solutions=[]
 for s in range(S):
  start=offset+s*(M+J);selected=np.flatnonzero(result.x[start:start+M]>.5).tolist();a=audit_schedule(p,x,available[s],selected);unserved=np.array([j in a['deferred'] for j in range(J)],float);np.testing.assert_allclose(result.x[start+M:start+M+J],unserved,atol=1e-5,rtol=0);solutions.append(a)
 values=np.array([r['recourse_cost'] for r in solutions]);book=booking_cost(p,x) if fixed is None else 0.;rebuilt=book+(values.max() if robust else values.mean())
 if not np.isclose(rebuilt,result.fun,atol=1e-6,rtol=1e-8):raise SolverFailure(4,'objective failed independent schedule accounting',float(result.fun))
 return dict(x=x.tolist(),objective=float(result.fun),booking_cost=float(book),eta=float(result.x[R]) if robust else None,scenario_solutions=solutions,solver_status=0,mip_gap=float(result.mip_gap) if getattr(result,'mip_gap',None) is not None else None,dual_bound=float(result.mip_dual_bound) if getattr(result,'mip_dual_bound',None) is not None else None,seconds=time.monotonic()-begin,variables=nv,constraints=len(lb))

def solve_recourse(p,allowed,x,time_limit=60.):
 result=solve_master(p,np.asarray(allowed)[None],mode='stochastic',fixed=x,time_limit=time_limit);answer=result['scenario_solutions'][0];return dict(**answer,solver_status=result['solver_status'],mip_gap=result['mip_gap'],seconds=result['seconds'])

def solve_ccg(p,available,time_limit=60.,guard=None,progress=None):
 available=np.asarray(available);S=len(available)
 if S<1:raise ValueError('empty scenario set')
 begin=time.monotonic();active=[0];cache={};upper=float('inf');best=None;trace=[];oracle_solves=0
 for iteration in range(1,S+2):
  if guard:guard()
  master=solve_master(p,available[active],'robust',time_limit=time_limit);lower=master['objective'];x=master['x'];key=tuple(x)
  if key not in cache:
   sols=[]
   for s in range(S):
    if guard:guard()
    sols.append(solve_recourse(p,available[s],x,time_limit));oracle_solves+=1
   cache[key]=sols
  sols=cache[key];q=np.array([r['recourse_cost'] for r in sols]);worst=int(np.argmax(q));candidate=booking_cost(p,x)+float(q[worst])
  if candidate<upper:upper=candidate;best=dict(x=x.copy(),solutions=sols,costs=q.copy())
  tolerance=1e-7*(1+abs(upper))
  if lower>upper+tolerance:raise SolverFailure(4,'C&CG lower bound exceeds upper bound',lower)
  row=dict(iteration=iteration,active_scenarios=active.copy(),resources=x.copy(),lower_bound=float(lower),upper_bound=float(upper),candidate_upper=float(candidate),worst_scenario=worst,oracle_solves=oracle_solves,master_seconds=master['seconds'],master_gap=master['mip_gap']);trace.append(row)
  if progress:progress(row)
  if upper-lower<=tolerance:
   return dict(x=best['x'],objective=float(upper),booking_cost=booking_cost(p,best['x']),lower_bound=float(lower),upper_bound=float(upper),gap=max(0.,upper-lower),trace=trace,iterations=len(trace),active_scenarios=active,oracle_solves=oracle_solves,recourse_costs=best['costs'].tolist(),scenario_solutions=best['solutions'],seconds=time.monotonic()-begin,solver_status=0,scope='finite scenario set, fully revealed stage-two weather, simulated resources/costs')
  if worst in active:raise SolverFailure(4,'unclosed gap despite already-active worst scenario')
  active.append(worst)
 raise SolverFailure(4,'finite scenario iteration bound exceeded')

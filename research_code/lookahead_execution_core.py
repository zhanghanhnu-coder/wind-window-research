"""Open-loop expected reward planning with causal current-step commitment."""
import time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from two_stage_core import make_problem,checked_resources,audit_schedule,SolverFailure

def window_grid(sc,config,threshold=10.):
 sc=np.asarray(sc);H=config['horizon'];times=range(0,H,config['step'])
 if sc.ndim!=3 or not len(sc) or sc.shape[1]<H or not np.isfinite(sc).all():raise ValueError('invalid scenarios')
 out=np.full((len(times),len(config['tasks'])),np.nan)
 for i,start in enumerate(times):
  for j,t in enumerate(config['tasks']):
   if start+t['duration']<=H:out[i,j]=(sc[:,start:start+t['duration'],t['node']]<=threshold).all(1).mean()
 return out

def plan_future(p,x,attempts,now,probability_grid,span,rule='economic',time_limit=60.):
 begin=time.monotonic();x=checked_resources(p,x);history=list(attempts);audit_schedule(p,x,np.ones(len(p.choices),bool),history);step=p.config['step'];grid=np.asarray(probability_grid,float)
 if not isinstance(now,(int,np.integer)) or isinstance(now,(bool,np.bool_)) or now<0 or now>=p.H or now%step:raise ValueError('invalid current time')
 if not isinstance(span,(int,np.integer)) or isinstance(span,(bool,np.bool_)) or span<=0 or span%step:raise ValueError('invalid planning span')
 if rule not in ['economic','guard80'] or grid.shape!=(len(range(0,p.H,step)),len(p.config['tasks'])):raise ValueError('invalid forecast grid/rule')
 if any(p.choices[i]['start']>=now for i in history):raise ValueError('old commitment starts in current/future step')
 seen={p.choices[i]['task'] for i in history};choices=[];values=[]
 for i,o in enumerate(p.choices):
  if o['task'] in seen or o['start']<now or o['start']>=now+span or not x[o['boat']] or not x[p.B+o['crew']]:continue
  prob=grid[(o['start']-now)//step,o['task']]
  if not np.isfinite(prob) or not 0<=prob<=1:raise ValueError('invalid candidate probability')
  value=float(p.config['tasks'][o['task']]['penalty']*prob-o['cost'])
  if value>0 and (rule!='guard80' or prob>=.8):choices.append(i);values.append(value)
 if not choices:return dict(planned=[],selected=[],expected_benefit=0.,dual_bound=0.,mip_gap=0.,solver_status=0,solver_called=False,candidate_count=0,seconds=time.monotonic()-begin)
 columns={i:j for j,i in enumerate(choices)};rr=[];cc=[];vv=[];upper=[];seenrows=set();previous=set(history)
 def add(indices,coeff,bound):
  cols=[];vals=[]
  for i,v in zip(indices,coeff):
   if i in columns:cols.append(columns[i]);vals.append(float(v))
  if not cols:return
  key=(tuple(cols),tuple(vals),float(bound))
  if key in seenrows:return
  seenrows.add(key);row=len(upper);upper.append(bound)
  for col,value in zip(cols,vals):rr.append(row);cc.append(col);vv.append(value)
 for ix in p.task_groups:add(ix,np.ones(len(ix)),1)
 for resource,ix,coef in p.resource_groups:add(ix,coef,x[resource]-sum(v for i,v in zip(ix,coef) if i in previous))
 for ix in p.node_groups:add(ix,np.ones(len(ix)),1-sum(i in previous for i in ix))
 matrix=coo_matrix((vv,(rr,cc)),shape=(len(upper),len(choices))).tocsc();gain=np.asarray(values);result=milp(-gain,integrality=np.ones(len(choices)),bounds=Bounds(np.zeros(len(choices)),np.ones(len(choices))),constraints=LinearConstraint(matrix,np.full(len(upper),-np.inf),np.array(upper)),options={'presolve':True,'time_limit':float(time_limit),'mip_rel_gap':0.})
 if result.status!=0:raise SolverFailure(result.status,result.message,getattr(result,'fun',None),getattr(result,'mip_gap',None))
 if result.x is None or not np.isfinite(result.x).all() or np.max(abs(result.x-np.rint(result.x)))>1e-5:raise SolverFailure(4,'invalid incumbent')
 selectedcols=np.flatnonzero(result.x>.5);planned=[choices[j] for j in selectedcols];audit_schedule(p,x,np.ones(len(p.choices),bool),history+planned);value=float(gain[selectedcols].sum())
 if not np.isclose(value,-result.fun,atol=1e-7,rtol=0) or not np.isclose(value,-result.mip_dual_bound,atol=1e-7,rtol=0):raise SolverFailure(4,'expected reward certificate mismatch')
 return dict(planned=planned,selected=[i for i in planned if p.choices[i]['start']==now],expected_benefit=value,dual_bound=float(-result.mip_dual_bound),mip_gap=float(result.mip_gap),solver_status=int(result.status),solver_called=True,candidate_count=len(choices),seconds=time.monotonic()-begin)

def roll_execution(config,x,probability_grids,quality,span,rule='economic',guard=None):
 p=make_problem(config);x=checked_resources(p,x);times=list(range(0,p.H,p.config['step']));grids=np.asarray(probability_grids,float);quality=np.asarray(quality)
 if grids.shape!=(len(times),len(times),len(p.config['tasks'])) or quality.shape!=(len(times),) or not np.isin(quality,[0,1]).all():raise ValueError('invalid update arrays')
 attempts=[];steps=[];previous_plan={};postponements=0;future_only=0
 for j,now in enumerate(times):
  if guard:guard()
  if quality[j]:decision=plan_future(p,x,attempts,now,grids[j],span,rule)
  else:decision=dict(planned=[],selected=[],expected_benefit=0.,dual_bound=0.,mip_gap=0.,solver_status=0,solver_called=False,candidate_count=0,seconds=0.)
  current_plan={p.choices[i]['task']:p.choices[i]['start'] for i in decision['planned']};postponements+=sum(task in previous_plan and start>previous_plan[task] for task,start in current_plan.items());future_only+=bool(decision['planned'] and not decision['selected']);previous_plan=current_plan;attempts.extend(decision['selected']);steps.append(dict(now=now,quality_ok=bool(quality[j]),**decision))
 audit_schedule(p,x,np.ones(len(p.choices),bool),attempts)
 return dict(resources=x.tolist(),span=int(span),rule=rule,attempts=attempts,steps=steps,skipped_updates=int((~quality.astype(bool)).sum()),future_only_updates=int(future_only),postponed_task_plans=int(postponements))

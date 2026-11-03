"""Hourly capacity MILP with scenario loss; costs are illustrative points, not money."""
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix

def validate(tasks,nodes,crews,travel,threshold,minimum_probability,policy,risk_weight):
 if isinstance(crews,bool) or int(crews)!=crews or not 1<=crews<=3:raise ValueError('维修队数量必须为 1 到 3')
 if int(travel)!=travel or not 0<=travel<=6:raise ValueError('单程路途占用必须为 0 到 6 个整小时')
 if not np.isfinite(threshold) or not 1<=threshold<=60:raise ValueError('风速试验阈值必须为 1 到 60')
 if not 0<=minimum_probability<=1:raise ValueError('概率门槛必须为 0 到 1')
 if not np.isfinite(risk_weight) or not 0<=risk_weight<=5:raise ValueError('风险权重必须为 0 到 5')
 if policy not in ['fifo','expected','cvar','robust']:raise ValueError('未知调度方法')
 if len(tasks)>40:raise ValueError('每次最多安排 40 项任务')
 ids=[]
 for t in tasks:
  if not isinstance(t['id'],str) or not 1<=len(t['id'])<=60:raise ValueError('任务编号须为 1 到 60 字符')
  ids.append(t['id'])
  if t['node'] not in nodes:raise ValueError('机组编号不在演示数据内')
  for k in ['duration','priority','release','due']:
   if isinstance(t[k],bool) or not np.isfinite(t[k]) or int(t[k])!=t[k]:raise ValueError('时长、优先级、释放和到期时间必须为整数')
  if not 1<=t['duration']<=24 or not 1<=t['priority']<=5:raise ValueError('时长须为 1 到 24 小时，优先级为 1 到 5')
  if not 0<=t['release']<=168 or not -10000<=t['due']<=10000:raise ValueError('任务时间超出范围')
 if len(ids)!=len(set(ids)):raise ValueError('任务编号不能重复')

def plan(scenarios,nodes,tasks,crews=1,travel=1,policy='expected',threshold=10,minimum_probability=.8,risk_weight=1,busy=None):
 validate(tasks,nodes,crews,travel,threshold,minimum_probability,policy,risk_weight)
 z=np.asarray(scenarios);busy=busy or []
 if z.ndim!=3 or z.shape[1]!=72 or z.shape[2]!=len(nodes) or not np.isfinite(z).all() or len(z)<2:raise ValueError('天气情景数组不完整')
 S,H,N=z.shape;availability=np.ones((crews,H),dtype=int);node_availability=np.ones((N,H),dtype=int)
 for k,reservation in enumerate(busy):
  if isinstance(reservation,dict):a,b,crew=reservation['start'],reservation['end'],reservation['crew']
  elif len(reservation)==3:a,b,crew=reservation
  else:a,b=reservation;crew=k+1
  if int(a)!=a or int(b)!=b or a<0 or b<=a or int(crew)!=crew or not 1<=crew<=crews:raise ValueError('已占用时间段或维修队编号不合法')
  availability[crew-1,max(0,a):min(H,b)]-=1
  if isinstance(reservation,dict) and reservation.get('node') is not None:
   if reservation['node'] not in nodes:raise ValueError('已有任务的机组编号不合法')
   node_availability[nodes.index(reservation['node']),max(0,a):min(H,b)]-=1
 if availability.min()<0:raise ValueError('同一维修队的已有任务互相重叠')
 if node_availability.min()<0:raise ValueError('同一机组的已有任务互相重叠')
 candidates=[];success=[];cost=[];weather_eligible=set();diagnostics={}
 for j,t in enumerate(tasks):
  duration=t['duration']+2*travel;node=nodes.index(t['node'])
  diagnostic={'id':t['id'],'best_probability':None,'best_start':None,'total_occupied_hours':duration,'required_probability':1. if policy=='robust' else minimum_probability};diagnostics[t['id']]=diagnostic
  for start in range(0,H-duration+1,3):
   if start<t['release']:continue
   ok=(z[:,start:start+duration,node]<=threshold).all(1);p=float(ok.mean())
   if diagnostic['best_probability'] is None or p>diagnostic['best_probability']:diagnostic['best_probability']=p;diagnostic['best_start']=start
   if (policy=='robust' and not ok.all()) or p+1e-10<minimum_probability:continue
   end=start+duration;weather_eligible.add(t['id'])
   if not node_availability[node,start:end].all():continue
   for crew in range(crews):
    if not availability[crew,start:end].all():continue
    candidates.append(dict(t,task_index=j,start=start,end=end,crew=crew+1,probability=p,travel_hours=2*travel,late_hours=max(0,end-t['due'])))
    success.append(ok)
    # End/deadline/priority tradeoffs explicitly in synthetic score units.
    cost.append(.012*duration+.003*start)
 status='empty';gap=None;chosen=[]
 if candidates:
  K=len(candidates);ok=np.array(success).T;weights=np.array([c['priority'] for c in candidates])
  # Deferring past the horizon also incurs lateness. Otherwise old tasks can
  # become permanently unattractive simply because they already missed a deadline.
  deferred_loss=np.array([c['priority']*(1+.025*max(0,H-c['due'])) for c in candidates])
  completed_loss=np.array([.005*c['priority']*c['late_hours'] for c in candidates])
  delta=np.where(ok,completed_loss-deferred_loss,2*weights)
  if policy=='fifo':
   usage=np.zeros((crews,H),dtype=int);node_usage=np.zeros((N,H),dtype=int);used=set()
   for k,c in enumerate(candidates):
    lane=c['crew']-1;node=nodes.index(c['node'])
    if c['id'] in used or np.any(usage[lane,c['start']:c['end']]+1>availability[lane,c['start']:c['end']]) or np.any(node_usage[node,c['start']:c['end']]+1>node_availability[node,c['start']:c['end']]):continue
    used.add(c['id']);chosen.append(k);usage[lane,c['start']:c['end']]+=1;node_usage[node,c['start']:c['end']]+=1
   status='fifo'
  else:
   tail=policy=='cvar';nv=K+(S+1 if tail else 0);node_begin=len(tasks)+crews*H;resource_end=node_begin+N*H;nr=resource_end+(S if tail else 0)
   baseline_loss=sum(t['priority']*(1+.025*max(0,H-t['due'])) for t in tasks)
   A=lil_matrix((nr,nv));ub=np.concatenate([np.ones(len(tasks)),availability.ravel(),node_availability.ravel(),np.full(S,-baseline_loss) if tail else []])
   for k,c in enumerate(candidates):
    A[c['task_index'],k]=1;offset=len(tasks)+(c['crew']-1)*H;A[offset+c['start']:offset+c['end'],k]=1
    offset=node_begin+nodes.index(c['node'])*H;A[offset+c['start']:offset+c['end'],k]=1
   objective=np.zeros(nv);objective[:K]=delta.mean(0)+np.array(cost)
   integrality=np.zeros(nv);integrality[:K]=1;upper=np.full(nv,np.inf);upper[:K]=1
   if tail:
    A[resource_end:,:K]=delta;A[resource_end:,K]=-1
    for s in range(S):A[resource_end+s,K+1+s]=-1
    objective[K]=risk_weight;objective[K+1:]=risk_weight/(.2*S)
   res=milp(objective,integrality=integrality,bounds=Bounds(np.zeros(nv),upper),constraints=LinearConstraint(A.tocsr(),np.full(nr,-np.inf),ub),options={'time_limit':3,'mip_rel_gap':.01})
   status={0:'optimal',1:'time_limit',2:'infeasible',3:'unbounded',4:'solver_error'}.get(res.status,str(res.status))
   gap=float(res.mip_gap) if getattr(res,'mip_gap',None) is not None and np.isfinite(res.mip_gap) else None
   if res.x is not None:chosen=np.flatnonzero(res.x[:K]>.5).tolist()
   elif res.status==1:status='time_limit_no_incumbent'
 # Validate every returned incumbent independently, including time-limited solves.
 scheduled=sorted([candidates[k] for k in chosen],key=lambda c:c['start']);usage=np.zeros((crews,H),int);node_usage=np.zeros((N,H),int);seen=set()
 for c in scheduled:
  if c['id'] in seen or c['start']<c['release'] or c['end']>H:raise RuntimeError('求解结果未通过任务边界复核')
  seen.add(c['id']);usage[c['crew']-1,c['start']:c['end']]+=1;node_usage[nodes.index(c['node']),c['start']:c['end']]+=1;c.pop('task_index',None)
 if np.any(usage>availability):raise RuntimeError('求解结果未通过维修队身份与资源冲突复核')
 if np.any(node_usage>node_availability):raise RuntimeError('求解结果未通过同机组排他复核')
 def reason(t):
  d=diagnostics[t['id']]
  if status=='time_limit_no_incumbent':return '求解超时且未取得方案，保留待办'
  if d['best_probability'] is None:return '释放时间与总占用时长超出72小时可安排范围'
  if t['id'] not in weather_eligible:return f"最高情景通过比例{d['best_probability']:.1%}，未达到{d['required_probability']:.1%}门槛"
  if not any(c['id']==t['id'] for c in candidates):return '可用天气时段内维修队或同一机组已被占用'
  return '资源、等待与风险权衡后暂缓'
 deferred=[dict(t,reason=reason(t),diagnostic=diagnostics[t['id']]) for t in tasks if t['id'] not in seen]
 return {'scheduled':scheduled,'deferred':deferred,'solver_status':status,'mip_gap':gap,'candidate_count':len(candidates),'policy':policy,'scenario_count':S,'diagnostics':list(diagnostics.values()),'scope':'模拟维修队和路途占用；风速条件代理，不代表船舶可达或安全许可'}

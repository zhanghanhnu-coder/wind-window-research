"""Rolling closed-loop simulation. Reveal weather only after an attempt ends."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scheduler import plan
from planning import make_plan
from forecast import Forecaster
from model import ART

def outcome(raw,start,end,node,threshold):
 # Attempt [start,end) occupies future hour-ending records start+1 ... end.
 values=raw[start+1:end+1,node]
 if len(values)!=end-start or not np.isfinite(values).all():return 'unknown'
 return 'completed' if (values<=threshold).all() else 'failed'

def replay(f,start='2024-11-01',days=30,policy='expected',model='graph',crews=1,travel=1,threshold=10,minimum_probability=.8,risk_weight=1,arrivals_per_day=1):
 if int(days)!=days or not 1<=days<=30:raise ValueError('回放天数必须为 1 到 30')
 if int(arrivals_per_day)!=arrivals_per_day or not 1<=arrivals_per_day<=3:raise ValueError('每日新增任务必须为 1 到 3')
 begin=f.locate(start)
 if f.times[begin].hour!=0:raise ValueError('回放从每日 00:00 开始')
 f.locate(str(f.times[begin]+pd.Timedelta(days=days-1)))
 tasks=[];attempts=[];daily=[];solver_counts={}
 def settle(now):
  for a in attempts:
   if a['outcome']=='running' and a['end_index']<=now:
    result=outcome(f.raw,a['start_index'],a['end_index'],f.nodes.index(a['node']),threshold);a['outcome']=result;a['revealed_index']=now
    t=next(t for t in tasks if t['id']==a['id']);t['state']='pending' if result=='failed' else result
    if result=='completed':t['completed_index']=a['end_index']
 def snapshot(now):
  counts={s:sum(t['state']==s for t in tasks) for s in ['pending','running','completed','unknown']}
  if sum(counts.values())!=len(tasks):raise RuntimeError('任务台账不守恒')
  return dict(time=str(f.times[now]),arrived=len(tasks),**counts)
 for day in range(days):
  now=begin+day*24;settle(now)
  for k in range(arrivals_per_day):
   number=day*arrivals_per_day+k
   tasks.append({'id':f'T{number+1:03d}','node':f.nodes[number%len(f.nodes)],'duration':6,'priority':number%3+1,'release_index':now,'due_index':now+72,'state':'pending','attempt_count':0})
  pending=sorted([t for t in tasks if t['state']=='pending'],key=lambda t:(t['due_index'],-t['priority'],t['id']))
  subset=pending[:40]
  ts=[{k:t[k] for k in ['id','node','duration','priority']}|{'release':max(0,t['release_index']-now),'due':t['due_index']-now} for t in subset]
  busy=[{'start':max(0,a['start_index']-now),'end':a['end_index']-now,'crew':a['crew'],'node':a['node']} for a in attempts if a['outcome']=='running' and a['end_index']>now]
  try:
   proposed=make_plan(f,str(f.times[now]),ts,model,crews,travel,policy,threshold,minimum_probability,risk_weight,busy)
  except ValueError as e:
   if '数据不足' not in str(e):raise
   proposed={'scheduled':[],'solver_status':'insufficient_data'}
  solver_counts[proposed['solver_status']]=solver_counts.get(proposed['solver_status'],0)+1
  # Only next-day starts are committed; later recommendations are recalculated tomorrow.
  for c in proposed['scheduled']:
   if c['start']>=24:continue
   t=next(t for t in tasks if t['id']==c['id']);t['state']='running';t['attempt_count']+=1
   attempts.append(dict(c,start_index=now+c['start'],end_index=now+c['end'],issued_index=now,outcome='running'))
  daily.append(snapshot(now))
 end=begin+days*24;settle(end);final=snapshot(end)
 completed=[t for t in tasks if t['state']=='completed'];known=[a for a in attempts if a['outcome'] in ['completed','failed']]
 waiting=[t['completed_index']-t['release_index'] for t in completed]
 result={'settings':dict(start=str(f.times[begin]),days=days,policy=policy,model=model,crews=crews,travel=travel,threshold=threshold,minimum_probability=minimum_probability,risk_weight=risk_weight,arrivals_per_day=arrivals_per_day),'summary':dict(final,attempts=len(attempts),failed_attempts=sum(a['outcome']=='failed' for a in attempts),unknown_attempts=sum(a['outcome']=='unknown' for a in attempts),known_attempts=len(known),failure_rate=sum(a['outcome']=='failed' for a in attempts)/len(known) if known else None,average_completion_hours=float(np.mean(waiting)) if waiting else None,overdue_pending=sum(t['state']=='pending' and t['due_index']<end for t in tasks),pending_task_days=sum(d['pending'] for d in daily),solver_counts=solver_counts),'daily':daily+[final],'tasks':tasks,'attempts':attempts,'notes':['每日模拟新增工单，6小时维修，72小时期限；失败任务重新排队；未知结果单独等待人工核对。','路途时间、任务和维修队均为模拟；只检验风速代理条件。','只在每天00:00获知已经结束的任务结果；每次只落实24小时内出发的任务。','到达总数 = 已完成 + 待安排 + 正在执行 + 结果未知。','每日最多将最早到期的40项待办送入求解器，其余留在队列。']}
 return result

def benchmark():
 f=Forecaster();out=[]
 for model in ['graph','persistence']:
  for policy in ['fifo','expected','cvar','robust']:
   print('REPLAY',model,policy,flush=True)
   r=replay(f,model=model,policy=policy);(ART/f'replay_{model}_{policy}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');out.append(r['settings']|r['summary'])
 pd.DataFrame(out).drop(columns=['solver_counts']).to_csv(ART/'replay_summary.csv',index=False,encoding='utf-8-sig')
 print(pd.DataFrame(out).to_string(index=False),flush=True)

if __name__=='__main__':benchmark()

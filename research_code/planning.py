"""One consistent input-quality gate for interactive plans and rolling replay."""
import hashlib
from scheduler import plan,validate

def make_plan(f,issue,tasks,model='graph',crews=1,travel=1,policy='expected',threshold=10,minimum_probability=.8,risk_weight=1,busy=None):
 validate(tasks,f.nodes,crews,travel,threshold,minimum_probability,policy,risk_weight)
 quality=f.quality(issue);allowed={q['node'] for q in quality if q['eligible']}
 eligible=[t for t in tasks if t['node'] in allowed];blocked=[t for t in tasks if t['node'] not in allowed]
 sc,_,selected=f.scenarios(issue,model)
 r=plan(sc,f.nodes,eligible,crews,travel,policy,threshold,minimum_probability,risk_weight,busy)
 r['deferred'] += [dict(t,reason='该机组近期数据不足或已过时，暂停自动安排',quality_blocked=True) for t in blocked]
 r.update(version='2.0',issue=issue,model=model,quality=quality,quality_blocked=len(blocked),scenario_fingerprint=hashlib.sha256(sc.tobytes()).hexdigest(),scenario_indices=selected.tolist(),settings={'issue':issue,'model':model,'crews':crews,'travel':travel,'policy':policy,'threshold':threshold,'minimum_probability':minimum_probability,'risk_weight':risk_weight},input_tasks=tasks)
 r['expected_successes']=sum(c['probability'] for c in r['scheduled']);r['expected_failures']=len(r['scheduled'])-r['expected_successes']
 r['interpretation']='期望完成/失败是同一组情景下的事前统计，不是实际维修结果。'
 return r

def compare_plans(f,issue,tasks,**settings):
 settings.pop('policy',None)
 results=[make_plan(f,issue,tasks,policy=p,**settings) for p in ['fifo','expected','cvar','robust']]
 assert len({r['scenario_fingerprint'] for r in results})==1
 return {'version':'2.0','issue':issue,'input_tasks':tasks,'settings':settings,'results':results,'scope':'同批任务、同一预测和情景；只比较事前方案，不调用未来实测。'}

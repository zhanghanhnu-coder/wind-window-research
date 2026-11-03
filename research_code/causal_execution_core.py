"""Small-batch causal dispatch: fixed bookings, irrevocable single attempts."""
import itertools
import numpy as np
from two_stage_core import make_problem,checked_resources,booking_cost,audit_schedule

def dispatch_now(p,x,attempts,now,probabilities,rule='economic'):
 x=checked_resources(p,x);attempts=list(attempts);history=audit_schedule(p,x,np.ones(len(p.choices),bool),attempts)
 if isinstance(now,(bool,np.bool_)) or not isinstance(now,(int,np.integer)) or now<0 or now>=p.H or now%p.config['step']:raise ValueError('invalid update time')
 if any(p.choices[i]['start']>=now for i in attempts):raise ValueError('previous attempts must precede current update')
 pr=np.asarray(probabilities,float)
 if pr.shape!=(len(p.config['tasks']),) or not np.isfinite(pr).all() or (pr<0).any() or (pr>1).any():raise ValueError('invalid completion probabilities')
 if rule not in ['economic','guard80']:raise ValueError('unknown execution rule')
 prior=[p.choices[i] for i in attempts];active=[o for o in prior if o['end']>now];seen={o['task'] for o in prior};busyboat={o['boat'] for o in active};busycrew={o['crew'] for o in active};busynode={o['node'] for o in active};remaining=x[p.B+p.C:]-history['spares_used'];candidates=[];gain={}
 for i,o in enumerate(p.choices):
  j=o['task']
  if o['start']!=now or j in seen or o['node'] in busynode or o['boat'] in busyboat or o['crew'] in busycrew:continue
  if not x[o['boat']] or not x[p.B+o['crew']] or o['quantity']>remaining[o['spare']]:continue
  if rule=='guard80' and pr[j]<.8:continue
  benefit=float(p.config['tasks'][j]['penalty']*pr[j]-o['cost'])
  if benefit>0:candidates.append(i);gain[i]=benefit
 best=();value=0.;maximum=min(int(x[:p.B].sum())-len(busyboat),int(x[p.B:p.B+p.C].sum())-len(busycrew),len(p.config['tasks'])-len(seen))
 for k in range(1,maximum+1):
  for ids in itertools.combinations(candidates,k):
   oo=[p.choices[i] for i in ids]
   if any(len({o[key] for o in oo})!=k for key in ['task','boat','crew','node']):continue
   if any(sum(o['quantity'] for o in oo if o['spare']==s)>remaining[s] for s in range(p.K)):continue
   benefit=float(sum(gain[i] for i in ids))
   if benefit>value or (benefit==value and ids<best):best=ids;value=benefit
 return dict(selected=list(best),expected_benefit=value,candidates=[dict(choice=i,expected_benefit=gain[i]) for i in candidates],remaining_spares=remaining.tolist(),active_attempts=[i for i in attempts if p.choices[i]['end']>now])

def plan_execution(config,x,probabilities,quality,rule='economic'):
 p=make_problem(config);x=checked_resources(p,x);times=list(range(0,p.H,p.config['step']));pr=np.asarray(probabilities,float);quality=np.asarray(quality)
 if pr.shape!=(len(times),len(p.config['tasks'])) or quality.shape!=(len(times),) or not np.isin(quality,[False,True]).all():raise ValueError('invalid update arrays')
 if rule not in ['economic','guard80']:raise ValueError('unknown execution rule')
 attempts=[];steps=[]
 for i,now in enumerate(times):
  if quality[i]:decision=dispatch_now(p,x,attempts,now,pr[i],rule);attempts.extend(decision['selected'])
  else:decision=dict(selected=[],expected_benefit=0.,candidates=[],remaining_spares=(x[p.B+p.C:]-audit_schedule(p,x,np.ones(len(p.choices),bool),attempts)['spares_used']).tolist(),active_attempts=[j for j in attempts if p.choices[j]['end']>now])
  steps.append(dict(now=now,quality_ok=bool(quality[i]),probabilities=pr[i].tolist() if quality[i] else None,**decision))
 audit_schedule(p,x,np.ones(len(p.choices),bool),attempts)
 return dict(resources=x.tolist(),rule=rule,attempts=attempts,steps=steps,skipped_updates=int((~quality.astype(bool)).sum()))

def evaluate_execution(p,x,attempts,truth,threshold=10.):
 truth=np.asarray(truth,float)
 if truth.ndim!=2 or truth.shape[0]!=p.H or truth.shape[1]<=max(t['node'] for t in p.config['tasks']) or not np.isfinite(threshold):raise ValueError('invalid evaluation weather')
 result=audit_schedule(p,x,np.ones(len(p.choices),bool),attempts);outcomes=[];failed=0;successful=0;unknown=0;known_penalty=result['unserved_penalty'];unknown_penalty=0.
 for i in attempts:
  o=p.choices[i];t=p.config['tasks'][o['task']];row=truth[o['start']:o['end'],o['node']]
  if (np.isfinite(row)&(row>threshold)).any():status='failed';failed+=1;known_penalty+=t['penalty']
  elif np.isfinite(row).all():status='successful';successful+=1
  else:status='unknown';unknown+=1;unknown_penalty+=t['penalty']
  outcomes.append(dict(choice=i,task=o['task'],status=status,revealed_at=o['end'],known_hours=int(np.isfinite(row).sum())))
 book=booking_cost(p,x);lower=book+result['operating_cost']+known_penalty
 return dict(booking_cost=book,operating_cost=result['operating_cost'],penalty_lower=float(known_penalty),penalty_upper=float(known_penalty+unknown_penalty),cost_lower=float(lower),cost_upper=float(lower+unknown_penalty),successful=successful,failed=failed,unknown=unknown,unattempted=len(result['deferred']),attempted=len(attempts),spares_used=result['spares_used'],outcomes=outcomes)

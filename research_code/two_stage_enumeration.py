"""Independent tiny-instance enumeration; reads original configuration, not MILP rows."""
from itertools import product
import numpy as np

def brute_recourse(c,weather,x,threshold=10.):
 H=c['horizon'];B=len(c['boat_booking']);C=len(c['crew_booking']);K=len(c['spare_cost']);choices=[]
 for t in c['tasks']:
  options=[None]
  for start in range(0,H-t['duration']+1,c['step']):
   if not np.all(weather[start:start+t['duration'],t['node']]<=threshold):continue
   for b in range(B):
    for cr in range(C):
     if x[b] and x[B+cr] and t['skill'] in c['crew_skills'][cr]:options.append((start,start+t['duration'],b,cr))
  choices.append(options)
 best=float('inf');best_choices=None
 for selected in product(*choices):
  used=np.zeros(K,int);valid=True;cost=0.
  for j,o in enumerate(selected):
   t=c['tasks'][j]
   if o is None:cost+=t['penalty'];continue
   start,end,b,cr=o;used[t['spare']]+=t['quantity'];cost+=(c['boat_hourly'][b]+c['crew_hourly'][cr])*(end-start)+c['late_fraction']*t['penalty']*start/H
   for jj,oo in enumerate(selected[:j]):
    if oo is None:continue
    if max(start,oo[0])<min(end,oo[1]) and (b==oo[2] or cr==oo[3] or t['node']==c['tasks'][jj]['node']):valid=False;break
   if not valid:break
  if valid and np.all(used<=np.asarray(x[B+C:])) and cost<best:best=cost;best_choices=selected
 return dict(recourse_cost=float(best),choices=best_choices)

def brute_two_stage(c,weather,mode='robust'):
 B=len(c['boat_booking']);C=len(c['crew_booking']);limits=[1]*(B+C)+c['spare_max'];fee=np.array(c['boat_booking']+c['crew_booking']+c['spare_cost']);best=float('inf');answer=None
 for x in product(*[range(m+1) for m in limits]):
  q=[brute_recourse(c,w,x)['recourse_cost'] for w in weather];cost=float(fee@x)+(max(q) if mode=='robust' else float(np.mean(q)))
  if cost<best:best=cost;answer=list(x)
 return dict(objective=float(best),x=answer)

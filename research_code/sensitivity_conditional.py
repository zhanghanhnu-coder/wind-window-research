"""Paired raw/smoothed diagnostic with identical event AND timestamp support.

Canonical unordered pair orientation matches the previous graph study's support.
This post-fit diagnostic never changes model features, identity, or graph selection.
"""
import json
import numpy as np
from research_conditional import OUT,ART,PAIRS,load,csv
from conditional_core import changes
from propagation_graph import lag_profile

def matched_profiles(raw,smooth,source,target,max_lag=12,minimum=36):
 valid=np.isfinite(raw)&np.isfinite(smooth);raw=np.where(valid,raw,np.nan);smooth=np.where(valid,smooth,np.nan);i,j=sorted([source,target]);fixed=1 if source==i else -1
 result={'fixed_lag':fixed}
 for name,values in [('raw',raw),('smooth30',smooth)]:result[name]=lag_profile(values[:,i],values[:,j],max_lag,'corr',minimum)
 assert result['raw'][2]==result['smooth30'][2]
 return result

def run():
 fine,times,nodes=load();raw=changes(fine['direction'].astype(float),True);smooth=np.load(ART/'event_propagation_v1/event_signals.npz')['turn_signal'];events=json.loads((ART/'event_propagation_v1/events.json').read_text(encoding='utf-8'));rows=[]
 for source,target in PAIRS[:4]:
  si=nodes.index(source);ti=nodes.index(target)
  for event in events:
   if event['kind']!='turn' or not event['window_eligible']:continue
   i=event['index'];r=matched_profiles(raw[i-24:i+49],smooth[i-24:i+49],si,ti);n=r['raw'][2];row={'source':source,'target':target,'event_id':event['event_id'],'month':event['month'],'time':event['time'],'matched_timestamps':n,'valid':n>=36}
   for name in ['raw','smooth30']:
    ls,scores,_=r[name];row[name+'_gain5']=float(scores[12+r['fixed_lag']]-scores[12]) if n>=36 else None;row[name+'_corr5']=float(scores[12+r['fixed_lag']]) if n>=36 else None
   rows.append(row)
 csv(rows,'smoothing_sensitivity_matched.csv');print('Matched event rows',len(rows),'eligible',sum(r['valid'] for r in rows))

if __name__=='__main__':run()

"""Matched-window Brier diagnostic, preserving seed-level squares."""
import numpy as np

CELLS=['AI','GI','AC','GC']
CONTRASTS={'interaction':[-1,1,1,-1], 'initial_model_difference':[1,-1,0,0],
 'current_model_difference':[0,0,1,-1], 'analog_update':[ -1,0,1,0], 'graph_update':[0,-1,0,1]}
KEY=['fold','issue','seed','task','start']

def affine(p,signs):
 p=np.asarray(p,float);v=np.asarray(signs,float)
 return np.sum(v*p*p,axis=-1),np.sum(v*(1-2*p),axis=-1)

def matched(d):
 d=d.copy();d['cell']=d.prediction.map({'AnalogFixed':'A','GraphFixed':'G'})+d.stage.map({'initial_candidate':'I','current_candidate':'C'})
 assert not d.duplicated(KEY+['cell']).any()
 current=d[d.stage=='current_candidate'];assert current.groupby(KEY).size().eq(2).all()
 queue=current[KEY].drop_duplicates()
 x=queue.merge(d,on=KEY,validate='one_to_many')
 assert x.groupby(KEY).size().eq(4).all()
 for col in ['label','complete']:
  assert x.groupby(KEY)[col].nunique(dropna=False).eq(1).all()
 p=x.pivot(index=KEY,columns='cell',values='probability')[CELLS].reset_index()
 meta=x.drop_duplicates(KEY)[KEY+['label','complete']]
 return p.merge(meta,on=KEY,validate='one_to_one')

def weights(d):
 """Windows within origin/seed, three seeds within origin, equal origins."""
 counts=d.groupby(['fold','issue','seed']).issue.transform('size').to_numpy()
 seedcounts=d[['fold','issue','seed']].drop_duplicates().groupby(['fold','issue']).size()
 assert seedcounts.eq(3).all()
 n=len(d[['fold','issue']].drop_duplicates())
 return 1/(counts*3*n)

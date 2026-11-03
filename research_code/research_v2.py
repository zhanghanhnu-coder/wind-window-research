"""Evaluate the fixed v2 replay protocol and unchanged v1 checkpoints on identical rows."""
import json
import numpy as np
import pandas as pd
from forecast import Forecaster
from model import ART,predict
from replay import replay

OUT=ART/'v2'
def context_comparison():
 f=Forecaster();d=np.load(ART/'data.npz');ids=d['test_ids'];truth=np.stack([f.raw[i+1:i+73] for i in ids]);rows=[]
 for name in ['graph','temporal','graph24','temporal24','mean24','random24','persistence']:
  errors=[];medians=[]
  for i in ids:
   sc,med,_=f.scenarios(str(f.times[i]),name);medians.append(med);actual=f.raw[i+1:i+73];es=[]
   for offset in [0,24,48,66]:
    t=actual[offset:offset+6];valid=np.isfinite(t).all(0);y=(t<=10).all(0);p=(sc[:,offset:offset+6]<=10).all(1).mean(0);es.append(np.where(valid,(p-y)**2,np.nan))
   errors.append(es)
  errors=np.array(errors);medians=np.array(medians)
  for month in [0,11,12]:
   mask=np.ones(len(ids),bool) if month==0 else f.times[ids].month==month
   rows.append({'model':name,'seed':20241101,'month':month,'brier_primary':float(np.nanmean(errors[mask])),'mae_all':float(np.nanmean(abs(medians[mask]-truth[mask]))),'valid_window_cells':int(np.isfinite(errors[mask]).sum())})
 pd.DataFrame(rows).to_csv(OUT/'context_comparison.csv',index=False,encoding='utf-8-sig');print('CONTEXT',rows,flush=True)

def benchmark():
 f=Forecaster();rows=[]
 for start,days in [('2024-11-01',30),('2024-12-01',28)]:
  for model in ['graph24','temporal24','persistence']:
   for policy in ['fifo','expected','cvar','robust']:
    r=replay(f,start=start,days=days,model=model,policy=policy)
    (OUT/f'replay_{start[5:7]}_{model}_{policy}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');rows.append(r['settings']|r['summary'])
    print('REPLAY2',start,model,policy,'completed',r['summary']['completed'],'fail',r['summary']['failed_attempts'],flush=True)
 pd.DataFrame(rows).drop(columns=['solver_counts']).to_csv(OUT/'replay_summary.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':context_comparison();benchmark()

"""Score finite-horizon longest-run distributions, not uncensored real-world windows."""
import numpy as np
import pandas as pd
from forecast import Forecaster
from model import ART

def longest_runs(good):
 current=np.zeros((good.shape[0],good.shape[2]),int);longest=current.copy()
 for h in range(good.shape[1]):current=np.where(good[:,h],current+1,0);longest=np.maximum(longest,current)
 return longest

def empirical_crps(samples,truth):
 # Score the empirical CDF: mean|X-y| - 1/2 mean|X-X'|.
 return np.mean(abs(samples-truth),axis=0)-.5*np.mean(abs(samples[:,None,:]-samples[None,:,:]),axis=(0,1))

def run():
 f=Forecaster();ids=np.load(ART/'data.npz')['test_ids'];records=[]
 for name in ['graph24','temporal24','persistence']:
  for i in ids:
   sc,_,_=f.scenarios(str(f.times[i]),name);actual=f.raw[i+1:i+73]
   for horizon in [24,48,72]:
    valid=np.isfinite(actual[:horizon]).all(0);pred=longest_runs(sc[:,:horizon]<=10);observed=longest_runs((actual[:horizon]<=10)[None])[0]
    crps=empirical_crps(pred,observed);p=(pred>=6).mean(0);y=observed>=6
    for j,node in enumerate(f.nodes):
     if valid[j]:records.append({'model':name,'issue':str(f.times[i]),'month':f.times[i].month,'node':node,'horizon':horizon,'length_crps':float(crps[j]),'any6h_brier':float((p[j]-y[j])**2),'pred_any6h':float(p[j]),'observed_any6h':int(y[j]),'observed_longest':int(observed[j])})
 frame=pd.DataFrame(records);frame.to_csv(ART/'v2/window_length_cases.csv',index=False,encoding='utf-8-sig')
 summary=frame.groupby(['model','horizon']).agg(length_crps=('length_crps','mean'),any6h_brier=('any6h_brier','mean'),known_cases=('node','size'),observed_any6h_rate=('observed_any6h','mean')).reset_index();summary['possible_cases']=len(ids)*len(f.nodes);summary.to_csv(ART/'v2/window_length_metrics.csv',index=False,encoding='utf-8-sig');print(summary.to_string(index=False))

if __name__=='__main__':run()

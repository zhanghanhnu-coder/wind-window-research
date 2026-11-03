"""Properties of continuous runs within a finite, hourly forecast horizon."""
import numpy as np

def summarize_windows(scenarios,nodes,threshold,duration):
 z=np.asarray(scenarios)
 if z.ndim!=3 or z.shape[1]!=72 or z.shape[2]!=len(nodes) or not np.isfinite(z).all():raise ValueError('窗口情景必须是完整的72小时数组')
 if int(duration)!=duration or not 1<=duration<=72 or not np.isfinite(threshold):raise ValueError('窗口时长或阈值不合法')
 duration=int(duration);good=z<=threshold;out=[]
 for j,node in enumerate(nodes):
  longest=[];first=[]
  for row in good[:,:,j]:
   run=0;maximum=0;earliest=None
   for h,yes in enumerate(row):
    run=run+1 if yes else 0;maximum=max(maximum,run)
    if run>=duration and earliest is None:earliest=h-duration+1
   longest.append(maximum);first.append(earliest)
  hist=np.bincount(longest,minlength=73)/len(longest);found=[s for s in first if s is not None]
  out.append({'node':node,'longest_hours_by_scenario':longest,'first_start_by_scenario':first,'longest_length_pmf':hist.tolist(),'longest_hours_quantiles':np.quantile(longest,[.1,.5,.9]).tolist(),'no_window_probability':1-len(found)/len(first),'conditional_first_start_quantiles':np.quantile(found,[.1,.5,.9]).tolist() if found else None,'within_horizon_probability':{str(h):sum(s is not None and s+duration<=h for s in first)/len(first) for h in [24,48,72]},'boundary':'仅统计72小时范围内的完整窗口；末端连续段可能仍在继续，不能据此推断真实总长度。'})
 return out

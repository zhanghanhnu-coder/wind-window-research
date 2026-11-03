"""Post-hoc diagnostics of the two short-lag candidates; never changes frozen graphs."""
import json
import numpy as np
import pandas as pd
from research_propagation import OUT,data,save_csv
from propagation_graph import difference,lag_profile,best_lag

def run():
 d,meta,times=data();nodes=meta['nodes'];pairs=pd.read_csv(OUT/'pair_summary.csv',dtype={'node_i':str,'node_j':str});fine=pd.read_csv(OUT/'five_minute_lags.csv',dtype={'node_i':str,'node_j':str});monthly=pd.read_csv(OUT/'monthly_lags.csv',dtype={'node_i':str,'node_j':str});blocks=pd.read_csv(OUT/'dtw_blocks.csv',dtype={'node_i':str,'node_j':str})
 stable=[]
 for _,row in pairs[pairs.signal=='change'].iterrows():
  ms=monthly[(monthly.node_i==row.node_i)&(monthly.node_j==row.node_j)&(monthly.method==row.method)&(monthly.n_common>=100)]
  stable.append({'node_i':row.node_i,'node_j':row.node_j,'method':row.method,'full_lag':row.lag_hours,'valid_months':len(ms),'same_lag_months':int((ms.lag_hours==row.lag_hours).sum()),'same_sign_months':int((np.sign(ms.lag_hours)==np.sign(row.lag_hours)).sum())})
 save_csv(stable,'monthly_stability.csv')
 # Training-selected 5min candidates only. Subsequent months are diagnostics, not tuning.
 f=np.load(OUT/'fine_data.npz');ft=pd.DatetimeIndex(f['timestamps']);w=f['wind'].astype(float);rows=[];weekly=[];profiles=[]
 selected=fine[(fine.signal=='change')&(fine.lag_minutes!=0)]
 for _,pair in selected.iterrows():
  i=nodes.index(pair.node_i);j=nodes.index(pair.node_j);fixed=int(pair.lag_minutes//5)
  for month in range(1,13):
   v=difference(w[ft.month==month]);ls,ss,n=lag_profile(v[:,i],v[:,j],12,'corr',1000);lag,score,_=best_lag(v[:,i],v[:,j],12,'corr',1000)
   rows.append({'node_i':pair.node_i,'node_j':pair.node_j,'month':month,'best_lag_minutes':lag*5,'best_score':score,'fixed_lag_minutes':fixed*5,'fixed_lag_score':float(ss[fixed+12]),'zero_score':float(ss[12]),'fixed_gain_over_zero':float(ss[fixed+12]-ss[12]),'n_common':n,'scope':'posthoc diagnostic; no graph refitting'})
  train=w[ft<pd.Timestamp('2024-09-01',tz='Asia/Shanghai')]
  for start in range(0,len(train)-2015,2016):
   v=difference(train[start:start+2016]);lag,score,n=best_lag(v[:,i],v[:,j],12,'corr',500)
   weekly.append({'node_i':pair.node_i,'node_j':pair.node_j,'week_start':str(ft[start]),'best_lag_minutes':lag*5,'score':score,'n_common':n})
  ls,ss,n=lag_profile(difference(train)[:,i],difference(train)[:,j],12,'corr',1000)
  profiles.extend({'node_i':pair.node_i,'node_j':pair.node_j,'lag_minutes':int(l)*5,'score':float(s),'n_common':n} for l,s in zip(ls,ss))
 save_csv(rows,'short_lag_monthly_followup.csv');save_csv(weekly,'short_lag_weekly_followup.csv');save_csv(profiles,'short_lag_profiles.csv')
 dt=pairs[pairs.method=='dtw'];ss=pd.DataFrame(stable)
 summary={'hourly_pairs':136,'hourly_corr_nonzero_pairs':int(((pairs.signal=='change')&(pairs.method=='corr')&(pairs.lag_hours!=0)).sum()),'hourly_mi_nonzero_pairs':int(((pairs.signal=='change')&(pairs.method=='mi')&(pairs.lag_hours!=0)).sum()),'dtw_pair_count_with_atleast3_blocks':int((dt.n_common>=3).sum()),'dtw_nonzero_candidate_pairs':int(((dt.n_common>=3)&(dt.lag_hours!=0)).sum()),'dtw_complete_pair_blocks':len(blocks),'dtw_blocks_nonzero_path_median':int((blocks.path_median_lag!=0).sum()),'fine_change_nonzero_pairs':len(selected),'monthly_corr_zero_count':int(((monthly.method=='corr')&(monthly.lag_hours==0)).sum()),'monthly_mi_zero_count':int(((monthly.method=='mi')&(monthly.lag_hours==0)).sum()),'monthly_per_method_count':int((monthly.method=='corr').sum()),'short_lag_warning':'two pairs selected after training diagnostics; posthoc; not physical validation or a new primary model','no_selected_graph_has_nonzero_lag':bool(all(int(v['nonzero_lag_edges'])==0 for k,v in json.loads((OUT/'graph_summary.json').read_text()).items() if k!='direction'))}
 (OUT/'diagnostic_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False));print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':run()

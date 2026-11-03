"""Descriptive post-run diagnostics; never select or change fitted predictors."""
import json
import numpy as np
import pandas as pd
from model import ART
from research_crps_distribution import CONFIGS,SEEDS
OUT=ART/'crps_distribution_v1'
def run():
 ev=np.load(OUT/'evaluation_data.npz');scale=json.loads((ART/'metadata.json').read_text(encoding='utf-8'))['scale'];rows=[];windows=[]
 for model in CONFIGS:
  for seed in SEEDS:
   pred=np.load(OUT/f'predictions_{model}_{seed}_budget40.npz');theta=np.load(OUT/f'bank_{model}_{seed}.npz')['theta']
   for period in ['cal','cal58','test']:
    key='cal' if period=='cal58' else period;truth=ev[key+'_truth'];q=pred[key]
    if period=='cal58':truth=truth[ev['complete_positions']];q=q[ev['complete_positions']]
    med=q[...,9]
    for a,b in [(0,6),(6,24),(24,48),(48,72)]:
     diff=truth[:,a:b]-med[:,a:b];valid=np.isfinite(diff);corrected=diff-theta[0]*scale
     rows.append(dict(model=model,seed=seed,period=period,lead_start=a+1,lead_end=b,observed_minus_raw_median=float(np.nanmean(diff)),observed_minus_adjusted_median=float(np.nanmean(corrected)),raw_median_mae=float(np.nanmean(abs(diff))),adjusted_median_mae=float(np.nanmean(abs(corrected))),n=int(valid.sum()),common_shift=float(theta[0]*scale)))
 for item in json.loads((OUT/'inventory.json').read_text(encoding='utf-8')):
  scores=np.load(OUT/f'scores_{item["id"]}.npz')
  for j,a in enumerate([0,24,48,66]):
   windows.append(dict(**item,offset=a,brier6=float(np.nanmean(scores['brier6'][:,j])),mean_probability=float(np.nanmean(scores['prob6'][:,j])),observed_rate=float(np.nanmean(scores['label6'][:,j])),n=int(np.isfinite(scores['brier6'][:,j]).sum())))
 pd.DataFrame(rows).to_csv(OUT/'diagnostic_bias_by_lead.csv',index=False,encoding='utf-8-sig');pd.DataFrame(windows).to_csv(OUT/'diagnostic_windows_by_lead.csv',index=False,encoding='utf-8-sig')
 print(pd.DataFrame(rows).query("model=='gated_graph_atoms'").groupby(['period','lead_start'])[['observed_minus_raw_median','observed_minus_adjusted_median']].mean().to_string())
 print(pd.DataFrame(windows).query("model in ['gated_graph_atoms','old_graph'] and method in ['affine_ecc57','residual57','old57']").groupby(['model','method','offset']).brier6.mean().to_string())
if __name__=='__main__':run()

"""Audit frozen model identity, chronology, saved scores and dependence controls."""
from pathlib import Path
import json,hashlib,re,subprocess,sys
import numpy as np
import pandas as pd
import torch
from model import ART,Net,predict
from probability_core import make_bank,scenarios,first_start,longest,window_prob,score_scenarios
from research_probability import OUT,SEEDS,METHODS,METRICS,sha,write_json

def main():
 old=json.loads((OUT/'previous_hashes.json').read_text(encoding='utf-8'))
 assert all(sha(ART/p)==v for p,v in old.items())
 identity=json.loads((OUT/'identity.json').read_text(encoding='utf-8'));assert all(sha(Path(p))==v for p,v in identity.items())
 tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ART.parent,capture_output=True,text=True,encoding='utf-8',errors='replace')
 (OUT/'tests.log').write_text(tests.stdout+tests.stderr,encoding='utf-8');assert tests.returncode==0,tests.stderr[-2500:]
 test_count=int(re.search(r'Ran (\d+) tests',tests.stderr).group(1))
 pred=np.load(OUT/'predictions.npz');d=np.load(ART/'data.npz');times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');tt=times[pred['test_ids']]
 assert np.array_equal(pred['test_ids'],d['test_ids'])
 fit=pred['inner_fit'];val=pred['inner_val'];assert max(pred['cal_ids'][fit]+72)<min(pred['cal_ids'][val]);assert max(pred['cal_ids']+72)<min(pred['test_ids'])
 metrics=pd.read_csv(OUT/'metrics.csv');checks=0;max_base=0.;control_diffs={};parameters=json.loads((OUT/'selected_parameters.json').read_text(encoding='utf-8'))
 oldloss=np.load(ART/'v2/paired_losses.npz')['graph24']
 valid_counts={}
 for seed_index,s in enumerate(SEEDS):
  ck=torch.load(ART/'v2'/('graph24.pt' if s==SEEDS[0] else f'graph24_{s}.pt'),weights_only=False,map_location='cpu');net=Net(ck['adj'],dilation=ck['dilation']);net.load_state_dict(ck['state'])
  for label in ['cal','test']:
   ids=pred[f'{label}_ids'];q=predict(net,d['x'],ids[[0,-1]])*ck['scale']+ck['center'];np.testing.assert_allclose(q,pred[f'{label}_q_{s}'][[0,-1]],rtol=0,atol=2e-5)
  bank=make_bank(pred[f'cal_q_{s}'],pred['cal_truth'],pred['cal_context'])
  samples=scenarios(bank,pred[f'test_q_{s}'][0],pred['test_context'][0],**parameters)
  for m in METHODS:
   with np.load(OUT/f'scores_{m}_{s}.npz') as a:
    for r in metrics[(metrics.seed==s)&(metrics.method==m)].itertuples():
     mask=np.ones(len(tt),bool) if r.month==0 else tt.month==r.month;v=a[r.metric][mask]
     np.testing.assert_allclose(np.nanmean(v),r.value,rtol=1e-12,atol=1e-12);assert np.isfinite(v).sum()==r.n;checks+=1
    if m=='joint32':
     err=np.nanmax(abs(a['brier6']-oldloss[seed_index]));max_base=max(max_base,float(err));assert err<1e-12
     valid_counts={k:int(np.isfinite(a[k]).sum()) for k in METRICS}
    # Fresh first-origin construction must agree with stored per-case results.
    sc,w=samples[m];fresh=score_scenarios(sc,pred['test_truth'][0],w)
    for k in fresh:np.testing.assert_allclose(fresh[k],a[k][0],rtol=1e-12,atol=1e-12,equal_nan=True)
  tuned=np.load(OUT/f'scores_tuned_ecc_{s}.npz')
  for name in ['time_broken','space_broken']:
   with np.load(OUT/f'scores_{name}_{s}.npz') as b:
    same=['crps','coverage','width','interval_score','low_miss','high_miss']
    if name=='space_broken':same+=['brier3','brier6','brier12','length_crps24','length_crps48','length_crps72','any6_brier24','any6_brier48','any6_brier72','first_cdf_brier','temporal_variogram']
    else:same+=['spatial_variogram']
    largest=0.
    for k in same:
     np.testing.assert_allclose(tuned[k],b[k],atol=1e-12,rtol=1e-12,equal_nan=True);largest=max(largest,float(np.nanmax(abs(tuned[k]-b[k]))))
    control_diffs[f'{name}_{s}']=largest
  for name,(sc,w) in samples.items():
   for h in [24,48,72]:assert np.all((longest(sc[:,:h]<=10)>=6)==np.isfinite(first_start(sc[:,:h]<=10)))
   assert np.all(window_prob(sc,3,0,w)>=window_prob(sc,6,0,w)-1e-12);assert np.all(window_prob(sc,6,0,w)>=window_prob(sc,12,0,w)-1e-12)
  if s==SEEDS[0]:
   arrays={m:sc for m,(sc,w) in samples.items()};arrays.update({m+'_weights':np.full(len(sc),1/len(sc)) if w is None else w for m,(sc,w) in samples.items()});arrays['truth']=pred['test_truth'][0]
   np.savez_compressed(OUT/'first_origin_scenarios.npz',**arrays)
 # Reconstruct every paired mean and bootstrap interval from saved case scores.
 paired=pd.read_csv(OUT/'paired_comparisons.csv');week=((tt-pd.Timestamp('2024-11-01',tz=tt.tz)).days//7).to_numpy();keys=np.unique(week);draw=np.random.default_rng(20261101).integers(0,len(keys),(2000,len(keys)))
 for r in paired.itertuples():
  differences=[]
  for s in SEEDS:
   with np.load(OUT/f'scores_{r.candidate}_{s}.npz') as a,np.load(OUT/f'scores_{r.baseline}_{s}.npz') as b:differences.append(a[r.metric]-b[r.metric])
  diff=np.mean(differences,axis=0);sums=np.array([np.nansum(diff[week==k]) for k in keys]);counts=np.array([np.isfinite(diff[week==k]).sum() for k in keys]);boot=sums[draw].sum(1)/counts[draw].sum(1)
  np.testing.assert_allclose([np.nanmean(diff),*np.quantile(boot,[.025,.975])],[r.difference,r.low95,r.high95],atol=1e-12)
 result=dict(unit_tests_passed=test_count,previous_artifacts_unchanged=len(old),identity_files=len(identity),metric_rows_recomputed=checks,paired_rows_recomputed=len(paired),baseline_max_case_difference=max_base,dependence_control_max_differences=control_diffs,valid_counts_per_seed=valid_counts,scope='All stored metric means/counts and paired intervals recomputed; first/last frozen-network inference and first-origin full-score reconstruction for all 24 combinations; not a second full experiment implementation.')
 write_json(OUT/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

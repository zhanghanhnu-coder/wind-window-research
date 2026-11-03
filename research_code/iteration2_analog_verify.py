"""Independent config-based capacity/accounting/MILP/prefix checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from iteration2_analog_execution import OUT,OLD,ROOT,ART,PREDS,EXECS,read,write,sha
from verify_review_cross_execution import independent_reward,independent_now
from precision_truth_core import load_truth
from two_stage_core import make_problem
from highs_readback_compat import install

def main():
 install();cfg=read(OUT/'protocol.json')['config'];p=make_problem(cfg);_,_,raw=load_truth();assert raw.dtype==np.float64
 ident=read(OUT/'source_identity.json')
 for path,h in ident.items():assert sha(path)==h,path
 inv=read(OUT/'trajectory_inventory.json');assert len(inv)==528;checks=steps=milps=prefix=nowchecks=0
 for item in inv:
  assert sha(item['path'])==item['sha256'];saved=read(item['path']);a=saved['trajectory'];case=saved['case'];x=a['x'];pred=item['prediction'];ex=item['executor'];meta=read(OUT/'inputs'/f'{case["key"]}.json');z=np.load(OUT/'inputs'/f'{case["key"]}.npz');grid=z[pred];history=[]
  assert x==meta['booking']['B']['x'];quality=z['quality'];oldz=np.load(OLD/'inputs'/f'{case["key"]}.npz');np.testing.assert_array_equal(z['GraphFixed'],oldz['B']);np.testing.assert_array_equal(quality,oldz['quality'])
  for k,step in enumerate(a['steps']):
   now=6*k;assert step['now']==now
   if not quality[k]:assert not step['selected']
   if ex=='myopic' and quality[k]:assert abs(independent_now(cfg,x,history,now,grid[k,0])-step['expected_benefit'])<1e-7;nowchecks+=1
   if ex=='lookahead72' and step.get('solver_called'):assert step['solver_status']==0 and step['mip_gap']<=1e-9 and abs(step['expected_benefit']-step['dual_bound'])<1e-6
   if case['position']==0 and case['seed']==20241101 and ex=='lookahead72' and k in [0,4,8] and quality[k]:assert abs(independent_reward(cfg,x,history,now,grid[k])-step['expected_benefit'])<1e-6;milps+=1
   for c in step['selected']:
    o=p.choices[c];assert o['start']==now and o['end']<=72 and o['task'] not in {h['task'] for h in history};assert cfg['tasks'][o['task']]['skill'] in cfg['crew_skills'][o['crew']]
   history.extend(p.choices[c] for c in step['selected']);steps+=1
  assert len(history)==len(a['attempts']);assert len({o['task'] for o in history})==len(history)
  ev=read(OUT/'evaluations'/f'{case["key"]}__fixed_B__{pred}__{ex}.json');truth=raw[case['issue']+1:case['issue']+73];boats=np.zeros((2,72));crews=np.zeros((3,72));spares=np.zeros(2);occupied={t['node']:np.zeros(72) for t in cfg['tasks']};bytask={o['task']:o for o in history};book=float(np.array(cfg['boat_booking']+cfg['crew_booking']+cfg['spare_cost'])@x);op=pl=pu=0
  for j,t in enumerate(cfg['tasks']):
   if j not in bytask:status='unattempted';pl+=t['penalty'];pu+=t['penalty']
   else:
    o=bytask[j];boats[o['boat'],o['start']:o['end']]+=1;crews[o['crew'],o['start']:o['end']]+=1;spares[o['spare']]+=o['quantity'];occupied[o['node']][o['start']:o['end']]+=1;y=truth[o['start']:o['end'],o['node']];known=np.isfinite(y);status='failed' if (y[known]>10).any() else 'successful' if known.all() else 'unknown';op+=(cfg['boat_hourly'][o['boat']]+cfg['crew_hourly'][o['crew']])*t['duration']+cfg['late_fraction']*t['penalty']*o['start']/72
    if status=='failed':pl+=t['penalty']
    if status!='successful':pu+=t['penalty']
   assert status==ev['tasks'][j]['status'];checks+=1
  assert np.all(boats<=np.array(x[:2])[:,None]) and np.all(crews<=np.array(x[2:5])[:,None]) and np.all(spares<=x[5:]) and all((v<=1).all() for v in occupied.values())
  assert abs(book+op+pl-ev['metrics']['cost_lower'])<1e-6 and abs(book+op+pu-ev['metrics']['cost_upper'])<1e-6
  if case['position']==0 and case['seed']==20241101:
   from cross_execution_review_core import execute
   modified=grid.copy();modified[2:]=0;new=execute(p,x,modified,quality,ex);assert [s['selected'] for s in new['steps'][:2]]==[s['selected'] for s in a['steps'][:2]];prefix+=1
 arms=pd.read_csv(OUT/'arms.csv');paired=pd.read_csv(OUT/'paired_cases.csv');assert len(arms)==528 and len(pd.read_csv(OUT/'task_ledger.csv'))==3168
 keyed={(r.key,r.prediction,r.executor):r for r in arms.itertuples()}
 for r in paired.itertuples():
  aa=keyed[r.key,'AnalogFixed',r.executor];bb=keyed[r.key,'GraphFixed',r.executor];assert abs(r.difference_lower-aa.cost_lower+bb.cost_upper)<1e-7 and abs(r.difference_upper-aa.cost_upper+bb.cost_lower)<1e-7;assert aa.x==bb.x
 # Independent marginal-to-scenario threshold probability reconstruction on every input.
 from lead_calibration_core import apply_groups
 probability_checks=0
 for fold in ['M05','M07','M09','M11']:
  f=np.load(ART/'seasonal_probability_v1'/fold/'data.npz');ids=f['test_ids'];lookup={int(v):i for i,v in enumerate(ids)};aq=apply_groups(np.load(ART/'multiscale_context_v1'/fold/'predictions_analog112_0.npz')['test'],np.load(ART/'multiscale_context_v1'/fold/'calibration_analog112_0.npz')['theta'],float(f['scale']),True)
  for seed in [20241101,20241102,20241103]:
   gq=np.load(ART/'seasonal_probability_v1'/fold/f'calibrated_gated_graph_atoms_{seed}.npz')['q'];rank=np.load(ART/'partial_copula_v1'/fold/f'ranks_gated_graph_atoms_{seed}_partial05.npz')['ranks']
   for c in [c for c in read(OUT/'protocol.json')['cases'] if c['fold']==fold and c['seed']==seed]:
    z=np.load(OUT/'inputs'/f'{c["key"]}.npz')
    for k in range(12):
     if not z['quality'][k]:continue
     pos=lookup[c['issue']+6*k]
     for pred,q in [('AnalogFixed',aq),('GraphFixed',gq)]:
      for ti,t in enumerate(cfg['tasks']):
       for h in range(0,73-t['duration'],6):
        vals=q[pos,h:h+t['duration'],t['node']];ix=rank[:,h:h+t['duration'],t['node']]//3;safe=np.take_along_axis(vals[None,:,:],ix[:,:,None],axis=2).squeeze(2)<=10;prob=safe.all(1).mean();assert prob==z[pred][k,h//6,ti];probability_checks+=1
 # Brier and support labels independently rebuilt from raw64; missing never classified as success.
 w=pd.read_csv(OUT/'support_window_ledger.csv');brier_checks=0
 for r in w.itertuples():
  t=cfg['tasks'][r.task];y=raw[r.issue+r.start+1:r.issue+r.start+t['duration']+1,t['node']];known=np.isfinite(y);label=0 if (y[known]>10).any() else 1 if known.all() else np.nan;assert np.isclose(label,r.label,equal_nan=True);brier_checks+=1
 # Legacy analog replay: first/middle/last cached origin per fold, fixed independently of outcomes.
 from multiscale_context_core import analog_forecast,context_features
 legacy=np.load(ART/'data.npz')['raw'];features=context_features(legacy);analog_replays=0
 for fold in ['M05','M07','M09','M11']:
  f=np.load(ART/'seasonal_probability_v1'/fold/'data.npz');ids=f['test_ids'];prior=np.load(ART/'multiscale_context_v1'/fold/'baseline_prior.npz')['q'];q=np.load(ART/'multiscale_context_v1'/fold/'predictions_analog112_0.npz')['test']
  for pos in [0,len(ids)//2,len(ids)-1]:
   issue=int(ids[pos]);replay,details=analog_forecast(legacy,issue,prior,float(f['scale']),features);np.testing.assert_array_equal(replay,q[pos]);prefixraw=legacy[:issue+1].copy();causal,_=analog_forecast(prefixraw,issue,prior,float(f['scale']));np.testing.assert_array_equal(causal,replay);assert all(s+72<issue for ss in details['source_ids'] for s in ss);analog_replays+=1
 result=dict(status='PASS',arms=528,tasks=checks,steps=steps,independent_myopic_batches=nowchecks,independent_MILP_optima=milps,causal_prefix_perturbations=prefix,independent_task_probabilities=probability_checks,independent_window_labels=brier_checks,analog_raw_replays_and_future_truncation=analog_replays,source_hashes_unchanged=len(ident),truth_dtype=str(raw.dtype))
 write(OUT/'verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

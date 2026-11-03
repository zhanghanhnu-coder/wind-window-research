"""Isolated fixed-resource analog execution experiment; frozen before raw64 outcomes."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,time,copy
import numpy as np
import pandas as pd
from review_cross_execution import ROOT,ART,SEEDS,FOLDS,sha,read,write
from cross_execution_review_core import probability_grid,execute,task_ledger,difference_bounds
from two_stage_core import default_config,make_problem
from crps_distribution_core import couple_atoms
from lead_calibration_core import apply_groups
from highs_readback_compat import install

OUT=ART/'review_iteration2_v1/analog_execution'
OLD=ART/'review_followup_v1/cross_execution'
BASELINES=ART/'review_followup_v1/baselines'
MULTI=ART/'multiscale_context_v1'
PREDS=['AnalogFixed','GraphFixed'];EXECS=['myopic','lookahead72']
METRICS=['booking_cost','operating_cost','cost_lower','cost_upper','successful','failed','unknown','unattempted']

def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def common(item):return {k:item[k] for k in ['key','fold','position','issue','seed','layer','prediction','executor']}
def frozen(path,ident,manifest=None):
 path=Path(path);h=sha(path)
 if manifest is not None:assert manifest[str(path)]==h,str(path)
 ident[str(path)]=h;return path

def plan():
 install();OUT.mkdir(parents=True,exist_ok=True);cases=read(OLD/'protocol.json')['cases'];assert len(cases)==132
 protocol=dict(status='FROZEN_BEFORE_EVALUATION',cases=cases,primary='AnalogFixed minus GraphFixed with lookahead72, fixed old cross B booking x',auxiliary='same comparison with myopic',predictions=PREDS,executors=EXECS,config=default_config(),planned_origins=44,planned_new_arms=264,planned_total_arms=528,seed_policy='one analog producer per fold; same three frozen partial05 rank templates; average seed losses inside origin',updates='12 updates every 6h over 72h; current history quality >=.65; no retry; no own resource booking',analog='multiscale_context_v1 analog112 raw test atoms with original preceding-month smooth calibration; legacy_frozen32/natural identity',graph='context1_graph legacy_frozen32 natural = gated_graph_atoms frozen = old cross fixed_B B',selection='all 44 old cross origins, unchanged; complete15 sensitivity fixed by six-task-node 72h weather coverage, not outcomes',scoring='all admissible initial task windows and quality-eligible current task windows in same 44 origins; three-seed losses averaged before origins; no outcome filtering',truth_loaded_in_planning=False,scope='Repeatedly explored 2024 development data; no new independent generalization; no tuning or external data')
 if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
 else:write(OUT/'protocol.json',protocol)
 ident={};oldmanifest=read(OLD/'final_manifest.json');bm=read(BASELINES/'final_manifest.json');mm=read(MULTI/'final_manifest.json');roll=read(ART/'seasonal_rolling_v1/inventory.json');bi=read(BASELINES/'inventory.json');mi=read(MULTI/'inventory.json')
 for p in [OLD/'protocol.json',OLD/'trajectory_inventory.json',OLD/'evaluation_inventory.json',BASELINES/'inventory.json',MULTI/'inventory.json',ART/'seasonal_rolling_v1/inventory.json',ROOT/'iteration2_analog_execution.py']:
  frozen(p,ident)
 oldinv=read(OLD/'trajectory_inventory.json');oldmap={(r['key'],r['executor']):r for r in oldinv if r['layer']=='fixed_B' and r['prediction']=='B'};rows=[];identity_checks=[]
 # All inputs and source identities are completed before executing any new arm.
 for fold in FOLDS:
  fp=frozen(ART/'seasonal_probability_v1'/fold/'data.npz',ident);f=np.load(fp);ids=f['test_ids'];lookup={int(v):i for i,v in enumerate(ids)};assert int((f['cal_ids']+72).max())<ids.min()
  aitem=next(r for r in bi if (r['fold'],r['family'],r['method'],r['regime'])==(fold,'analog112','legacy_frozen32','natural'))
  ap=frozen(aitem['source'],ident,bm);theta=frozen(MULTI/fold/'calibration_analog112_0.npz',ident,mm);apack=np.load(ap);aq=apply_groups(apack[aitem['source_key']],np.load(theta)['theta'],float(f['scale']),True)
  oldtheta=np.load(frozen(aitem['parameters'],ident,bm))['legacy_frozen32'];np.testing.assert_array_equal(oldtheta,np.load(theta)['theta'])
  af=next(r for r in mi if (r['fold'],r['model'],r['method'])==(fold,'analog112','frozen'));np.testing.assert_array_equal(aq,np.load(frozen(af['source'],ident,mm))[af['source_key']])
  asp=frozen(MULTI/fold/'statistical_sources_analog112.npz',ident,mm);asrc=np.load(asp);aslookup={int(v):i for i,v in enumerate(asrc['ids'])}
  for seed in SEEDS:
   gitem=next(r for r in roll if (r['fold'],r['seed'],r['model'],r['method'])==(fold,seed,'gated_graph_atoms','frozen'))
   gp=frozen(gitem['source'],ident);assert sha(gp)==gitem['source_sha256'];gq=np.load(gp)[gitem['source_key']]
   gb=next(r for r in bi if (r['fold'],r['seed'],r['family'],r['method'],r['regime'])==(fold,seed,'context1_graph','legacy_frozen32','natural'))
   graw=np.load(frozen(gb['source'],ident,bm))[gb['source_key']];gt=np.load(frozen(gb['parameters'],ident,bm))['legacy_frozen32'];np.testing.assert_array_equal(gq,apply_groups(graw,gt,float(f['scale']),True))
   rp=frozen(gitem['ranks_source'],ident);assert sha(rp)==gitem['ranks_sha256'];ranks=np.load(rp)['ranks'];assert ranks.shape==(57,72,17)
   ab=next(r for r in bi if (r['fold'],r['seed'],r['family'],r['method'],r['regime'])==(fold,seed,'analog112','legacy_frozen32','natural'));assert ab['source']==aitem['source'];assert ab['ranks']==gb['ranks']==str(rp)
   for case in [c for c in cases if c['fold']==fold and c['seed']==seed]:
    oldpack=frozen(OLD/'inputs'/f'{case["key"]}.npz',ident,oldmanifest);oz=np.load(oldpack);meta=read(frozen(oldpack.with_suffix('.json'),ident,oldmanifest));grids={p:np.full((12,12,6),np.nan) for p in PREDS};checks=[]
    for k,now in enumerate(range(0,72,6)):
     issue=case['issue']+now;assert bool(oz['quality'][k])==(float(f['x'][issue-23:issue+1,:,-1].mean())>=.65)
     if issue not in lookup:
      assert not oz['quality'][k];checks.append(dict(now=now,issue=issue,quality_ok=False,latest_analog_target=None,cached_forecast=False));continue
     pos=lookup[issue]
     source_ids=asrc['source_ids'][aslookup[issue]];chosen=source_ids[source_ids>=0];assert len(chosen)==0 or int((chosen+72).max())<issue
     assert len(chosen)==0 or int(chosen.min())>=issue-112*24
     checks.append(dict(now=now,issue=issue,quality_ok=bool(oz['quality'][k]),latest_analog_target=None if len(chosen)==0 else int((chosen+72).max())))
     if not oz['quality'][k]:continue
     for pred,q in [('AnalogFixed',aq),('GraphFixed',gq)]:grids[pred][k]=probability_grid(couple_atoms(q[pos].astype(float),ranks),default_config())
    np.testing.assert_array_equal(grids['GraphFixed'],oz['B'])
    ip=OUT/'inputs'/f'{case["key"]}.npz';ip.parent.mkdir(exist_ok=True);np.savez_compressed(ip,**grids,quality=oz['quality'])
    write(ip.with_suffix('.json'),dict(case=case,updates=checks,booking={'B':meta['booking']['B']},rank_source=str(rp),original_cross_input=str(oldpack),original_cross_input_sha256=sha(oldpack)))
    for ex in EXECS:
     old=oldmap[case['key'],ex];frozen(old['path'],ident,oldmanifest);assert sha(old['path'])==old['sha256'];saved=read(old['path']);assert saved['trajectory']['x']==meta['booking']['B']['x']
     rows.append(dict(**case,layer='fixed_B',prediction='GraphFixed',executor=ex,path=old['path'],sha256=old['sha256'],reused=True))
   identity_checks.append(dict(fold=fold,seed=seed,graph_full_q_bitwise=True,graph_all_update_grids_bitwise=True,analog_full_q_bitwise=True,shared_rank_identity=True))
  print('SOURCES',fold,flush=True)
 write(OUT/'source_identity.json',ident);write(OUT/'source_checks.json',identity_checks)
 p=make_problem(default_config());start=time.monotonic()
 for case in cases:
  ip=OUT/'inputs'/f'{case["key"]}.npz';z=np.load(ip);meta=read(ip.with_suffix('.json'));x=meta['booking']['B']['x']
  for ex in EXECS:
   path=OUT/'trajectories'/f'{case["key"]}__fixed_B__AnalogFixed__{ex}.json'
   if path.exists():saved=read(path);assert saved['input_sha256']==sha(ip)
   else:
    trajectory=execute(p,x,z['AnalogFixed'],z['quality'],ex);saved=dict(case=case,layer='fixed_B',prediction='AnalogFixed',executor=ex,trajectory=trajectory,input_sha256=sha(ip));write(path,saved)
   rows.append(dict(**case,layer='fixed_B',prediction='AnalogFixed',executor=ex,path=str(path),sha256=sha(path),reused=False))
  print('PLANNED',case['key'],round(time.monotonic()-start,2),flush=True)
 assert len(rows)==528
 write(OUT/'trajectory_inventory.json',rows);write(OUT/'planning_complete.json',dict(arms=528,new_arms=264,reused_graph_arms=264,truth_loaded=False,seconds=time.monotonic()-start))

def evaluate():
 from precision_truth_core import load_truth
 from cross_support_review import label_window,candidate_record,fixed_action_costs,interval_outside
 install();inv=read(OUT/'trajectory_inventory.json');assert len(inv)==528;assert read(OUT/'planning_complete.json')['truth_loaded'] is False
 times,nodes,raw=load_truth();assert raw.dtype==np.float64;p=make_problem(default_config());cfg=p.config
 olde={Path(r['path']).name:r for r in read(OLD/'evaluation_inventory.json')};rows=[];tasks=[];evals=[];outcomes={}
 for item in inv:
  assert sha(item['path'])==item['sha256'];saved=read(item['path']);case=saved['case'];tt,metrics=task_ledger(p,saved['trajectory'],raw[case['issue']+1:case['issue']+73]);graphname=f'{case["key"]}__fixed_B__B__{item["executor"]}.json';oe=olde[graphname];assert sha(oe['path'])==oe['sha256'];old=read(oe['path']);ideal=old['ideal']
  if item['prediction']=='GraphFixed':
   assert tt==old['tasks'];assert metrics==old['metrics']
  assert saved['trajectory']['x']==read(OLD/'inputs'/f'{case["key"]}.json')['booking']['B']['x']
  c=common(item);row=dict(**c,**metrics,complete=ideal['complete'],ideal_lower=ideal['lower'],ideal_upper=ideal['upper'],execution_minus_ideal_lower=metrics['cost_lower']-ideal['upper'],execution_minus_ideal_upper=metrics['cost_upper']-ideal['lower'],x=json.dumps(saved['trajectory']['x']))
  rows.append(row);tasks.extend(dict(**c,**t) for t in tt);outcomes[case['key'],item['prediction'],item['executor']]=row
  ep=OUT/'evaluations'/f'{case["key"]}__fixed_B__{item["prediction"]}__{item["executor"]}.json';write(ep,dict(**c,metrics=metrics,tasks=tt,ideal=ideal));evals.append(dict(path=str(ep),sha256=sha(ep)))
 csv('arms.csv',rows);csv('task_ledger.csv',tasks);write(OUT/'evaluation_inventory.json',evals)
 pairs=[]
 for case in read(OUT/'protocol.json')['cases']:
  for ex in EXECS:
   a=outcomes[case['key'],'AnalogFixed',ex];b=outcomes[case['key'],'GraphFixed',ex];lo,hi=difference_bounds(a,b)
   pairs.append(dict(**case,executor=ex,comparison='AnalogFixed_minus_GraphFixed_'+ex,difference_lower=lo,difference_upper=hi,complete=a['complete'],**{m+'_difference':a[m]-b[m] for m in ['successful','failed','unknown','unattempted','booking_cost','operating_cost']}))
 csv('paired_cases.csv',pairs);df=pd.DataFrame(pairs);valuecols=['difference_lower','difference_upper']+[m+'_difference' for m in ['successful','failed','unknown','unattempted','booking_cost','operating_cost']]
 origin=df.groupby(['fold','position','issue','executor','comparison'],sort=False).agg({**{c:'mean' for c in valuecols},'complete':'all'}).reset_index();csv('paired_origins.csv',origin);summary=[];arm_summary=[]
 for scope in ['ALL']+FOLDS:
  for population in ['all','complete_task_weather']:
   z=origin if scope=='ALL' else origin[origin.fold==scope]
   if population!='all':z=z[z.complete]
   for (ex,comp),g in z.groupby(['executor','comparison'],sort=False):summary.append(dict(scope=scope,population=population,executor=ex,comparison=comp,origins=len(g),**{c:float(g[c].mean()) for c in valuecols}))
   adf=pd.DataFrame(rows);a=adf if scope=='ALL' else adf[adf.fold==scope]
   if population!='all':a=a[a.complete]
   for (pred,ex),g in a.groupby(['prediction','executor']):
    gg=g.groupby(['fold','issue'])[METRICS].mean();arm_summary.append(dict(scope=scope,population=population,prediction=pred,executor=ex,origins=len(gg),**gg.mean().to_dict()))
 csv('summary.csv',summary);csv('arm_summary.csv',arm_summary)
 # Same 44 origins, initial and current candidate task-window probabilities, not selected by actions.
 windows=[];ranges=[];cases=read(OUT/'protocol.json')['cases'];scache={}
 for fold in FOLDS:
  f=np.load(ART/'seasonal_probability_v1'/fold/'data.npz');aq=apply_groups(np.load(MULTI/fold/'predictions_analog112_0.npz')['test'],np.load(MULTI/fold/'calibration_analog112_0.npz')['theta'],float(f['scale']),True)
  for case in [c for c in cases if c['fold']==fold]:
   z=np.load(OUT/'inputs'/f'{case["key"]}.npz');ranks=np.load(ART/'partial_copula_v1'/fold/f'ranks_gated_graph_atoms_{case["seed"]}_partial05.npz')['ranks'];meta=read(OLD/'inputs'/f'{case["key"]}.json');book=meta['booking']['B'];pp=Path(book['source']);fc=pp.with_name(pp.name.replace('plan_','forecast_').replace('__tv025.json','.npz'));assert sha(fc)==book['forecast_sha256']
   for pred in PREDS:
    sc=couple_atoms(aq[case['position']],ranks) if pred=='AnalogFixed' else np.load(fc)['scenarios'];scache[case['key'],pred]=sc
    cc={**case,'prediction':pred,'complete':outcomes[case['key'],pred,'myopic']['complete']}
    for task,t in enumerate(cfg['tasks']):
     for start in range(0,73-t['duration'],6):
      y=label_window(raw[case['issue']+start+1:case['issue']+start+t['duration']+1,t['node']]);prob=float((sc[:,start:start+t['duration'],t['node']]<=10).all(1).mean())
      windows.append(candidate_record(cc,'initial_candidate',task,start,prob,y))
      if z['quality'][start//6]:windows.append(candidate_record(cc,'current_candidate',task,start,z[pred][start//6,0,task],y))
 for item in inv:
  saved=read(item['path']);cc=common(item);ev=read(OUT/'evaluations'/f'{item["key"]}__fixed_B__{item["prediction"]}__{item["executor"]}.json');costs=fixed_action_costs(cfg,saved['trajectory']['x'],ev['tasks'],scache[item['key'],item['prediction']]);lo=ev['metrics']['cost_lower'];hi=ev['metrics']['cost_upper']
  ranges.append(dict(**cc,complete=ev['ideal']['complete'],stage='fixed_saved_action_execution',observed_lower=lo,observed_upper=hi,support_min=float(costs.min()),support_max=float(costs.max()),**interval_outside(lo,hi,float(costs.min()),float(costs.max()))))
 csv('support_window_ledger.csv',windows);csv('support_cost_ledger.csv',ranges);ws=pd.DataFrame(windows);bs=[];support=[]
 for scope in ['ALL']+FOLDS:
  for population in ['all','complete_task_weather']:
   w=ws if scope=='ALL' else ws[ws.fold==scope]
   if population!='all':w=w[w.complete]
   for (pred,stage),g in w.groupby(['prediction','stage']):
    known=g.label.notna();prob=g.probability.to_numpy();y=g.label.to_numpy();loss=(prob-y)**2;lo=np.where(known,loss,np.minimum(prob**2,(1-prob)**2));hi=np.where(known,loss,np.maximum(prob**2,(1-prob)**2));bs.append(dict(scope=scope,population=population,prediction=pred,stage=stage,origins=g.issue.nunique(),seed_window_rows=len(g),known=int(known.sum()),unknown=int((~known).sum()),brier_lower=float(lo.mean()),brier_upper=float(hi.mean()),known_brier=float(np.nanmean(loss)),all_available_known=int((g.all_available&known).sum()),all_available_actual_failure=int(g.all_available_actual_failure.sum()),all_unavailable_known=int((g.all_unavailable&known).sum()),all_unavailable_actual_success=int(g.all_unavailable_actual_success.sum())))
   cost=pd.DataFrame(ranges);cost=cost if scope=='ALL' else cost[cost.fold==scope]
   if population!='all':cost=cost[cost.complete]
   for (pred,ex),g in cost.groupby(['prediction','executor']):support.append(dict(scope=scope,population=population,prediction=pred,executor=ex,origin_seed_cases=len(g),**{c:int(g[c].sum()) for c in ['certified_above','certified_below','possible_above','possible_below']}))
 csv('task_window_brier.csv',bs);csv('support_cost_summary.csv',support)
 write(OUT/'evaluation_complete.json',dict(arms=len(rows),tasks=len(tasks),truth_dtype=str(raw.dtype),complete_origins=int(origin[origin.executor=='lookahead72'].complete.sum()),support_windows=len(windows),support_costs=len(ranges),new_booking=False))
 print('EVALUATED',len(rows),len(tasks),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['plan','evaluate']);args=parser.parse_args();globals()[args.stage]()

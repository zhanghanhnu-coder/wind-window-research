"""Retrospective 2x2 mechanism diagnostic; read-only frozen forecast inputs."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import json,hashlib,argparse
import numpy as np
import pandas as pd
from iteration3_matched_core import CELLS,CONTRASTS,matched,affine,weights
from iteration2_bounds_core import window_term,solve_bounds
from precision_truth_core import load_truth,SOURCE

ROOT=Path(__file__).resolve().parent;ART=ROOT/'artifacts'
OLD=ART/'review_iteration2_v1/analog_execution'
OUT=ART/'review_iteration3_v1/matched_updates'
FOLDS=['M05','M07','M09','M11']
GROUPS={'ALL':list(range(0,72,6)),'start0':[0],'start6_18':[6,12,18],'start24_42':[24,30,36,42],'start48_66':[48,54,60,66]}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(name,x):pd.DataFrame(x).to_csv(OUT/name,index=False,encoding='utf-8-sig')

def freeze():
 OUT.mkdir(parents=True,exist_ok=True)
 protocol=dict(status='FROZEN_BEFORE_MATCHED_OUTCOME_ANALYSIS',created_utc=datetime.now(timezone.utc).isoformat(),
  design='Retrospective mechanism diagnostic motivated by old unmatched stage differences, not a new confirmatory trial.',
  cells=CELLS,cell_order='AI Analog initial, GI Graph initial, AC Analog current, GC Graph current',
  primary='interaction=(AC-GC)-(AI-GI); positive means Graph benefits more from updates',contrasts=CONTRASTS,
  matching='Exact fold,issue,seed,task,start on quality-eligible current queue; same target, node and duration in all four cells. No probability averaging before Brier squares.',
  aggregation='Within each origin/seed average available matched windows; average all three frozen rank seeds; average origins equally within scope. Empty support reported, not silently replaced.',
  populations=['all44_primary','complete15_auxiliary'],scopes=['ALL']+FOLDS,start_groups=GROUPS,
  bounds='Brier(p,y)=p^2+(1-2p)y per cell, signed sum only after individual scores. Shared unknown physical hour/node threshold states AND within physical window via unmodified iteration2_bounds_core.',
  coarse='Independent label per physical window, shared across seeds and four cells; no cross-window weather constraint; must contain strict bounds.',
  negative_control='start0 initial and current probabilities exactly equal within each model; update and interaction identically zero.',
  no_changes='Frozen probabilities, calibration, ranks, quality gates and actions unchanged; no forecasting, tuning, dispatch or outcome filtering.',
  limitations='Cannot establish true sea-state operational gain or a general causal mechanism. Bounds conditional on fixed forecasts and unrestricted missing-hour binary completion; not confidence intervals.',
  tests='RED observed missing iteration3_matched_core; GREEN synthetic validity tests before real analysis')
 if (OUT/'protocol.json').exists():
  old=read(OUT/'protocol.json');protocol['created_utc']=old['created_utc'];assert old==protocol
 else:write(OUT/'protocol.json',protocol)
 paths=[OLD/'support_window_ledger.csv',OLD/'protocol.json',OLD/'source_identity.json',OLD/'source_checks.json',SOURCE,ART/'metadata.json',ART/'data.npz']
 paths += [ROOT/n for n in ['iteration3_matched_core.py','iteration3_matched_run.py','iteration3_matched_test.py','iteration2_bounds_core.py','precision_truth_core.py','lead_calibration_core.py']]
 # Freeze every original probability and quality input before reading real labels.
 for fold in FOLDS:
  paths += [ART/'seasonal_probability_v1'/fold/'data.npz',ART/'multiscale_context_v1'/fold/'predictions_analog112_0.npz',ART/'multiscale_context_v1'/fold/'calibration_analog112_0.npz']
  for seed in [20241101,20241102,20241103]:
   paths += [ART/'seasonal_probability_v1'/fold/f'calibrated_gated_graph_atoms_{seed}.npz',ART/'partial_copula_v1'/fold/f'ranks_gated_graph_atoms_{seed}_partial05.npz']
 for case in read(OLD/'protocol.json')['cases']:
  ip=OLD/'inputs'/f'{case["key"]}.npz';meta=ip.with_suffix('.json');paths += [ip,meta]
  book=read(meta)['booking']['B'];pp=Path(book['source']);fc=pp.with_name(pp.name.replace('plan_','forecast_').replace('__tv025.json','.npz'));paths.append(fc)
 evidence=ART/'review_iteration2_v1/data_gaps/partial_hour_evidence.npz'
 if evidence.exists():paths.append(evidence)
 ident={str(p):sha(p) for p in dict.fromkeys(paths)}
 if (OUT/'source_identity.json').exists():assert read(OUT/'source_identity.json')==ident
 else:write(OUT/'source_identity.json',ident)
 write(OUT/'tests_red.json',dict(status='EXPECTED_FAIL',command='python iteration3_matched_test.py',exit_code=1,error="ModuleNotFoundError: No module named 'iteration3_matched_core'"))
 print('FROZEN',len(ident),flush=True)

def audit_sources(d,raw,cfg):
 """Independent rank-to-atom reconstruction, every original four-cell row."""
 from lead_calibration_core import apply_groups
 checks=0;gatechecks=0
 for fold in FOLDS:
  f=np.load(ART/'seasonal_probability_v1'/fold/'data.npz');lookup={int(v):i for i,v in enumerate(f['test_ids'])}
  aq=apply_groups(np.load(ART/'multiscale_context_v1'/fold/'predictions_analog112_0.npz')['test'],np.load(ART/'multiscale_context_v1'/fold/'calibration_analog112_0.npz')['theta'],float(f['scale']),True)
  for seed in [20241101,20241102,20241103]:
   gq=np.load(ART/'seasonal_probability_v1'/fold/f'calibrated_gated_graph_atoms_{seed}.npz')['q'];ranks=np.load(ART/'partial_copula_v1'/fold/f'ranks_gated_graph_atoms_{seed}_partial05.npz')['ranks']
   for case in [c for c in read(OLD/'protocol.json')['cases'] if c['fold']==fold and c['seed']==seed]:
    z=np.load(OLD/'inputs'/f'{case["key"]}.npz')
    for k in range(12):assert bool(z['quality'][k])==(float(f['x'][case['issue']+6*k-23:case['issue']+6*k+1,:,-1].mean())>=.65);gatechecks+=1
   for r in d[(d.fold==fold)&(d.seed==seed)].itertuples():
    t=cfg['tasks'][r.task];node=t['node'];dur=t['duration'];values=raw[r.issue+r.start+1:r.issue+r.start+dur+1,node];known=np.isfinite(values)
    y=0. if (values[known]>10).any() else 1. if known.all() else np.nan
    assert np.isclose(y,r.label,equal_nan=True)
    for cell,q,offset,lead in [('AI',aq,0,r.start),('GI',gq,0,r.start),('AC',aq,r.start,0),('GC',gq,r.start,0)]:
     vals=q[lookup[r.issue+offset],lead:lead+dur,node];ix=ranks[:,lead:lead+dur,node]//3
     safe=np.take_along_axis(vals[None,:,:],ix[:,:,None],axis=2).squeeze(2)<=10
     probability=float(safe.all(1).mean());assert abs(probability-getattr(r,cell))<1e-14,(r,cell,probability)
     checks+=1
    if r.start==0:assert r.AI==r.AC and r.GI==r.GC
 return dict(independent_probability_rows=checks,independent_label_windows=len(d),independent_gate_checks=gatechecks)

def direct(d,w,sign,raw,witness,cfg):
 # Fill one common weather completion, then directly square each cell.
 full=raw.copy()
 for (h,node),good in witness.items():full[h,node]=0. if good else 11.
 full[~np.isfinite(full)]=11.
 ys=[]
 for r in d.itertuples():
  t=cfg['tasks'][r.task];ys.append(float((full[r.issue+r.start+1:r.issue+r.start+t['duration']+1,t['node']]<=10).all()))
 losses=(d[CELLS].to_numpy()-np.array(ys)[:,None])**2
 return float(np.dot(w,losses@sign))

def evaluate():
 assert (OUT/'source_identity.json').exists();identity=read(OUT/'source_identity.json')
 for p,h in identity.items():assert sha(p)==h
 _,nodes,raw=load_truth();cfg=read(OLD/'protocol.json')['config'];ledger=pd.read_csv(OLD/'support_window_ledger.csv');d=matched(ledger)
 assert len(d[['fold','issue']].drop_duplicates())==44
 for (f,i,s),g in d.groupby(['fold','issue','start']):assert g.seed.nunique()==3
 audit=audit_sources(d,raw,cfg)
 d['node']=d.task.map(lambda x:cfg['tasks'][x]['node']);d['duration']=d.task.map(lambda x:cfg['tasks'][x]['duration'])
 csv('matched_windows.csv',d)
 terms=[window_term(raw,int(r.issue),int(r.start),int(r.start+r.duration),int(r.node)) for r in d.itertuples()]
 rows=[];empty=[];maxerr=0.;certificates=0
 for pop in ['all44_primary','complete15_auxiliary']:
  for scope in ['ALL']+FOLDS:
   for group,starts in GROUPS.items():
    mask=d.start.isin(starts)&((d.fold==scope) if scope!='ALL' else True)&(d.complete if pop!='all44_primary' else True)
    sub=d.loc[mask].copy();ix=np.flatnonzero(mask)
    base=dict(population=pop,scope=scope,start_group=group,origins=len(sub[['fold','issue']].drop_duplicates()),seed_window_rows=len(sub),known_window_rows=int(sub.label.notna().sum()),unknown_window_rows=int(sub.label.isna().sum()))
    if sub.empty:empty.append(dict(**base,status='EMPTY_SUPPORT'));continue
    w=weights(sub);p=sub[CELLS].to_numpy();kk=[terms[j] for j in ix]
    # Same support for four cell marginals and all five contrasts.
    scores={**CONTRASTS,**{cell:[int(j==i) for j in range(4)] for i,cell in enumerate(CELLS)}}
    for name,sign in scores.items():
     c,b=affine(p,sign);constant=float(w@c);tt=defaultdict(float);coarse_terms=defaultdict(float);clo=constant;chi=constant
     for r,key,coef in zip(sub.itertuples(),kk,w*b):
      if key is True:constant+=coef;clo+=coef;chi+=coef
      elif key is not False:
       tt[key]+=float(coef);coarse_terms[(r.issue+r.start,r.node,r.duration)]+=float(coef)
     clo+=sum(min(0,v) for v in coarse_terms.values());chi+=sum(max(0,v) for v in coarse_terms.values())
     result=solve_bounds(constant,tt)
     for side in ['lower','upper']:
      witness={(h,n):g for h,n,g in result[side+'_witness']};actual=direct(sub,w,sign,raw,witness,cfg);err=abs(actual-result[side]);maxerr=max(maxerr,err);assert err<1e-9
      result[side+'_certificate']['direct_completed_weather_brier']=actual
     assert clo-1e-9<=result['lower']<=result['upper']+1e-9 and result['upper']<=chi+1e-9
     if group=='start0' and name in ['interaction','analog_update','graph_update']:assert abs(result['lower'])<1e-12 and abs(result['upper'])<1e-12
     record=dict(**base,comparison=name,role='primary' if name=='interaction' else 'auxiliary',shared_lower=result['lower'],shared_upper=result['upper'],window_label_lower=clo,window_label_upper=chi,unknown_hours=result['unknown_hours'],window_terms=result['window_terms'],certificate='PASS')
     rows.append(record);write(OUT/'certificates'/f'{pop}__{scope}__{group}__{name}.json',dict(**record,optimization=result));certificates+=1
    print('DONE',pop,scope,group,flush=True)
 csv('bounds_summary.csv',rows);csv('empty_support.csv',empty)
 badpath=ART/'review_iteration2_v1/data_gaps/partial_hour_evidence.npz';sensitivity={}
 if badpath.exists():
  bad=np.load(badpath)['known_bad'];unique_unknown={key for key in terms if key is not True and key is not False};hit={k for k in unique_unknown if any(bad[h,n] for h,n in k)}
  sensitivity=dict(known_bad_total=int(bad.sum()),unknown_distinct_hoursets=len(unique_unknown),hit_distinct_hoursets=len(hit),unknown_seed_window_rows=sum(k is not True and k is not False for k in terms),hit_seed_window_rows=sum(k in hit for k in terms if k is not True and k is not False),scope='Intersection count only; main bounds use original raw64. No partial-evidence sensitivity bounds run.')
 write(OUT/'partial_evidence_overlap.json',sensitivity)
 for path,h in identity.items():assert sha(path)==h,path
 write(OUT/'verification.json',dict(status='PASS',**audit,matched_seed_windows=len(d),origins=44,complete_origins=len(d[d.complete][['fold','issue']].drop_duplicates()),certificates=certificates,max_direct_endpoint_error=maxerr,source_hashes_unchanged=len(identity),no_refit=True,no_action_change=True,tests='6 synthetic tests GREEN after recorded missing-module RED'))
 print(pd.DataFrame(rows).query("population=='all44_primary' and scope=='ALL' and comparison=='interaction'").to_string(index=False),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','evaluate']);args=parser.parse_args();globals()[args.stage]()

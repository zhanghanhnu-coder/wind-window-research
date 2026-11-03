"""Original-scale 8/10/12 sensitivity of the frozen fair graph-mean contrast."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from precision_truth_core import load_truth,window_labels,SOURCE
from research_review_baselines import source
from lead_calibration_core import apply_groups
from closure_joint_core import probabilities
from closure_outage_core import paired_bounds
from missing_window_bounds_core import physical_mapping
from review_baselines_core import OFFSETS,GROUPS

ROOT=Path(__file__).resolve().parent;ART=ROOT/'artifacts';OLD=ART/'review_followup_v1/baselines'
OUT=ART/'review_closure_v1/threshold_h1';FOLDS=['M05','M07','M09','M11'];SEEDS=[20241101,20241102,20241103]
def main():
    OUT.mkdir(parents=True,exist_ok=True);identity={}
    def track(p):p=Path(p);identity[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
    def write(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
    protocol=dict(thresholds=[8,10,12],contrast='context1_graph minus context1_mean',families='same24h history frozen weights',
      evidence='raw64',calibration='existing natural refit64, unchanged at all thresholds',cohorts=['all880','complete_input392'],
      outcomes=['single6h','group6h'],offsets=OFFSETS,aggregation='equal forecast records after averaging seed losses; same physical unknown event shared',
      scope='Sensitivity of original-scale proxy, not independent confirmation, physical safety limits or new calibration')
    protocol=json.loads(json.dumps(protocol));p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==protocol
    else:write('protocol.json',protocol)
    for n in ['closure_threshold_h1.py','closure_joint_core.py','closure_outage_core.py','precision_truth_core.py']:track(ROOT/n)
    track(SOURCE);_,_,raw=load_truth();pop={};probs={};natural_checks=0
    for fold in FOLDS:
        f=np.load(track(ART/'seasonal_probability_v1'/fold/'data.npz'));ids=f['test_ids']
        z={'ids':ids,'complete':np.array([(f['x'][i-23:i+1,:,2]==1).all() for i in ids])}
        for kind in ['single','group']:z[kind+'_inverse']=physical_mapping(ids,kind)[1]
        for t in [8,10,12]:
            z['single',t]=np.concatenate([window_labels(raw,ids,o,threshold=t) for o in OFFSETS],axis=1)
            z['group',t]=np.stack([window_labels(raw,ids,o,threshold=t,node_indices=list(g),joint=True) for o in OFFSETS for g in GROUPS],axis=1)
        pop[fold]=z
        for family in ['context1_graph','context1_mean']:
          for seed in SEEDS:
            q=apply_groups(np.load(track(source(fold,family,seed)))['test'],np.load(track(OLD/fold/f'calibration_{family}_{seed}.npz'))['refit64'],float(f['scale']),True)
            ranks=np.load(track(ART/'partial_copula_v1'/fold/f'ranks_gated_graph_atoms_{seed}_partial05.npz'))['ranks']
            natural=np.load(track(OLD/fold/f'probabilities_{family}_{seed}_refit64_natural.npz'))
            fields={}
            for t in [8,10,12]:
                pair=probabilities(q,ranks,t)
                for kind,v in zip(['single','group'],pair):
                    probs[fold,family,seed,kind,t]=v;fields[f'{kind}_{t}']=v
                    if t==10:np.testing.assert_array_equal(v,natural[kind]);natural_checks+=1
            np.savez_compressed(OUT/f'{fold}_{family}_{seed}.npz',**fields)
        print('THRESHOLD',fold,flush=True)
    rows=[]
    for cohort in ['all880','complete_input392']:
      for scope in ['ALL']+FOLDS:
       ff=FOLDS if scope=='ALL' else [scope]
       for t in [8,10,12]:
        for kind in ['single','group']:
            masks={f:pop[f]['complete'] if cohort=='complete_input392' else np.ones(len(pop[f]['ids']),bool) for f in ff}
            y=np.concatenate([pop[f][kind,t][masks[f]] for f in ff]);ixs=[];shift=0
            for f in ff:
                ix=pop[f][kind+'_inverse'];ixs.append(ix[masks[f]]+shift);shift+=int(ix.max())+1
            ix=np.concatenate(ixs)
            a=np.stack([np.concatenate([probs[f,'context1_graph',s,kind,t][masks[f]] for f in ff]) for s in SEEDS])
            b=np.stack([np.concatenate([probs[f,'context1_mean',s,kind,t][masks[f]] for f in ff]) for s in SEEDS])
            lo,hi=paired_bounds(a,b,y,ix)
            rows.append(dict(cohort=cohort,scope=scope,threshold=t,kind=kind,origins=len(y),records=y.size,physical_windows=len(np.unique(ix)),
                known_records=int(np.isfinite(y).sum()),unknown_records=int(np.isnan(y).sum()),lower=lo,upper=hi))
    pd.DataFrame(rows).to_csv(OUT/'paired_bounds.csv',index=False,encoding='utf-8-sig')
    original=pd.read_csv(OLD/'paired_brier.csv');selected=original[(original.method=='refit64')&(original.regime=='natural')&(original.candidate=='context1_graph')&(original.baseline=='context1_mean')]
    for r in selected.itertuples():
        row=next(v for v in rows if (v['cohort'],v['scope'],v['threshold'],v['kind'])==('all880',r.scope,10,r.kind))
        assert abs(row['lower']-r.lower)<1e-12 and abs(row['upper']-r.upper)<1e-12
    for path,digest in identity.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    write('source_identity.json',identity);write('verification.json',dict(status='PASS',rows=len(rows),natural_probability_checks=natural_checks,baseline_comparisons_reproduced=len(selected),source_files_unchanged=len(identity)))
    print(pd.DataFrame(rows).query("scope=='ALL' and kind=='single'").to_string(index=False),flush=True)

if __name__=='__main__':main()

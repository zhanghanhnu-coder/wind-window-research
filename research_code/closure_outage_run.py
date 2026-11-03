"""Frozen graph/mean/own forecasts under synchronized causal input outages."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time
import numpy as np
import pandas as pd
import torch
from closure_outage_core import outage_history,paired_bounds
from research_review_baselines import infer,checkpoint,source
from lead_calibration_core import apply_groups
from review_baselines_core import window_probabilities
from precision_truth_core import load_truth,SOURCE

ROOT=Path(__file__).resolve().parent;ART=ROOT/'artifacts'
OLD=ART/'review_followup_v1/baselines';OUT=ART/'review_closure_v1/outage_stress'
FOLDS=['M05','M07','M09','M11'];SEEDS=[20241101,20241102,20241103]
FAMILIES=['context1_graph','context1_mean','context1_own'];HOURS=[0,6,12,24]
IDENT={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):p=Path(p);IDENT[str(p)]=sha(p);return p
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),models=FAMILIES,seeds=SEEDS,outage_hours=HOURS,
      primary='Graph minus mean, 6h whole-farm outage, single window, all880; other doses/cohorts/own/group auxiliary stress',
      labels='strict raw64 threshold10 original units; unchanged outcomes',
      calibration='frozen natural-input refit64; no fit on stress',
      aggregation='seed loss average then equal forecast record weight; same physical window shares unknown label; no cross-window hourly AND constraint',
      cohorts=['all880','complete_input392'],scope='Controlled stress, not estimated missing mechanism, action eligibility, confidence interval or external validation')
    dest=OUT/'protocol.json'
    if dest.exists():
        old=json.loads(dest.read_text(encoding='utf-8'));protocol['created_utc']=old['created_utc'];assert old==protocol
    else:write(dest,protocol)
    for name in ['closure_outage_core.py','closure_outage_run.py','closure_outage_test.py','review_baselines_core.py','research_review_baselines.py','lead_calibration_core.py']:track(ROOT/name)
    track(SOURCE);_,_,raw=load_truth();assert raw.dtype==np.float64
    pop={};support=[];probs={};checks=0;maxerr=0.;streams=0
    for fold in FOLDS:
        f=np.load(track(ART/'seasonal_probability_v1'/fold/'data.npz'));z=dict(np.load(track(OLD/fold/'population.npz')))
        ids=z['ids'];assert np.array_equal(ids,f['test_ids'])
        complete=np.array([(f['x'][i-23:i+1,:,2]==1).all() for i in ids]);z['complete']=complete;pop[fold]=z
        histories={h:outage_history(f['x'],ids,h) for h in HOURS}
        for h in HOURS:
            support.append(dict(fold=fold,outage_hours=h,origins=len(ids),complete_input_origins=int(complete.sum()),
              all_last24_valid_fraction=float(histories[h][:,-24:,:,2].mean()),complete_last24_valid_fraction=float(histories[h][complete,-24:,:,2].mean())))
        # Future poisoning uses only one current issue; not an all-ids global cutoff shortcut.
        for pos in [0,len(ids)//2,len(ids)-1]:
            t=int(ids[pos]);poison=f['x'].copy();poison[t+1:]=np.nan
            for h in HOURS:np.testing.assert_array_equal(outage_history(poison,np.array([t]),h),histories[h][pos:pos+1]);checks+=1
        for family in FAMILIES:
            for seed in SEEDS:
                ck=torch.load(track(checkpoint(fold,family,seed)),map_location='cpu',weights_only=False)
                theta=np.load(track(OLD/fold/f'calibration_{family}_{seed}.npz'))['refit64']
                ranks=np.load(track(ART/'partial_copula_v1'/fold/f'ranks_gated_graph_atoms_{seed}_partial05.npz'))['ranks']
                natural=np.load(track(OLD/fold/f'probabilities_{family}_{seed}_refit64_natural.npz'))
                original=np.load(track(source(fold,family,seed)))['test']
                for h in HOURS:
                    dest=OUT/f'{fold}_{family}_{seed}_outage{h}.npz'
                    if dest.exists():a=dict(np.load(dest))
                    else:
                        qraw=infer(ck,family,histories[h]);q=apply_groups(qraw,theta,float(f['scale']),True)
                        single,group=window_probabilities(q,ranks)
                        a=dict(ids=ids,single=single,group=group)
                        np.savez_compressed(dest,**a)
                        if h==0:
                            err=float(np.max(abs(qraw-original)));assert err<1e-5;maxerr=max(maxerr,err)
                    assert np.array_equal(a['ids'],ids)
                    for kind in ['single','group']:
                        if h==0:np.testing.assert_array_equal(a[kind],natural[kind])
                        probs[fold,family,seed,h,kind]=a[kind]
                    streams+=1
                print('OUTAGE',fold,family,seed,round(time.monotonic()-started,1),flush=True)
    assert sum(int(z['complete'].sum()) for z in pop.values())==392
    rows=[]
    for cohort in ['all880','complete_input392']:
      for scope in ['ALL']+FOLDS:
       selected=FOLDS if scope=='ALL' else [scope]
       for kind in ['single','group']:
        labels=[];inverses=[];offset=0;origin_count=0
        for fold in selected:
            z=pop[fold];mask=z['complete'] if cohort=='complete_input392' else np.ones(len(z['ids']),bool)
            y=z[kind+'_labels'][mask];ix=z[kind+'_inverse'][mask]
            labels.append(y);inverses.append(ix+offset);offset+=int(z[kind+'_inverse'].max())+1;origin_count+=int(mask.sum())
        y=np.concatenate(labels);ix=np.concatenate(inverses)
        def load(family,h):
            return np.stack([np.concatenate([probs[f,family,s,h,kind][pop[f]['complete'] if cohort=='complete_input392' else np.ones(len(pop[f]['ids']),bool)] for f in selected]) for s in SEEDS])
        for h in HOURS:
            contrasts=[('graph_minus_mean',load(FAMILIES[0],h),load(FAMILIES[1],h)),('graph_minus_own',load(FAMILIES[0],h),load(FAMILIES[2],h))]
            contrasts += [(f'{fam}_outage_minus_natural',load(fam,h),load(fam,0)) for fam in FAMILIES]
            for contrast,a,b in contrasts:
                lo,hi=paired_bounds(a,b,y,ix)
                known=np.isfinite(y);kd=np.mean((a[:,known]-y[known])**2-(b[:,known]-y[known])**2) if known.any() else None
                rows.append(dict(cohort=cohort,scope=scope,kind=kind,outage_hours=h,contrast=contrast,origins=origin_count,
                  forecast_records=int(y.size),physical_windows=int(len(np.unique(ix))),known_records=int(known.sum()),unknown_records=int((~known).sum()),
                  lower=lo,upper=hi,known_only_difference=kd,certificate='PASS'))
    pd.DataFrame(rows).to_csv(OUT/'paired_bounds.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(support).to_csv(OUT/'support.csv',index=False,encoding='utf-8-sig')
    for path,digest in IDENT.items():assert sha(path)==digest,path
    write(OUT/'source_identity.json',IDENT)
    write(OUT/'verification.json',dict(status='PASS',streams=streams,source_files_unchanged=len(IDENT),
      causal_poison_checks=checks,natural_max_raw_atom_error=maxerr,zero_outage_probabilities_exact=True,
      paired_rows=len(rows),independent_event_endpoint_checks=len(rows)*2,complete_input_origins=392,total_origins=880,
      seconds=time.monotonic()-started,scope='Frozen model input stress; no reoptimization, retraining or outcome masking'))
    print(pd.DataFrame(rows).query("scope=='ALL' and kind=='single' and contrast=='graph_minus_mean'").to_string(index=False),flush=True)

if __name__=='__main__':main()

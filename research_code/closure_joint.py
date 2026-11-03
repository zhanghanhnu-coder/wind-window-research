"""Frozen H2 dependence ablation and original-scale threshold sensitivity."""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
import pandas as pd
from precision_truth_core import load_truth,window_labels,SOURCE
from lead_calibration_core import apply_groups
from missing_window_bounds_core import physical_mapping,coefficients,sharp_bounds,collapse_labels
from review_baselines_core import OFFSETS,GROUPS
from closure_joint_core import intervene,probabilities,direct_probabilities,materialize

BASE=Path(__file__).resolve().parent
OUT=BASE/'artifacts/review_closure_v1/joint_structure'
OLD=BASE/'artifacts/review_followup_v1/baselines'
THRESHOLDS=(8.,10.,12.);SHUFFLES=(2026110201,2026110202,2026110203)
SEEDS=(20241101,20241102,20241103);FAMILIES=('context1_graph','analog112')
TRACK={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):TRACK[str(p)]=sha(p);return Path(p)
def js(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def protocol():
    OUT.mkdir(parents=True,exist_ok=True)
    spec=dict(version='closure_joint_v1',thresholds=THRESHOLDS,offsets=OFFSETS,shuffle_seeds=SHUFFLES,model_seeds=SEEDS,families=FAMILIES,scenarios=57,primary='context1_graph threshold10 single6h original minus time-shuffled, equal forecast-record weight over all four offsets',secondary='original minus space-shuffled group6h; both-shuffled; thresholds8/12; Graph-minus-analog',calibration='reuse natural refit64 smooth affine parameters, NO refit or training',repetitions='score each permutation and model seed separately then arithmetic mean of scores; no averaged probabilities before Brier',permutation='one frozen permutation per lead/node shared across origins, folds, models; time keeps contemporaneous spatial vectors, space keeps whole node trajectories; finite random pairing approximates independence',unknown='one shared binary label per same physical window; conservative bounds, no shared-hour constraints between distinct windows or thresholds or event kinds',scope='all 2024 origins remain development evidence; no confidence interval or deployment claim',source_hashes={str(BASE/n):sha(BASE/n) for n in ['closure_joint.py','closure_joint_core.py','closure_joint_tests.py','precision_truth_core.py','lead_calibration_core.py','missing_window_bounds_core.py','review_baselines_core.py']})
    # JSON-normalize tuples before frozen comparison.
    spec=json.loads(json.dumps(spec))
    p=OUT/'protocol.json'
    if p.exists():assert read(p)==spec,'frozen protocol differs'
    else:js(p,spec)
    return spec

def run():
    spec=protocol();start=time.monotonic();times,nodes,raw=load_truth();track(SOURCE)
    inv=read(track(OLD/'inventory.json'))
    streams=[r for r in inv if r['family'] in FAMILIES and r['regime']=='natural' and r['method']=='refit64']
    assert len(streams)==24
    audit=[];populations={}
    for fold in dict.fromkeys(r['fold'] for r in streams):
        ids=np.load(track(OLD/fold/'population.npz'))['ids'];fields={'ids':ids}
        for kind in ('single','group'):
            keys,ix=physical_mapping(ids,kind);fields[kind+'_keys']=keys;fields[kind+'_inverse']=ix
        for threshold in THRESHOLDS:
            fields[f'single_{int(threshold)}']=np.concatenate([window_labels(raw,ids,off,threshold=threshold) for off in OFFSETS],1)
            fields[f'group_{int(threshold)}']=np.stack([window_labels(raw,ids,off,threshold=threshold,node_indices=list(g),joint=True) for off in OFFSETS for g in GROUPS],1)
        populations[fold]=fields
        np.savez_compressed(OUT/f'population_{fold}.npz',**fields)
    for r in streams:
        fold,family,seed=r['fold'],r['family'],r['seed'];dest=OUT/f'probabilities_{fold}_{family}_{seed}.npz'
        pack=np.load(track(r['source']));params=np.load(track(r['parameters']))['refit64'];ranks=np.load(track(r['ranks']))['ranks']
        scale=float(np.load(track(BASE/'artifacts/seasonal_probability_v1'/fold/'data.npz'))['scale'])
        q=apply_groups(pack[r['source_key']],params,scale,True)
        assert q.dtype==np.float64 and ranks.shape==(57,72,17)
        expected=np.broadcast_to(np.arange(57)[:,None,None],ranks.shape)
        assert np.array_equal(np.sort(ranks,axis=0),expected)
        # Evaluate original plus all nine declared permutations. No winner selection.
        variants=[('original',0,ranks)]
        for kind in ('time','space','both'):
            for rep,shuffle_seed in enumerate(SHUFFLES,1):
                rr,ix=intervene(ranks,kind,shuffle_seed)
                variants.append((kind,rep,rr))
                np.savez_compressed(OUT/f'indices_{kind}_{rep}.npz',indices=ix)
        fields={};base_sample=materialize(q[[0,-1]],ranks)
        for kind,rep,rr in variants:
            # Actual value equality audit, plus all-origin algebraic certificate
            # above: each coordinate is a bijection of the complete 57 atoms.
            np.testing.assert_array_equal(np.sort(materialize(q[[0,-1]],rr),axis=1),np.sort(base_sample,axis=1))
            for threshold in THRESHOLDS:
                p,g=probabilities(q,rr,threshold)
                fields[f'single_{int(threshold)}_{kind}_{rep}']=p
                fields[f'group_{int(threshold)}_{kind}_{rep}']=g
                for j in [0,len(q)-1]:
                    dp,dg=direct_probabilities(q[j],rr,threshold)
                    np.testing.assert_array_equal(dp,p[j]);np.testing.assert_array_equal(dg,g[j])
                if kind=='original' and threshold==10:
                    archived=np.load(track(r['probabilities']))
                    np.testing.assert_array_equal(p,archived['single']);np.testing.assert_array_equal(g,archived['group'])
            if kind=='original':
                globalr,_=intervene(ranks,'global',SHUFFLES[0])
                for threshold in THRESHOLDS:
                    gp,gg=probabilities(q,globalr,threshold)
                    np.testing.assert_array_equal(gp,fields[f'single_{int(threshold)}_original_0'])
                    np.testing.assert_array_equal(gg,fields[f'group_{int(threshold)}_original_0'])
        np.savez_compressed(dest,**fields)
        audit.append(dict(fold=fold,family=family,seed=seed,origins=len(q),coordinate_multisets_exact=True,global_negative_control_exact=True,independent_direct_checks=60,actual_value_multiset_checks=20,crps_invariant='exact marginal index certificate for every origin/lead/node; actual values additionally checked at two origins'))
        print('H2',fold,family,seed,'seconds',round(time.monotonic()-start,1),flush=True)
    assert sum(len(p['ids']) for p in populations.values())==880
    for p,h in TRACK.items():assert sha(p)==h,p
    js(OUT/'source_identity.json',TRACK);js(OUT/'validation.json',dict(audits=audit,origins=880,streams=24,seconds=time.monotonic()-start,protocol_sha=sha(OUT/'protocol.json')))
    evaluate()

def evaluate():
    folds=['M05','M07','M09','M11'];pop={f:dict(np.load(OUT/f'population_{f}.npz')) for f in folds}
    cache={(f,m,s):dict(np.load(OUT/f'probabilities_{f}_{m}_{s}.npz')) for f in folds for m in FAMILIES for s in SEEDS}
    absolute=[];paired=[];support=[]
    for scope in [*folds,'ALL']:
        ff=folds if scope=='ALL' else [scope]
        seeds_sets=[('mean',SEEDS)] if scope=='ALL' else [(str(s),(s,)) for s in SEEDS]+[('mean',SEEDS)]
        for threshold in THRESHOLDS:
            for kind,width in [('single',17),('group',3)]:
                for offset in ['all',*OFFSETS]:
                    cols=np.arange(width*4) if offset=='all' else np.arange(OFFSETS.index(offset)*width,(OFFSETS.index(offset)+1)*width)
                    ys=[];ixs=[];shift=0
                    for f in ff:
                        y=pop[f][f'{kind}_{int(threshold)}'][:,cols];rawix=pop[f][kind+'_inverse'][:,cols]
                        _,ix=np.unique(rawix,return_inverse=True);ix=ix.reshape(y.shape)+shift;shift=int(ix.max())+1
                        ys.append(y);ixs.append(ix)
                    y=np.concatenate(ys);ix=np.concatenate(ixs);unique=collapse_labels(y,ix)
                    meta=dict(scope=scope,threshold=threshold,kind=kind,offset=str(offset),records=y.size,known=int(np.isfinite(y).sum()),unknown=int(np.isnan(y).sum()),unknown_rate=float(np.isnan(y).mean()),physical_windows=len(unique),unknown_physical_windows=int(np.isnan(unique).sum()),origins=len(y))
                    support.append(meta)
                    for seedlabel,seeds in seeds_sets:
                        def get(model,variant,rep):
                            return np.stack([np.concatenate([cache[f,model,s][f'{kind}_{int(threshold)}_{variant}_{rep}'][:,cols] for f in ff]) for s in seeds])
                        def record(pa,pb):
                            c,d=coefficients(pa,pb);b=sharp_bounds(c,d,y,ix);known=np.isfinite(y)
                            loss=np.mean((pa-y)**2,axis=0)
                            if pb is not None:loss-=np.mean((pb-y)**2,axis=0)
                            return dict(lower=b['lower'],upper=b['upper'],known_brier_or_delta=float(loss[known].mean()))
                        for model in FAMILIES:
                            base=get(model,'original',0)
                            absolute.append(dict(**meta,seed=seedlabel,model=model,variant='original',repeat='0',**record(base,None)))
                            for variant in ('time','space','both'):
                                reps=[('mean',(1,2,3))] if scope=='ALL' else [(str(i),(i,)) for i in (1,2,3)]+[('mean',(1,2,3))]
                                for replabel,rr in reps:
                                    comparison=np.concatenate([get(model,variant,i) for i in rr]);a=np.concatenate([base for _ in rr])
                                    absolute.append(dict(**meta,seed=seedlabel,model=model,variant=variant,repeat=replabel,**record(comparison,None)))
                                    primary=scope=='ALL' and model=='context1_graph' and threshold==10 and kind=='single' and offset=='all' and variant=='time'
                                    paired.append(dict(**meta,seed=seedlabel,model=model,comparison='original-minus-'+variant,repeat=replabel,primary=primary,**record(a,comparison)))
                        for variant in ('original','time','space','both'):
                            rr=(0,) if variant=='original' else (1,2,3)
                            a=np.concatenate([get('context1_graph',variant,i) for i in rr]);b=np.concatenate([get('analog112',variant,i) for i in rr])
                            paired.append(dict(**meta,seed=seedlabel,model='graph-minus-analog',comparison=variant,repeat='mean',primary=False,**record(a,b)))
    for name,rows in [('absolute_brier',absolute),('paired_brier',paired),('label_support',support)]:pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    primary=[r for r in paired if r['primary']];assert len(primary)==1
    js(OUT/'complete.json',dict(status='COMPLETE',primary=primary,absolute_rows=len(absolute),paired_rows=len(paired),support_rows=len(support),interpretation='development-only descriptive effects; missing-label identification bounds, not confidence intervals'))
    print('PRIMARY',primary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','evaluate'],default='all');args=p.parse_args()
    run() if args.phase=='all' else evaluate()

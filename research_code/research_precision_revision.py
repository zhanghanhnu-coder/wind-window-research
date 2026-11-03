"""Versioned authoritative labels and fixed-forecast rescoring, no model refit."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from precision_truth_core import load_truth,window_labels,SOURCE

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts'
OUT=ART/'review_followup_v1/precision'
OFFSETS=[0,24,48,66]
GROUPS=[[0,1,2],[8,9,10],[13,15,16]]
FOLDS=['M05','M07','M09','M11']
SEEDS=[20241101,20241102,20241103]
CASES=['B','R','W28','M50','G28','K4','K6']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    spec=dict(version='review_truth64_v1',truth_precision='float64 from original strict hourly max',
              offsets=OFFSETS,groups=GROUPS,thresholds=[8,10,12],duration=6,
              forecast_policy='all existing forecasts and calibration parameters frozen; evaluation-only change',
              decision_threshold=.8,score_policy='mean per-seed score; decisions use mean probability',
              scope='2024 development data, not independent validation')
    protocol=OUT/'protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text(encoding='utf-8'))==spec
    else:protocol.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    times,nodes,raw=load_truth()
    inputs={str(SOURCE):sha(SOURCE),str(ART/'data.npz'):sha(ART/'data.npz'),str(ROOT/'precision_truth_core.py'):sha(ROOT/'precision_truth_core.py')}
    np.savez_compressed(OUT/'hourly_truth64.npz',raw=raw,timestamps=np.array([str(t) for t in times]),nodes=nodes)
    rows=[];changes=[];splits=[];details=[];metric_records=[]
    for fold in FOLDS:
        data_path=ART/'seasonal_probability_v1'/fold/'data.npz';inputs[str(data_path)]=sha(data_path)
        with np.load(data_path) as pack:
            ids=pack['test_ids']; arrays={k:pack[k] for k in ['train_ids','val_ids','cal_ids','test_ids']}
        for split in ['train','val','cal','test']:
            split_ids=arrays[split+'_ids'];y=raw[split_ids[:,None]+np.arange(1,73)[None,:]]
            splits.append(dict(fold=fold,split=split,origins=len(split_ids),known_targets=int(np.isfinite(y).sum()),first_issue=str(times[split_ids[0]]),last_target=str(times[split_ids[-1]+72])))
            if split in ['cal','test']:arrays[split+'_truth64']=y
        population=ART/'state_gate_v1'/fold/'population.npz';inputs[str(population)]=sha(population)
        old=np.load(population)
        for threshold in [8,10,12]:
            single=np.concatenate([window_labels(raw,ids,offset=o,threshold=threshold) for o in OFFSETS],axis=1)
            group=np.stack([window_labels(raw,ids,offset=o,threshold=threshold,node_indices=g,joint=True) for o in OFFSETS for g in GROUPS],axis=1)
            for kind,y in [('single',single),('group',group)]:
                # Independent oracle: inspect raw intervals directly, without the new labeller.
                oracle=[]
                for issue in ids:
                    current=[]
                    for offset in OFFSETS:
                        segment=raw[issue+offset+1:issue+offset+7]
                        objects=[segment[:,j] for j in range(len(nodes))] if kind=='single' else [segment[:,g].ravel() for g in GROUPS]
                        for obj in objects:
                            current.append(0. if np.any(obj[np.isfinite(obj)]>threshold) else 1. if np.isfinite(obj).all() else np.nan)
                    oracle.append(current)
                np.testing.assert_array_equal(y,np.array(oracle))
                arrays[f'{kind}_labels_threshold{threshold}']=y
                if threshold!=10:continue
                previous=old[kind+'_labels'];inverse=old[kind+'_inverse'];keys=old[kind+'_keys']
                arrays[kind+'_inverse']=inverse;arrays[kind+'_keys']=keys
                changed=~((y==previous)|(np.isnan(y)&np.isnan(previous)))
                changes.append(dict(fold=fold,kind=kind,records=y.size,known=int(np.isfinite(y).sum()),unknown=int(np.isnan(y).sum()),changed=int(changed.sum()),changed_unique_windows=int(np.unique(inverse[changed]).size)))
                for r,c in np.argwhere(changed):
                    key=keys[inverse[r,c]]
                    details.append(dict(fold=fold,kind=kind,issue=str(times[ids[r]]),target_start=str(times[key[0]]),node_or_group=int(key[1]),old_label=previous[r,c],corrected_label=y[r,c]))
                for case in CASES:
                    probs=[]
                    for seed in SEEDS:
                        f=ART/'state_gate_v1'/fold/f'diagnostic_{case}_{seed}.npz';inputs[str(f)]=sha(f)
                        probs.append(np.load(f)[kind])
                    ps=np.asarray(probs);p=ps.mean(axis=0);accept=p>=.8-1e-12
                    known=np.isfinite(y)
                    score=np.mean((ps-y[None])**2,axis=0)
                    old_score=np.mean((ps-previous[None])**2,axis=0)
                    unique_labels=np.full(len(keys),np.nan)
                    unique_labels[inverse[known]]=y[known]
                    const=(ps*ps).mean(axis=0);coeff=1-2*p
                    grouped=np.bincount(inverse.ravel(),weights=coeff.ravel(),minlength=len(keys))
                    k=np.isfinite(unique_labels);fixed=const.sum()+np.sum(grouped[k]*unique_labels[k])
                    lower=(fixed+np.minimum(grouped[~k],0).sum())/y.size
                    upper=(fixed+np.maximum(grouped[~k],0).sum())/y.size
                    row=dict(fold=fold,kind=kind,case=case,n=y.size,known=int(known.sum()),unknown=int((~known).sum()),brier_known=float(np.nanmean(score)),brier_known_legacy=float(np.nanmean(old_score)),brier_all_lower=float(lower),brier_all_upper=float(upper),accepted=int(accept.sum()),tp=int((accept&(y==1)).sum()),fp=int((accept&(y==0)).sum()),fn=int((~accept&(y==1)).sum()),tn=int((~accept&(y==0)).sum()),unknown_accepted=int((accept&~known).sum()))
                    rows.append(row)
        np.savez_compressed(OUT/f'{fold}_truth64.npz',**arrays)
    frame=pd.DataFrame(rows)
    for (kind,case),g in frame.groupby(['kind','case']):
        total={k:int(g[k].sum()) for k in ['n','known','unknown','accepted','tp','fp','fn','tn','unknown_accepted']}
        total.update(fold='ALL',kind=kind,case=case,brier_known=float(np.average(g.brier_known,weights=g.known)),brier_known_legacy=float(np.average(g.brier_known_legacy,weights=g.known)),brier_all_lower=float(np.average(g.brier_all_lower,weights=g.n)),brier_all_upper=float(np.average(g.brier_all_upper,weights=g.n)))
        metric_records.append(total)
    frame=pd.concat([frame,pd.DataFrame(metric_records)],ignore_index=True)
    frame.to_csv(OUT/'fixed_forecast_scores64.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(changes).to_csv(OUT/'label_changes.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(details).to_csv(OUT/'changed_windows.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(splits).to_csv(OUT/'split_support.csv',index=False,encoding='utf-8-sig')
    for path,digest in inputs.items():assert sha(path)==digest,path
    result=dict(status='PASS',source_inputs=inputs,script_sha256=sha(__file__),independent_window_oracle='all four folds and three thresholds match',origins=sum(r['origins'] for r in splits if r['split']=='test'),changed_single=sum(r['changed'] for r in changes if r['kind']=='single'),changed_group=sum(r['changed'] for r in changes if r['kind']=='group'),scope=spec)
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(frame[(frame.fold=='ALL')&(frame.kind=='single')][['case','brier_known','fp','fn']].to_string(index=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_inputs','scope']},ensure_ascii=False))


if __name__=='__main__':main()

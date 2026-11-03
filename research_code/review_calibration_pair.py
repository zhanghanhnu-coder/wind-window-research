"""Fixed W28 minus B window comparison with original precision and target blocks."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from precision_truth_core import load_truth
from review_baselines_core import loss_event_totals

ART=Path(__file__).resolve().parent/'artifacts'
OUT=ART/'review_followup_v1/precision'
FOLDS=['M05','M07','M09','M11'];SEEDS=[20241101,20241102,20241103]


def main():
    protocol=dict(candidate='W28',baseline='B',threshold=10,duration=6,offsets=[0,24,48,66],replicates=2000,seed=20261105,blocks='within-fold seven-day target-start blocks; all seeds averaged before resampling',truth='original float64',scope='fixed forecast development-data comparison; no new calibration or training')
    p=OUT/'calibration_comparison_protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:p.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    times,_,_=load_truth();rng=np.random.default_rng(protocol['seed']);sources={};results=[]
    accum={k:dict(lo=np.zeros(2000),hi=np.zeros(2000),n=np.zeros(2000),pl=0.,pu=0.,pn=0,known_sum=0.,known_n=0) for k in ['single','group']}
    for fold in FOLDS:
        file=OUT/f'{fold}_truth64.npz';sources[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest();z=np.load(file)
        anchor=times[z['single_keys'][:,0].min()].normalize()
        blockcount=int(((times[z['single_keys'][:,0]]-anchor).total_seconds()//(7*86400)).max())+1
        draws=rng.integers(0,blockcount,(2000,blockcount))
        for kind in accum:
            y=z[f'{kind}_labels_threshold10'];ix=z[kind+'_inverse'];keys=z[kind+'_keys'];ps={}
            for case in ['B','W28']:
                vals=[]
                for seed in SEEDS:
                    f=ART/'state_gate_v1'/fold/f'diagnostic_{case}_{seed}.npz';sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest();vals.append(np.load(f)[kind])
                ps[case]=np.asarray(vals)
            lo,hi,count=loss_event_totals(ps['W28'],ps['B'],y,ix)
            blocks=((times[keys[:,0]]-anchor).total_seconds()//(7*86400)).astype(int)
            b_lo=np.bincount(blocks,weights=lo,minlength=blockcount);b_hi=np.bincount(blocks,weights=hi,minlength=blockcount);b_n=np.bincount(blocks,weights=count,minlength=blockcount)
            ds=np.mean((ps['W28']-y[None])**2-(ps['B']-y[None])**2,axis=0)
            a=accum[kind];a['lo']+=b_lo[draws].sum(1);a['hi']+=b_hi[draws].sum(1);a['n']+=b_n[draws].sum(1);a['pl']+=lo.sum();a['pu']+=hi.sum();a['pn']+=y.size;a['known_sum']+=np.nansum(ds);a['known_n']+=np.isfinite(y).sum()
            results.append(dict(scope=fold,kind=kind,known_difference=float(np.nanmean(ds)),lower=float(lo.sum()/y.size),upper=float(hi.sum()/y.size),outer_low=float(np.quantile(b_lo[draws].sum(1)/b_n[draws].sum(1),.025)),outer_high=float(np.quantile(b_hi[draws].sum(1)/b_n[draws].sum(1),.975)),records=y.size,known=int(np.isfinite(y).sum()),blocks=blockcount))
    for kind,a in accum.items():
        results.append(dict(scope='ALL',kind=kind,known_difference=float(a['known_sum']/a['known_n']),lower=float(a['pl']/a['pn']),upper=float(a['pu']/a['pn']),outer_low=float(np.quantile(a['lo']/a['n'],.025)),outer_high=float(np.quantile(a['hi']/a['n'],.975)),records=a['pn'],known=int(a['known_n']),blocks=36))
    pd.DataFrame(results).to_csv(OUT/'calibration_W28_minus_B.csv',index=False,encoding='utf-8-sig')
    for file,digest in sources.items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest
    (OUT/'calibration_comparison_verification.json').write_text(json.dumps(dict(status='PASS',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs=sources,rows=len(results)),indent=2),encoding='utf-8')
    print(pd.DataFrame(results).tail(2).to_string(index=False))


if __name__=='__main__':main()

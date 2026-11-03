"""Independent coefficient-form verification of the exact cross forecast pair."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from precision_truth_core import load_truth,window_labels
from missing_window_bounds_core import coefficients
from window_error_attribution_core import selected_bounds

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts/review_followup_v1/fixed_rolling'
FOLDS=['M05','M07','M09','M11'];SEEDS=[20241101,20241102,20241103]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def run():
    times,_,raw=load_truth();cache={}
    for f in FOLDS:
        z=dict(np.load(OUT/f/'population.npz'));ids=z['ids'];d=np.load(OUT/f/'draws.npz');anchor=pd.Timestamp(str(d['anchor']))
        single=np.concatenate([window_labels(raw,ids,x) for x in [0,24,48,66]],1);group=np.stack([window_labels(raw,ids,x,node_indices=g,joint=True) for x in [0,24,48,66] for g in [[0,1,2],[8,9,10],[13,15,16]]],1)
        np.testing.assert_array_equal(single,z['single_labels']);np.testing.assert_array_equal(group,z['group_labels'])
        p={m:{k:np.stack([np.load(OUT/f/f'probabilities_{m}_{s}.npz')[k] for s in SEEDS]) for k in ['single','group']} for m in ['fixed_B','rolling28']}
        cache[f]=(z,p,anchor,d['draws'])
    rows=pd.read_csv(OUT/'paired.csv');nchecked=0
    primary=rows[rows.primary];assert len(primary)==1 and primary.iloc[0]['scope']=='ALL' and primary.iloc[0]['kind']=='single' and primary.iloc[0]['metric']=='brier'
    for row in rows.itertuples():
        fs=FOLDS if row.scope=='ALL' else [row.scope];lower=upper=0.;n=0;bl=np.zeros(2000);bu=np.zeros(2000);bn=np.zeros(2000)
        for f in fs:
            z,p,anchor,draw=cache[f];y=z[row.kind+'_labels'];ix=z[row.kind+'_inverse'];pa=p['rolling28'][row.kind];pb=p['fixed_B'][row.kind]
            if row.metric=='brier':c,d=coefficients(pa,pb)
            else:
                a=(pa.mean(0)>=.8-1e-12).astype(float);b=(pb.mean(0)>=.8-1e-12).astype(float)
                c=a-b if row.metric=='FP' else np.zeros_like(a);d=b-a
            bound=selected_bounds(c,d,y,ix,np.ones_like(y,bool));lower+=bound['lower']*y.size;upper+=bound['upper']*y.size;n+=y.size
            # Shared coefficients aggregated at physical-event level, separately from runner's outcome enumeration.
            eventc=np.bincount(ix.ravel(),weights=c.ravel());eventd=np.bincount(ix.ravel(),weights=d.ravel());count=np.bincount(ix.ravel());label=np.full(len(count),np.nan);known=np.isfinite(y);label[ix[known]]=y[known];ky=np.isfinite(label)
            fixed=eventc.copy();fixed[ky]+=eventd[ky]*label[ky];lo=fixed.copy();hi=fixed.copy();lo[~ky]+=np.minimum(eventd[~ky],0);hi[~ky]+=np.maximum(eventd[~ky],0)
            block=((times[z[row.kind+'_keys'][:,0]]-anchor).total_seconds()//(7*86400)).astype(int);k=draw.shape[1]
            ls=np.bincount(block,weights=lo,minlength=k);us=np.bincount(block,weights=hi,minlength=k);ns=np.bincount(block,weights=count,minlength=k)
            bl+=ls[draw].sum(1);bu+=us[draw].sum(1);bn+=ns[draw].sum(1)
        np.testing.assert_allclose([lower/n,upper/n,*np.quantile(bl/bn,[.025]),*np.quantile(bu/bn,[.975])],[row.lower,row.upper,row.outer_low,row.outer_high],atol=1e-12,rtol=0);assert row.n==n;nchecked+=1
    identity=read(OUT/'source_identity.json')
    for p,h in identity.items():assert sha(p)==h,p
    version=read(OUT/'version_identity.json');assert len(version['sources'])==24 and len(version['context1_graph_identity_checks'])==12
    result=dict(status='PASS',paired_rows=nchecked,all_bootstrap_replicates_checked=2000,truth_folds_checked=4,source_files_unchanged=len(identity),source_identity='exact cross frozen/rolling paths and keys, not state_gate B/W28',independent_method='native selected_bounds and seed-loss coefficients versus runner direct binary-outcome enumeration')
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='final_manifest.json'}
    for p in [ROOT/'review_fixed_rolling_pair.py',Path(__file__),ROOT/'precision_truth_core.py',ROOT/'review_baselines_core.py',ROOT/'artifacts/seasonal_rolling_v1/inventory.json',ROOT/'artifacts/review_followup_v1/cross_execution/source_identity.json',ROOT/'artifacts/review_followup_v1/cross_execution/protocol.json',ROOT.parent/'研究笔记/基线实验_2026-11-01/严格完整小时风速.pkl']:manifest[str(p)]=sha(p)
    manifest.update(identity);(OUT/'final_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(result)
if __name__=='__main__':run()

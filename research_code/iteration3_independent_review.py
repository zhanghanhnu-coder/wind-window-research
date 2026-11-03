"""Independent reviewer: exhaustive W checks and direct matched endpoint checks."""
from pathlib import Path
import itertools, json, hashlib
import numpy as np
import pandas as pd
from iteration3_acquisition_core import worst_width
from precision_truth_core import load_truth

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/review_iteration3_v1/independent_review'
OUT.mkdir(parents=True,exist_ok=True)
result={}

# Independent brute force groups values by observed query return vector.
rng=np.random.default_rng(726203)
checks=0
for trial in range(12):
    n=int(rng.integers(1,6)); ids=[(j,trial%3) for j in range(n)]
    terms={tuple(ids[j] for j in range(n) if mask>>j&1):float(rng.integers(-6,7)) for mask in range(1,1<<n)}
    const=float(rng.integers(-9,10))
    assignments=list(itertools.product([0,1],repeat=n))
    values=[const+sum(c*np.prod([z[ids.index(k)] for k in key]) for key,c in terms.items()) for z in assignments]
    previous=None
    for budget in range(n+1):
        q=ids[:budget];groups={}
        for z,value in zip(assignments,values):groups.setdefault(z[:budget],[]).append(value)
        truth=max(max(v)-min(v) for v in groups.values())
        cert=worst_width(const,terms,q)
        assert cert['certificate']=='PASS' and abs(cert['worst_remaining_width']-truth)<1e-7
        if previous is not None:assert truth<=previous+1e-9
        previous=truth;checks+=1
result['independent_exhaustive_width_checks']=checks

base=ROOT/'artifacts/review_iteration3_v1/matched_updates'
if (base/'bounds_summary.csv').exists():
    d=pd.read_csv(base/'matched_windows.csv'); ledger=pd.read_csv(ROOT/'artifacts/review_iteration2_v1/analog_execution/support_window_ledger.csv')
    keys=['fold','issue','seed','task','start']; cells={'AI':('AnalogFixed','initial_candidate'),'GI':('GraphFixed','initial_candidate'),'AC':('AnalogFixed','current_candidate'),'GC':('GraphFixed','current_candidate')}
    for cell,(pred,stage) in cells.items():
        src=ledger[(ledger.prediction==pred)&(ledger.stage==stage)]
        joined=d.merge(src[keys+['probability','label','complete']],on=keys,suffixes=('','_src'),validate='one_to_one')
        assert len(joined)==len(d)
        assert np.allclose(joined[cell],joined.probability,rtol=0,atol=1e-15)
        assert np.allclose(joined.label,joined.label_src,equal_nan=True)
        assert (joined.complete==joined.complete_src).all()
    z=d[d.start==0];assert (z.AI==z.AC).all() and (z.GI==z.GC).all()
    _,_,raw=load_truth()
    signs={'interaction':[-1,1,1,-1],'initial_model_difference':[1,-1,0,0],'current_model_difference':[0,0,1,-1],'analog_update':[-1,0,1,0],'graph_update':[0,-1,0,1]}
    signs.update({c:[int(i==j) for i in range(4)] for j,c in enumerate(cells)})
    starts={'ALL':range(0,72,6),'start0':[0],'start6_18':[6,12,18],'start24_42':[24,30,36,42],'start48_66':[48,54,60,66]}
    maxerr=0.;certs=list((base/'certificates').glob('*.json'))
    for path in certs:
        a=json.loads(path.read_text(encoding='utf-8'));sub=d[d.start.isin(starts[a['start_group']])]
        if a['scope']!='ALL':sub=sub[sub.fold==a['scope']]
        if a['population']!='all44_primary':sub=sub[sub.complete]
        assert a['window_label_lower']-1e-9<=a['shared_lower']<=a['shared_upper']+1e-9<=a['window_label_upper']+2e-9
        for side in ['lower','upper']:
            witness={(h,n):v for h,n,v in a['optimization'][side+'_witness']};loss=[]
            for r in sub.itertuples():
                y=all((raw[h,r.node]<=10 if np.isfinite(raw[h,r.node]) else witness.get((h,r.node),0)) for h in range(r.issue+r.start+1,r.issue+r.start+r.duration+1))
                loss.append(float(np.dot((np.array([getattr(r,c) for c in cells])-y)**2,signs[a['comparison']])))
            # Direct group means, independent of implementation's row weights.
            x=sub[keys].copy();x['loss']=loss
            value=x.groupby(['fold','issue','seed']).loss.mean().groupby(['fold','issue']).mean().mean()
            err=abs(value-a['shared_'+side]);assert err<1e-9;maxerr=max(maxerr,err)
    result.update(matched_source_cells=len(d)*4,matched_certificates=len(certs),matched_endpoints=len(certs)*2,max_matched_direct_error=maxerr,start0_rows=len(z))
result['status']='PASS'
result['review_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))

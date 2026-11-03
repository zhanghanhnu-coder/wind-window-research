"""Independent ledger reconstruction and acquisition certificate audit."""
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import json,hashlib
import numpy as np
import pandas as pd
from precision_truth_core import load_truth

ROOT=Path(__file__).resolve().parent;ART=ROOT/'artifacts';BASE=ART/'review_iteration3_v1/acquisition'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
_,_,raw=load_truth();cases=read(ART/'review_followup_v1/cross_execution/protocol.json')['cases']
poly=defaultdict(Fraction);constant=Fraction();ledgers=[]
for c in cases:
    pair=[]
    for sign,folder,arm in [(1,'review_iteration2_v1/analog_execution','AnalogFixed'),(-1,'review_followup_v1/cross_execution','B')]:
        ev=read(ART/folder/'evaluations'/f"{c['key']}__fixed_B__{arm}__lookahead72.json");pair.append(ev)
        constant+=sign*(Fraction(str(ev['metrics']['booking_cost']))+sum((Fraction(str(t['operation_cost']))+Fraction(str(t['penalty'])) for t in ev['tasks']),Fraction()))/132
        for t in ev['tasks']:
            if t['choice'] is None:continue
            hours=range(c['issue']+t['start']+1,c['issue']+t['end']+1)
            if any(np.isfinite(raw[h,t['node']]) and raw[h,t['node']]>10 for h in hours):continue
            unknown=tuple((h,t['node']) for h in hours if not np.isfinite(raw[h,t['node']]))
            coef=-sign*Fraction(str(t['penalty']))/132
            if unknown:poly[unknown]+=coef
            else:constant+=coef
    ledgers.append((c,*pair))
poly={k:v for k,v in poly.items() if v};stored=pd.read_csv(BASE/'merged_terms.csv')
storedpoly={tuple(map(tuple,json.loads(r.unknown_hours))):Fraction(r.coefficient_exact) for r in stored.itertuples()}
assert poly==storedpoly
hours=set().union(*map(set,poly));assert len(hours)==234 and len(poly)==82
oldhours=pd.read_csv(ART/'review_iteration2_v1/data_gaps/targeted_missing_hours.csv')
assert hours<=set(zip(oldhours.hour_index,oldhours.node_index))
def val(z):return float(constant+sum((v for k,v in poly.items() if all(z[h] for h in k)),Fraction()))
maxerr=0.;certs=list((BASE/'certificates').glob('*.json'))
for path in certs:
    a=read(path);assert a['certificate']=='PASS' and a['status']==0
    zp={(h,n):z for h,n,z in a['witness_plus']};zm={(h,n):z for h,n,z in a['witness_minus']}
    assert set(zp)==hours==set(zm) and set(zp.values())<=set([0,1]) and set(zm.values())<=set([0,1])
    assert all(zp[tuple(h)]==zm[tuple(h)] for h in a['query'])
    dp=val(zp);dm=val(zm);w=dp-dm
    for error in [abs(w-a['worst_remaining_width']),abs(dp-a['direct_symbolic_plus']),abs(dm-a['direct_symbolic_minus']),abs(w-a['dual_upper'])]:assert error<1e-7;maxerr=max(maxerr,error)
dependency=read(BASE/'essential_dependency_witnesses.json')['witnesses'];assert len(dependency)==234
for a in dependency:
    h=tuple(a['hour']);zp={g:int(g in map(tuple,a['ones_when_h_zero'])) for g in hours};zm=dict(zp);zp[h]=1
    delta=val(zp)-val(zm);assert abs(delta)>1e-12 and abs(delta-float(Fraction(a['exact_change'])))<1e-7
curves=pd.read_csv(BASE/'budget_curves.csv')
for _,g in curves.groupby('strategy'):
    g=g.sort_values('budget_hours');assert (np.diff(g.worst_remaining_width)<=1e-7).all()
    assert abs(g.iloc[0].worst_remaining_width-110.60606060606061)<1e-7 and abs(g.iloc[-1].worst_remaining_width)<1e-7
plans=pd.read_csv(BASE/'query_plans.csv')
essential=pd.read_csv(BASE/'essential_hours.csv',dtype={'node':str})
oldorder={(int(r.hour_index),int(r.node_index)):int(r.request_order) for r in oldhours.itertuples()}
sensitivity={h:sum((abs(v) for k,v in poly.items() if h in k),Fraction()) for h in hours}
node_names={(int(r.hour_index),int(r.node_index)):r.node for r in essential.itertuples()}
orders={'timestamp_node':sorted(hours,key=lambda h:(h[0],node_names[h])),
        'previous_request_order':sorted(hours,key=lambda h:oldorder[h]),
        'coefficient_sensitivity':sorted(hours,key=lambda h:(-sensitivity[h],h[0],node_names[h]))}
for plan,g in plans.groupby('plan_id'):
    cert=read(BASE/'certificates'/f'{plan}.json');assert set(map(tuple,cert['query']))==set(zip(g.hour_index,g.node_index))
    ordered=g.sort_values('query_order');strategy=ordered.strategy.iloc[0];budget=int(ordered.budget_hours.iloc[0])
    assert list(zip(ordered.hour_index,ordered.node_index))==orders[strategy][:budget]
packages=pd.read_csv(BASE/'single_package_ranking.csv');assert packages.hours.sum()==234 and packages.node_index.notna().all()
for r in packages.itertuples():
    cert=read(BASE/'certificates'/f'package_{r.package_id}.json');q=sorted(map(tuple,cert['query']))
    assert len(q)==r.hours and all(n==r.node_index for h,n in q) and all(b[0]-a[0]==1 for a,b in zip(q,q[1:]))
result=dict(status='PASS',independently_reconstructed_exact_terms=len(poly),essential_hours=len(hours),independent_two_completion_certificates=len(certs),dependency_witnesses=len(dependency),budget_strategies=curves.strategy.nunique(),packages=len(packages),max_exact_polynomial_certificate_error=maxerr,scope='Original task ledgers independently converted to exact Fraction polynomial; no implementation canonical/arm_expression used',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(BASE/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))

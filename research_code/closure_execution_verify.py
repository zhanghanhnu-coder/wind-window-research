"""Independent saved-certificate and rational task/resource readback verifier."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json
import numpy as np
import pandas as pd
from precision_truth_core import load_truth
from two_stage_core import make_problem,default_config,audit_schedule,booking_cost

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/review_closure_v1/execution_sensitivity'
OLD=ROOT/'artifacts/review_followup_v1/cross_execution'
ANALOG=ROOT/'artifacts/review_iteration2_v1/analog_execution'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def rational(x):return F(str(x))

def main():
    for path,digest in read(OUT/'sources_sha256.json').items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    config=default_config();p=make_problem(config);raw=load_truth()[2]
    cases=read(OLD/'protocol.json')['cases'];protocol=read(OUT/'protocol.json')
    cache={};checked=0;failed_checked=0;maxerror=0.
    def load(c,arm):
        nonlocal checked,failed_checked
        root=ANALOG if arm[1]=='AnalogFixed' else OLD
        name=f"{c['key']}__{'__'.join(arm)}.json";path=root/'evaluations'/name
        if path in cache:return cache[path]
        e=read(path);traj=read(root/'trajectories'/name)['trajectory']
        audit_schedule(p,traj['x'],np.ones(len(p.choices),bool),traj['attempts'])
        assert sorted(t['choice'] for t in e['tasks'] if t['choice'] is not None)==sorted(traj['attempts'])
        op=F(0);book=sum(rational(v)*int(x) for v,x in zip(p.resource_cost,traj['x']))
        for t in e['tasks']:
            original=config['tasks'][t['task']]
            assert t['penalty']==original['penalty'] and t['duration']==original['duration'] and t['node']==original['node']
            if t['choice'] is None:
                assert t['status']=='unattempted' and t['operation_cost']==0
                continue
            o=p.choices[t['choice']]
            assert t['start']==o['start'] and t['end']==o['end'] and o['task']==t['task']
            fee=(rational(config['boat_hourly'][o['boat']])+rational(config['crew_hourly'][o['crew']]))*t['duration']+rational(config['late_fraction'])*rational(t['penalty'])*o['start']/p.H
            assert abs(float(fee)-t['operation_cost'])<1e-9
            op+=fee
            window=raw[c['issue']+t['start']+1:c['issue']+t['end']+1,t['node']]
            status='failed' if np.any(window[np.isfinite(window)]>10) else 'successful' if np.isfinite(window).all() else 'unknown'
            assert status==t['status']
            if status=='failed':failed_checked+=1
        assert abs(float(op)-e['metrics']['operating_cost'])<1e-9
        assert abs(float(book)-booking_cost(p,traj['x']))<1e-9
        assert abs(float(book)-e['metrics']['booking_cost'])<1e-9
        e['_exact_constant']=book+op;checked+=1;cache[path]=e;return e
    complete={(c['fold'],c['issue']) for c in cases if load(c,['fixed_B','B','myopic'])['ideal']['complete']}
    def scope_cases(scope):
        if scope=='ALL':return cases
        if scope=='COMPLETE15':return [c for c in cases if (c['fold'],c['issue']) in complete]
        if scope.startswith('COMPLETE15_'):return [c for c in cases if c['fold']==scope.split('_')[1] and (c['fold'],c['issue']) in complete]
        if '_issue' in scope:
            fold,issue=scope.split('_issue');return [c for c in cases if c['fold']==fold and c['issue']==int(issue)]
        return [c for c in cases if c['fold']==scope]
    def value(e,c,w,metric):
        result=F(0)
        for t in e['tasks']:
            good=t['choice'] is not None
            if good:
                for hour in range(c['issue']+t['start']+1,c['issue']+t['end']+1):
                    z=raw[hour,t['node']]
                    if not (z<=10 if np.isfinite(z) else w.get((hour,t['node']),0)):
                        good=False
            result+=rational(t['penalty'])*(not good) if metric=='penalty' else int(good)
        return result
    exact=[];cert_count=0
    for path in sorted((OUT/'certificates').glob('*.json')):
        cert=read(path);sub=scope_cases(cert['scope']);arms=protocol['comparisons'][cert['comparison']]
        assert len(sub)==cert['seed_cases'] and len(sub)==3*cert['origins']
        pairs=[(c,load(c,arms[0]),load(c,arms[1])) for c in sub];weight=F(1,len(sub))
        C=sum(weight*(a['_exact_constant']-b['_exact_constant']) for c,a,b in pairs)
        assert abs(float(C)-cert['cost_constant_difference'])<1e-9
        results={}
        for metric in ['penalty','success']:
            opt=cert['optimization'][metric]
            for side in ['lower','upper']:
                triples=opt[side+'_witness'];w={(h,n):bit for h,n,bit in triples}
                assert len(w)==len(triples) and all(bit in [0,1] for bit in w.values())
                assert all(not np.isfinite(raw[h,n]) for h,n in w)
                result=sum(weight*(value(a,c,w,metric)-value(b,c,w,metric)) for c,a,b in pairs)
                error=abs(float(result)-opt[side]);assert error<1e-7;maxerror=max(maxerror,error)
                result_key=metric+'_'+side;results[result_key]=result
                saved=opt[side+'_certificate'];assert saved['status']==0 and saved['absolute_gap']<=1e-7 and saved['mip_gap']<=1e-9
        pl,pu=results['penalty_lower'],results['penalty_upper']
        roots=sorted({-C/P for P in [pl,pu] if P and -C/P>0})
        exact.append(dict(comparison=cert['comparison'],scope=cert['scope'],domain='lambda>=0',
                          cost_constant=str(C),penalty_lower=str(pl),penalty_upper=str(pu),
                          success_lower=str(results['success_lower']),success_upper=str(results['success_upper']),
                          cost_lower=f'({C}) + lambda * ({pl})',cost_upper=f'({C}) + lambda * ({pu})',
                          positive_roots=[str(x) for x in roots],positive_roots_decimal=[float(x) for x in roots]))
        cert_count+=1
    write(OUT/'exact_rational_affine.json',exact)
    result=dict(status='PASS',certificate_count=cert_count,unique_saved_arms=checked,
                failed_attempts_full_duration_fee_and_resources_verified=failed_checked,
                source_hashes_match=True,physical_hour_witnesses_shared=True,
                direct_rational_certificate_max_error=maxerror,
                independent_method='No symbolic window_term/expression/direct imported. Original config reconstructs all selected choice fees, duration/spares and booking. Fraction accounting independently evaluates every saved global witness and reports exact rational affine formulas/roots.')
    write(OUT/'independent_verification.json',result)
    sealed=[x for x in OUT.rglob('*') if x.is_file() and x.name!='final_manifest.json']
    sealed += [ROOT/'closure_execution_core.py',ROOT/'closure_execution_run.py',Path(__file__),ROOT.parent/'研究笔记/阶段收尾_2026-11-02/执行费用与服务敏感性.md']
    write(OUT/'final_manifest.json',{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(sealed)})
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()

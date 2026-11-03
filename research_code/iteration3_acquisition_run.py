"""Prespecified query plans on the single frozen primary cost difference."""
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from precision_truth_core import load_truth,SOURCE
from iteration2_bounds_run import arm_expression,direct_cost
from iteration3_acquisition_core import canonical,evaluate,worst_width

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/review_iteration3_v1/acquisition'
DOC=ROOT.parent/'研究笔记/第三轮实证推进_2026-11-02'
OLD=ROOT/'artifacts/review_followup_v1/cross_execution'
SECOND=ROOT/'artifacts/review_iteration2_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')

def reconstruct():
    times,nodes,raw=load_truth();cases=read(OLD/'protocol.json')['cases']
    origins={(c['fold'],c['issue']) for c in cases};assert len(origins)==44 and len(cases)==132
    identity={str(p):sha(p) for p in [SOURCE,OLD/'protocol.json',ROOT/'precision_truth_core.py',ROOT/'iteration2_bounds_run.py',ROOT/'iteration2_bounds_core.py',SECOND/'data_gaps/targeted_missing_hours.csv',SECOND/'shared_bounds/certificates/analog_graph_lookahead72__ALL.json']}
    old_sources={str((ROOT/Path(p)).resolve()):v for p,v in read(SECOND/'shared_bounds/analog_sources.json').items()}
    terms=defaultdict(float);exact=defaultdict(Fraction);constant=0.;weighted=[]
    for c in cases:
        assert sum(d['fold']==c['fold'] and d['issue']==c['issue'] for d in cases)==3
        paths=[SECOND/'analog_execution/evaluations'/f"{c['key']}__fixed_B__AnalogFixed__lookahead72.json",OLD/'evaluations'/f"{c['key']}__fixed_B__B__lookahead72.json"]
        evals=[]
        for sign,p in zip([1,-1],paths):
            identity[str(p)]=sha(p);assert identity[str(p)]==old_sources[str(p)]
            ev=read(p);evals.append(ev);cc,tt=arm_expression(ev,c,raw);constant+=sign*cc/132
            for ks,v in tt.items():
                key=tuple(sorted(set(ks)));terms[key]+=sign*v/132;exact[key]+=sign*Fraction(str(v))/132
        weighted.append((c,*evals))
    constant,terms=canonical(constant,terms);exact={k:v for k,v in exact.items() if v}
    assert set(exact)==set(terms) and all(abs(float(v)-terms[k])<1e-12 for k,v in exact.items())
    terms={k:float(v) for k,v in sorted(exact.items())}
    assert len(terms)==82 and len(set().union(*map(set,terms)))==234
    for p,d in identity.items():
        if p in old_sources:assert d==old_sources[p]
    return times,nodes,raw,cases,constant,terms,exact,weighted,identity

def main():
    OUT.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),comparison='AnalogFixed minus GraphFixed, fixed B resources, lookahead72, all 44 origins x 3 seeds',definition='W(Q)=max D(z_plus)-D(z_minus) subject to z_plus[h]=z_minus[h] for every requested h. All missing threshold states unrestricted binary.',interpretation='Worst remaining range width across all complete possible query returns; not expected information value, not a conditional bound with fixed values.',budgets=[0,12,24,48,96,192,234],strategies=['timestamp_node','previous_request_order','coefficient_sensitivity'],sensitivity='sum(abs(nonzero merged coefficient)) over windows containing each hour; ties timestamp,node',packages='maximal consecutive essential hour runs on same node; enumerate single packages only; no optimal multi-package search',solver='Two complete AND copies plus equality constraints. SciPy HiGHS, 120 seconds per solve, require status 0 and absolute primal-dual gap <=1e-7, mip_gap<=1e-9.',scope='Frozen saved predictions, quality gates, actions and original strict hourly evidence. No forecasts or actions changed; no new weather; no missing value assigned safety.',return_requirement='Guarantee requires all requested states resolved by complete valid records or sufficient threshold evidence; receipt of a file with remaining missing values is insufficient.')
    write(OUT/'protocol.json',protocol)
    times,nodes,raw,cases,constant,terms,exact,weighted,identity=reconstruct()
    previous=pd.read_csv(SECOND/'data_gaps/targeted_missing_hours.csv');assert len(previous)==380
    old_lookup={(int(r.hour_index),int(r.node_index)):r for r in previous.itertuples()}
    hours=sorted(set().union(*map(set,terms)));assert set(hours)<=set(old_lookup)
    timekey=lambda h:(times[h[0]].value,nodes[h[1]])
    sensitivity_exact={h:sum((abs(v) for ks,v in exact.items() if h in ks),Fraction(0)) for h in hours}
    sensitivity={h:float(v) for h,v in sensitivity_exact.items()}
    termrows=[dict(term_id=i,coefficient=v,coefficient_exact=str(exact[ks]),unknown_hour_count=len(ks),unknown_hours=json.dumps(ks)) for i,(ks,v) in enumerate(terms.items())]
    csv('merged_terms.csv',termrows)
    records=[]
    for h,j in sorted(hours,key=timekey):
        r=old_lookup[h,j]
        records.append(dict(hour_index=h,node_index=j,node=nodes[j],hour_start=(times[h]-pd.Timedelta(hours=1)).isoformat(),hour_end=times[h].isoformat(),previous_request_order=int(r.request_order),coefficient_sensitivity=sensitivity[h,j],field='feature_A',missing_or_invalid_5min_count=int(r.missing_or_invalid_5min_count),request='补交完整有效原始记录及时间戳质量码，或足以确定是否超过原始量纲阈值10（单位待确认）的证据'))
    csv('essential_hours.csv',records)
    packs=[]
    for j in sorted({j for h,j in hours}):
        chunks=[]
        for h in sorted(h for h,n in hours if n==j):
            if not chunks or h!=chunks[-1][-1]+1:chunks.append([])
            chunks[-1].append(h)
        for chunk in chunks:
            packs.append(dict(package_id=f'P{len(packs)+1:03}',node=nodes[j],node_index=j,start=(times[chunk[0]]-pd.Timedelta(hours=1)).isoformat(),end=times[chunk[-1]].isoformat(),hours=len(chunk),query=[(h,j) for h in chunk]))
    old=read(SECOND/'shared_bounds/certificates/analog_graph_lookahead72__ALL.json');width=old['shared_width']
    rows=[];cache={}
    def solve(name,q):
        key=tuple(sorted(q))
        if key not in cache:
            r=worst_width(constant,terms,key)
            if r['certificate']=='PASS':
                for side in ['plus','minus']:
                    z={(h,n):v for h,n,v in r['witness_'+side]}
                    ledger=sum((direct_cost(a,c,raw,z)-direct_cost(b,c,raw,z))/132 for c,a,b in weighted)
                    assert abs(ledger-r['direct_symbolic_'+side])<1e-7
                    r['independent_ledger_'+side]=ledger
                assert -1e-7<=r['worst_remaining_width']<=width+1e-7
            cache[key]=r
        r=dict(cache[key]);r['plan_id']=name
        write(OUT/'certificates'/f'{name}.json',r)
        print(name,r['certificate'],r.get('worst_remaining_width'),flush=True)
        return r
    empty=solve('invariant_empty',[]);full=solve('invariant_all234',hours)
    assert empty['certificate']=='PASS' and abs(empty['worst_remaining_width']-width)<1e-7
    assert full['certificate']=='PASS' and abs(full['worst_remaining_width'])<1e-7
    irrelevant=solve('invariant_irrelevant_hour',[(0,0)]);assert irrelevant['worst_remaining_width']==empty['worst_remaining_width']
    for p in packs:
        r=solve('package_'+p['package_id'],p['query']);row={k:v for k,v in p.items() if k!='query'}
        row.update(certificate=r['certificate'],worst_remaining_width=r.get('worst_remaining_width'),guaranteed_reduction=width-r['worst_remaining_width'] if r['certificate']=='PASS' else None)
        row['reduction_per_hour']=row['guaranteed_reduction']/p['hours'] if row['guaranteed_reduction'] is not None else None;rows.append(row)
    frame=pd.DataFrame(rows).sort_values(['guaranteed_reduction','reduction_per_hour','package_id'],ascending=[False,False,True]);frame.insert(0,'rank',range(1,len(frame)+1));frame.to_csv(OUT/'single_package_ranking.csv',index=False,encoding='utf-8-sig')
    strategies={'timestamp_node':sorted(hours,key=timekey),'previous_request_order':sorted(hours,key=lambda h:old_lookup[h].request_order),'coefficient_sensitivity':sorted(hours,key=lambda h:(-sensitivity_exact[h],*timekey(h)))}
    curves=[];plans=[]
    for strategy,order in strategies.items():
        previous_w=width
        for budget in protocol['budgets']:
            q=order[:budget];name=f'{strategy}_b{budget:03}';r=solve(name,q)
            w=r.get('worst_remaining_width')
            if w is not None:assert w<=previous_w+1e-7;previous_w=w
            curves.append(dict(strategy=strategy,budget_hours=budget,certificate=r['certificate'],worst_remaining_width=w,guaranteed_reduction=width-w if w is not None else None,remaining_fraction=w/width if w is not None else None,absolute_gap=r.get('absolute_gap'),seconds=r['seconds']))
            for rank,h in enumerate(q,1):plans.append(dict(plan_id=name,strategy=strategy,budget_hours=budget,query_order=rank,hour_index=h[0],node_index=h[1],node=nodes[h[1]],hour_start=(times[h[0]]-pd.Timedelta(hours=1)).isoformat(),hour_end=times[h[0]].isoformat(),coefficient_sensitivity=sensitivity[h],previous_request_order=int(old_lookup[h].request_order)))
    csv('budget_curves.csv',curves);csv('query_plans.csv',plans);csv('plans_48hours.csv',[r for r in plans if r['budget_hours']==48])
    # Minimal containing monomial yields an explicit two-completion dependency witness.
    witnesses=[]
    for h in hours:
        ks=min((k for k in terms if h in k),key=lambda k:(len(k),k));minus={g:int(g in ks and g!=h) for g in hours};plus=dict(minus);plus[h]=1
        delta=evaluate(constant,terms,plus)-evaluate(constant,terms,minus)
        assert abs(delta-terms[ks])<1e-7 and abs(delta)>1e-12
        witnesses.append(dict(hour=list(h),minimal_monomial=list(ks),exact_change=str(exact[ks]),change=delta,ones_when_h_zero=[list(g) for g in ks if g!=h],all_other_hours_zero=True))
    write(OUT/'essential_dependency_witnesses.json',dict(proof='For each h choose a minimum-cardinality nonzero monomial T containing h. Set other variables to one exactly on T minus h and zero elsewhere. The h-derivative retains only coefficient(T), since every proper subset containing h has smaller cardinality and zero coefficient. Thus D changes when only h changes. Every essential variable is necessary for any fixed nonadaptive query set that determines D for ALL completions. This proves 234 is minimal for exact D, not for its sign or a winner.',witnesses=witnesses))
    for p,d in identity.items():assert sha(p)==d
    for p in [ROOT/'iteration3_acquisition_core.py',ROOT/'iteration3_acquisition_run.py',ROOT/'iteration3_acquisition_test.py']:identity[str(p)]=sha(p)
    write(OUT/'source_sha256.json',identity)
    write(OUT/'test_red_green.json',dict(red='Before implementation, iteration3_acquisition_test.py failed ModuleNotFoundError: iteration3_acquisition_core (exit 1).',green='2 tests passed, including 24 random five-variable problems at every query prefix (144 MILPs), exhaustive enumeration agreement, subset monotonicity, full resolution and irrelevant-hour invariance.'))
    best=min((r for r in curves if r['budget_hours']==48 and r['certificate']=='PASS'),key=lambda r:r['worst_remaining_width'])
    summary=dict(status='PASS' if all(r['certificate']=='PASS' for r in curves+rows) else 'INCOMPLETE',comparison=protocol['comparison'],origins=44,seeds=3,essential_hours=234,nonzero_merged_terms=82,previous_request_hours=380,is_subset_of_previous380=True,irrelevant_previous_hours=146,indivisible_packages=len(packs),old_lower=old['shared_lower'],old_upper=old['shared_upper'],old_width=width,empty_width=empty['worst_remaining_width'],full_width=full['worst_remaining_width'],best_prespecified_48hour_strategy=best,unique_query_solves=len(cache),all_source_sha_rechecked=True,query_return_requirement=protocol['return_requirement'],scope=protocol['scope'])
    write(OUT/'summary.json',summary)
    report=f'''# 固定行动主比较的补录决策实验

对象仅为 AnalogFixed−GraphFixed / lookahead72 / all44，三随机种子在每起点平均，再等权起点平均。复用第二轮 arm_expression 和原 evaluations；源码、严格小时原表与264份行动评估 SHA-256 见 source_sha256.json。重建合并后82个非零AND项、234个essential机组小时；用 Fraction 精确合并存储十进制系数，与浮点模型逐项核对。它们全部属于旧380机组小时请求，其中146小时对本主比较零作用；此处未声称覆盖其他六项比较。

原费用差界 [{old['shared_lower']:.12f}, {old['shared_upper']:.12f}]，宽 {width:.12f}。对预先固定补录集合Q，W(Q)=max D(z+)−D(z−)，两补全在Q上相同；复制两份AND模型并添加Q等式约束，一次MILP求最坏剩余宽度。它覆盖所有可能完整补录结果，不读取未观测真值、不假设概率，不是预期信息价值，也不是固定已知取值后的条件界。保证缩宽为原宽−W(Q)。

全部{len(packs)}个同机组连续时段包分别单独求解；排名和每小时效率见 single_package_ranking.csv。包不能拆分，该排名不证明任何多包预算下的全局最优组合。三条固定小时排序（时间、旧request_order、系数绝对敏感度）均保存0/12/24/48/96/192/234小时曲线和具体小时计划。预定48小时中保证缩宽最佳者为 {best['strategy']}：最坏剩余宽 {best['worst_remaining_width']:.12f}、保证缩宽 {best['guaranteed_reduction']:.12f}。这是有限三条计划中的数据检索工程比较，非统计发现或全局最优查询策略。

W(空集)={empty['worst_remaining_width']:.12f}，W(全部234小时)={full['worst_remaining_width']:.12f}；已核对嵌套计划单调性、非负与原宽上界、无关小时零作用。全部证书保存原始MILP状态、原/对偶界、间隙和两补全；每个见证再用原task ledger直接重算双方成本差。小例先RED（模块缺失），实现后144个随机查询小例与穷举一致。

234是对所有二元补全都要完全确定D值时，固定非自适应阈值查询的最小充分集。每个变量选择包含它的最小非零单项式，其他变量仅在该单项式内置一，则翻转该变量的费用变化等于该项非零系数；234份显式见证和精确系数见 essential_dependency_witnesses.json。因此不查询任何一个essential变量，都存在两份补全无法区分而D不同。此结论不涉及确定D符号所需的最小查询，也不保证补录后某方胜出。

保证以每个请求小时得到完整有效记录、或得到足以确定是否超过原始量纲阈值10（单位待确认）的证据为前提；只收到文件而仍缺有效值，不满足保证条件。此界只条件于冻结预测、质量门控与单次行动；未重新预测、未更改行动、未补造天气或认定缺测安全。所有求解限时120秒；未取得最优证书时只报告UNAVAILABLE并保留可行值/对偶界，不冒称精确结果。
'''
    (DOC/'补录决策实验.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

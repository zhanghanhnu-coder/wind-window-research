"""All seven frozen-action comparisons: costs, service, affine lambda ranges."""
from pathlib import Path
from collections import defaultdict
import hashlib, json
import numpy as np
import pandas as pd
from precision_truth_core import load_truth, SOURCE
from iteration2_bounds_core import solve_bounds
from iteration2_bounds_run import COMPARISONS
from closure_execution_core import expression, direct, affine_direction, exhaustive_tests

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/review_closure_v1/execution_sensitivity'
OLD=ROOT/'artifacts/review_followup_v1/cross_execution'
ANALOG=ROOT/'artifacts/review_iteration2_v1/analog_execution'
BASE=ROOT/'artifacts/review_iteration2_v1/shared_bounds'
REPORT=ROOT.parent/'研究笔记/阶段收尾_2026-11-02/执行费用与服务敏感性.md'
LAMBDAS=[0.,.5,1.,2.,4.]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(p,rows):pd.DataFrame(rows).to_csv(OUT/p,index=False,encoding='utf-8-sig')

def main():
    protocol=dict(status='FROZEN_BEFORE_REPRICING', lambdas=LAMBDAS,
      comparisons={**COMPARISONS,**{f'analog_graph_{x}':(('fixed_B','AnalogFixed',x),('fixed_B','B',x)) for x in ['lookahead72','myopic']}},
      primary='analog_graph_lookahead72', secondary='analog_graph_myopic',
      estimand='candidate minus baseline; three paired seed averages inside each origin, equal origin weights',
      intervention='Fixed saved actions ex post repricing only: booking and entire operating cost fixed; lambda multiplies original unmet-task penalty, including failed, unattempted and unresolved failures. No reoptimization, no lambda selection.',
      operating_definition='Full duration boat+crew hourly cost plus late_fraction * ORIGINAL penalty * start / 72. Even late surcharge remains fixed.',
      weather='Original raw64 <=10 AND success; one missing absolute (hour,node) bit globally shared across methods, seeds and origins within every estimand. Three seeds are not new weather.',
      scope='ALL44, four folds, all 44 individual origins, complete15 and its nonempty fold subsets. Complete15 auxiliary only. Reused 2024 development data; no real money, confidence interval or new external generalization.',
      formulas='For lambda>=0 only: Delta cost lower/upper = C + lambda * min/max Delta unmet-original-penalty. Service: unit-weight successful task difference, solved separately on same shared weather domain.',
      resource_definition='Single attempt per task; failed attempts consume full scheduled duration and spares; no failure cancellation or retry.')
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==json.loads(json.dumps(protocol))
    else:write(path,protocol)
    tests=exhaustive_tests();write(OUT/'unit_tests.json',tests)
    sources={}
    def track(p):p=Path(p);sources[str(p)]=sha(p);return p
    for p in [Path(__file__),ROOT/'closure_execution_core.py',ROOT/'closure_execution_verify.py', ROOT/'iteration2_bounds_core.py', ROOT/'iteration2_bounds_run.py',ROOT/'cross_execution_review_core.py',ROOT/'two_stage_core.py',ROOT/'lookahead_execution_core.py',ROOT/'causal_execution_core.py',ROOT/'precision_truth_core.py',SOURCE,OLD/'protocol.json',ANALOG/'protocol.json']:
        track(p)
    cases=read(OLD/'protocol.json')['cases'];assert len(cases)==132
    times,nodes,raw=load_truth();assert raw.dtype==np.float64
    original=pd.concat([pd.read_csv(track(BASE/f'{kind}_bounds.csv')) for kind in ['cross','analog']]).set_index(['comparison','scope'])
    cache={}
    def load(c,arm):
        folder=ANALOG if arm[1]=='AnalogFixed' else OLD
        p=(folder/'evaluations')/f"{c['key']}__{'__'.join(arm)}.json"
        if p not in cache:
            cache[p]=read(track(p))
            track(folder/'trajectories'/p.name)
        return cache[p]
    # The original complete condition is independent of method/seed outcome.
    complete={(c['fold'],c['issue']) for c in cases if load(c,('fixed_B','B','myopic'))['ideal']['complete']}
    assert len(complete)==15
    scopes=[('ALL',cases)]+[(f,[c for c in cases if c['fold']==f]) for f in sorted({c['fold'] for c in cases})]
    for fold,issue in sorted({(c['fold'],c['issue']) for c in cases}):
        scopes.append((f'{fold}_issue{issue}',[c for c in cases if (c['fold'],c['issue'])==(fold,issue)]))
    scopes.append(('COMPLETE15',[c for c in cases if (c['fold'],c['issue']) in complete]))
    for fold in sorted({c['fold'] for c in cases}):
        sub=[c for c in cases if c['fold']==fold and (c['fold'],c['issue']) in complete]
        if sub:scopes.append(('COMPLETE15_'+fold,sub))
    rows=[];lambda_rows=[];armrows=[];allregions=[];maxerror=0.;maxgap=0.;old_checks=0;partial_hits=0
    evidence=ROOT/'artifacts/review_iteration2_v1/data_gaps/partial_hour_evidence.npz'
    bad=np.load(track(evidence))['known_bad']
    for name,arms in protocol['comparisons'].items():
        for scope,sub in scopes:
            origin_keys=sorted({(c['fold'],c['issue']) for c in sub})
            for origin in origin_keys:
                seeds=[c['seed'] for c in sub if (c['fold'],c['issue'])==origin]
                assert len(seeds)==len(set(seeds))==3
            weight=1/(3*len(origin_keys));weighted=[(c,load(c,arms[0]),load(c,arms[1])) for c in sub]
            constants=[]
            for arm_index,arm_name in enumerate(['candidate','baseline']):
                stats=defaultdict(float)
                for c,a,b in weighted:
                    e=[a,b][arm_index];m=e['metrics']
                    for key in ['booking_cost','operating_cost','successful','failed','unknown','unattempted']:
                        stats[key]+=weight*m[key]
                    stats['attempted']+=weight*sum(t['choice'] is not None for t in e['tasks'])
                    stats['operation_task_hours']+=weight*sum(t['duration'] for t in e['tasks'] if t['choice'] is not None)
                    stats['unmet_penalty_lower']+=weight*sum(t['penalty_lower'] for t in e['tasks'])
                    stats['unmet_penalty_upper']+=weight*sum(t['penalty_upper'] for t in e['tasks'])
                    d=direct(e,c,raw,{})
                    assert abs(d['operation']-m['operating_cost'])<1e-7
                    assert abs(sum(t['duration'] for t in e['tasks'] if t['choice'] is not None)-d['hours'])<1e-7
                    if scope=='ALL':
                        for t in e['tasks']:
                            if t['status']=='unknown':partial_hits+=int(bad[c['issue']+t['start']+1:c['issue']+t['end']+1,t['node']].any())
                constants.append(stats['booking_cost']+stats['operating_cost'])
                armrows.append(dict(comparison=name,scope=scope,arm=arm_name,origins=len(origin_keys),**stats))
            constant=constants[0]-constants[1]
            record=dict(comparison=name,scope=scope,origins=len(origin_keys),seed_cases=len(sub),cost_constant_difference=constant)
            certificate={}
            for metric in ['penalty','success']:
                const=0.;terms=defaultdict(float)
                for c,a,b in weighted:
                    for sign,e in [(1,a),(-1,b)]:
                        intercept,tt=expression(e,c,raw,metric);const+=weight*sign*intercept
                        for ks,v in tt.items():terms[ks]+=weight*sign*v
                opt=solve_bounds(const,terms)
                for side in ['lower','upper']:
                    assignment={(h,node):g for h,node,g in opt[side+'_witness']}
                    value=sum(weight*(direct(a,c,raw,assignment)[metric]-direct(b,c,raw,assignment)[metric]) for c,a,b in weighted)
                    error=abs(value-opt[side]);maxerror=max(maxerror,error);assert error<1e-7
                    opt[side+'_certificate']['direct_task_ledger']=value
                    maxgap=max(maxgap,opt[side+'_certificate']['absolute_gap'])
                record[metric+'_lower']=opt['lower'];record[metric+'_upper']=opt['upper'];certificate[metric]=opt
            for lam in LAMBDAS:
                lo=constant+lam*record['penalty_lower'];hi=constant+lam*record['penalty_upper']
                for side,endpoint in [('lower',lo),('upper',hi)]:
                    witness={(h,node):g for h,node,g in certificate['penalty'][side+'_witness']}
                    direct_value=0.
                    for c,a,b in weighted:
                        da,db=direct(a,c,raw,witness),direct(b,c,raw,witness)
                        direct_value+=weight*(da['constant']-db['constant']+lam*(da['penalty']-db['penalty']))
                    error=abs(direct_value-endpoint);maxerror=max(maxerror,error);assert error<1e-7
                if lam==0:assert lo==hi==constant
                if lam==1 and (name,scope) in original.index:
                    old=original.loc[name,scope];assert abs(lo-old.shared_lower)<1e-7 and abs(hi-old.shared_upper)<1e-7;old_checks+=1
                lambda_rows.append(dict(comparison=name,scope=scope,origins=len(origin_keys),lambda_value=lam,cost_lower=lo,cost_upper=hi))
            regions=affine_direction(constant,record['penalty_lower'],record['penalty_upper'])
            allregions.append(dict(comparison=name,scope=scope,**regions))
            write(OUT/'certificates'/f'{name}__{scope}.json',dict(**record,optimization=certificate,affine_regions=regions,cost_lambda_endpoints_direct_recomputed=True))
            rows.append(record)
        print('DONE',name,flush=True)
    assert old_checks==343 and partial_hits==0
    csv('bounds.csv',rows);csv('lambda_costs.csv',lambda_rows);csv('arm_service.csv',armrows)
    csv('summary.csv',[r for r in rows if '_issue' not in r['scope']])
    write(OUT/'affine_regions.json',allregions)
    verify=dict(status='PASS',comparisons=7,origins=44,seeds_per_origin=3,complete_origins=15,
                scope_count=len(scopes),bound_rows=len(rows),lambda_rows=len(lambda_rows),old_lambda1_pairs_reproduced=old_checks,
                zero_lambda_fixed_cost=True,max_direct_endpoint_error=maxerror,max_primal_dual_gap=maxgap,
                partial_evidence_unknown_attempt_intersections=partial_hits,
                missing_hour_key='absolute_hour,node only; no arm, origin or seed component',
                unit_tests=tests,raw_dtype=str(raw.dtype))
    write(OUT/'verification.json',verify)
    make_report(rows,lambda_rows,armrows,allregions,verify)
    for p,digest in sources.items():assert sha(p)==digest
    write(OUT/'sources_sha256.json',sources)
    print(json.dumps(verify,ensure_ascii=False),flush=True)


def make_report(rows,lambda_rows,armrows,regions,verify):
    main=next(r for r in rows if r['comparison']=='analog_graph_lookahead72' and r['scope']=='ALL')
    text=['# 执行费用与服务敏感性：阶段封存', '',
      '本轮固定全部保存预测、质量门控、预订和执行动作，仅事后重计价。主比较为 AnalogFixed − GraphFixed，固定同一 B 资源、lookahead72；myopic 和旧 cross 五比较全部保留。44 起报×3 配对种子，先种子平均再起报均权；三种子不是三套新天气。完整15起报仅为辅助。', '',
      'λ 固定为 0、0.5、1、2、4，倍乘原未完成罚损：失败、未尝试各付原罚额，未知按共享天气决定。booking 与整个 operating 固定。原 operating 包括全时长船/crew费及 late_fraction×原penalty×start/72，因此迟开工项也固定，不随λ倍乘。失败照样占满 duration 和备件；动作不会因结果撤销，不重试。费用为假设成本单位，不能解释为真实金钱或新费用下的重优化结果。', '',
      '每个缺测绝对(小时,测点)只有一个二元变量，全部方法、种子、起报共用；raw64 任一已知 >10 先判失败，全部 <=10 判成功，剩余未知逐小时 AND。端点由全范围共同天气 MILP 求得；服务任务数另用每任务权重1求界，不能从罚额差除常数推回任务数。', '',
      '对于 λ≥0，精确界为 [C+λP下, C+λP上]。λ=0退化为已知运营+预订差；λ=1复现旧343个比较/范围端点。此公式没有延伸到负λ。差值为候选减基线，严格小于0才保证候选费用更低。', '',
      '主比较ALL精确公式为 [2045/3168 − (530/11)λ, 2045/3168 + (2060/33)λ]：0≤λ<409/30528时保证候选更贵；λ=409/30528时下端恰为0；λ>409/30528时费用方向不能识别。不存在λ≥0区间能保证ALL候选更便宜。完整15辅助界为 −5527/5400 − (16/3)λ，在全部λ≥0均较低，不能据此覆盖ALL不确定结论。M11费用转折精确为25247/97200与25247/31200，而成功任务数差为[−5/33,−1/33]，体现低费用不能替代服务指标。', '',
      '| 范围 | 起报数 | C | 原罚损差共享界 | 成功任务数差共享界 |', '|---|---:|---:|---|---|']
    for r in rows:
        if r['comparison']=='analog_graph_lookahead72' and '_issue' not in r['scope']:
            text.append(f"| {r['scope']} | {r['origins']} | {r['cost_constant_difference']:.6f} | [{r['penalty_lower']:.6f}, {r['penalty_upper']:.6f}] | [{r['success_lower']:.6f}, {r['success_upper']:.6f}] |")
    text+=['','主比较 ALL 的预设费用权重结果：','','| λ | 费用差共享界 |','|---:|---|']
    for r in lambda_rows:
        if r['comparison']=='analog_graph_lookahead72' and r['scope']=='ALL':text.append(f"| {r['lambda_value']:g} | [{r['cost_lower']:.6f}, {r['cost_upper']:.6f}] |")
    text+=['','主比较费用方向随λ的完整正半轴分段（精确根由机器表中的 −C/P 表达式确定，下列仅显示小数）：','']
    for r in regions:
        if r['comparison']=='analog_graph_lookahead72' and '_issue' not in r['scope']:
            parts=[f"({x['lambda_left']:.9g}, {'∞' if x['lambda_right'] is None else format(x['lambda_right'],'.9g')}): {x['direction']}" for x in r['intervals']]
            points=[f"λ={x['lambda_value']:.9g}: {x['direction']}" for x in r['points']]
            text.append(f"- {r['scope']}："+'；'.join(parts+points))
    text+=['','主比较 ALL 双方平均服务与投入（任务小时为各已尝试任务duration之和，非72小时日历跨度；全部动作固定所以不随λ变化）：','','| 方 | 尝试 | 未尝试 | 已知成功 | 已知失败 | 未知 | 操作任务小时 | operating | booking |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in armrows:
        if r['comparison']=='analog_graph_lookahead72' and r['scope']=='ALL':
            text.append('| '+r['arm']+' | '+' | '.join(f'{r[k]:.6f}' for k in ['attempted','unattempted','successful','failed','unknown','operation_task_hours','operating_cost','booking_cost'])+' |')
    text+=['','上述服务界与投入共同防止“少干活自然低费”解释被遗漏；没有据λ挑选动作或重新训练。全部7比较、ALL、4fold、44起报与完整15辅助的双方服务账在 arm_service.csv，罚损/成功界在 bounds.csv，5组费用在 lambda_costs.csv，正半轴转折点与开区间在 affine_regions.json，每范围端点证书在 certificates/。', '',
      f"独立小例穷举35组×32种天气及配对任务账本4种完成通过；直接任务账本重算全部费用/服务端点，最大误差 {verify['max_direct_endpoint_error']:.3g}，最大原始/对偶差 {verify['max_primal_dual_gap']:.3g}。λ=1复现343对旧界；λ=0全部退化固定差。partial已观测超阈值证据与旧unknown尝试交集为0，因此不会收紧这些固定动作界。", '',
      '限制：仅对保存动作和无限制共享缺测阈值状态的条件费用/服务界；端点的罚损、服务目标可对应不同完成天气，不是联合矩形可达性声明。不是物理天气模型、置信区间、真实费用估计、重新优化政策或独立外部检验。当前44起报均来自反复研究的2024开发数据，完整15可能偏向易观测天气，不能替代ALL。', '',
      '复现：在 风电运维Demo 目录先运行 `python closure_execution_run.py`，再运行 `python closure_execution_verify.py`。来源摘要在 sources_sha256.json；核验汇总 verification.json 与 independent_verification.json。独立核验从原配置、已保存动作重建全时长费用与资源消耗，不导入符号表达式或窗口分类函数；exact_rational_affine.json 给出所有范围精确有理数C/P、成功任务界、公式及转折根。']
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__':main()

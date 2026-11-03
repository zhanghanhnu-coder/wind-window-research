"""Describe the fixed benchmark without selecting a winning family."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from lead_calibration_core import apply_groups

BASE=Path(__file__).resolve().parent;OUT=BASE/'artifacts/review_followup_v1/baselines'
REPORT=BASE.parent/'研究笔记/复审后实证推进_2026-11-02/强基线实验.md'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def run():
    assert read(OUT/'verification.json')['status']=='PASS'
    inventory=read(OUT/'inventory.json');seen=set();changes=[]
    for r in inventory:
        key=(r['fold'],r['family'],r['producer_seed'],r['regime'])
        if key in seen:continue
        seen.add(key);params=np.load(r['parameters']);a=params['legacy_frozen32'];b=params['refit64'];q=np.load(r['source'])[r['source_key']]
        scale=float(np.load(BASE/'artifacts/seasonal_probability_v1'/r['fold']/'data.npz')['scale'])
        old=apply_groups(q,a,scale,True);new=apply_groups(q,b,scale,True)
        seeds=[x['seed'] for x in inventory if (x['fold'],x['family'],x['producer_seed'],x['regime'],x['method'])==(*key,'refit64')]
        delta=0.;flips=0;records=0
        for seed in seeds:
            path=OUT/r['fold'];p0=np.load(path/f"probabilities_{r['family']}_{seed}_legacy_frozen32_{r['regime']}.npz")['single'];p1=np.load(path/f"probabilities_{r['family']}_{seed}_refit64_{r['regime']}.npz")['single']
            delta=max(delta,float(abs(p0-p1).max()));flips+=int(((p0>=.8-1e-12)!=(p1>=.8-1e-12)).sum());records+=p0.size
        changes.append(dict(fold=r['fold'],family=r['family'],producer_seed=r['producer_seed'],regime=r['regime'],maximum_delta_parameter_change=float(abs(a[:,0]-b[:,0]).max()),maximum_gamma_parameter_change=float(abs(a[:,1]-b[:,1]).max()),maximum_atom_change=float(abs(old-new).max()),mean_atom_change=float(abs(old-new).mean()),maximum_single_probability_change=delta,single_seed_record_decision_flips=flips,single_seed_records=records))
    diff=pd.DataFrame(changes);diff.to_csv(OUT/'calibration_precision_effects.csv',index=False,encoding='utf-8-sig')
    fitfiles=list(OUT.glob('*/calibration_*.json'));fitgroups=sum(len(read(p)) for p in fitfiles);fallbacks=sum(sum(not x['success'] for x in read(p)) for p in fitfiles)
    pa=pd.read_csv(OUT/'paired_brier.csv');ab=pd.read_csv(OUT/'absolute_brier.csv');cr=pd.read_csv(OUT/'crps.csv');mask=pd.read_csv(OUT/'mask_support.csv');verification=read(OUT/'verification.json')
    primary=pa[pa.primary].iloc[0]
    analogy=pa[(pa.scope=='ALL')&(pa.method=='refit64')&(pa.regime=='natural')&(pa.kind=='single')&(pa.candidate=='context1_graph')&(pa.baseline=='analog112')].iloc[0]
    def interval(row):return f"[{row.lower:+.6f}, {row.upper:+.6f}]；外包[{row.outer_low:+.6f}, {row.outer_high:+.6f}]"
    lines=['# 复审后强基线实证（2026-11-02）','',f"唯一预设主比较是相同24小时历史下，相关图context1_graph相对全场均值context1_mean，在自然缺测输入、原始float64评价真值与refit64校准下的ALL单机6小时窗口Brier差。结果为{interval(primary)}。差为候选减对照，负数有利。",'', '全部结果是反复开发的2024年数据证据，不是新的独立最终测试；没有重训或选择新网络，没有根据本轮结果改图、选块长或选模型。原冻结实验不改写。', '', '## 固定设计与资产复用', '', '四折M05/M07/M09/M11合计880起报、17台机组、四偏移0/24/48/66、阈值原刻度10。每个模型输出相同19等权原子，并统一使用逐折/种子的固定partial05共57条秩模板。所有模型共用相同起报和缺失真值；主终点每类方法59,840条单机6小时记录，三机共同10,560条另作辅助。记录数不是独立天气样本数。', '', '资产包括context1_own（共享逐机时序权重、不混合其他机组）、context1_mean（全场均匀均值聚合）、context1_graph（训练期相关图）；另现成context3_graph和context7_graph/own/mean作上下文敏感性；climate28为过去28天经验分布，analog112为过去112天固定历史相似度候选。9家族按固定顺序列示，不作为任意挑选冠军的排行榜。神经家族各3种子；统计家族每折仅1个生产者，其相同边际配3个固定秩模板，不能称3份独立统计训练。', '', '三种24小时神经对照使用相同训练、验证、40轮预算、45,984参数和matched初始参数，校准月相同，仅聚合矩阵不同；7日对照也匹配同一参数预算。沿用冻结权重，新增信息来自评价/校准标签精度，未引入新的气象数据。全体共用图模型派生的秩模板以固定依赖；这不是各模型最优联合关系之间的比较。', '', '## 校准版本与精度效应', '', 'legacy_frozen32保留旧float32真值拟合的冻结分提前量平滑affine参数，但评价使用原始float64；refit64从同一校准月原始float64真值重拟合相同4组affine，目标与正则项及求解预算不变。两者分表保留。训练权重、旧统计生产者、固定秩模板仍来自旧链，因此refit64只表示校准和新评价用64位真值，不称全模型64位重训。所有cal末目标严格早于首测试起报。', '', f"全量参数变化：最大标准化平移变化{diff.maximum_delta_parameter_change.max():.9g}、最大尺度系数变化{diff.maximum_gamma_parameter_change.max():.9g}；全部自然/遮挡预测原子最大变化{diff.maximum_atom_change.max():.9g}（原字段刻度），最大单机窗口概率变化{diff.maximum_single_probability_change.max():.9g}；三种子逐记录0.8判定合计变化{int(diff.single_seed_record_decision_flips.sum())}次（该数重复计同一物理事件，不能作新样本量）。详见calibration_precision_effects.csv，含逐fold/family/producer/regime。微小真值改动也会改变非光滑CRPS优化路径；不将数值差异解释为新增气象信息。",'', '## 自然输入主终点', '', '| 家族 | legacy校准 Brier缺测界 | refit64校准 Brier缺测界 | refit64已知标签CRPS |', '|---|---|---|---|']
    order=['context1_graph','context1_mean','context1_own','climate28','analog112','context3_graph','context7_graph','context7_mean','context7_own']
    for fam in order:
        row=ab[(ab.scope=='ALL')&(ab.regime=='natural')&(ab.kind=='single')&(ab.family==fam)];a=row[row.method=='legacy_frozen32'].iloc[0];b=row[row.method=='refit64'].iloc[0];c=cr[(cr.scope=='ALL')&(cr.regime=='natural')&(cr.family==fam)&(cr.method=='refit64')].iloc[0]
        lines.append(f'| {fam} | [{a.lower:.6f}, {a.upper:.6f}] | [{b.lower:.6f}, {b.upper:.6f}] | {c.crps:.6f} |')
    lines+=['',f'共{len(fitfiles)}个生产者、{fitgroups}个提前量组完成float64重新拟合，求解失败/恒等回退{fallbacks}组。参数最大变化需与这项求解信息一起解释。','', '最大0.175439窗口概率变化出现在M07 climate28：原子仅微调约0.000177，但离散情景的阈值指示函数在10处不连续，多个小时的跨阈值会放大为10/57的窗口成功概率跳变；不是增加了气象信息。ALL校准精度变化仅引起3条逐种子0.8判定翻转，不能以最大单例突跳包装成整体改进。','',f'强基线反证必须同时报告：24h图模型相对analog112的Brier差为{interval(analogy)}，本批同口径开发数据上analog112更好。因此“相关图比全场均值有帮助”不能扩展成“图模型优于简单概率基线”。图相对own的自然输入外包跨0，3/7日上下文也未带来更好的结果。','', '本表context1_graph的legacy_frozen32自然输入预测，与cross_execution的fixed_B（inventory method=frozen）逐元素相同；refit64沿用同一gated_graph_atoms权重，仅重拟合校准月64位真值。本表未纳入seasonal_rolling_v1的rolling28或residual_warm_v1的WT。precision/state表的B、W28分别指后两条链，与cross表的B=frozen、W28=rolling28不能按缩写直接连接。','', '完整配对结果（含全部折、两校准版本、单机/共同）在paired_brier.csv。以下仅汇总预设自然输入refit64的ALL单机比较，前一行为唯一主比较，其余均是辅助或敏感性：','','| 对照 | 缺测差界；目标时刻分块外包 |','|---|---|']
    for r in pa[(pa.scope=='ALL')&(pa.regime=='natural')&(pa.kind=='single')&(pa.method=='refit64')].itertuples():lines.append(f'| {r.candidate} − {r.baseline} | {interval(r)} |')
    complete=read(OUT/'complete_history_sensitivity.json');assert complete['status']=='PASS'
    lines+=['','## 完整24小时输入历史子群','','自然输入包含实际缺测，不能与完整历史混称。按复审规程另固定筛选起报前24小时×全部17节点输入mask均为1的392起报，不依据未来标签或模型损失筛选。只用已有自然/refit64概率，比较单机context1_graph−context1_mean，不重推理或调参。这是主效应的完整输入子群敏感性，不替换880起报主总体，也不是新的模型选择终点。','','| 范围 | 完整历史起报 | 预测窗口记录 | 缺测差界；目标分块外包 |','|---|---:|---:|---|']
    for r in complete['comparisons']:
        lines.append(f"| {r['scope']} | {r['origins']} | {r['n']} | [{r['lower']:+.6f}, {r['upper']:+.6f}]；外包[{r['outer_low']:+.6f}, {r['outer_high']:+.6f}] |")
    lines+=['','完整历史子群的ALL优势主要来自M09；M05/M07/M11的外包跨0，M11缺测差界甚至略为正，因此不能宣称四折一致改善。按既定目标时间抽样保留各折9个日历块，子群有记录块为8/7/7/9；M05/M07末块为0记录，未删除或重锚。全部抽样2000次有效，最少非空块7。子群共享未知真值的系数法与直接y=0/1物理窗口损失穷举式汇总、全部重采样分子独立吻合；详见complete_history_paired.csv和complete_history_sensitivity.json。']
    lines+=['','## 一次冻结的经验缺测模板压力','','自然输入保留实际已有缺测，不能称全场传感器齐全。为每个测试起报从本fold训练期起报等概率、有放回抽一个168小时×17节点有效性模板（seed=20261104+fold顺序）。模板与当前有效性取AND，只把原可见输入改成缺失，不补造缺失值；同一起报所有神经模型共享同一模板，24h模型取同一模板最后24h。输入特征均置训练标准化0且mask置0，真实评价标签不遮挡。未按未来天气或损失筛模板，未尝试其他强度或挑最好seed。','','遮挡预测仍用自然输入拟合的legacy/refit64参数，不在遮挡数据重新拟合。climate/analog不做遮挡推理，因此压力表只比较同压力神经模型，不和未遮挡简单基线混排。训练期经验模板保留联合缺测形态，但这种叠加压力仍是受控离线遮挡，不能称现场真实故障试验。旧22轮单机6/12/24h人为遮挡不混入本轮队列。','','| fold | 自然168h有效率 | 压力168h有效率 | 自然24h有效率 | 压力24h有效率 |','|---|---:|---:|---:|---:|']
    for r in mask.itertuples():lines.append(f'| {r.fold} | {r.natural_valid_fraction:.4%} | {r.masked_valid_fraction:.4%} | {r.natural_last24_valid_fraction:.4%} | {r.masked_last24_valid_fraction:.4%} |')
    lines+=['','| 同压力对照，ALL/refit64单机 | 缺测差界；目标时刻分块外包 |','|---|---|']
    for r in pa[(pa.scope=='ALL')&(pa.regime=='empirical_mask')&(pa.kind=='single')&(pa.method=='refit64')].itertuples():lines.append(f'| {r.candidate} − {r.baseline} | {interval(r)} |')
    lines+=['','## 统计与独立核验','','每折锚至第一目标窗口日期当地零时，按目标起始时刻7日块抽2000次（seed20261103）。同一物理窗口全部重复预测共用一次块权重与未知真值；分母仍为全部预测记录，种子先算损失再平均。全部非空末端块保留；每fold至少4非空块、1900有效抽样才输出外包。相邻目标块仍可能相关，重采样不消除多轮开发选择偏差。缺测界针对当前有限总体，外包范围不是现场概率或选择校正后的95%保证。', '',f"独立核验已通过：{verification['probability_streams']}条概率流全部从有序原子/秩索引独立重建；{verification['masked_raw_streams_reinferred']}条遮挡原子流的首/中/末起报走原multiscale输入路径重新推理；{verification['paired_rows']}行配对界及重采样由y=0/1两种损失直接重算；{verification['direct_CRPS_origins']}起报用两两原子CRPS恒等式核验；float64标签另独立从raw小时分类。{verification['source_files_unchanged']}个消费源保持不变，其中{verification['sources_matching_previous_frozen_manifest']}个与旧冻结manifest匹配。", '', '运行命令：`python research_review_baselines.py`；独立核验：`python verify_review_baselines.py`；报告：`python summarize_review_baselines.py`。模块测试8项通过（含先RED记录）：共享未知真值穷举、种子平方顺序、绝对缺测界、训练模板无未来、只能减少可见性、输入不改写、缺测不混入粗尺度均值、原子/情景一致性。', '', '全部机器产物仅位于artifacts/review_followup_v1/baselines；protocol.json固定唯一主比较与辅助边界，source_identity.json保存来源，final_manifest.json冻结本次产物。没有改变旧结果、旧模型或程序默认。']
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest=read(OUT/'final_manifest.json')
    for p in [*OUT.rglob('*'),REPORT,Path(__file__),BASE/'review_complete_history_sensitivity.py']:
        if p.is_file() and p.name!='final_manifest.json':manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'final_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(REPORT),flush=True)
if __name__=='__main__':run()

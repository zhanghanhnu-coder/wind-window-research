"""Read-only checks of frozen H2 artifacts and the stage report generator."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from closure_joint import OUT,BASE,read,js,sha,SEEDS,FAMILIES

def main():
    complete=read(OUT/'complete.json');assert complete['status']=='COMPLETE'
    for p,h in read(OUT/'source_identity.json').items():assert sha(p)==h,p
    for p,h in read(OUT/'protocol.json')['source_hashes'].items():assert sha(p)==h,p
    count=0;diagnostics=[]
    for fold in ['M05','M07','M09','M11']:
        for family in FAMILIES:
            for seed in SEEDS:
                z=dict(np.load(OUT/f'probabilities_{fold}_{family}_{seed}.npz'))
                for key,value in z.items():
                    assert np.isfinite(value).all() and (value>=0).all() and (value<=1).all()
                    np.testing.assert_allclose(value*57,np.round(value*57),atol=1e-14,rtol=0)
                    if '_8_' in key:
                        assert (value<=z[key.replace('_8_','_10_')]).all()
                        assert (z[key.replace('_8_','_10_')]<=z[key.replace('_8_','_12_')]).all()
                    count+=1
                for threshold in (8,10,12):
                    for kind in ('single','group'):
                        original=z[f'{kind}_{threshold}_original_0']
                        for variant in ('time','space','both'):
                            for rep in (1,2,3):
                                delta=z[f'{kind}_{threshold}_{variant}_{rep}']-original
                                diagnostics.append(dict(fold=fold,family=family,seed=seed,threshold=threshold,kind=kind,variant=variant,repeat=rep,records=delta.size,changed_records=int((delta!=0).sum()),mean_abs_probability_change=float(abs(delta).mean()),max_abs_probability_change=float(abs(delta).max())))
    pd.DataFrame(diagnostics).to_csv(OUT/'intervention_diagnostics.csv',index=False,encoding='utf-8-sig')
    pairs=pd.read_csv(OUT/'paired_brier.csv');absolute=pd.read_csv(OUT/'absolute_brier.csv');support=pd.read_csv(OUT/'label_support.csv')
    selected=pairs[(pairs.scope=='ALL')&(pairs.offset=='all')&(pairs.seed=='mean')]
    selected.to_csv(OUT/'summary_effects.csv',index=False,encoding='utf-8-sig')
    main=selected[(selected.model=='context1_graph')&(selected.threshold==10)&(selected.kind=='single')&(selected.comparison=='original-minus-time')].iloc[0]
    secondary=selected[(selected.model=='context1_graph')&(selected.threshold==10)&(selected.kind=='group')&(selected.comparison=='original-minus-space')].iloc[0]
    def fmt(v):return f'{v:.6f}'
    def tab(df,cols):
        lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
        for _,row in df.iterrows():lines.append('| '+' | '.join(fmt(row[c]) if isinstance(row[c],(float,np.floating)) else str(row[c]) for c in cols)+' |')
        return '\n'.join(lines)
    report='''# 联合结构与阈值实验（阶段封存）

本轮全部为2024年开发证据。实验不依赖外部缺失资料，不训练模型、不重新拟合边际、不搜索阈值或执行规则。原始量纲尚未核定，8/10/12均按原始刻度表述。

## 冻结设计

880个起报、四fold、17节点；context1_graph与analog112采用既有自然输入refit64边际、四段平滑校准及三个partial05秩模板（20241101/02/03）。两模型共用Graph的partial05秩模板；Graph−analog敏感性反映这一共同联合模板下的边际差异，不能单独归因为图结构。每流57个情景、72小时、19原子各出现三次。四窗口偏移0/24/48/66、单机6h及三组共同6h均完整保留。主终点为Graph、阈值10、四偏移等记录权重下原联合减去打散时间的单机6h Brier；组共同6h原联合减去打散空间为辅助。负数表示原联合评分较低。

fold名称表示双月评价段，定义来自seasonal_probability_core.py的FOLDS：M05为2024年5–6月[2024-05-01,2024-07-01)，M07为7–8月[2024-07-01,2024-09-01)，M09为9–10月[2024-09-01,2024-11-01)，M11为11–12月[2024-11-01,2025-01-01)。括号表示起始日含、结束日不含；M09不等同于9月单月，逐fold结果不表述为逐单月结果。

冻结shuffle种子为2026110201/02/03，三次均运行且逐次保留，不挑最好。单个shuffle在各起报、fold和模型之间共享，以固定有限情景干预。模型种子和shuffle重复先各自计算Brier再平均；不是先平均概率形成新的部署集成。种子和重复不增加独立天气样本数。

## 同边际干预实际改变什么

| 干预 | 保留 | 改变 |
| --- | --- | --- |
| 原联合 | partial05跨小时、跨节点配对 | 无 |
| 打散时间time | 各lead的完整同期跨节点向量；每lead,node的57值多重集 | 不同lead使用独立scenario排列，改变跨时配对 |
| 打散空间space | 每node整段72h轨迹；每lead,node多重集 | 不同node使用独立scenario排列，改变节点之间配对 |
| 同时both | 每lead,node多重集 | 每个lead,node独立排列 |
| 全局负控 | 全部情景整条轨迹 | 只改scenario编号 |

打散对照改变相关6h/共同事件概率，不改变边际、原子权重和边际CRPS。space保留各节点整段轨迹，所以单机6h事件概率及评分精确不变，这也是结构负控。57情景的随机重配只是有限情景的独立性近似，残余相关和随机误差仍存在，不能称为精确独立分布。time仍保留同期空间关系，space仍保留单节点时间关系；它们不是新的预测模型或部署方法。

每个stream的每lead,node均精确检查排序后的57整数秩索引完全相等。由于原秩恰为0至56的排列，此证书对该stream全部起报的任意q成立，逐格证明值多重集和CRPS恒等；另在每stream首末起报直接排序实际float64场景值验证。全局scenario重排的所有阈值事件概率完全相等。每stream全部十种原/干预、三个阈值、首末起报还用独立标量循环直接从q与秩重算概率，共1440次，与批量计算精确一致。阈值10原联合概率与旧自然输入refit64产物精确相等。

## 结果

'''
    report+=f"主H2：Graph阈值10单机6h，原联合−打散时间，已知标签差为 **{fmt(main.known_brier_or_delta)}**，共享未知物理窗口的保守配对识别界为 **[{fmt(main.lower)}, {fmt(main.upper)}]**。\n\n"
    absprimary=absolute[(absolute.scope=='ALL')&(absolute.offset=='all')&(absolute.seed=='mean')&(absolute.model=='context1_graph')&(absolute.threshold==10)&(absolute.kind=='single')&(absolute.variant.isin(['original','time']))]
    report+='主终点的绝对评分：\n\n'+tab(absprimary,['variant','lower','upper','known_brier_or_delta'])+'\n\n'
    report+=f"辅助：Graph阈值10共同6h，原联合−打散空间，已知标签差为 **{fmt(secondary.known_brier_or_delta)}**，识别界 **[{fmt(secondary.lower)}, {fmt(secondary.upper)}]**。\n\n"
    graph=selected[(selected.model=='context1_graph')&(((selected.kind=='single')&(selected.comparison=='original-minus-time'))|((selected.kind=='group')&(selected.comparison=='original-minus-space')))]
    report+=tab(graph,['threshold','kind','comparison','lower','upper','known_brier_or_delta'])+'\n\n'
    foldmain=pairs[(pairs.scope!='ALL')&(pairs.offset=='all')&(pairs.seed=='mean')&(pairs.repeat=='mean')&(pairs.model=='context1_graph')&(pairs.threshold==10)&(pairs.kind=='single')&(pairs.comparison=='original-minus-time')]
    report+='主终点逐fold（逐seed及shuffle先评分后平均）：\n\n'+tab(foldmain,['scope','records','known','unknown','lower','upper','known_brier_or_delta'])+'\n\n'
    report+='主效应存在季节反转：M09原联合评分较差，其余三个fold较好，不能表述为所有季节稳定获益。阈值8/10/12的ALL主对比识别界均为负只支持开发集平均改善。阈值10下Graph−analog原联合单机已知分数差为+0.037684，识别界[+0.026174,+0.044338]；Graph仍差于强基线，同边际联合结构收益不等于图模型优胜。\n\n'
    report+='阈值10完整干预及模型敏感性：\n\n'+tab(selected[selected.threshold==10],['model','kind','comparison','lower','upper','known_brier_or_delta'])+'\n\n'
    report+='## 标签与有效数\n\n'+tab(support[(support.scope=='ALL')&(support.offset=='all')],['threshold','kind','origins','records','known','unknown','unknown_rate','physical_windows','unknown_physical_windows'])+'\n\n'
    report+='''标签直接由严格float64原始小时值生成：任一已知超阈值为失败、全已知达标为成功、其余未知。未知不填安全；同一物理窗口在不同起报中共享一个二元标签，配对界基于同一未知标签优化。不同窗口之间共享小时、不同threshold嵌套、单机与组窗口之间包含关系未施加联合限制，因此这些界保守，不能称为全物理约束下的sharp界。识别界不是置信区间；没有外部95%保证、显著性或部署升级结论。已知标签分数只说明已知子集，不代表完整缺失目标总体。

所有fold、模型seed、shuffle重复、threshold及offset分数保存在absolute_brier.csv和paired_brier.csv。ALL聚合按预测记录计权，物理窗口数另外报告，未把重复起报当作新的独立天气事件。label_support.csv提供全部offset和fold的标签数。intervention_diagnostics.csv逐重复保存改变概率的记录数及平均/最大幅度；不据此筛选shuffle。

## 验证与复现

数学测试24项通过，其中3项RED真实检测错误：重复秩索引导致边际变化、先平均概率再Brier、float32跨阈值；其余GREEN验证边际实际值、结构保留、直接概率、负控、float64标签和共享未知配对界枚举。math_test_log.json、validation.json和final_verification.json保存证据。protocol.json在读取结果前冻结，含核心代码SHA；source_identity.json逐项追踪旧输入SHA，结束时全部复核。旧文件/产物只读。

在项目根目录运行：

```powershell
& 'python' '风电运维Demo/closure_joint_tests.py'
& 'python' '风电运维Demo/closure_joint.py'
& 'python' '风电运维Demo/closure_joint_report.py'
```

只重算评分可用closure_joint.py --phase evaluate。产物均在风电运维Demo/artifacts/review_closure_v1/joint_structure。代码为closure_joint.py、closure_joint_core.py、closure_joint_tests.py与closure_joint_report.py。阈值核定、真实海况/船规、历史发布预报、真实费用工单和独立时段仍须另行补齐；本实验不能补足这些证据。
'''
    note=BASE.parent/'研究笔记/阶段收尾_2026-11-02/联合结构与阈值实验.md';note.parent.mkdir(parents=True,exist_ok=True);note.write_text(report,encoding='utf-8')
    js(OUT/'final_verification.json',dict(status='PASS',probability_arrays=count,source_hashes_verified=True,probabilities_quantized_and_monotone=True,math_checks=len(read(OUT/'math_test_log.json')),report=str(note),report_sha=sha(note)))
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='final_manifest.json']+[BASE/n for n in ['closure_joint.py','closure_joint_core.py','closure_joint_tests.py','closure_joint_report.py']]+[note]
    js(OUT/'final_manifest.json',{str(p):sha(p) for p in files})
    print('FINAL PASS',count,'probability arrays',len(files),'hashed files');print(selected[(selected.threshold==10)&(selected.model=='context1_graph')].to_string(index=False))

if __name__=='__main__':main()

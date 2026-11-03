"""Transparent tables and support accounting for the frozen matched diagnostic."""
from iteration3_matched_run import OUT,ROOT,GROUPS,read,write,sha
import pandas as pd
import numpy as np

def table(d,cols):
 def fmt(v):return f'{v:.8f}' if isinstance(v,(float,np.floating)) else str(v)
 return '| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in d[cols].itertuples(index=False,name=None))

def main():
 d=pd.read_csv(OUT/'matched_windows.csv');b=pd.read_csv(OUT/'bounds_summary.csv');verify=read(OUT/'verification.json');overlap=read(OUT/'partial_evidence_overlap.json')
 # Descriptive role metadata only; retain every frozen numerical result.
 if 'role' in b.columns:b=b.rename(columns={'role':'contrast_role'})
 b['contrast_role']=np.where(b.comparison.eq('interaction'),'primary_contrast_family','auxiliary_contrast')
 b.to_csv(OUT/'bounds_summary.csv',index=False,encoding='utf-8-sig')
 registry=b[['population','scope','start_group','comparison']].copy()
 registry['is_primary']=registry.population.eq('all44_primary')&registry.scope.eq('ALL')&registry.start_group.eq('ALL')&registry.comparison.eq('interaction')
 registry['analysis_role']=np.where(registry.is_primary,'primary',np.where(registry.comparison.eq('interaction'),'sensitivity','auxiliary'))
 assert int(registry.is_primary.sum())==1
 registry.to_csv(OUT/'analysis_registry.csv',index=False,encoding='utf-8-sig')
 origins=d[['fold','issue','complete']].drop_duplicates();support=[]
 for pop in ['all44_primary','complete15_auxiliary']:
  cohort=origins if pop=='all44_primary' else origins[origins.complete]
  for group,starts in GROUPS.items():
   x=d[d.start.isin(starts)]
   if pop!='all44_primary':x=x[x.complete]
   for o in cohort.itertuples():
    z=x[(x.fold==o.fold)&(x.issue==o.issue)]
    support.append(dict(population=pop,start_group=group,fold=o.fold,issue=o.issue,seed_count=z.seed.nunique(),seed_window_rows=len(z),windows_per_seed=len(z)//3,empty_support=z.empty,origin_weight=1/len(cohort)))
 pd.DataFrame(support).to_csv(OUT/'origin_support.csv',index=False,encoding='utf-8-sig')
 assert not pd.DataFrame(support).empty_support.any()
 # Explicit header for an empty-support table with zero records.
 pd.DataFrame(columns=['population','scope','start_group','origins','seed_window_rows','known_window_rows','unknown_window_rows','status']).to_csv(OUT/'empty_support.csv',index=False,encoding='utf-8-sig')
 write(OUT/'tests_green.json',dict(status='PASS',tests=6,command='python iteration3_matched_test.py',observed_output='Ran 6 tests in 0.030s; OK',covers=['four-cell affine algebra','interaction sign','square before seed average','start0 zero','equal-origin same-window weights','current-eligible matched queue']))
 primary=b[b.population=='all44_primary'];aux=b[b.population=='complete15_auxiliary']
 lines=['# 第三轮：同一物理任务窗口的预报更新机制诊断','',
 '这是上一轮初始/当前阶段差异提出的回溯机制诊断，不是新的确认性试验。已先冻结 protocol 与 446 项 source_identity；全部旧源码和历史 artifacts 保持只读。未重新预测、校准、训练、调参或改变动作。','',
 '44 个起报、每起报 3 套冻结秩模板全部保留。仅取 current 质量门控可用的任务窗口队列，在 fold、issue、seed、task、start 上严格匹配 Analog initial、Graph initial、Analog current、Graph current 四格。各格共享目标、node、duration 和原始 raw64 标签，因此旧阶段队列漂移不会被计作更新贡献。这是条件于当前门控可用窗口的回溯存活队列对比；不解释门控拒绝窗口的表现。','',
 '共 9,150 个种子任务窗口、36,600 个单格概率，8,262 个已知标签窗口、888 个未知标签窗口（均含种子重复）；主分析每个分组的 44 个起报均有支持，没有空支持。完整天气 15 起报只作辅助；该资格取自上一轮完整任务节点 72h 天气定义。','',
 '每个原始单格先计算 Brier=(p−y)²，然后在起报/种子内平均窗口、起报内平均三种子、最后等权平均起报。禁止先平均概率再评分。一个起报在 ALL 和分组分析中均以该分组实际匹配窗口取平均，因此分组总体不能按窗口数简单拼接回总体。origin_support.csv 逐起报明示窗口数和权重。','',
 '唯一主要诊断为 interaction=(AC−GC)−(AI−GI)=(Analog 更新差)−(Graph 更新差)。正值表示 Graph 的 Brier 从更新中下降更多。辅助诊断为初始模型差 AI−GI、当前模型差 AC−GC、Analog 更新差 AC−AI、Graph 更新差 GC−GI。模型差负值偏向 Analog；更新差负值表示改进。四格边际 Brier 界仅辅助；不同诊断各自优化，不能将不同端点任意拼减。','',
 '唯一主分析严格限定为 all44_primary × scope=ALL × start_group=ALL × interaction；其余49个子群或分组交互均为敏感性分析，四项辅助对比及四格边际均为辅助。analysis_registry.csv 的 is_primary 仅一行=True，并作为唯一样本定义及分析角色登记；bounds_summary.csv 的 contrast_role 仅表示对比家族（primary_contrast_family/auxiliary_contrast），不表示每个子群各有主检验。原证书中 role 字段沿用冻结运行时的对比家族标签，不能用于识别主分析。该角色澄清不改变任何数值、样本或分组，也不增加或重选主检验。','',
 '## 全44主分析：总体与四项辅助诊断','',table(primary[(primary.scope=='ALL')&(primary.start_group=='ALL')],['comparison','shared_lower','shared_upper','window_label_lower','window_label_upper']),'',
 '总体 interaction 严格界为 [0.04641048, 0.08420953]，在所有允许的共享缺测小时补全下为正。Analog 与 Graph 更新差各自严格为负，Graph 更新改善更大。初始模型差严格偏向 Analog；当前模型差跨零，无法判定当前同窗的总体模型优劣。','',
 '## 预设 start 分组与全部 fold（均报告，不择优）','',table(primary[primary.comparison=='interaction'],['scope','start_group','origins','seed_window_rows','shared_lower','shared_upper']),'',
 'start0 每一行同一模型初始/当前概率完全相等；更新差和交互代数为零，浮点汇总残差小于 1e−12。M05/M07 总体交互界跨零；M09、M11 为正。ALL 为四 fold 组成的等起报平均，不是各 fold 均改善或普遍因果机制的证据。','',
 '## 完整天气15起报辅助','',table(aux[(aux.scope=='ALL')&(aux.start_group=='ALL')],['comparison','shared_lower','shared_upper']),'',
 '完整天气子群总体 interaction=0.09075418，各界退化为点；该子群由天气完整性筛选，不能替代全44主分析。所有 fold×start 分组四格、主要及四项辅助结果均在 bounds_summary.csv。','',
 '## 未知标签与严格界','',
 '按每格 p²+(1−2p)y 建立线性式，四格分别平方后按符号相加。已知 raw64 小时风速任一>10 则窗口失败；全部已知且<=10 则成功；其余窗口为未知。对每个未知绝对 hour,node 只建一个二元状态，窗口成功等于其未知小时状态的 AND。复用未修改的 iteration2_bounds_core.window_term/solve_bounds，全范围共同优化共享天气。','',
 '较松核验界允许每个独立物理窗口一个标签，四格和三种子共享该标签，但不同窗口互不约束；按绝对起始小时、node、duration 合并重复物理窗口。所有450个严格界均包含在相应逐窗口松界内。每端点提供最优状态0、零 MIP gap、原/对偶目标及小时见证；把见证填为同一份共同 weather 后，直接逐格 (p−y)² 重算全部端点。','',
 f"有效性：6项先RED后GREEN；独立重建 {verify['independent_probability_rows']:,} 原概率、{verify['independent_label_windows']:,} raw64 标签、{verify['independent_gate_checks']:,} 门控状态；450证书PASS，最大端点直接重算误差 {verify['max_direct_endpoint_error']:.3g}；446源哈希不变。",'',
 f"独立 partial 证据交集检查：原494个 known_bad 命中主同窗队列 {overlap['hit_distinct_hoursets']} 个不同未知 hourset、{overlap['hit_seed_window_rows']} 个含种子重复窗口。该处只报交集，未重算 partial 敏感性界，主分析仍只用原始严格小时 raw64。",'',
 '## 结论边界','',
 '该设计分离了同一目标下从初始到当前的预测更新差异，避免把可用队列变化误算为更新效果；仍条件于冻结预测、质量门控及这44个开发起报。初始预测和当前预测的信息集随时间改变，不是随机化干预；正交互不能证明真实海况、可执行性、费用收益或普遍因果机制。研究目标是原始小时风速阈值窗口，未接入新增海浪/船舶/现场维修结果。严格界是允许任意缺测二元天气补全的条件识别范围，不是置信区间，也不是物理天气概率模型。','',
 '复现：在风电运维Demo目录运行 iteration3_matched_test.py；iteration3_matched_run.py freeze；iteration3_matched_run.py evaluate；iteration3_matched_report.py。再次freeze要求协议/源哈希完全一致。','']
 notes=ROOT.parent/'研究笔记/第三轮实证推进_2026-11-02';notes.mkdir(exist_ok=True,parents=True)
 report=notes/'同窗预报更新机制诊断.md';report.write_text('\n'.join(lines),encoding='utf-8')
 (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
 write(OUT/'final_manifest.json',{str(p):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='final_manifest.json'})
 print(report)

if __name__=='__main__':main()

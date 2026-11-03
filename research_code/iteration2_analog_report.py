"""Equal-origin probability sensitivity, explicit development-scope report, manifest."""
import json
import numpy as np
import pandas as pd
from iteration2_analog_execution import ROOT,OUT,read,write,sha,PREDS,EXECS

DOC=ROOT.parent/'研究笔记/第二轮实证推进_2026-11-02/强基线固定资源执行实验.md'

def main():
 windows=pd.read_csv(OUT/'support_window_ledger.csv');originrows=[];pairs=[]
 for (fold,issue,stage),g in windows.groupby(['fold','issue','stage'],sort=False):
  pp={p:g[g.prediction==p].sort_values(['seed','task','start']) for p in PREDS};a=pp['AnalogFixed'];b=pp['GraphFixed'];assert np.array_equal(a[['seed','task','start']],b[['seed','task','start']]);complete=bool(a.complete.iloc[0]);n=len(a)//3
  y=a.label.to_numpy().reshape(3,n);assert np.allclose(y,y[0],equal_nan=True);y=y[0];known=np.isfinite(y)
  probs={p:pp[p].probability.to_numpy().reshape(3,n) for p in PREDS}
  for pred,p in probs.items():
   c=np.mean(p*p,0);d=1-2*p.mean(0);lo=c+np.where(known,d*np.nan_to_num(y),np.minimum(d,0));hi=c+np.where(known,d*np.nan_to_num(y),np.maximum(d,0));originrows.append(dict(fold=fold,issue=issue,stage=stage,prediction=pred,complete=complete,windows=n,known=int(known.sum()),unknown=int((~known).sum()),brier_lower=float(lo.mean()),brier_upper=float(hi.mean()),known_brier=float((c[known]+d[known]*y[known]).mean()) if known.any() else np.nan))
  pa=probs['AnalogFixed'];pb=probs['GraphFixed'];c=(pa*pa-pb*pb).mean(0);d=-2*(pa-pb).mean(0);lo=c+np.where(known,d*np.nan_to_num(y),np.minimum(d,0));hi=c+np.where(known,d*np.nan_to_num(y),np.maximum(d,0));pairs.append(dict(fold=fold,issue=issue,stage=stage,complete=complete,windows=n,lower=float(lo.mean()),upper=float(hi.mean())))
 pd.DataFrame(originrows).to_csv(OUT/'task_window_brier_origins.csv',index=False,encoding='utf-8-sig');pd.DataFrame(pairs).to_csv(OUT/'paired_task_window_brier_origins.csv',index=False,encoding='utf-8-sig');summary=[];paired=[]
 for scope in ['ALL','M05','M07','M09','M11']:
  for pop in ['all','complete_task_weather']:
   for df,by,cols,dest in [(pd.DataFrame(originrows),['stage','prediction'],['brier_lower','brier_upper','known_brier'],summary),(pd.DataFrame(pairs),['stage'],['lower','upper'],paired)]:
    z=df if scope=='ALL' else df[df.fold==scope]
    if pop!='all':z=z[z.complete]
    for key,g in z.groupby(by):
     key=key if isinstance(key,tuple) else (key,);dest.append(dict(scope=scope,population=pop,**dict(zip(by,key)),origins=len(g),**{c:float(g[c].mean()) for c in cols}))
 pd.DataFrame(summary).to_csv(OUT/'task_window_brier_equal_origin.csv',index=False,encoding='utf-8-sig');pd.DataFrame(paired).to_csv(OUT/'paired_task_window_brier.csv',index=False,encoding='utf-8-sig')
 s=pd.read_csv(OUT/'summary.csv');arms=pd.read_csv(OUT/'arm_summary.csv');ver=read(OUT/'verification.json');ev=read(OUT/'evaluation_complete.json');text=['# AnalogFixed固定资源执行实验','', '本轮唯一主比较为 **AnalogFixed − GraphFixed，lookahead72执行，共用旧cross固定B资源x**；myopic为辅助。全部44个预定起报×3种子，新增264个类比执行臂并精确复用264个旧GraphFixed臂，合计528臂、3168条任务。差为候选减对照，负数表示类比费用更低。','', '## 费用主结果','', '| 执行 | 完整15起报费用差 | 全44起报保守差界 | 完整15成功任务差 | 完整15已知失败差 |','|---|---:|---:|---:|---:|']
 for ex in ['lookahead72','myopic']:
  a=s[(s.scope=='ALL')&(s.population=='complete_task_weather')&(s.executor==ex)].iloc[0];b=s[(s.scope=='ALL')&(s.population=='all')&(s.executor==ex)].iloc[0];text.append(f'| {ex} | {a.difference_lower:+.3f} | [{b.difference_lower:+.3f}, {b.difference_upper:+.3f}] | {a.successful_difference:+.3f} | {a.failed_difference:+.3f} |')
 text+=['', '**负结果与异质性：全44例主比较仍跨0，不能证实稳定费用节约。** 完整15例主费用差按fold为M05(2例)−18.919、M07(1例)−103.436、M09(4例)+11.481、M11(8例)约0。M07只有1例却贡献明显，M09反而增加费用，不能宣称时间外一致收益。辅助myopic在M07完整1例增加155.503，在M09增加18.284。']
 text+=['','## 各臂费用与任务指标','','| 人群 | 预测 | 执行 | 费用下/上界 | 作业费 | 成功 | 已知失败 | 未知 | 未尝试 |','|---|---|---|---:|---:|---:|---:|---:|---:|']
 for r in arms[arms.scope=='ALL'].itertuples():text.append(f'| {r.population} | {r.prediction} | {r.executor} | {r.cost_lower:.3f} / {r.cost_upper:.3f} | {r.operating_cost:.3f} | {r.successful:.3f} | {r.failed:.3f} | {r.unknown:.3f} | {r.unattempted:.3f} |')
 text+=['','预订费用在每个匹配起报/种子上完全一致；费用是模拟分值。完整15例由6任务节点72h完整原始天气固定，非随机子样本，不能替代全44例。未知天气差界是保守外包，不是置信区间；共享未知窗口的进一步收紧由独立shared_bounds模块提供。','', '## 同任务人群的预测窗口评分','', '上轮强基线Brier覆盖880起报的固定6小时单机窗，本轮补充同44起报、6任务各自真实时长的全部可开工窗口。initial_candidate使用初始预报全部可排窗；current_candidate只纳入质量门控允许时点的当前任务窗，未按动作、天气标签或效果选窗。下面先算种子损失再在每个起报平均窗口，最后等起报平均。task_window_brier.csv另存候选窗加权均值以明确质量门控导致的权重差异。','', '| 人群 | 窗口 | Analog−Graph Brier差界 |','|---|---|---:|']
 for r in pd.DataFrame(paired).query("scope == 'ALL'").itertuples():text.append(f'| {r.population} | {r.stage} | [{r.lower:+.6f}, {r.upper:+.6f}] |')
 text+=['', '**初始任务窗优势没有稳定延续到滚动当前任务窗。** 初始窗all44差界全负，完整15差−0.085684；当前窗all44差界跨0，完整15反而+0.005070，即Analog略差。该现象直接限制“较好初始概率必然改善执行”的主线推断。', '', 'p=1但实际失败：Analog初始156/当前24，Graph初始989/当前47；p=0但实际成功：Analog初始36/当前15，Graph初始1/当前4。上述按起报×种子×任务候选窗计数，不能作为独立天气样本量；详见task_window_brier.csv。固定动作费用高于初始57情景支持最大值的确证次数为：Analog前瞻6、即时3，Graph前瞻22、即时20（均132起报种子case）。Analog仍有支持外费用，支持越界减少不构成风险覆盖保证。']
 text+=['', '即使任务窗Brier较好，也不能推出当前费用结构和固定资源下的执行费用较低；资源、经济阈值和失败作业费用会改变传导。完整与缺测人群均保留，全部fold结果见summary.csv、paired_task_window_brier.csv。','', '## 身份、因果与边界','', 'AnalogFixed使用multiscale_context_v1 analog112原始test原子和它自身前月固定分提前量平滑仿射系数（legacy_frozen32），与上轮强基线inventory的natural来源及参数逐位吻合，也与旧multiscale frozen全q逐位相等。没有套用图模型校准系数，没有重新拟合。GraphFixed是context1_graph同版本自然输入，整份q与旧gated_graph_atoms frozen逐位相等；12次更新概率网格与旧cross B逐位相等。三个种子仅为同一analog生产者0配三份旧partial05秩，不能称独立统计训练。','', '逐次检查类比匹配历史source+72严格早于当次issue，历史回溯不超过112天；候选身份完整保存在冻结statistical_sources_analog112.npz并核SHA。全部可用更新缓存来自同fold test_ids；M09低质量且无缓存更新直接门控跳过，与旧cross一致。起报前质量、资源、费用、一次尝试无重试、每6h更新/72h执行均不变。','', '先写protocol和source_identity、验证所有来源与Graph输入，再生成全部264新轨迹并保存planning_complete.json(truth_loaded=false)，之后才调用load_truth读取原始float64评价。264旧Graph轨迹引用原路径和原SHA，旧evaluation核SHA并与新64位任务台账逐字段一致；没有重跑或重订Graph资源。','', '命名隔离：旧cross B=fixed_month/frozen，W28=rolling28；state_gate B=rolling28、W28=残差WT。本轮只消费旧cross B，不跨简称连接。','', '## 核验与失败记录','', f'独立核验 {ver["status"]}：{ver["tasks"]}条任务、{ver["steps"]}更新步骤，{ver["independent_myopic_batches"]}即时批次穷举，{ver["independent_MILP_optima"]}个预定前瞻MILP重建，{ver["causal_prefix_perturbations"]}次后续预报扰动不改变前缀，{ver["independent_task_probabilities"]}个原子/秩概率独立重建，{ver["independent_window_labels"]}任务窗口raw64标签重算，{ver["source_hashes_unchanged"]}源SHA不变。', '', '首轮plan在M09质量不足且无预测缓存的issue=6461触发KeyError；当时尚未执行任何新臂/载入评价真值。修复为沿用原cross质量门控跳过后重新从全范围plan，未删起报、未补未来信息。之后plan与evaluate完整成功，无求解失败或求解重试；该接口失败不作为模型效果证据。','', '支持诊断完整保存全部候选窗口的p=0/1已知错漏及528臂固定已保存动作的初始57情景费用支持范围。固定动作回放支持不是自适应策略的预测分布；不能据此声称TV集合覆盖真实风险或倒选参数。','', '全部证据仍来自反复开发的2024年数据，没有新外部数据、没有独立留出测试，不声称新独立泛化、确认性显著性或真实现场费用收益。','', '运行：`python iteration2_analog_execution.py plan`，`python iteration2_analog_execution.py evaluate`，`python iteration2_analog_verify.py`，`python iteration2_analog_report.py`。所有新产物仅在review_iteration2_v1/analog_execution，旧源码/产物/默认保持不变。','']
 text+= [f'附加因果回放：按每fold首/中/末固定起报，共{ver.get("analog_raw_replays_and_future_truncation",0)}个analog原始生产者预测逐位重算通过；截断全部issue之后数据后预测仍逐位相同。','']
 DOC.parent.mkdir(parents=True,exist_ok=True);DOC.write_text('\n'.join(text),encoding='utf-8')
 manifest={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='final_manifest.json'}
 manifest[str(DOC)]=sha(DOC)
 for name in ['iteration2_analog_execution.py','iteration2_analog_verify.py','iteration2_analog_report.py']:manifest[str(ROOT/name)]=sha(ROOT/name)
 write(OUT/'final_manifest.json',manifest);print('REPORT',str(DOC),len(manifest),flush=True)

if __name__=='__main__':main()

"""Final text/numerical/calendar audit; no PDF or image layout claims."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
from seasonal_probability_core import FOLDS
ROOT=Path(__file__).resolve().parent
NOTES=ROOT.parent/'研究笔记/阶段收尾_2026-11-02'
ART=ROOT/'artifacts/review_closure_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    docs=[NOTES/'实证结果报告.md',NOTES/'论文整合稿_现阶段结论.md',NOTES/'同步中断与阈值敏感性.md',ROOT.parent/'00_现阶段完成总结与等待资料_2026-11-02.md',NOTES/'统一口径勘误.md']
    texts={p:p.read_text(encoding='utf-8') for p in docs}
    expected={'M05':('2024-05-01','2024-07-01'),'M07':('2024-07-01','2024-09-01'),'M09':('2024-09-01','2024-11-01'),'M11':('2024-11-01','2025-01-01')}
    for f,(start,end) in expected.items():assert (FOLDS[f]['test_start'],FOLDS[f]['test_end'])==(start,end)
    for p in docs[:2]:
        text=texts[p]
        for s in ['9–10月','11–12月','双月','625','固定动作','35.68%']:assert s in text,(p,s)
        assert '完整月份' not in text
    assert '四个双月fold' in texts[docs[2]] and '9–10月存在反向' in texts[docs[3]]
    assert '| fold(双月) |' in (NOTES/'联合结构独立审查.md').read_text(encoding='utf-8')
    testcounts=[]
    for file,n in [('完整测试日志.txt',599),('增补测试日志.txt',26)]:
        text=(NOTES/file).read_text(encoding='utf-8')
        assert int(re.findall(r'Ran (\d+) tests? in',text)[-1])==n and text.rstrip().endswith('OK')
        testcounts.append(n)
    reviewpaths=[ART/'independent_review/joint.json',ART/'outage_stress/independent_verification.json',ART/'execution_sensitivity/independent_verification.json',ART/'update_robustness/independent_artifact_audit.json']
    reviews={str(p):read(p) for p in reviewpaths};assert all(v['status']=='PASS' for v in reviews.values())
    for p,digest in reviews[str(reviewpaths[0])]['source_hashes_read_only'].items():assert sha(p)==digest,p
    for directory in ['execution_sensitivity','update_robustness']:
        manifest=read(ART/directory/'final_manifest.json');files=manifest.get('files',manifest)
        for p,digest in files.items():assert sha(p)==digest,p
    selection=pd.read_csv(ART/'update_robustness/selection_by_origin.csv')
    foldcounts={f:{'complete':int(z.complete.sum()),'other':int((~z.complete).sum())} for f,z in selection.groupby('fold')}
    assert [foldcounts[f]['complete'] for f in expected]==[2,1,4,8]
    assert [foldcounts[f]['other'] for f in expected]==[9,10,7,3]
    months={m:int(z.complete.sum()) for m,z in selection.groupby('month') if z.complete.any()}
    assert months=={'2024-05':2,'2024-07':1,'2024-09':1,'2024-10':3,'2024-11':3,'2024-12':5}
    support=pd.read_csv(ART/'outage_stress/support.csv')
    inputs={k:int(v) for k,v in support[support.outage_hours==0].set_index('fold').complete_input_origins.to_dict().items()}
    assert [inputs[f] for f in expected]==[61,49,109,173] and sum(inputs.values())==392
    u=pd.read_csv(ART/'update_robustness/bounds_summary.csv')
    inter=u[(u.comparison=='interaction')&(u.estimand=='equal_origin')&(u.evidence=='raw64')]
    base=inter[inter.scope=='ALL'].iloc[0];removed=inter[inter.scope=='leave_fold_M09'].iloc[0]
    np.testing.assert_allclose([base.lower,base.upper],[.04641048394511018,.08420952700731113],atol=1e-12,rtol=0)
    np.testing.assert_allclose([removed.lower,removed.upper],[-.007656834803649325,.031607993934863625],atol=1e-12,rtol=0)
    loo=inter[inter.kind=='leave_issue'];seeds=inter[inter.kind=='seed']
    assert len(loo)==44 and len(seeds)==3 and loo.lower.min()>0 and seeds.lower.min()>0
    assert abs(loo.lower.min()-.03508755)<5e-9
    phy=u[(u.scope=='ALL')&(u.comparison=='interaction')&(u.estimand=='physical_event')&(u.evidence=='raw64')].iloc[0]
    np.testing.assert_allclose([phy.lower,phy.upper],[.04743038211000615,.08361099758311516],atol=1e-12,rtol=0)
    partial=read(ART/'update_robustness/partial_evidence_nesting.json')
    assert partial['known_bad_total']==494 and partial['unknown_to_known_unique_physical_windows']==1
    assert partial['unknown_to_known_seed_window_rows']==3 and partial['unknown_to_known_four_cell_rows']==12
    h1=pd.read_csv(ART/'threshold_h1/paired_bounds.csv');outage=pd.read_csv(ART/'outage_stress/paired_bounds.csv')
    for t,target in [(8,(-.014676,-.012012)),(10,(-.006691,-.004739)),(12,(-.002270,-.000888))]:
        r=h1[(h1.scope=='ALL')&(h1.cohort=='all880')&(h1.kind=='single')&(h1.threshold==t)].iloc[0]
        np.testing.assert_allclose([r.lower,r.upper],target,atol=5.1e-7,rtol=0)
    for h,target in [(6,(-.007513,-.005445)),(12,(-.008523,-.006158)),(24,(-.011360,-.008583))]:
        r=outage[(outage.scope=='ALL')&(outage.cohort=='all880')&(outage.kind=='single')&(outage.outage_hours==h)&(outage.contrast=='graph_minus_mean')].iloc[0]
        np.testing.assert_allclose([r.lower,r.upper],target,atol=5.1e-7,rtol=0)
    acqpath=ROOT/'artifacts/review_iteration3_v1/acquisition/summary.json';acq=read(acqpath)
    assert (acq['essential_hours'],acq['nonzero_merged_terms'],acq['irrelevant_previous_hours'])==(234,82,146)
    reduction=acq['best_prespecified_48hour_strategy']['guaranteed_reduction']/acq['old_width']
    assert abs(reduction-.35684931506849315)<1e-12
    assert '有效' in texts[docs[0]] and '当前费用范围未缩小' in texts[docs[0]]
    report=NOTES/'最终综合审查.md'
    report.write_text('''# 最终综合审查

最终Markdown综合文稿通过文字、数值、时间口径及解释范围审查。对象为实证结果报告、论文整合稿、同步中断与阈值敏感性、根完成入口及统一口径勘误。本审查没有视觉检查PDF或图片版面，版面验收由主代理另行完成。

已核对seasonal_probability_core.py的FOLDS：M05=5–6月[2024-05-01,2024-07-01)，M07=7–8月[2024-07-01,2024-09-01)，M09=9–10月[2024-09-01,2024-11-01)，M11=11–12月[2024-11-01,2025-01-01)，均左闭右开。M09反向及删除M09不是9月单月结论。完整15的fold分布2/1/4/8、其他29的9/10/7/3和完整输入392的61/49/109/173均复核；真实日历月完整15为5月2、7月1、9月1、10月3、11月3、12月5。

625项软件回归为599+26，两份实际unittest日志末尾均OK；独立数学检查、穷举、投毒、证书端点及负控另列，不并入625，更不当成新增天气样本。

H2全量13740行配对、12720行绝对、60行summary与150行标签表独立复核仍通过；主界[-0.069799541,-0.036300529]、辅助共同窗口空间界[-0.065658140,-0.013303083]一致。正文保留M09反向、Graph差于强类比及有限57情景只是独立性近似，没有全面或跨年优越的结论。

400行中断及60行公平阈值表与独立核验一致。24h全零的72组输出跨起报恒定，正文明确只反映冻结模型/自然校准先验，不能作为实时空间信息收益。自然负控及未来毒化的实际检查范围准确。

更新原交互、物理事件等权、删除M09、部分失败证据及完整性构成与机器表一致。44次逐一起报和三个单种子交互下界均为正，逐一起报最小下界0.03508755；删除M09跨零，单点稳定与双月fold构成敏感并不矛盾。9150种子窗口、36600概率、3050物理事件/机会、494部分超限小时以及1物理窗口/3种子窗口/12单格相交均保持原范围。

执行378组界、1890组费用、343组旧界复现及1056动作/1557失败尝试独立核验相符。主费用公式仅用于λ≥0的固定动作事后重计价，整个运营及迟开工项固定。费用与成功任务端点分别求解，不当成任意组合的联合矩形。当前费用界仍跨零，不能宣称真实经济收益。

234必要小时、82项、原380中的其他146小时及48小时保证缩宽35.6849%与既有证书一致。保证要求全部请求阈值状态得到有效解析，仅缩小识别范围，不是费用节约。本轮没有新记录，实际范围没有缩小。

最终停止条件一致：全部为同年开发证据，字段单位及时间可用性、真实海况船规和成本、独立年度仍待补证。本阶段停止扩张模型/参数搜索，等待资料后先验收、冻结预测动作重评分，再开展外部确认。旧封存历史不覆盖，双月简称由统一勘误解释。

机器快照为artifacts/review_closure_v1/independent_review/final_delivery.json。根代理随后生成交付核验与总清单，并独立完成PDF版面验收；本审查不重复声称这些步骤已由本审查完成。
''',encoding='utf-8')
    sources=docs+reviewpaths+[Path(__file__),ROOT/'seasonal_probability_core.py',NOTES/'完整测试日志.txt',NOTES/'增补测试日志.txt',ART/'update_robustness/bounds_summary.csv',ART/'update_robustness/selection_by_origin.csv',ART/'outage_stress/support.csv',ART/'threshold_h1/paired_bounds.csv',ART/'outage_stress/paired_bounds.csv',acqpath]
    result=dict(status='PASS',scope='Final Markdown/numerical/calendar/log review; PDF and images not visually inspected',documents={str(p):sha(p) for p in docs},
                fold_test_intervals_left_closed_right_open=expected,complete15_by_fold=foldcounts,complete15_true_calendar_month_counts=months,
                complete_input392_by_fold=inputs,unittest_regressions=dict(existing=599,added=26,total=625,separate_mathematical_checks_not_added=True),
                branches=dict(joint=dict(paired=13740,absolute=12720,summary=60,support=150,primary_count=1,status='PASS'),
                  outage_threshold=dict(outage_rows=400,threshold_rows=60,status='PASS'),
                  execution=dict(saved_arms=1056,failed_attempts=1557,bounds=378,lambda_rows=1890,old_lambda1_reproductions=343,status='PASS'),
                  updates=dict(certificates=1120,endpoints=2240,leave_one_origin=44,single_seeds=3,loo_minimum_lower=float(loo.lower.min()),status='PASS')),
                acquisition_existing_evidence=dict(essential_hours=234,terms=82,other_previous_hours=146,query_hours=48,guaranteed_width_reduction_fraction=reduction,new_records_received=False,actual_bounds_not_shrunk=True),
                reviewed_source_sha256={str(p):sha(p) for p in sources},final_review_report=str(report),final_review_report_sha256=sha(report),
                pdf_layout_review='NOT_PERFORMED_BY_THIS_REVIEWER',root_final_index_and_history_seal='Root handles final index generation and history40/33391 audit; not independently rerun by this reviewer.')
    (ART/'independent_review/final_delivery.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',documents=len(docs),unittest_total=sum(testcounts),loo_minimum_lower=float(loo.lower.min()),source_snapshots=len(sources),pdf_visual_review=False),ensure_ascii=False))

if __name__=='__main__':main()

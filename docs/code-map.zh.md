[中文](./code-map.zh.md) · [English](./code-map.en.md) · [README](../README.md)

# 代码导航与实验阅读顺序

## 先区分两个入口

`demo/server.py`是可直接启动的公开合成演示。`research_code/`是脱敏研究存档，不是演示入口，也不是完整可安装的软件包。部分脚本需要未发布的内部模块、冻结缓存和模型。先读核心函数，再读编排脚本；不要把文件名中的历史模型称谓当作本轮主结论。

## 研究代码地图

| 模块 | 核心/入口 | 辅助脚本 | 阅读重点 |
|---|---|---|---|
| 标签与窗口 | [precision_truth_core.py](../research_code/precision_truth_core.py) | [research_precision_revision.py](../research_code/research_precision_revision.py) | 原始精度与三值事件；历史张量与真值分开 |
| 公平与强基线 | [review_baselines_core.py](../research_code/review_baselines_core.py) | [research_review_baselines.py](../research_code/research_review_baselines.py) | 信息预算、输入掩码与窗口概率 |
| 联合依赖 | [closure_joint_core.py](../research_code/closure_joint_core.py) | [closure_joint.py](../research_code/closure_joint.py) | 边际集合不变的时间/空间排列 |
| 缺测压力 | [closure_outage_core.py](../research_code/closure_outage_core.py) | [closure_outage_run.py](../research_code/closure_outage_run.py) | 匹配完整输入子群的同步遮挡 |
| 固定/滚动校准 | [review_fixed_rolling_pair.py](../research_code/review_fixed_rolling_pair.py) | [verify_review_fixed_rolling_pair.py](../research_code/verify_review_fixed_rolling_pair.py) | 校准比较与配对核验 |
| 更新稳健性 | [closure_robustness.py](../research_code/closure_robustness.py) | [closure_robustness_independent_review.py](../research_code/closure_robustness_independent_review.py) | 同目标、物理权重与季节删除诊断 |
| 执行与资源 | [causal_execution_core.py](../research_code/causal_execution_core.py) | [review_cross_execution.py](../research_code/review_cross_execution.py) | 不可撤回尝试、固定资源交叉 |
| 共享状态界 | [iteration2_bounds_core.py](../research_code/iteration2_bounds_core.py) | [iteration2_analog_execution.py](../research_code/iteration2_analog_execution.py) | 同一物理小时共享未知状态 |
| 补录价值 | [iteration3_acquisition_core.py](../research_code/iteration3_acquisition_core.py) | [iteration3_acquisition_core.py](../research_code/iteration3_acquisition_core.py) | 固定查询下的最坏剩余宽度 |
| 有界求解形式 | [bounded_distribution_core.py](../research_code/bounded_distribution_core.py) | [research_bounded_distribution.py](../research_code/research_bounded_distribution.py) | 非负费用结构下的有限变量界 |
| 情景归并 | [scenario_equivalence_core.py](../research_code/scenario_equivalence_core.py) | [research_scenario_equivalence.py](../research_code/research_scenario_equivalence.py) | 约束/费用等价情景归并 |
| 支配支持 | [support_dominance_core.py](../research_code/support_dominance_core.py) | [research_support_dominance.py](../research_code/research_support_dominance.py) | 零质量支配支持的工程验证 |

## 建议阅读路径

1. 从`precision_truth_core.py`理解真值精度与时间轴断言。
2. 阅读`review_baselines_core.py`和`closure_joint_core.py`，区分预测边际、情景秩与连续事件。
3. 阅读`causal_execution_core.py`和相关规划实现，区分预测时信息、不可撤回动作和事后评价。
4. 阅读`iteration2_bounds_core.py`，理解窗口的AND结构和共享物理状态。
5. 阅读`iteration3_acquisition_core.py`，理解固定补录查询的最坏剩余宽度。
6. 最后阅读求解工程分支，避免把数值形式的改进解释为预测效果提升。

## 文件类别如何理解

| 类别 | 常见作用 | 注意事项 |
|---|---|---|
| `*_core.py` | 核心数学、数据契约或执行逻辑 | 不都可脱离依赖独立导入 |
| `research_*.py`、`review_*.py`、`closure_*.py` | 不同阶段的实验编排与报告 | 可能有缓存写入和历史约定，不批量执行 |
| `verify_*.py`、`*_verify.py` | 特定分支的断言与证书检查 | 需要该分支的完整私有输入 |
| `*_tables.py`、`*_report.py` | 表格与报告生成 | 源表未发布时无法重建 |
| `tests/test_demo.py` | 公开合成核心契约 | 不验证研究模型或实证结论 |

部分文件带有旧模型名字、原目录结构及既往实验设定；这是保留演化过程的代码存档，主研究口径以[方法](research.zh.md)和[证据](evidence.zh.md)为准。匿名`feature_A/B/C/D`不提供真实字段字典，授权用户需在私有环境自行映射。

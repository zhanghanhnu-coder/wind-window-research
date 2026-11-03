[中文](./reproducibility.zh.md) · [English](./reproducibility.en.md) · [README](../README.md)

# 复现指南与公开边界

## 三种可复现程度

| 层次 | 当前公开材料支持什么 | 入口 |
|---|---|---|
| 合成演示 | 本地运行、调整输入、检查确定性输出 | `demo/server.py` |
| 方法阅读与汇总核对 | 阅读核心实现、比对图注和已发布汇总表 | `research_code/`、`results/`、`figures/` |
| 完整实证重跑 | 当前不能仅靠公开仓库完成 | 需要授权数据、缓存、模型及完整私有环境 |

公开合成演示与实证数据生产链是独立入口。不要用演示的12台机组和解析路径去“复现”研究的17台机组、880起报和44执行起报。

## 1. 获取并运行公开演示

```sh
git clone https://github.com/zhanghanhnu-coder/wind-window-research.git
cd wind-window-research
python --version
python demo/server.py
```

要求Python 3.10+和支持WebGL的浏览器。Python端只用标准库，无需安装研究依赖。打开`http://127.0.0.1:8788`；结束时Ctrl+C。Windows可在仓库根目录双击`start_demo.cmd`。

也可以下载[v0.1.0运行包](https://github.com/zhanghanhnu-coder/wind-window-research/releases/tag/v0.1.0)，解压后按相同方式运行。它是需要Python的源码运行包，不是免安装的Windows可执行程序。该历史版本包内文档可能早于main分支，最新说明以本仓库为准。

## 2. 执行公开检查

在仓库根目录运行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q demo research_code
```

当前6个单元检查覆盖已知超限优先、未知不等于安全、完整成功、相同边际不同联合窗口概率、容量上限和合成输出确定性。`compileall`只验证语法，不导入全部依赖，也不运行私有研究流程。[GitHub Actions](../.github/workflows/checks.yml)使用Python 3.11执行这两类检查。

## 3. 查看结果与图

- 五项主对照：[结果CSV](../results/aggregate_comparisons.csv)。每行的方向、指标和范围必须一起解读。
- 26组图：[中文图集](experiments.zh.md)，矢量原件在`figures/zh/`与`figures/en/`。
- 图名/图注元数据：[catalog.json](../figures/catalog.json)，不含原始图数据或证据路径。
- 本地直接打开SVG即可检查图示；浏览器缩放保持矢量清晰度。

## 4. 研究源码不是即装即跑的软件包

`research_code/`保存脱敏的研究脚本与算法实现，是供阅读和审查的存档。部分脚本仍依赖原私有项目中的包（例如`v3`）、缓存目录和冻结记录；本库没有打包完整私有运行环境。`research_code/requirements.txt`记录研究依赖，不表示安装它即可重现全部结果。

授权研究者需要先恢复原环境中的模块和私有输入契约，核对时间、机组顺序、精度、清单指纹和冻结状态，再确定适用脚本。不要批量执行所有`research_*.py`：其中有历史分支，可能写入缓存或采用旧标签版本。历史结果不能不加说明地与修正后的主结果合并。

## 5. 授权环境中的复评顺序

1. 在仓库外验收原始记录、字段语义、质量码和时间区间定义。
2. 保留原始float64真值、缺失掩码和源文件指纹，确认机组顺序及时间轴一致。
3. 恢复冻结的起报清单、训练/校准分界、模型输出和动作账本。
4. 先重建三值标签，核对已知超限优先规则；记录标签变化对应的物理目标。
5. 在预测与动作不变条件下，重算相同目标的评分和共享状态约束下的费用/服务界。
6. 报告补证前后范围、分母、季节组成和失败项；不将标签修正误说成模型改善。
7. 如需改模型或业务规则，建立新版本，再用未参与开发的独立年度确认。

这是一份验收流程，不是公开仓库可直接执行的命令链。公开数据需求见[后续计划](roadmap.zh.md)，具体机组/日期清单保留私下交付。

## 6. 修改后的最低核验

文档修改检查相对链接、中英口径与范围方向；演示修改运行6个单元检查并实际查看页面；研究算法修改还需要授权环境的对应验证器。不要把公开CI绿色解释成实证正确性或安全许可。

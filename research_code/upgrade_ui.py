"""Idempotent v2 UI extension; original UI is archived in artifacts/v1_source."""
from pathlib import Path
P=Path(__file__).resolve().parent/'static'
def run():
 p=P/'index.html';s=p.read_text(encoding='utf-8')
 if 'id="windowDetails"' not in s:
  s=s.replace('</nav>','<button data-tab="research">05 · 项目研究路线</button></nav>')
  s=s.replace('<option value="graph">时空图模型 ST-GNN</option>','<option value="graph24" selected>新版图候选（24h）</option><option value="temporal24">新版无图对照（24h）</option><option value="mean24">全场平均图对照（24h）</option><option value="random24">打乱关系对照（24h）</option><option value="graph">原图模型（9h）</option>')
  s=s.replace('不使用图的时序模型','原无图模型（9h）').replace('由训练期数据计算的关联，不是地理位置或天气传播路径。','<span id="graphDescription">当前模型实际使用的关系；不是地理位置。</span>').replace('每台机组选取 4 个相关邻居。点击节点可切换预测曲线。','点击节点可切换预测曲线。无图模型只连接自身；均匀图与打乱关系图用于检验图结构价值。')
  s=s.replace('<article class="panel"><div class="panel-title"><div><h2>连续窗口', '<article class="panel"><div class="panel-title"><div><h2>窗口能持续多久，要等多久</h2><small id="forecastStamp"></small></div><span class="pill">完整路径分析</span></div><div id="windowDetails"></div></article><article class="panel"><h2>逐台机组的数据情况</h2><p class="hint">预测曲线可供参考；近期数据不足的机组暂停自动安排，其任务保留待办。</p><details><summary id="qualitySummary">展开逐台检查</summary><div id="qualityTable"></div></details></article><article class="panel"><div class="panel-title"><div><h2>连续窗口')
  s=s.replace('<button id="planBtn" class="primary">计算安排</button>','<button id="planBtn" class="primary">计算安排</button><button id="compareBtn">同批任务比较四种方法</button>')
  s=s.replace('<div id="planResult">','<div id="compareResult"></div><div id="planResult">')
  s=s.replace('<button id="benchmarkBtn">查看固定对照实验</button>','<button id="benchmarkV2Btn">查看新版两个月对照</button><button id="benchmarkBtn">查看上一版固定对照</button>')
  s=s.replace('<footer>', '''<section class="tab" id="research"><article class="panel"><div class="eyebrow">研究主线 / 可检验的问题</div><h2>从气象信息到可持续作业窗口，再到维修决策</h2><p>子项目对应母项目的运维资源调配部分。上游提供维修需求、优先级和截止时间；本项目提供窗口分析与安排；下游接收任务执行和等待情况。当前用明确的模拟工单连接这些接口。</p><div class="researchSteps"><span>① 数据口径与质量</span><b>→</b><span>② 机组联合预测</span><b>→</b><span>③ 连续窗口与等待</span><b>→</b><span>④ 资源安排与回放</span></div><div class="notice">新增功能不等于效果提高。新版候选和原模型并列保留；当前尚不能证明相关性图有独立的额外价值。</div></article><article class="panel"><h2>三项研究问题与验收方式</h2><div id="questionTable"></div></article><article class="panel"><h2>新结构消融：改变图关系，其他条件保持一致</h2><p class="hint">4种关系 × 3组初始化 = 12次训练；1—8月训练，9月选择轮数，10月情景库，11—12月探索性评估。主指标为阈值10、连续6小时、出发偏移0/24/48/66合并的Brier，越低越好。</p><div id="ablationTable"></div><div id="pairedTable"></div><p class="hint">区间按9个周块重抽样2,000次，先在每个起点内对3个种子求平均；不是把机组或种子当作独立天气样本。</p></article><article class="panel"><h2>新版与原版、简单基线的共同样本比较</h2><p class="hint">以下固定种子20241101；用于发现代价，不按结果重新挑选种子。</p><div id="contextTable"></div></article><article class="panel"><h2>研究工作包与完成条件</h2><div id="roadmapTable"></div></article></section><footer>''')
  s=s.replace('<script src="/app.js"></script>','<script src="/app.js"></script><script src="/research.js"></script>')
  s=s.replace('2024 实测数据 · 本地运行','V2 · 2024 实测 · 本地运行')
  p.write_text(s,encoding='utf-8')
 p=P/'app.js';s=p.read_text(encoding='utf-8')
 if 'drawWindowDetails' not in s:
  s=s.replace('drawGraph(j)}',"drawGraph(j);if(typeof drawWindowDetails==='function')drawWindowDetails(j)}")
  s=s.replace('meta?.scope,forecast:',"meta?.scope,version:'2.0',comparison:typeof compareData==='undefined'?null:compareData,forecast:")
  s=s.replace('evidence();replayHint();await updateForecast()',"evidence();if(typeof renderResearch==='function')renderResearch();replayHint();await updateForecast()")
  s=s.replace("${esc(names[cfg.policy])} · 色块", "${esc(names[cfg.policy])} · 风速阈值 ${c.threshold} · 最低比例 ${fmt(100*cfg.minimum_probability,0)}% · 风险权重 ${cfg.risk_weight} · 色块")
  s=s.replace("function benchmark(){const r=meta.replay_metrics;", "function benchmark(){const r=meta.replay_metrics;")
  s=s.replace('固定对照：2024 年 11 月，连续 30 天','上一版固定对照：2024 年 11 月，连续 30 天（不含新版质量门控）')
  p.write_text(s,encoding='utf-8')

if __name__=='__main__':run()

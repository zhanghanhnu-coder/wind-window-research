"""Independent artifact consistency audit and descriptive closure supplements."""
from pathlib import Path
import json,hashlib,subprocess,sys
import numpy as np
import pandas as pd
from precision_truth_core import load_truth
from closure_robustness import ROOT,OLD,OUT,NOTE,CONTRASTS,sha,read,write,csv

def main():
 r=pd.read_csv(OUT/'bounds_summary.csv');v=read(OUT/'verification.json');assert v['status']=='PASS' and len(r)==1120
 for p,h in read(OUT/'source_identity.json').items():assert sha(p)==h,p
 green=subprocess.run([sys.executable,str(ROOT/'closure_robustness_test.py')],capture_output=True,text=True)
 assert green.returncode==0;write(OUT/'tests_green.json',dict(status='PASS',command='python closure_robustness_test.py',exit_code=green.returncode,stdout=green.stdout,stderr=green.stderr))
 extra=subprocess.run([sys.executable,str(ROOT/'closure_robustness_nested_test.py')],capture_output=True,text=True)
 assert extra.returncode==0;write(OUT/'tests_nested_green.json',dict(status='PASS',command='python closure_robustness_nested_test.py',exit_code=extra.returncode,stdout=extra.stdout,stderr=extra.stderr,source_sha=sha(ROOT/'closure_robustness_nested_test.py')))
 maxerr=0.;maxgap=0.
 for row in r.itertuples():
  cert=read(OUT/'certificates'/f'{row.scope}__{row.estimand}__{row.evidence}__{row.comparison}.json')
  for side in ['lower','upper']:
   assert abs(cert[side]-getattr(row,side))<1e-14
   x=cert['optimization'][side+'_certificate'];assert x['status']==0
   err=abs(x['direct_four_square_weather_value']-getattr(row,side));maxerr=max(maxerr,err)
   maxgap=max(maxgap,x['absolute_gap']);assert err<1e-9 and x['absolute_gap']<1e-7
 impact=[]
 for keys,g in r[r.kind.isin(['leave_issue','leave_fold','seed'])].groupby(['evidence','estimand','kind','comparison']):
  impact.append(dict(zip(['evidence','estimand','kind','comparison'],keys))|dict(cases=len(g),lower_min=float(g.lower.min()),lower_max=float(g.lower.max()),upper_min=float(g.upper.min()),upper_max=float(g.upper.max()),crosses_zero=int(((g.lower<=0)&(g.upper>=0)).sum()),strictly_positive=int((g.lower>0).sum()),strictly_negative=int((g.upper<0).sum())))
 csv('influence_summary.csv',impact)
 # Verify M09 contribution/decomposition without assuming unrelated optimized endpoints add.
 d=pd.read_csv(OLD/'matched_windows.csv');d['event']=[f'{int(x.issue+x.start+1)}:{int(x.node)}:{int(x.duration)}' for x in d.itertuples()]
 shared_events=int((d[['event','fold','issue']].drop_duplicates().groupby('event').size()>1).sum());assert shared_events==0
 hourly_sets={f:set((h,int(x.node)) for x in g.itertuples() for h in range(int(x.issue+x.start+1),int(x.issue+x.start+x.duration+1))) for f,g in d.groupby('fold')}
 from itertools import combinations
 assert all(not hourly_sets[a]&hourly_sets[b] for a,b in combinations(hourly_sets,2))
 decomposition=[]
 for (evidence,estimand,comparison),g in r.groupby(['evidence','estimand','comparison']):
  overall=g[g.scope=='ALL'].iloc[0];folds=g[g.kind=='fold'];sums={side:0. for side in ['lower','upper']}
  for x in folds.itertuples():
   weight=(x.origins/overall.origins) if estimand=='equal_origin' else x.physical_events/overall.physical_events
   decomposition.append(dict(evidence=evidence,estimand=estimand,comparison=comparison,fold=x.removed_or_selected,weight=weight,weighted_lower=weight*x.lower,weighted_upper=weight*x.upper))
   for side in sums:sums[side]+=weight*getattr(x,side)
  for side in sums:assert abs(sums[side]-overall[side])<1e-12
 csv('fold_contributions.csv',decomposition)
 # Raw-hour observation rates differ from known binary-window labels.
 times,nodes,raw=load_truth();cfg=read(ROOT/'artifacts/review_iteration2_v1/analog_execution/protocol.json')['config'];tasknodes=sorted({int(t['node']) for t in cfg['tasks']})
 hourly=[]
 for (f,i),g in d.groupby(['fold','issue']):
  full=raw[int(i)+1:int(i)+73,tasknodes];hs={(h,int(x.node)) for x in g.itertuples() for h in range(int(x.issue+x.start+1),int(x.issue+x.start+x.duration+1))}
  hourly.append(dict(fold=f,issue=i,complete=bool(g.complete.iloc[0]),full72_tasknode_hour_cells=full.size,full72_observed_hour_cells=int(np.isfinite(full).sum()),matched_distinct_hour_cells=len(hs),matched_observed_hour_cells=sum(np.isfinite(raw[h,n]) for h,n in hs)))
 csv('selection_hour_observation_by_origin.csv',hourly)
 hours=[]
 for complete,g in pd.DataFrame(hourly).groupby('complete'):
  hours.append(dict(group='complete15' if complete else 'other29',origins=len(g),full72_observed_fraction=g.full72_observed_hour_cells.sum()/g.full72_tasknode_hour_cells.sum(),matched_observed_fraction=g.matched_observed_hour_cells.sum()/g.matched_distinct_hour_cells.sum()))
 csv('selection_hour_observation_summary.csv',hours)
 q=pd.DataFrame(impact);chosen=q[(q.comparison=='interaction')&(q.kind.isin(['leave_issue','leave_fold','seed']))].copy()
 supplement='\n\n## 全量影响诊断补充核验\n\n'+chosen.to_markdown(index=False,floatfmt='.8f')+'\n\n'+pd.DataFrame(hours).to_markdown(index=False,floatfmt='.8f')
 supplement+='\n\n小时观测率与已知窗口标签率分开报告：已有超限可使缺小时窗口标签已知。full72按每起报72h×原任务节点统计；matched按每起报当前队列的不同(hour,node)统计，起报之间不声称独立。物理事件跨起报重复数实际为0；四fold目标小时集合两两不交，全部5对比×2权重×2证据的总体端点均独立核验可按fold权重精确分解。fold_contributions.csv给出M09及其他月份贡献。\n'
 text=NOTE.read_text(encoding='utf-8');marker='\n\n## 全量影响诊断补充核验';text=text.split(marker)[0]+supplement
 text=text.replace('## 月份与单一起报影响','## 时间fold与单一起报影响').replace('四月份没有共享目标小时','四个fold没有共享目标小时').replace('删整个月份是不同扰动尺度','删整个fold是不同扰动尺度')
 text+='\nfold名称不是单一日历月份：M05/M07/M09/M11分别包含5–6/7–8/9–10/11–12月的起报；selection_summary.csv按真实日历月独立报告完整性构成。因此“删M09”指删9–10月fold，而非只删9月。\n'
 text+='\n五对比完整影响汇总见influence_summary.csv。原raw等起报下，两模型更新差在44留起报、4留fold、3种子中均严格负；当前模型差始终跨0；初始模型差仅删M09后跨0，其余留fold、全部留起报与种子均严格负。交互仅删M09后跨0，其余留fold、全部44留起报与三种子均严格正。物理事件权重和partial证据的对应分类一致，逐case端点均保留。\n'
 text+='\n额外嵌套窗口数学测试也通过：成功较长窗口必然成功其子窗口；独立窗口标签给出的虚假极端被共享AND排除，与穷举一致（tests_nested_green.json）。\n'
 NOTE.write_text(text,encoding='utf-8');(OUT/'report.md').write_text(text,encoding='utf-8')
 write(OUT/'independent_artifact_audit.json',dict(status='PASS',certificates=1120,endpoints=2240,max_csv_certificate_and_direct_error=maxerr,max_absolute_mip_gap=maxgap,shared_events_across_issues=shared_events,fold_hour_sets_disjoint=True,fold_decomposition_cases=20,finalizer_source_sha=sha(__file__),old_source_hashes_rechecked=True))
 manifest={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='final_manifest.json'};manifest[str(NOTE)]=sha(NOTE)
 write(OUT/'final_manifest.json',dict(status='PASS',files=manifest))
 print(chosen.to_string(index=False));print(pd.DataFrame(hours).to_string(index=False))

if __name__=='__main__':main()

"""Round40 historical common-path availability, not a forecasting experiment."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
from seasonal_probability_core import FOLDS
from history_availability_core import complete_history_paths,max_disjoint
from research_crps_distribution import guard

OUT=ART/'history_availability_v1';PARENT=ART/'seasonal_probability_v1';PLAN=ART.parent.parent/'研究笔记/历史完整天气片段可用性方案_2026-11-02.md'
GROUPS={'all17':list(range(17)),'tasks6':[0,3,6,9,12,16],'group_a':[0,1,2],'group_b':[8,9,10],'group_c':[13,15,16]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def specifications():return [(g,n,h,d) for g,n in GROUPS.items() for h in [6,12,24,72] for d in [28,56,112]]+[(f'node_{n:02d}',[n],72,112) for n in range(17)]
def run():
 guard();assert not (OUT/'final_manifest.json').exists();assert not (OUT/'records.csv').exists(),'preserve original diagnostic';manifest=read(PARENT/'final_manifest.json')
 for file,h in manifest.items():assert sha(file)==h,file
 OUT.mkdir(exist_ok=True);(OUT/'protocol_frozen.md').write_text(PLAN.read_text(encoding='utf-8').split('## 执行记录')[0],encoding='utf-8')
 files=[Path(__file__),OUT/'protocol_frozen.md',Path(__file__).parent/'history_availability_core.py',Path(__file__).parent/'tests/test_history_availability.py',Path(__file__).parent/'verify_history_availability.py',PARENT/'final_manifest.json',ART/'data.npz']+[PARENT/fold/'data.npz' for fold in FOLDS];identity={str(p):sha(p) for p in files};js(OUT/'identity.json',identity);js(OUT/'recipe.json',dict(groups=GROUPS,horizons=[6,12,24,72],lookback_days=[28,56,112],single_nodes_horizon=72,single_nodes_lookback_days=112,grid_offset=23,grid_stride=6,maturity='j+H<i',whole_lookback='j+1>=i-24D',expected_records=67760,scope='historical validity mask only; no forecasting or savings evaluation',started_utc=datetime.now(timezone.utc).isoformat()))
 raw=np.load(ART/'data.npz')['raw'];valid=np.isfinite(raw);rows=[];future_checks=0
 for fold in FOLDS:
  ids=np.load(PARENT/fold/'data.npz')['test_ids'];assert len(np.unique(ids))==len(ids)
  for index,issue in enumerate(ids):
   guard();issue=int(issue)
   for group,nodes,h,d in specifications():
    complete=complete_history_paths(valid,issue,h,d*24,nodes);candidate=np.arange(23,issue-h,6);candidate=candidate[candidate+1>=issue-d*24]
    rows.append(dict(fold=fold,issue_id=issue,group=group,node_count=len(nodes),horizon=h,lookback_days=d,candidate_count=len(candidate),complete_count=len(complete),complete_fraction=len(complete)/len(candidate) if len(candidate) else np.nan,maximum_disjoint=max_disjoint(complete,h),at_least19=len(complete)>=19,at_least57=len(complete)>=57,primary=group=='all17' and h==72 and d==112))
    if index in [0,len(ids)//2,len(ids)-1]:
     polluted=valid.copy();polluted[issue:]=~polluted[issue:];np.testing.assert_array_equal(complete,complete_history_paths(polluted,issue,h,d*24,nodes));future_checks+=1
  print('COUNTED',fold,len(ids),len(rows),flush=True)
 assert len(rows)==67760 and len({r['issue_id'] for r in rows})==880 and future_checks==924;df=pd.DataFrame(rows);df.to_csv(OUT/'records.csv',index=False,encoding='utf-8-sig');summary=[]
 for scope in ['ALL',*FOLDS]:
  selected=df if scope=='ALL' else df[df.fold==scope]
  for (group,h,d),g in selected.groupby(['group','horizon','lookback_days'],sort=False):
   summary.append(dict(scope=scope,group=group,node_count=int(g.node_count.iloc[0]),horizon=int(h),lookback_days=int(d),n=len(g),complete_min=int(g.complete_count.min()),complete_q25=float(g.complete_count.quantile(.25)),complete_median=float(g.complete_count.median()),complete_q75=float(g.complete_count.quantile(.75)),complete_max=int(g.complete_count.max()),fraction_at_least19=float(g.at_least19.mean()),fraction_at_least57=float(g.at_least57.mean()),zero_fraction=float((g.complete_count==0).mean()),disjoint_min=int(g.maximum_disjoint.min()),disjoint_median=float(g.maximum_disjoint.median()),disjoint_max=int(g.maximum_disjoint.max()),primary=group=='all17' and h==72 and d==112 and scope=='ALL'))
 assert len(summary)==385;pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
 for file,h in identity.items():assert sha(file)==h,file
 for file,h in manifest.items():assert sha(file)==h,file
 js(OUT/'complete.json',dict(status='COMPLETE',records=67760,summary_rows=385,unique_origins=880,future_pollution_checks=924,source_files_unchanged=len(identity),parent_files_unchanged=len(manifest),utc=datetime.now(timezone.utc).isoformat()));print('COMPLETE',flush=True)

if __name__=='__main__':run()

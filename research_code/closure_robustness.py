"""Closure-only sensitivity of frozen matched Brier contrasts; no model fitting."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import json,hashlib,argparse
import numpy as np
import pandas as pd
from iteration3_matched_core import CELLS,CONTRASTS,matched
from iteration2_bounds_core import window_term,solve_bounds
from precision_truth_core import load_truth,SOURCE

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'artifacts/review_iteration3_v1/matched_updates'
OUT=ROOT/'artifacts/review_closure_v1/update_robustness'
NOTE=ROOT.parent/'研究笔记/阶段收尾_2026-11-02/更新诊断稳健性.md'
FOLDS=['M05','M07','M09','M11']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(name,data):pd.DataFrame(data).to_csv(OUT/name,index=False,encoding='utf-8-sig')

def aggregation_weights(d,estimand):
 """Event -> issue opportunity -> seed -> duplicate task rows; square first."""
 if estimand=='equal_origin':
  counts=d.groupby(['fold','issue','seed']).issue.transform('size').to_numpy()
  seeds=d.groupby(['fold','issue']).seed.transform('nunique').to_numpy()
  return 1/(counts*seeds*len(d[['fold','issue']].drop_duplicates()))
 event_count=d.event.nunique()
 opportunities=d[['event','fold','issue']].drop_duplicates().groupby('event').size()
 seeds=d.groupby(['event','fold','issue']).seed.transform('nunique').to_numpy()
 duplicates=d.groupby(['event','fold','issue','seed']).seed.transform('size').to_numpy()
 return 1/(event_count*d.event.map(opportunities).to_numpy()*seeds*duplicates)

def partial_terms(raw,bad):
 assert raw.dtype==np.float64 and bad.shape==raw.shape
 # No partial value is interpreted as safe; measured known raw64 stays authoritative.
 out=raw.copy();fix=(~np.isfinite(raw))&bad.astype(bool);out[fix]=11.
 assert np.array_equal(out[np.isfinite(raw)],raw[np.isfinite(raw)])
 return out,int(fix.sum())

def evaluate_bounds(d,w,sign,terms,raw):
 p=d[CELLS].to_numpy();v=np.asarray(sign);constant=float(w@(p*p@v));b=(1-2*p)@v
 coefficients=defaultdict(float)
 for key,coef in zip(terms,w*b):
  if key is True:constant+=float(coef)
  elif key is not False:coefficients[key]+=float(coef)
 result=solve_bounds(constant,coefficients);maxerr=0.
 for side in ['lower','upper']:
  witness={(h,n):s for h,n,s in result[side+'_witness']}
  # Reconstruct each target from one completed physical weather, then four squares.
  values=raw.copy()
  for (h,n),s in witness.items():values[h,n]=0. if s else 11.
  values[~np.isfinite(values)]=11.
  y=np.array([float((values[int(r.issue+r.start+1):int(r.issue+r.start+r.duration+1),int(r.node)]<=10).all()) for r in d.itertuples()])
  actual=float(w@(((p-y[:,None])**2)@v));err=abs(actual-result[side]);maxerr=max(maxerr,err)
  assert err<1e-9
  result[side+'_certificate']['direct_four_square_weather_value']=actual
 return result,maxerr

def freeze():
 OUT.mkdir(parents=True,exist_ok=True)
 protocol=dict(status='FROZEN_BEFORE_CLOSURE_EVALUATION',created_utc=datetime.now(timezone.utc).isoformat(),
  comparisons=CONTRASTS,probabilities='same 36600 frozen probabilities, 44 issues, 3 ranks; each original cell square precedes aggregation',
  cases=['ALL','leave each of 4 folds','leave each of 44 issues','each of 3 seeds','each fold descriptive'],
  estimands=dict(equal_origin='Windows equally within issue/seed; seeds equally within issue; issues equally. In single-seed runs only that seed is retained.',
   physical_event='Unique event=(absolute first target hour issue+start+1,node,duration). Equal events; within event equal retained (fold,issue) forecast opportunities; within opportunity equal retained seeds; within seed equal duplicate task mappings. Four probabilities remain separate until after squares; cross-issue forecasts are never averaged before scoring. After deletions all weights are normalized over remaining support. This changes the target estimand, not an unbiased correction to equal-origin.'),
  evidence=['raw64','partial_bad_only'],partial_rule='Only unknown raw64 hour,node with existing known_bad=True is fixed failed; no safe imputation. Existing raw64 observations unchanged. Other unknown states unrestricted. Feasible set is raw64 set intersect {h=0 for known_bad unknown hours}.',
  inference='Conditional identification bounds, not confidence intervals; seeds are rank sensitivity on same weather, not independent weather; issue/fold deletion is influence, not external CV.',
  selection='Complete15 vs other29: descriptive counts, months, window availability, known-label fraction only; no MNAR probability model.')
 if (OUT/'protocol.json').exists():
  protocol['created_utc']=read(OUT/'protocol.json')['created_utc'];assert read(OUT/'protocol.json')==protocol
 else:write(OUT/'protocol.json',protocol)
 files=list(OLD.rglob('*'));files=[p for p in files if p.is_file()]
 files+=list(map(Path,read(OLD/'source_identity.json')))
 files+=[ROOT/'closure_robustness.py',ROOT/'closure_robustness_test.py']
 identity={str(p):sha(p) for p in dict.fromkeys(files)}
 if (OUT/'source_identity.json').exists():assert identity==read(OUT/'source_identity.json')
 else:write(OUT/'source_identity.json',identity)
 write(OUT/'tests_red.json',dict(status='EXPECTED_FAIL_OBSERVED',command='python closure_robustness_test.py',exit_code=1,error="ModuleNotFoundError: No module named 'closure_robustness'",scope='Three mathematical examples before real data evaluation'))
 print('FROZEN',len(identity),flush=True)

def run():
 identity=read(OUT/'source_identity.json')
 for p,h in identity.items():assert sha(p)==h,p
 times,nodes,raw=load_truth();bad=np.load(ROOT/'artifacts/review_iteration2_v1/data_gaps/partial_hour_evidence.npz')['known_bad'];assert int(bad.sum())==494
 partial,nfix=partial_terms(raw,bad)
 d=pd.read_csv(OLD/'matched_windows.csv')
 ledger=pd.read_csv(ROOT/'artifacts/review_iteration2_v1/analog_execution/support_window_ledger.csv');recon=matched(ledger)
 pd.testing.assert_frame_equal(d[recon.columns],recon,check_dtype=False,check_exact=False,atol=1e-14,rtol=0)
 from iteration3_matched_run import audit_sources
 cfg=read(ROOT/'artifacts/review_iteration2_v1/analog_execution/protocol.json')['config']
 audit=audit_sources(d,raw,cfg)
 assert len(d)==9150 and len(d)*4==36600 and d[['fold','issue']].drop_duplicates().shape[0]==44
 d['event']=[f'{int(r.issue+r.start+1)}:{int(r.node)}:{int(r.duration)}' for r in d.itertuples()]
 d['opportunity']=[f'{r.event}:{r.fold}:{r.issue}' for r in d.itertuples()]
 d['month']=[str(times[int(i)].tz_convert('Asia/Shanghai').strftime('%Y-%m')) for i in d.issue]
 terms={key:[window_term(arr,int(r.issue),int(r.start),int(r.start+r.duration),int(r.node)) for r in d.itertuples()] for key,arr in [('raw64',raw),('partial_bad_only',partial)]}
 # Every partial term either retains its original identity or becomes known failure.
 changed=[]
 for j,(a,b) in enumerate(zip(terms['raw64'],terms['partial_bad_only'])):
  assert a==b or (a is not True and a is not False and b is False)
  if a!=b:changed.append(j)
 conversion=dict(known_bad_total=int(bad.sum()),unknown_hours_fixed_failed=nfix,unknown_to_known_seed_window_rows=len(changed),unknown_to_known_four_cell_rows=4*len(changed),unknown_to_known_forecast_opportunities=int(d.iloc[changed].opportunity.nunique()),unknown_to_known_unique_physical_windows=int(d.iloc[changed].event.nunique()),feasible_set_nested=True,proof='partial feasible set = raw64 feasible set intersect all fixed unknown bad-hour states=0; no unknown becomes success, no known raw64 changes')
 write(OUT/'partial_evidence_nesting.json',conversion)
 csv('partial_changed_windows.csv',d.iloc[changed])
 scopes=[('ALL','all',None)]+[(f'leave_fold_{f}','leave_fold',f) for f in FOLDS]+[(f'leave_issue_{i}','leave_issue',int(i)) for i in d.issue.drop_duplicates()]+[(f'seed_{s}','seed',int(s)) for s in sorted(d.seed.unique())]+[(f'fold_{f}','fold',f) for f in FOLDS]
 rows=[];supports=[];maxerr=0.;nested=0
 for scope,kind,value in scopes:
  mask=np.ones(len(d),dtype=bool)
  if kind=='leave_fold':mask=d.fold.ne(value).to_numpy()
  if kind=='leave_issue':mask=d.issue.ne(value).to_numpy()
  if kind=='seed':mask=d.seed.eq(value).to_numpy()
  if kind=='fold':mask=d.fold.eq(value).to_numpy()
  sub=d.loc[mask].copy();ix=np.flatnonzero(mask)
  for estimand in ['equal_origin','physical_event']:
   w=aggregation_weights(sub,estimand);assert abs(w.sum()-1)<1e-12
   if estimand=='physical_event':assert np.max(np.abs(pd.Series(w,index=sub.index).groupby(sub.event).sum().to_numpy()-1/sub.event.nunique()))<1e-12
   else:assert np.max(np.abs(pd.Series(w,index=sub.index).groupby([sub.fold,sub.issue]).sum().to_numpy()-1/len(sub[['fold','issue']].drop_duplicates())))<1e-12
   base=dict(scope=scope,kind=kind,removed_or_selected='' if value is None else value,estimand=estimand,origins=len(sub[['fold','issue']].drop_duplicates()),seed_window_rows=len(sub),physical_events=sub.event.nunique(),forecast_opportunities=sub.opportunity.nunique(),weight_sum=float(w.sum()))
   supports.append(base)
   for evidence,weather in [('raw64',raw),('partial_bad_only',partial)]:
    tt=[terms[evidence][j] for j in ix]
    for name,sign in CONTRASTS.items():
     result,err=evaluate_bounds(sub,w,sign,tt,weather);maxerr=max(maxerr,err)
     row=dict(**base,evidence=evidence,comparison=name,lower=result['lower'],upper=result['upper'],unknown_hour_variables=result['unknown_hours'],unknown_window_terms=result['window_terms'],known_seed_window_rows=sum(k is True or k is False for k in tt),certificate='PASS')
     rows.append(row);write(OUT/'certificates'/f'{scope}__{estimand}__{evidence}__{name}.json',dict(**row,optimization=result))
   for name in CONTRASTS:
    a=next(r for r in rows[-10:] if r['evidence']=='raw64' and r['comparison']==name);b=next(r for r in rows[-10:] if r['evidence']=='partial_bad_only' and r['comparison']==name)
    assert a['lower']-1e-9<=b['lower']<=b['upper']+1e-9 and b['upper']<=a['upper']+1e-9;nested+=1
  print('DONE',scope,flush=True)
 result=pd.DataFrame(rows);csv('bounds_summary.csv',result);csv('case_support.csv',supports)
 # Equal-origin ALL exactly reproduces original five scores, including both endpoints.
 old=pd.read_csv(OLD/'bounds_summary.csv').query("population=='all44_primary' and scope=='ALL' and start_group=='ALL'")
 reproduce=[]
 for name in CONTRASTS:
  a=old[old.comparison==name].iloc[0];b=result.query("scope=='ALL' and estimand=='equal_origin' and evidence=='raw64' and comparison==@name").iloc[0]
  for side in ['lower','upper']:
   err=abs(float(a['shared_'+side])-float(b[side]));assert err<1e-12;reproduce.append(dict(comparison=name,side=side,absolute_error=err))
 csv('original_all_reproduction.csv',reproduce)
 # Completeness selection: different retained queue and observed-label coverage, no missingness model.
 origin=[]
 for (f,i),g in d.groupby(['fold','issue']):
  known=g.label.notna();events=g.drop_duplicates('event');rterms=[terms['raw64'][j] for j in g.index];pterms=[terms['partial_bad_only'][j] for j in g.index]
  origin.append(dict(fold=f,issue=int(i),month=g.month.iloc[0],complete=bool(g.complete.iloc[0]),seed_window_rows=len(g),forecast_windows_per_seed=len(g)/3,physical_events=g.event.nunique(),known_seed_window_rows=int(known.sum()),raw_known_fraction=float(known.mean()),partial_known_fraction=sum(t is True or t is False for t in pterms)/len(g),available_starts=g.start.nunique(),event_known_fraction=float(events.label.notna().mean())))
 origins=pd.DataFrame(origin);csv('selection_by_origin.csv',origins)
 selection=[]
 for month in ['ALL']+sorted(origins.month.unique()):
  for complete in [True,False]:
   g=origins[(origins.complete==complete)&((origins.month==month) if month!='ALL' else True)]
   if g.empty:continue
   selection.append(dict(month=month,group='complete15' if complete else 'other29',origins=len(g),seed_window_rows=int(g.seed_window_rows.sum()),available_windows_per_seed=int(g.forecast_windows_per_seed.sum()),available_windows_min=float(g.forecast_windows_per_seed.min()),available_windows_max=float(g.forecast_windows_per_seed.max()),known_seed_window_rows=int(g.known_seed_window_rows.sum()),raw_known_fraction=float(g.known_seed_window_rows.sum()/g.seed_window_rows.sum()),origin_mean_known_fraction=float(g.raw_known_fraction.mean())))
 csv('selection_summary.csv',selection)
 for p,h in identity.items():assert sha(p)==h,p
 write(OUT/'verification.json',dict(status='PASS',**audit,certificates=len(rows),endpoint_checks=2*len(rows),max_direct_endpoint_error=maxerr,nesting_checks=nested,source_hashes_unchanged=len(identity),original_all_checks=len(reproduce),physical_weight_checks=len(scopes),old_inputs_readonly=True,**conversion))
 report(result,origins,pd.DataFrame(selection),conversion,maxerr,len(rows))

def report(r,o,s,conversion,err,ncert):
 def table(frame,cols):
  x=frame[cols].copy()
  for c in x.select_dtypes('float'):x[c]=x[c].map(lambda v:f'{v:.8f}')
  return x.to_markdown(index=False)
 q=r.query("evidence=='raw64' and estimand=='equal_origin'")
 interaction=q.query("comparison=='interaction'");loo=interaction.query("kind=='leave_issue'");lf=interaction.query("kind=='leave_fold'")
 excluded=lf[lf.removed_or_selected.astype(str)=='M09'].iloc[0]
 full=interaction[interaction.scope=='ALL'].iloc[0];m09=interaction[interaction.scope=='fold_M09'].iloc[0]
 lines=['# 阶段收尾：同窗预报更新诊断稳健性','',
 '冻结既有36,600概率、44起报×3秩、原raw64与共享小时AND界。主要对象仍为原ALL等起报交互；本轮所有新增分析均为事后稳健性/影响诊断，不增加确认性主检验。原三轮文件全部只读；未训练、校准、调参或改动作。',
 '', '5对比为交互、初始模型差、当前模型差、Analog更新差、Graph更新差。每个case的四格和全部种子共享同一物理天气；各对比各自优化。端点见证分别填回共同天气，独立逐格平方重算。上下界为条件识别范围，不能解释为置信区间。',
 '', '## 权重与目标量',
 '原等起报：起报/种子内窗口等权，起报内三个种子等权，最后起报等权。单种子case仅保留该种子，仍等起报。',
 '物理目标等权：事件key=(issue+start+1绝对首小时,node,duration)。每独特事件等权；事件内跨起报预测机会(fold,issue)等权；机会内种子等权；同机会同种子内如有多task映射同事件则等权。四格先各自平方再汇总，绝不先平均概率。删除后按剩余支持重归一。这与原等起报是不同estimand；不同起报对同一事件的预报保留，不能把重复目标误当独立天气。',
 table(r.query("scope=='ALL' and evidence=='raw64' and comparison=='interaction'"),['estimand','origins','physical_events','forecast_opportunities','seed_window_rows','weight_sum']),
 '', '## 原ALL复现及完整清单',table(q.query("scope=='ALL'"),['comparison','lower','upper']),
 '原5对比上下端点全部复现误差<1e-12。冻结清单含ALL、4留fold、44留起报、3单种子、4个fold描述，全部分别运行2权重×2证据×5对比；全量CSV与证书保留，不按结果择选。',
 '', '## 月份与单一起报影响',table(lf,['scope','lower','upper']),
 f'M09自身交互范围[{m09.lower:.8f}, {m09.upper:.8f}]；原等起报权重为1/4，因此其加权贡献范围[{m09.lower/4:.8f}, {m09.upper/4:.8f}]。四月份没有共享目标小时，整体端点可按各fold的1/4分解；删M09后三fold范围[{excluded.lower:.8f}, {excluded.upper:.8f}]，跨0。',
 f'44个逐一留起报：下界范围[{loo.lower.min():.8f}, {loo.lower.max():.8f}]；上界范围[{loo.upper.min():.8f}, {loo.upper.max():.8f}]；跨0案例{int(((loo.lower<=0)&(loo.upper>=0)).sum())}个。见bounds_summary.csv逐起报全表。',
 table(loo.sort_values('lower').head(3),['scope','lower','upper']),
 '留起报/留fold是固定开发起报的影响诊断，不是外部交叉验证；未对月份或起报作概率抽样推断。',
 '', '## 秩模板敏感性与物理目标权重',table(interaction.query("kind=='seed'"),['scope','lower','upper']),
 '三套秩使用相同天气与预测资料，不能当成三次独立天气实验；其范围是算法秩敏感性。',
 table(r.query("scope=='ALL' and evidence=='raw64'"),['estimand','comparison','lower','upper']),
 '', '## 既有partial证据与可行集嵌套',
 f'494个known_bad中，{conversion["unknown_hours_fixed_failed"]}个原未知小时固定失败；不填任何安全小时，不改变原raw64已知小时。主队列未知→已知：{conversion["unknown_to_known_seed_window_rows"]}个含种子窗口行、{conversion["unknown_to_known_four_cell_rows"]}个单格行、{conversion["unknown_to_known_forecast_opportunities"]}个预报机会、{conversion["unknown_to_known_unique_physical_windows"]}个独特物理窗口。',
 'partial可行集=raw64可行集∩{已有有效超限未知小时状态=失败}，故严格嵌套。每个case/权重/对比均验证raw下界≤partial下界≤partial上界≤raw上界。raw与partial分表列出，partial不替代主分析。',
 table(r.query("scope=='ALL' and comparison=='interaction'"),['estimand','evidence','lower','upper']),
 '', '## 完整15与其余29：描述性缺测选择',table(s,['month','group','origins','available_windows_per_seed','available_windows_min','available_windows_max','raw_known_fraction']),
 '完整资格来自原72h任务节点完整性，而非按本轮结果筛选。窗口可用量由冻结质量门控决定；已知标签包含已有超限导致的确定失败，并非全部小时已观测。两组月份构成、可用窗口和已知率不同，完整15不能代替44；不拟合无依据的MNAR概率模型。selection_by_origin.csv保留逐起报描述。',
 '', '## 验证与复现',f'先真实执行缺模块RED，再通过3项小例GREEN（物理目标/机会权重、partial只固定未知坏小时、穷举共享AND端点与四格平方）。独立重建36,600概率、9,150标签与1,584质量门控；{ncert}份端点证书全部通过，直接重算最大误差{err:.3g}。source_identity.json在运行前后核验旧源与旧artifacts不变。',
 '运行：closure_robustness_test.py；closure_robustness.py freeze；closure_robustness.py run。证书、协议、源SHA、完整CSV、verification均位于artifacts/review_closure_v1/update_robustness。',
 '', '整体正交互明显受M09驱动；只删一个起报的影响与删整个月份是不同扰动尺度。该结论只适用于冻结预测、门控队列与所允许的缺测天气补全，尚不能外推到真实海况、船舶约束、维修结果或收益。']
 text='\n\n'.join(lines)+'\n';NOTE.parent.mkdir(parents=True,exist_ok=True);NOTE.write_text(text,encoding='utf-8');(OUT/'report.md').write_text(text,encoding='utf-8')
 print(q.query("scope=='ALL' or (comparison=='interaction' and kind=='leave_fold')").to_string(index=False),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run']);a=p.parse_args();globals()[a.stage]()

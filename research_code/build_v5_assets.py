"""Build only the prespecified 48-case Demo subset from finalized research chains."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from model import ART
BASE=ART.parent.parent;FOLDS=['M05','M07','M09','M11'];SEEDS=[20241101,20241102,20241103];G='gated_graph_atoms';O='gated_temporal_atoms'
PARENTS={25:'seasonal_probability_v1',26:'partial_copula_v1',27:'seasonal_rolling_v1',29:'probability_pooling_v1',30:'window_pooling_v1',31:'missing_window_bounds_v1',32:'multiscale_context_v1',33:'distributional_schedule_v1',35:'rolling_copula_v1'}
MODELS=[dict(id='sync_raw',label='同步图 · 原始参考',round=27,history_hours=24,graph_role='预测网络的训练期同步图'),dict(id='sync_fixed',label='同步图 · 前月固定校准',round=27,history_hours=24,graph_role='预测网络的训练期同步图'),dict(id='sync_rolling',label='同步图 · 每周校准',round=27,history_hours=24,graph_role='预测网络的训练期同步图'),dict(id='own_rolling',label='无邻居 · 每周校准对照',round=27,history_hours=24,graph_role='预测网络不聚合邻居；联合天气仍有时空相关'),dict(id='window_pool',label='同步图 · 兼顾窗口组合',round=30,history_hours=24,graph_role='四个专家共享同一同步图网络'),dict(id='long7_raw',label='7天背景同步图 · 原始',round=32,history_hours=168,graph_role='共享编码器的训练期同步图'),dict(id='long7_rolling',label='7天背景同步图 · 每周校准',round=32,history_hours=168,graph_role='共享编码器的训练期同步图'),dict(id='climate28',label='近期28天经验分布 · 简单基线',round=32,history_hours=672,graph_role='不运行图网络；联合秩使用共同同步图模板'),dict(id='analog112',label='112天历史类比 · 简单基线',round=32,history_hours=24,graph_role='不运行图网络；按历史特征类比，联合秩共用'),dict(id='sync_dynamic',label='同步图 · 校准与相关结构每周更新',round=35,history_hours=24,graph_role='预测网络同步图固定；联合相关结构每周更新')]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def build(destination):
 out=Path(destination);out.mkdir(parents=True,exist_ok=True);assert not (out/'asset_manifest.json').exists(),'never overwrite a previously built asset';parenthash={}
 for number,folder in PARENTS.items():
  root=ART/folder;manifest=read(root/'final_manifest.json');assert read(root/'verification.json')['status']=='PASS'
  for file,h in manifest.items():assert sha(file)==h,file
  parenthash[str(number)]=dict(folder=folder,manifest_sha256=sha(root/'final_manifest.json'),files=len(manifest))
  print('PARENT',number,len(manifest),flush=True)
 data=np.load(ART/'data.npz');raw=data['raw'];times=pd.DatetimeIndex(pd.to_datetime(data['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');nodes=read(ART/'metadata.json')['nodes'];inventories={n:read(ART/PARENTS[n]/'inventory.json') for n in [27,30,32,35]};lookup={n:{(r['fold'],r['model'],r['seed'],r['method']):r for r in rows} for n,rows in inventories.items()};arrays={};dedup={};origins={};future={};cases=[];records={};source_hashes={}
 def source_record(file,key,kind,**extra):
  file=Path(file);h=sha(file)
  if str(file) in source_hashes:assert source_hashes[str(file)]==h
  source_hashes[str(file)]=h;return dict(kind=kind,source_path=str(file),source_sha256=h,source_key=key,**extra)
 def add(a,origin):
  a=np.ascontiguousarray(a);identity=hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
  if identity not in dedup:key=f'a_{len(arrays):04d}';dedup[identity]=key;arrays[key]=a;origins[key]=[]
  key=dedup[identity]
  if origin not in origins[key]:origins[key].append(origin)
  return key
 qcache={};rankcache={};weightcache={}
 def qref(file,key,positions):
  cachekey=(str(file),key,tuple(positions))
  if cachekey not in qcache:qcache[cachekey]=add(np.load(file)[key][positions],source_record(file,key,'selected_rows',positions=positions.tolist()))
  return qcache[cachekey]
 def rankref(file):
  if str(file) not in rankcache:rankcache[str(file)]=add(np.load(file)['ranks'],source_record(file,'ranks','whole'))
  return rankcache[str(file)]
 for fold in FOLDS:
  fpath=ART/PARENTS[25]/fold/'data.npz';f=np.load(fpath);positions=np.floor(np.linspace(0,len(f['test_ids'])-1,12)).astype(int);ids=f['test_ids'][positions];history=np.stack([raw[i-23:i+1] for i in ids]);historykey=add(history,source_record(ART/'data.npz','raw','history',ids=ids.tolist()));futurekey=f'future_{fold}';future[futurekey]=np.stack([raw[i+1:i+73] for i in ids]);graphkey=add(f['adj'],source_record(fpath,'adj','whole'));ownkey=add(np.eye(len(nodes)),dict(kind='identity',nodes=len(nodes)));schedule=read(ART/PARENTS[27]/fold/'schedule.json');block_for={int(pos):entry for entry in schedule for pos in entry['positions']}
  for row,(pos,issue) in enumerate(zip(positions,ids)):
   cid=f'{fold}_{pos:03d}';cases.append(dict(id=cid,fold=fold,position=int(pos),issue_id=int(issue),issue=str(times[issue]),history_start=str(times[issue-23]),history_key=historykey,row=row,observation_key=futurekey))
   for model in MODELS:
    mid=model['id'];parent=model['round'];network=O if mid=='own_rolling' else G
    for seed in SEEDS:
     if parent==27:
      method={'sync_raw':'raw','sync_fixed':'frozen','sync_rolling':'rolling28','own_rolling':'rolling28'}[mid];item=lookup[27][fold,network,seed,method];refs=[qref(item['source'],item['source_key'],positions)];rankfile=item['ranks_source'];weightkey=None
     elif parent==30:
      item=lookup[30][fold,G,seed,'window_exact1'];refs=[qref(e['source'],e['source_key'],positions) for e in item['experts']];rankfile=item['ranks_source'];weightkey=qref(item['weights_source'],'weights',positions)
     elif parent==32:
      family,method={'long7_raw':('context7_graph','raw'),'long7_rolling':('context7_graph','rolling28'),'climate28':('climate28','raw'),'analog112':('analog112','raw')}[mid];item=lookup[32][fold,family,seed,method];refs=[qref(item['source'],item['source_key'],positions)];rankfile=item['ranks_source'];weightkey=None
     else:
      item=lookup[35][fold,'graph_rolling28',seed,'rolling28'];refs=[qref(item['source'],item['source_key'],positions)];rankfile=next(r['source'] for r in item['ranks'] if int(pos) in r['positions']);weightkey=None
     rkey=rankref(rankfile);entry=block_for[int(pos)];rolling=mid in ['sync_rolling','own_rolling','window_pool','long7_rolling','sync_dynamic'];source=dict(parent_round=parent,parent_manifest_sha256=parenthash[str(parent)]['manifest_sha256'],forecast_sources=[copy for key in refs for copy in origins[key] if copy.get('kind')=='selected_rows'],rank_source=dict(file=Path(rankfile).name,sha256=sha(rankfile)),calibration_update=dict(block=int(entry['block']),issue_id=int(entry['issue_id']),time=str(times[entry['issue_id']])) if rolling else None,graph_role=model['graph_role'],graph_physical_coordinates=False)
     records[f'{cid}|{mid}|{seed}']=dict(q_keys=refs,row=row,rank_key=rkey,weight_key=weightkey,adj_key=ownkey if mid=='own_rolling' else graphkey,source=source)
  print('ASSET_FOLD',fold,'arrays',len(arrays),flush=True)
 assert len(cases)==48 and len(records)==1440;index=dict(version='5.0',nodes=nodes,seeds=SEEDS,models=MODELS,cases=cases,records=records,scope=['48个起报为四时段各12个固定等距位置，不按效果选择；完整评价仍有880个独特起报。','所有模型仅使用2024数据；同年已经多轮研究，属于探索性回放。','风速刻度及传感器含义待厂方确认；缺海浪、NWP、真实地理位置及作业资料。','同步图是统计关系，不是已验证天气传播图；候选方法不自动成为优胜模型。','TV半径控制概率质量变化，不是置信水平；费用为模拟分值。']);np.savez_compressed(out/'forecasts.npz',**arrays);np.savez_compressed(out/'observations.npz',**future);js(out/'index.json',index);js(out/'array_provenance.json',origins)
 from build_v5_booking_cache import build_cache
 cache_file,cache_sources=build_cache(out);source_hashes.update(cache_sources);js(out/'source_hashes.json',source_hashes);js(out/'asset_manifest.json',dict(version='5.0',prediction_files={n:sha(out/n) for n in ['forecasts.npz','index.json']},observation_files={'observations.npz':sha(out/'observations.npz')},planning_files={cache_file.name:sha(cache_file)},parent_manifests=parenthash));js(out/'build_summary.json',dict(status='BUILT_NOT_YET_VERIFIED',cases=48,models=10,seeds=3,forecasts=1440,unique_arrays=len(arrays),source_files=len(source_hashes),cached_plans=1392,forecast_bytes=(out/'forecasts.npz').stat().st_size,observation_bytes=(out/'observations.npz').stat().st_size,cache_bytes=cache_file.stat().st_size))
 for file,h in source_hashes.items():assert sha(file)==h,file
 print('BUILT_NOT_YET_VERIFIED',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(Path(__file__).parent/'v5/data'));args=parser.parse_args();build(args.out)

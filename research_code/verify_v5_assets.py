"""Independent source reconstruction before publishing any V5 asset PASS marker."""
from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager
import argparse,gzip,hashlib,json,shutil,tempfile
import numpy as np
import pandas as pd
from model import ART
from v5.store import StudyStore
from v5.forecasting import StudyForecaster
from v5.cache import BookingCache
from v5.planning import compute_booking

FOLDS=['M05','M07','M09','M11'];SEEDS=[20241101,20241102,20241103]
MAP={'sync_raw':(27,'gated_graph_atoms','raw'),'sync_fixed':(27,'gated_graph_atoms','frozen'),'sync_rolling':(27,'gated_graph_atoms','rolling28'),'own_rolling':(27,'gated_temporal_atoms','rolling28'),'window_pool':(30,'gated_graph_atoms','window_exact1'),'long7_raw':(32,'context7_graph','raw'),'long7_rolling':(32,'context7_graph','rolling28'),'climate28':(32,'climate28','raw'),'analog112':(32,'analog112','raw'),'sync_dynamic':(35,'graph_rolling28','rolling28')}
FOLDERS={25:'seasonal_probability_v1',26:'partial_copula_v1',27:'seasonal_rolling_v1',29:'probability_pooling_v1',30:'window_pooling_v1',31:'missing_window_bounds_v1',32:'multiscale_context_v1',33:'distributional_schedule_v1',35:'rolling_copula_v1'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
@lru_cache(maxsize=6)
def arr(file,key):
 with np.load(file) as z:return z[key]
def same(a,b):np.testing.assert_array_equal(a,b)

@contextmanager
def isolated_runtime_assets(root,temporary_parent=None):
 root=Path(root)
 # Long audits should not depend on a globally managed system TEMP lifetime.
 # An explicit parent remains available for the real cross-drive isolation test.
 parent=root.parent.resolve() if temporary_parent is None else Path(temporary_parent).resolve()
 assert parent.is_dir()
 with tempfile.TemporaryDirectory(prefix='xx风电场_v5_verify_',dir=parent) as tmp:
  temp=Path(tmp)
  assert temp.resolve().is_relative_to(parent) and temp.resolve()!=root.resolve()
  for file in ['asset_manifest.json','index.json','forecasts.npz','observations.npz','booking_cache.json.gz']:shutil.copy2(root/file,temp/file)
  js(temp/'verification.json',dict(status='PASS',asset_manifest_sha256=sha(root/'asset_manifest.json'),temporary_internal_stage=True))
  yield temp

def verify(root):
 root=Path(root);assert not (root/'verification.json').exists(),'do not overwrite prior published verification';manifest=read(root/'asset_manifest.json');index=read(root/'index.json');sources=read(root/'source_hashes.json');provenance=read(root/'array_provenance.json');counts={}
 for section in ['prediction_files','observation_files','planning_files']:
  for file,h in manifest[section].items():assert sha(root/file)==h
 assert set(manifest['parent_manifests'])==set(map(str,FOLDERS))
 for n,folder in FOLDERS.items():
  p=ART/folder;assert sha(p/'final_manifest.json')==manifest['parent_manifests'][str(n)]['manifest_sha256'];assert read(p/'verification.json')['status']=='PASS'
  for file,h in read(p/'final_manifest.json').items():assert sha(file)==h,file
 for file,h in sources.items():assert sha(file)==h,file
 raw=arr(str(ART/'data.npz'),'raw');times=pd.DatetimeIndex(pd.to_datetime(arr(str(ART/'data.npz'),'timestamps').tolist(),utc=True)).tz_convert('Asia/Shanghai');nodes=read(ART/'metadata.json')['nodes'];assert index['nodes']==nodes and index['seeds']==SEEDS;assert {m['id'] for m in index['models']}==set(MAP);assert len(index['cases'])==48 and len(index['records'])==1440
 with np.load(root/'forecasts.npz') as pack,np.load(root/'observations.npz') as future:
  assert set(pack.files)==set(provenance);assert set(future.files)=={f'future_{f}' for f in FOLDS}
  for key,origins in provenance.items():
   assert origins
   for o in origins:
    if o['kind']=='identity':expected=np.eye(o['nodes'])
    else:
     assert sources[o['source_path']]==o['source_sha256'];a=arr(o['source_path'],o['source_key'])
     if o['kind']=='whole':expected=a
     elif o['kind']=='selected_rows':expected=a[o['positions']]
     elif o['kind']=='history':expected=np.stack([a[i-23:i+1] for i in o['ids']])
     else:raise AssertionError(o['kind'])
    same(pack[key],expected);assert pack[key].dtype==expected.dtype
  for fold in FOLDS:
   ids=arr(str(ART/FOLDERS[25]/fold/'data.npz'),'test_ids');positions=np.floor(np.linspace(0,len(ids)-1,12)).astype(int);cases=[c for c in index['cases'] if c['fold']==fold];assert [c['position'] for c in cases]==positions.tolist();assert [c['issue_id'] for c in cases]==ids[positions].tolist()
   for row,c in enumerate(cases):
    assert c['id']==f'{fold}_{positions[row]:03d}' and c['row']==row and c['issue']==str(times[c['issue_id']]);same(pack[c['history_key']][row],raw[c['issue_id']-23:c['issue_id']+1]);same(future[c['observation_key']][row],raw[c['issue_id']+1:c['issue_id']+73])
 counts.update(source_files=len(sources),asset_arrays=len(provenance),cases=48,observation_arrays=4)
 inventories={n:{(r['fold'],r['model'],r['seed'],r['method']):r for r in read(ART/FOLDERS[n]/'inventory.json')} for n in [27,30,32,35]}
 # Runtime checks use an isolated, temporary proof after the byte/source checks above.
 # The publishable directory receives no PASS until all forecast and cache checks end.
 with isolated_runtime_assets(root) as temp:
  print('ISOLATED_AUDIT_COPY',str(temp),flush=True)
  store=StudyStore(temp);forecaster=StudyForecaster(store);cache=BookingCache(temp);checked=0
  def no_future(*a,**k):raise AssertionError('prediction or booking accessed future observations')
  store.observations=no_future
  for case in index['cases']:
   fold=case['fold'];pos=case['position']
   for mid,(n,model,method) in MAP.items():
    for seed in SEEDS:
     item=inventories[n][fold,model,seed,method];rfile=next(r['source'] for r in item['ranks'] if pos in r['positions']) if n==35 else item['ranks_source'];ranks=arr(rfile,'ranks');experts=item['experts'] if n==30 else [item];q=[arr(e['source'],e['source_key'])[pos] for e in experts];weights=arr(item['weights_source'],'weights')[pos] if n==30 else np.array([1.]);h=np.arange(72)[None,:,None];j=np.arange(len(nodes))[None,None,:];expected=np.concatenate([a[h,j,ranks//3] for a in q]).astype(float);expected_weights=np.repeat(weights/57,57);expected_weights/=expected_weights.sum();request=dict(case_id=case['id'],model=mid,seed=seed,threshold=10.,duration=6,group_nodes=nodes[:3]);sc,w,info=store.scenarios(request);same(sc,expected);same(w,expected_weights);assert info['source']['parent_round']==n and info['source']['graph_physical_coordinates'] is False
     p=forecaster.payload(request);assert p['scenario_count']==(228 if n==30 else 57);assert 'observed_after_issue' not in p;w=w/w.sum();good=expected<=10;single=np.array([np.sum(good[:,s:s+6].all(1)*w[:,None],axis=0) for s in range(67)]);joint=np.array([np.sum(good[:,s:s+6,:3].all((1,2))*w) for s in range(67)]);np.testing.assert_allclose(p['probabilities'],single,rtol=0,atol=3e-12);np.testing.assert_allclose(p['joint_probabilities'],joint,rtol=0,atol=3e-12)
     for k,level in [('q10',.1),('q50',.5),('q90',.9)]:
      order=np.argsort(expected,axis=0,kind='stable');sorted_values=np.take_along_axis(expected,order,axis=0);cw=np.cumsum(np.take_along_axis(np.broadcast_to(w[:,None,None],expected.shape),order,axis=0),axis=0);ix=np.argmax(cw>=level-1e-12,axis=0);same(np.asarray(p[k]),np.take_along_axis(sorted_values,ix[None],axis=0)[0])
     checked+=1
   print('FORECAST_ASSET_AUDIT',case['id'],checked,flush=True)
  assert checked==1440;counts['forecasts_reconstructed']=checked
  src=ART/FOLDERS[33];cached=json.loads(gzip.decompress((root/'booking_cache.json.gz').read_bytes()));assert cached['config']==read(src/'recipe.json')['config'];assert len(cached['records'])==1392;nc=0
  for item in read(src/'inventory.json'):
   for policy in item['policies']:
    key=f'{item["key"]}__{policy}';c=cached['records'][key];original=read(src/'cases'/f'plan_{key}.json');assert c['provenance']['source_sha256']==sha(src/'cases'/f'plan_{key}.json');assert c['provenance']['parent_manifest_sha256']==sha(src/'final_manifest.json')
    for field,value in c['plan'].items():
     if field=='trace':assert value==[{k:r[k] for k in ['iteration','lower_bound','upper_bound']} for r in original['plan']['trace']]
     else:assert value==original['plan'][field],(key,field)
    for field,value in c['dual_reference'].items():assert value==original['dual_reference'][field]
    assert not any(k in c for k in ['observed_after_issue','outcome','evaluation']);nc+=1
  counts['cached_plans_exact']=nc;bookings=0
  for fold in FOLDS:
   case=next(c for c in index['cases'] if c['fold']==fold)
   for mid in ['sync_fixed','sync_rolling','own_rolling']:
    for radius in [0.,.25,1.]:
     b=compute_booking(forecaster,dict(forecast=dict(case_id=case['id'],model=mid,seed=SEEDS[0]),radius=radius,algorithm='ccg'),cache=cache);assert b['source_mode']=='frozen_round33' and b['certificate']['passed'];bookings+=1
   print('BOOKING_ASSET_AUDIT',fold,bookings,flush=True)
  counts['cached_runtime_bookings']=bookings
 for file,h in sources.items():assert sha(file)==h
 for section in ['prediction_files','observation_files','planning_files']:
  for file,h in manifest[section].items():assert sha(root/file)==h
 counts.update(status='PASS',asset_manifest_sha256=sha(root/'asset_manifest.json'),future_reads_during_prediction_booking=0,parent_manifests=len(FOLDERS));js(root/'verification.json',counts);print(json.dumps(counts,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).parent/'v5/data'));a=p.parse_args();verify(a.root)

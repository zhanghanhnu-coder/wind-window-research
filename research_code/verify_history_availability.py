"""Direct rolling all-mask and independent interval scheduling audit for round40."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from model import ART
from research_crps_distribution import guard
OUT=ART/'history_availability_v1';PARENT=ART/'seasonal_probability_v1';FOLDS=['M05','M07','M09','M11'];GROUPS={'all17':list(range(17)),'tasks6':[0,3,6,9,12,16],'group_a':[0,1,2],'group_b':[8,9,10],'group_c':[13,15,16]}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
 guard();assert read(OUT/'complete.json')['status']=='COMPLETE';identity=read(OUT/'identity.json');manifest=read(PARENT/'final_manifest.json')
 for file,h in identity.items():assert sha(file)==h,file
 for file,h in manifest.items():assert sha(file)==h,file
 df=pd.read_csv(OUT/'records.csv',float_precision='round_trip');assert len(df)==67760 and not df.duplicated(['issue_id','group','horizon','lookback_days']).any();raw=np.load(ART/'data.npz')['raw'];valid=np.isfinite(raw);rolling={};foldids={f:set(np.load(PARENT/f/'data.npz')['test_ids'].tolist()) for f in FOLDS};expected_specs={(g,h,d) for g in GROUPS for h in [6,12,24,72] for d in [28,56,112]}|{(f'node_{n:02d}',72,112) for n in range(17)};assert len(expected_specs)==77
 for group,h,d in expected_specs:
  nodes=GROUPS[group] if group in GROUPS else [int(group.split('_')[1])]
  if (group,h) not in rolling:rolling[group,h]=sliding_window_view(valid[:,nodes].all(1),h).all(-1)
  selected=df[(df.group==group)&(df.horizon==h)&(df.lookback_days==d)];assert len(selected)==880 and set(selected.issue_id)==set.union(*foldids.values())
 counts=0;sample=np.random.default_rng(20261126).choice(len(df),24,replace=False);sample_set=set(sample.tolist());sliced_checks=0
 for idx,r in enumerate(df.itertuples(index=False)):
  if idx%1000==0:guard()
  assert r.issue_id in foldids[r.fold];nodes=GROUPS[r.group] if r.group in GROUPS else [int(r.group.split('_')[1])];assert r.node_count==len(nodes);j=np.arange(23,len(valid)-r.horizon,6);j=j[(j+r.horizon<r.issue_id)&(j+1>=r.issue_id-r.lookback_days*24)];good=j[rolling[r.group,r.horizon][j+1]];remaining=good.copy();disjoint=0
  while len(remaining):end=int(remaining[0])+r.horizon;disjoint+=1;remaining=remaining[remaining+1>end]
  assert r.candidate_count==len(j) and r.complete_count==len(good) and r.maximum_disjoint==disjoint
  np.testing.assert_allclose(r.complete_fraction,len(good)/len(j) if len(j) else np.nan,rtol=0,atol=1e-14,equal_nan=True);assert r.at_least19==(len(good)>=19) and r.at_least57==(len(good)>=57) and r.primary==(r.group=='all17' and r.horizon==72 and r.lookback_days==112)
  if idx in sample_set:
   direct=[v for v in j if np.isfinite(raw[v+1:v+r.horizon+1,nodes]).all()];np.testing.assert_array_equal(good,direct);sliced_checks+=1
  counts+=1
 summary=[]
 for scope in ['ALL',*FOLDS]:
  selected=df if scope=='ALL' else df[df.fold==scope]
  for (group,h,d),g in selected.groupby(['group','horizon','lookback_days']):
   a=np.sort(g.complete_count.to_numpy());n=len(a);q=np.quantile(a,[.25,.5,.75]);dis=np.sort(g.maximum_disjoint.to_numpy());summary.append(dict(scope=scope,group=group,node_count=int(g.node_count.iloc[0]),horizon=int(h),lookback_days=int(d),n=n,complete_min=int(a[0]),complete_q25=q[0],complete_median=q[1],complete_q75=q[2],complete_max=int(a[-1]),fraction_at_least19=sum(a>=19)/n,fraction_at_least57=sum(a>=57)/n,zero_fraction=sum(a==0)/n,disjoint_min=int(dis[0]),disjoint_median=float(np.median(dis)),disjoint_max=int(dis[-1]),primary=group=='all17' and h==72 and d==112 and scope=='ALL'))
 a=pd.read_csv(OUT/'summary.csv',float_precision='round_trip').sort_values(['scope','group','horizon','lookback_days']).reset_index(drop=True);b=pd.DataFrame(summary).sort_values(['scope','group','horizon','lookback_days']).reset_index(drop=True);assert len(a)==len(b)==385
 for col in b:
  if b[col].dtype.kind in 'biufc':np.testing.assert_allclose(a[col],b[col],rtol=0,atol=1e-13)
  else:assert a[col].tolist()==b[col].tolist()
 assert counts==67760 and sliced_checks==24
 for file,h in identity.items():assert sha(file)==h,file
 proof=dict(status='PASS',records_independently_recounted=counts,summary_rows=385,raw_slice_spot_checks=sliced_checks,rolling_masks=len(rolling),future_pollution_checks_from_producer=read(OUT/'complete.json')['future_pollution_checks'],source_files_unchanged=len(identity),parent_files_unchanged=len(manifest),scope='Availability diagnostic only; no effect or probability calibration claim');(OUT/'verification.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof,indent=2))

if __name__=='__main__':run()

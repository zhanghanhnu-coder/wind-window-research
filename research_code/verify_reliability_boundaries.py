"""Independent searchsorted bins and unchanged-source audit for endpoint correction."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
OUT=ART/'reliability_boundary_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(name):
 root=ART/name;dest=OUT/name;inventory=json.loads((root/'inventory.json').read_text(encoding='utf-8'));hashes=json.loads((dest/'source_hashes.json').read_text(encoding='utf-8'));counts=[];rows=[]
 for file,h in hashes.items():assert sha(Path(file))==h,file
 for item in inventory:
  path=Path(item['score_source']) if 'score_source' in item else root/item['fold']/f'scores_{item["model"]}_{item["seed"]}_{item["method"]}.npz';s=np.load(path);common={k:item[k] for k in ['fold','model','seed','method']}
  for target,pk,yk in [('hourly','probability','label'),('single_6h','prob6','label6'),('group_6h','group_probability','group_label')]:
   known=np.isfinite(s[yk]);p=s[pk][known];y=s[yk][known];assert np.isfinite(p).all() and ((p>=-1e-12)&(p<=1+1e-12)).all();bins=np.searchsorted(np.arange(1,10)/10,p,side='right');assert (bins>=0).all() and (bins<=9).all();n=np.bincount(bins,minlength=10);ps=np.bincount(bins,weights=p,minlength=10);ys=np.bincount(bins,weights=y,minlength=10)
   for b in range(10):rows.append(dict(**common,target=target,bin=b,n=int(n[b]),mean_probability=ps[b]/n[b] if n[b] else np.nan,observed_rate=ys[b]/n[b] if n[b] else np.nan))
   outside=int(((p<0)|(p>1)).sum());counts.append(dict(**common,target=target,valid_targets=len(p),new_binned=int(n.sum()),old_binned=int(((p>=0)&(p<=1)).sum()),old_unbinned=outside,roundoff_outside_unit_interval=outside,maximum_unit_interval_excess=float(max(0,p.max()-1,-p.min())) if len(p) else 0.))
 def compare(file,values,keys):
  a=pd.read_csv(dest/file).sort_values(keys).reset_index(drop=True);b=pd.DataFrame(values).sort_values(keys).reset_index(drop=True);assert len(a)==len(b)
  for col in b:
   if b[col].dtype.kind in 'biufc':np.testing.assert_allclose(a[col],b[col],rtol=0,atol=3e-11,equal_nan=True)
   else:assert a[col].tolist()==b[col].tolist()
 keys=['fold','model','seed','method','target'];compare('reliability_complete_bins.csv',rows,keys+['bin']);compare('bin_coverage.csv',counts,keys);old=pd.read_csv(root/'reliability.csv').sort_values(keys+['bin']).reset_index(drop=True);new=pd.DataFrame(rows).sort_values(keys+['bin']).reset_index(drop=True);middle=old.bin.between(1,8)
 for col in ['n','mean_probability','observed_rate']:np.testing.assert_allclose(old.loc[middle,col],new.loc[middle,col],rtol=0,atol=3e-11,equal_nan=True)
 result=dict(status='PASS',round=name,source_files_unchanged=len(hashes),bin_rows=len(rows),coverage_rows=len(counts),valid_targets=sum(r['valid_targets'] for r in counts),old_unbinned=sum(r['old_unbinned'] for r in counts),maximum_excess=max(r['maximum_unit_interval_excess'] for r in counts));(dest/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 if (root/'reliability_complete_bins.csv').exists():
  assert sha(root/'reliability_complete_bins.csv')==sha(dest/'reliability_complete_bins.csv');assert not (root/'final_manifest.json').exists();(root/'reliability_correction_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 import sys
 verify(sys.argv[1])

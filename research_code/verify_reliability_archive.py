"""Read-only archival audit; safe after canonical parent directories are frozen.

The numerical audit mirrors the already-reviewed independent searchsorted method.
Unlike the original preparation-time checker, it never creates or replaces files.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from model import ART

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def verify(name,art=ART):
 art=Path(art);root=art/name;dest=art/'reliability_boundary_v1'/name;inventory=read(root/'inventory.json');hashes=read(dest/'source_hashes.json');counts=[];rows=[]
 for file,h in hashes.items():assert sha(file)==h,file
 for item in inventory:
  path=Path(item['score_source']) if 'score_source' in item else root/item['fold']/f'scores_{item["model"]}_{item["seed"]}_{item["method"]}.npz';common={k:item[k] for k in ['fold','model','seed','method']}
  with np.load(path) as s:
   for target,pk,yk in [('hourly','probability','label'),('single_6h','prob6','label6'),('group_6h','group_probability','group_label')]:
    known=np.isfinite(s[yk]);p=s[pk][known];y=s[yk][known];assert np.isfinite(p).all() and np.isin(y,[0,1]).all() and ((p>=-1e-12)&(p<=1+1e-12)).all();bins=np.searchsorted(np.arange(1,10)/10,p,side='right');assert (bins>=0).all() and (bins<=9).all();n=np.bincount(bins,minlength=10);ps=np.bincount(bins,weights=p,minlength=10);ys=np.bincount(bins,weights=y,minlength=10)
    for b in range(10):rows.append(dict(**common,target=target,bin=b,n=int(n[b]),mean_probability=ps[b]/n[b] if n[b] else np.nan,observed_rate=ys[b]/n[b] if n[b] else np.nan))
    outside=int(((p<0)|(p>1)).sum());counts.append(dict(**common,target=target,valid_targets=len(p),new_binned=int(n.sum()),old_binned=int(((p>=0)&(p<=1)).sum()),old_unbinned=outside,roundoff_outside_unit_interval=outside,maximum_unit_interval_excess=float(max(0,p.max()-1,-p.min())) if len(p) else 0.))
 def compare(file,values,keys):
  a=pd.read_csv(dest/file).sort_values(keys).reset_index(drop=True);b=pd.DataFrame(values).sort_values(keys).reset_index(drop=True);assert len(a)==len(b)
  for col in b:
   if b[col].dtype.kind in 'biufc':np.testing.assert_allclose(a[col],b[col],rtol=0,atol=3e-11,equal_nan=True)
   else:assert a[col].tolist()==b[col].tolist()
 keys=['fold','model','seed','method','target'];compare('reliability_complete_bins.csv',rows,keys+['bin']);compare('bin_coverage.csv',counts,keys);old=pd.read_csv(root/'reliability.csv').sort_values(keys+['bin']).reset_index(drop=True);new=pd.DataFrame(rows).sort_values(keys+['bin']).reset_index(drop=True);assert old[keys+['bin']].equals(new[keys+['bin']]);middle=old.bin.between(1,8)
 for col in ['n','mean_probability','observed_rate']:np.testing.assert_allclose(old.loc[middle,col],new.loc[middle,col],rtol=0,atol=3e-11,equal_nan=True)
 if (root/'reliability_complete_bins.csv').exists():assert sha(root/'reliability_complete_bins.csv')==sha(dest/'reliability_complete_bins.csv')
 for file,h in hashes.items():assert sha(file)==h,file
 return dict(status='PASS',round=name,read_only=True,parent_frozen=(root/'final_manifest.json').exists(),source_files_unchanged=len(hashes),bin_rows=len(rows),coverage_rows=len(counts),valid_targets=sum(r['valid_targets'] for r in counts),old_unbinned=sum(r['old_unbinned'] for r in counts),maximum_excess=max(r['maximum_unit_interval_excess'] for r in counts))

if __name__=='__main__':
 import sys
 names=sys.argv[1:] or ['seasonal_rolling_v1','probability_pooling_v1','window_pooling_v1','multiscale_context_v1']
 for name in names:print(json.dumps(verify(name),indent=2),flush=True)

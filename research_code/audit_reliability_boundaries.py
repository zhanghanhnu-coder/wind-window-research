"""Append a corrected descriptive table; never mutate frozen forecasts or old tables."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
from reliability_boundaries_core import reliability_rows
OUT=ART/'reliability_boundary_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(root,attach=False):
 root=Path(root);inventory=json.loads((root/'inventory.json').read_text(encoding='utf-8'));rows=[];audits=[];hashes={str(root/'reliability.csv'):sha(root/'reliability.csv')}
 for item in inventory:
  source=Path(item['score_source']) if 'score_source' in item else root/item['fold']/f'scores_{item["model"]}_{item["seed"]}_{item["method"]}.npz';hashes[str(source)]=sha(source);s=np.load(source);common={k:item[k] for k in ['fold','model','seed','method']}
  for target,pk,yk in [('hourly','probability','label'),('single_6h','prob6','label6'),('group_6h','group_probability','group_label')]:
   bins,info=reliability_rows(s[pk],s[yk]);rows.extend([dict(**common,target=target,**b) for b in bins]);audits.append(dict(**common,target=target,**info))
 corrected=pd.DataFrame(rows);counts=pd.DataFrame(audits);old=pd.read_csv(root/'reliability.csv');keys=['fold','model','seed','method','target'];oldcounts=old.groupby(keys).n.sum().reset_index(name='observed_old_count');joined=counts.merge(oldcounts,on=keys);np.testing.assert_array_equal(joined.old_binned,joined.observed_old_count);assert counts.old_unbinned.sum()==counts.roundoff_outside_unit_interval.sum();destination=OUT/root.name;destination.mkdir(parents=True,exist_ok=True);corrected.to_csv(destination/'reliability_complete_bins.csv',index=False,encoding='utf-8-sig');counts.to_csv(destination/'bin_coverage.csv',index=False,encoding='utf-8-sig');(destination/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8');summary=dict(source_round=root.name,score_flows=len(inventory),valid_targets=int(counts.valid_targets.sum()),previous_unbinned=int(counts.old_unbinned.sum()),maximum_excess=float(counts.maximum_unit_interval_excess.max()),corrected_binned=int(counts.new_binned.sum()),changed='bin membership at numerical endpoints only; stored probabilities, labels, scores and original tables unchanged');(destination/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 for file,h in hashes.items():assert sha(Path(file))==h,file
 if attach:
  assert not (root/'final_manifest.json').exists(),'never attach into already frozen round'
  corrected.to_csv(root/'reliability_complete_bins.csv',index=False,encoding='utf-8-sig');counts.to_csv(root/'reliability_bin_coverage.csv',index=False,encoding='utf-8-sig');(root/'reliability_correction.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print(json.dumps(summary,indent=2),flush=True);return summary
if __name__=='__main__':
 import sys
 audit(ART/sys.argv[1],attach='--attach' in sys.argv)

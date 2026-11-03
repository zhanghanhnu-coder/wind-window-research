"""Copy checked plan evidence, never actual future outcomes, into the portable Demo."""
from pathlib import Path
import gzip,hashlib,json
import numpy as np
from model import ART

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build_cache(destination):
 root=ART/'distributional_schedule_v1';assert read(root/'verification.json')['status']=='PASS';manifest_sha=sha(root/'final_manifest.json');records={};sources={}
 for item in read(root/'inventory.json'):
  pack=np.load(item['forecast_source']);sc=pack['scenarios'].astype(float);w=pack['probability'].astype(float);w/=w.sum();w/=w.sum();sch=hashlib.sha256(sc.tobytes()).hexdigest();wh=hashlib.sha256(w.tobytes()).hexdigest()
  for policy in item['policies']:
   file=root/'cases'/f'plan_{item["key"]}__{policy}.json';record=read(file);plan=record['plan'];compact={k:plan[k] for k in ['x','objective','lower_bound','upper_bound','gap','iterations','scenario_solutions','recourse_costs','adversary']};compact['trace']=[{k:r[k] for k in ['iteration','lower_bound','upper_bound']} for r in plan['trace']];full={k:record['dual_reference'][k] for k in ['x','objective','dual_bound','solver_status','mip_gap']};h=sha(file);sources[str(file)]=h;records[f'{item["key"]}__{policy}']=dict(plan=compact,dual_reference=full,provenance=dict(parent_round=33,parent_manifest_sha256=manifest_sha,source_file=file.name,source_sha256=h,forecast_source_sha256=sha(item['forecast_source'])),scenario_sha256=sch,weight_sha256=wh)
 assert len(records)==1392;payload=dict(version='5.0',config=read(root/'recipe.json')['config'],records=records);file=Path(destination)/'booking_cache.json.gz';file.write_bytes(gzip.compress(json.dumps(payload,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8'),compresslevel=9,mtime=0));return file,sources

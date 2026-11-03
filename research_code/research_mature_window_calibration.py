"""Causal weekly event recalibration of frozen synchronous-graph forecasts."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
import numpy as np
import pandas as pd
from model import ART
from seasonal_probability_core import FOLDS
from research_crps_distribution import guard,SEEDS
from mature_window_calibration_core import fit_mature_map,apply_window_map,ENDS
from mature_window_grid_core import canonical_equal_probability
from mature_window_tables import analyze,reliability
OUT=ART/'mature_window_calibration_v2';PARENT=ART/'missing_window_bounds_v1';BASE=ART.parent.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def run():
 guard();assert read(ART/'scenario_equivalence_v1/complete.json')['status']=='COMPLETE' and read(ART/'scenario_equivalence_v1/verification.json')['status']=='PASS';assert not (OUT/'final_manifest.json').exists();parent=read(PARENT/'final_manifest.json')
 for file,h in parent.items():assert sha(file)==h,file
 OUT.mkdir(exist_ok=True);protocol=OUT/'protocol_frozen.md';plan=BASE/'研究笔记/已兑现窗口单调校准方案_2026-11-02.md'
 if protocol.exists():assert protocol.read_bytes()==plan.read_bytes()
 else:protocol.write_bytes(plan.read_bytes())
 archive=ART/'mature_window_calibration_v1/attempt1_archive_manifest.json'
 for file,h in read(archive).items():assert sha(file)==h,file
 files=[Path(__file__),protocol,PARENT/'final_manifest.json',ART/'data.npz',archive]+[ART.parent/n for n in ['mature_window_calibration_core.py','mature_window_grid_core.py','mature_window_tables.py','verify_mature_window_calibration.py','tests/test_mature_window_calibration.py','tests/test_mature_window_grid.py','missing_window_bounds_core.py','reliability_boundaries_core.py','verify_missing_window_bounds.py']]
 identity={str(p.resolve()):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==identity
 else:js(OUT/'identity.json',identity)
 if not (OUT/'recipe.json').exists():js(OUT/'recipe.json',dict(source_case='graph_rolling28',folds=list(FOLDS),seeds=SEEDS,window_hours=6,offsets=[0,24,48,66],lookback_hours=672,minimum_origins=20,prior_weight=20,prior_grid=np.linspace(0,1,21).tolist(),equal_scenarios=57,calibration_input='recover integer success count /57 within 1e-12; original baseline scores unchanged',maturity='window end < update; target end >= update-672',weekly='first eligible issue per frozen seven-day block',candidate='one origin-weighted common PAV map plus fixed triplet Frechet projection',bootstrap_seed=20261129,bootstrap_replicates=2000,primary='ALL candidate-minus-base shared-unknown-label Brier bounds: single and group',start_utc=datetime.now(timezone.utc).isoformat()))
 inventory=[];audits=[];pollution=0
 for fold in FOLDS:
  folder=OUT/fold;folder.mkdir(exist_ok=True);sourcepop=PARENT/fold/'population.npz';destpop=folder/'population.npz'
  if destpop.exists():assert sha(destpop)==sha(sourcepop)
  else:shutil.copy2(sourcepop,destpop)
  pop=np.load(destpop);ids=pop['ids'];blocks=pop['blocks'];ys=np.where(pop['single_complete'],pop['single_labels'],np.nan);yg=np.where(pop['group_complete'],pop['group_labels'],np.nan)
  for seed in SEEDS:
   guard();src=PARENT/fold/f'probabilities_graph_rolling28_{seed}.npz';base=folder/f'probabilities_base_{seed}.npz'
   if base.exists():assert sha(base)==sha(src)
   else:shutil.copy2(src,base)
   z=np.load(base);ps=canonical_equal_probability(z['single'],57);pg=canonical_equal_probability(z['group'],57);out_s=np.empty_like(ps);out_g=np.empty_like(pg)
   for block in np.unique(blocks):
    guard();positions=np.flatnonzero(blocks==block);update=int(ids[positions[0]]);fit=fit_mature_map(ps,pg,ys,yg,ids,update);a,b,projection=apply_window_map(ps[positions],pg[positions],fit);out_s[positions]=a;out_g[positions]=b;prefix=f'map_{seed}_w{int(block)}';npz=folder/(prefix+'.npz');record=dict(fold=fold,seed=seed,block=int(block),applied_origins=len(positions),**fit['audit'],**projection)
    # Poison all not-yet-mature targets and future-issued probabilities.
    sy=ys.copy();gy=yg.copy();sp=ps.copy();gp=pg.copy();sy[ids[:,None]+np.repeat(ENDS,17)>=update]=9;gy[ids[:,None]+np.repeat(ENDS,3)>=update]=9;sp[ids>=update]=np.nan;gp[ids>=update]=np.nan;check=fit_mature_map(sp,gp,sy,gy,ids,update)
    for name in ['knots','values','training']:np.testing.assert_array_equal(fit[name],check[name])
    pollution+=1
    if npz.exists():
     old=np.load(npz)
     for name in ['knots','values','training']:np.testing.assert_array_equal(old[name],fit[name])
     assert read(folder/(prefix+'.json'))==record
    else:np.savez_compressed(npz,knots=fit['knots'],values=fit['values'],training=fit['training']);js(folder/(prefix+'.json'),record)
    audits.append(record)
   destination=folder/f'probabilities_calibrated_{seed}.npz'
   if destination.exists():
    old=np.load(destination);np.testing.assert_array_equal(old['single'],out_s);np.testing.assert_array_equal(old['group'],out_g)
   else:np.savez_compressed(destination,single=out_s,group=out_g)
   for case,path in [('base',base),('calibrated',destination)]:inventory.append(dict(fold=fold,seed=seed,case=case,origins=len(ids),probability_source=str(path),probability_sha256=sha(path),base_source=str(src),base_sha256=sha(src)))
   print('FORECAST',fold,seed,len(ids),flush=True)
 assert len(inventory)==24 and len(audits)==108;js(OUT/'inventory.json',inventory);pd.DataFrame(audits).to_csv(OUT/'update_audit.csv',index=False,encoding='utf-8-sig');js(OUT/'prediction_complete.json',dict(status='COMPLETE',flows=24,forecast_origins=sum(i['origins'] for i in inventory),unique_origins=880,updates=len(audits),future_pollution_checks=pollution,fallback_updates=sum(r['fallback'] for r in audits),utc=datetime.now(timezone.utc).isoformat()))
 analyze();reliability()
 for file,h in identity.items():assert sha(file)==h,file
 for file,h in parent.items():assert sha(file)==h,file
 print('COMPLETE',flush=True)
if __name__=='__main__':run()

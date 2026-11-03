"""Post-hoc, prespecified block sensitivity of every primary contrast in rounds32/35."""
from pathlib import Path
from functools import lru_cache
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from model import ART
from research_crps_distribution import guard
from resampling_sensitivity_core import origin_totals,calendar_blocks,bootstrap_totals
OUT=ART/'resampling_sensitivity_v1';BASE=ART.parent.parent;PARENTS={32:'multiscale_context_v1',35:'rolling_copula_v1'};FOLDS={'M05':'2024-05-01','M07':'2024-07-01','M09':'2024-09-01','M11':'2024-11-01'};SEEDS=[20241101,20241102,20241103];SETTINGS=[(d,o) for d in [3,7,14,28] for o in [0,d//2]]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
@lru_cache(maxsize=12)
def loss(file,metric):
 with np.load(file) as z:return z['crps'][:,48:72] if metric=='crps_49_72' else z[metric]

def run():
 guard();parentproof={}
 for n,name in {25:'seasonal_probability_v1',**PARENTS}.items():
  root=ART/name;manifest=read(root/'final_manifest.json');assert read(root/'verification.json')['status']=='PASS'
  for file,h in manifest.items():assert sha(file)==h,file
  parentproof[str(n)]=dict(folder=name,manifest_sha256=sha(root/'final_manifest.json'),files=len(manifest))
 OUT.mkdir(exist_ok=True);assert not (OUT/'identity.json').exists(),'do not overwrite a frozen sensitivity run';protocol=(BASE/'研究笔记/时间分块敏感性审计方案_2026-11-02.md').read_text(encoding='utf-8').split('## 执行记录')[0];(OUT/'protocol_frozen.md').write_text(protocol,encoding='utf-8');code=[Path(__file__),ART.parent/'resampling_sensitivity_core.py',ART.parent/'verify_resampling_sensitivity.py',ART.parent/'tests/test_resampling_sensitivity.py',OUT/'protocol_frozen.md'];js(OUT/'identity.json',{str(p):sha(p) for p in code});js(OUT/'recipe.json',dict(parents=parentproof,folds=FOLDS,seeds=SEEDS,settings=[dict(days=d,offset_days=o) for d,o in SETTINGS],replicates=2000,minimum_blocks_per_fold=4,minimum_valid_draws=1900,source_primary_counts={'32':12,'35':10},original_seeds={'32':20261119,'35':20261123},new_seed_rule='SeedSequence([20261124,round,days,offset_days])',post_hoc=True,start_utc=datetime.now(timezone.utc).isoformat()));sourcefiles={str(ART/'data.npz')};time=pd.DatetimeIndex(pd.to_datetime(np.load(ART/'data.npz')['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');elapsed={};ids={}
 for fold,start in FOLDS.items():
  path=ART/'seasonal_probability_v1'/fold/'data.npz';sourcefiles.add(str(path));ids[fold]=np.load(path)['test_ids'];elapsed[fold]=np.asarray((time[ids[fold]]-pd.Timestamp(start,tz=time.tz)).total_seconds()/3600)
 allrows=[];totals={};drawpack={};replicates={};contrasts=[];design=[]
 for n,name in PARENTS.items():
  root=ART/name;sourcefiles.update([str(root/'inventory.json'),str(root/'paired_comparisons.csv')]);frame=pd.read_csv(root/'paired_comparisons.csv');main=frame[frame.primary].reset_index(drop=True);assert len(main)==(12 if n==32 else 10);lookup={(r['fold'],r['model'],r['seed'],r['method']):r for r in read(root/'inventory.json')};vectors={}
  for index,r in main.iterrows():
   cid=f'r{n}_c{index:02d}';contrast=dict(id=cid,round=n,**{k:r[k] for k in ['candidate','method','baseline','baseline_method','metric']},original_difference=float(r.difference),original_low95=float(r.low95),original_high95=float(r.high95),original_n=int(r.n));contrasts.append(contrast);vectors[cid]={}
   for fold in FOLDS:
    aa=[];bb=[]
    for seed in SEEDS:
     pa=lookup[fold,r.candidate,seed,r.method]['score_source'];pb=lookup[fold,r.baseline,seed,r.baseline_method]['score_source'];sourcefiles.update([pa,pb]);aa.append(loss(pa,r.metric));bb.append(loss(pb,r.metric))
    sums,counts=origin_totals(np.stack(aa),np.stack(bb));assert len(sums)==len(ids[fold]);vectors[cid][fold]=dict(sums=sums,counts=counts);totals[f'{cid}_{fold}_sums']=sums;totals[f'{cid}_{fold}_counts']=counts
  for days,offset in SETTINGS:
   rng=np.random.default_rng((20261119 if n==32 else 20261123) if (days,offset)==(7,0) else np.random.SeedSequence([20261124,n,days,offset]));setting={}
   for fold in FOLDS:
    blocks=calendar_blocks(elapsed[fold],days,offset);k=int(blocks.max())+1;draw=rng.integers(0,k,(2000,k));setting[fold]=dict(blocks=blocks,draws=draw);drawpack[f'r{n}_d{days}_o{offset}_{fold}']=draw;design.append(dict(round=n,days=days,offset_days=offset,fold=fold,calendar_blocks=k,origins=len(blocks),minimum_block_origins=int(np.bincount(blocks,minlength=k).min()),maximum_block_origins=int(np.bincount(blocks,minlength=k).max())))
   for contrast in [c for c in contrasts if c['round']==n]:
    guard();cid=contrast['id'];inputs=[dict(**vectors[cid][fold],**setting[fold]) for fold in FOLDS];r=bootstrap_totals(inputs);key=f'{cid}_d{days}_o{offset}';replicates[key]=r.pop('replicates');row=dict(contrast_id=cid,round=n,days=days,offset_days=offset,**r);np.testing.assert_allclose(row['difference'],contrast['original_difference'],rtol=0,atol=3e-12);assert row['n']==contrast['original_n']
    if (days,offset)==(7,0):np.testing.assert_allclose([row['low95'],row['high95']],[contrast['original_low95'],contrast['original_high95']],rtol=0,atol=3e-12)
    row['direction']='unsupported' if not r['supported'] else 'negative' if r['high95']<0 else 'positive' if r['low95']>0 else 'cross_zero';allrows.append(row)
   print('SETTING',n,days,offset,flush=True)
 assert len(contrasts)==22 and len(allrows)==176;df=pd.DataFrame(allrows);summary=[]
 for c in contrasts:
  g=df[df.contrast_id==c['id']];s=g[g.supported];summary.append(dict(**c,supported_settings=len(s),negative_settings=int((g.direction=='negative').sum()),positive_settings=int((g.direction=='positive').sum()),cross_zero_settings=int((g.direction=='cross_zero').sum()),unsupported_settings=int((g.direction=='unsupported').sum()),minimum_low95=float(s.low95.min()),maximum_high95=float(s.high95.max()),all_supported_negative=bool(len(s) and (s.high95<0).all()),all_supported_positive=bool(len(s) and (s.low95>0).all())))
 for file,rows in [('sensitivity.csv',allrows),('contrast_summary.csv',summary),('design_summary.csv',design)]:pd.DataFrame(rows).to_csv(OUT/file,index=False,encoding='utf-8-sig')
 np.savez_compressed(OUT/'origin_totals.npz',**totals);np.savez_compressed(OUT/'bootstrap_draws.npz',**drawpack);np.savez_compressed(OUT/'bootstrap_replicates.npz',**replicates);js(OUT/'contrasts.json',contrasts);js(OUT/'source_hashes.json',{p:sha(p) for p in sorted(sourcefiles)})
 for n,proof in parentproof.items():
  root=ART/proof['folder'];assert sha(root/'final_manifest.json')==proof['manifest_sha256']
  for file,h in read(root/'final_manifest.json').items():assert sha(file)==h,file
 js(OUT/'evaluation_complete.json',dict(status='COMPLETE',contrasts=22,settings=8,rows=176,unique_origins=880,parents=parentproof,utc=datetime.now(timezone.utc).isoformat()));print('COMPLETE',flush=True)

if __name__=='__main__':run()

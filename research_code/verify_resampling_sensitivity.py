"""Independent source losses, multinomial block counts and percentile audit."""
from pathlib import Path
from functools import lru_cache
import hashlib,json
import numpy as np
import pandas as pd
from model import ART
from research_crps_distribution import guard
OUT=ART/'resampling_sensitivity_v1';FOLDS={'M05':'2024-05-01','M07':'2024-07-01','M09':'2024-09-01','M11':'2024-11-01'};PARENTS={32:'multiscale_context_v1',35:'rolling_copula_v1'};SEEDS=[20241101,20241102,20241103]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,tol=3e-12):np.testing.assert_allclose(a,b,rtol=0,atol=tol,equal_nan=True)
@lru_cache(maxsize=12)
def loss(file,metric):
 with np.load(file) as z:return np.asarray(z['crps'][:,48:72] if metric=='crps_49_72' else z[metric],dtype=np.float64)

def run():
 guard();assert read(OUT/'evaluation_complete.json')['status']=='COMPLETE';recipe=read(OUT/'recipe.json');assert recipe['replicates']==2000 and recipe['minimum_blocks_per_fold']==4 and recipe['minimum_valid_draws']==1900 and recipe['post_hoc']
 for file,h in read(OUT/'identity.json').items():assert sha(file)==h,file
 for p in recipe['parents'].values():
  root=ART/p['folder'];assert sha(root/'final_manifest.json')==p['manifest_sha256']
  for file,h in read(root/'final_manifest.json').items():assert sha(file)==h,file
 for file,h in read(OUT/'source_hashes.json').items():assert sha(file)==h,file
 contrasts=read(OUT/'contrasts.json');assert len(contrasts)==22;stored=np.load(OUT/'origin_totals.npz');sdraw=np.load(OUT/'bootstrap_draws.npz');srep=np.load(OUT/'bootstrap_replicates.npz');df=pd.read_csv(OUT/'sensitivity.csv');summary=pd.read_csv(OUT/'contrast_summary.csv');design=pd.read_csv(OUT/'design_summary.csv');assert len(df)==176 and len(summary)==22 and len(design)==64;times=pd.DatetimeIndex(pd.to_datetime(np.load(ART/'data.npz')['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');elapsed={};vectors={};countvectors=0
 for fold,start in FOLDS.items():
  ids=np.load(ART/'seasonal_probability_v1'/fold/'data.npz')['test_ids'];elapsed[fold]=np.array([(times[i]-pd.Timestamp(start,tz=times.tz)).total_seconds()/86400 for i in ids])
 for n,name in PARENTS.items():
  inv={(i['fold'],i['model'],i['seed'],i['method']):i for i in read(ART/name/'inventory.json')};primary=pd.read_csv(ART/name/'paired_comparisons.csv');primary=primary[primary.primary].reset_index(drop=True);cc=[c for c in contrasts if c['round']==n];assert len(cc)==len(primary)
  for index,c in enumerate(cc):
   old=primary.iloc[index];assert c['id']==f'r{n}_c{index:02d}'
   for field in ['candidate','method','baseline','baseline_method','metric']:assert c[field]==old[field]
   close([c['original_difference'],c['original_low95'],c['original_high95']],[old.difference,old.low95,old.high95]);assert c['original_n']==old.n
   for fold in FOLDS:
    sums=np.zeros(len(elapsed[fold]));counts=None;mask0=None
    for seed in SEEDS:
     a=loss(inv[fold,c['candidate'],seed,c['method']]['score_source'],c['metric']);b=loss(inv[fold,c['baseline'],seed,c['baseline_method']]['score_source'],c['metric']);assert a.shape==b.shape;mask=np.isfinite(a);np.testing.assert_array_equal(mask,np.isfinite(b))
     if mask0 is None:mask0=mask;counts=mask.reshape(len(a),-1).sum(1)
     else:np.testing.assert_array_equal(mask,mask0)
     for pos in range(len(a)):sums[pos]+=float(np.sum((a[pos]-b[pos])[mask[pos]]))/3
    close(stored[f'{c["id"]}_{fold}_sums'],sums,3e-10);np.testing.assert_array_equal(stored[f'{c["id"]}_{fold}_counts'],counts);vectors[c['id'],fold]=(sums,counts);countvectors+=1
  for d in [3,7,14,28]:
   for offset in [0,d//2]:
    guard();rng=np.random.default_rng((20261119 if n==32 else 20261123) if (d,offset)==(7,0) else np.random.SeedSequence([20261124,n,d,offset]));blocks={};histograms={}
    for fold in FOLDS:
     block=np.array([int((t+offset)//d) for t in elapsed[fold]]);k=int(block.max())+1;draw=rng.integers(0,k,(2000,k));np.testing.assert_array_equal(draw,sdraw[f'r{n}_d{d}_o{offset}_{fold}']);histograms[fold]=np.stack([np.bincount(sample,minlength=k) for sample in draw]);blocks[fold]=block;record=design[(design['round']==n)&(design.days==d)&(design.offset_days==offset)&(design.fold==fold)].iloc[0];assert record.calendar_blocks==k and record.origins==len(block) and record.minimum_block_origins==np.bincount(block,minlength=k).min() and record.maximum_block_origins==np.bincount(block,minlength=k).max()
    for c in cc:
     total=0.;nobs=0;nums=np.zeros(2000);dens=np.zeros(2000,np.int64);nonempty=[]
     for fold in FOLDS:
      s,count=vectors[c['id'],fold];block=blocks[fold];k=int(block.max())+1;ss=np.array([sum(s[block==i]) for i in range(k)]);nn=np.array([sum(count[block==i]) for i in range(k)],np.int64);nums+=histograms[fold]@ss;dens+=histograms[fold]@nn;total+=sum(s);nobs+=sum(count);nonempty.append(int(np.count_nonzero(nn)))
     valid=dens>0;boot=np.full(2000,np.nan);boot[valid]=nums[valid]/dens[valid];close(boot,srep[f'{c["id"]}_d{d}_o{offset}']);supported=min(nonempty)>=4 and valid.sum()>=1900;lo,hi=np.quantile(boot[valid],[.025,.975]) if supported else [np.nan,np.nan];r=df[(df.contrast_id==c['id'])&(df.days==d)&(df.offset_days==offset)].iloc[0];assert r['round']==n and r.supported==supported and r.n==nobs and r.minimum_blocks==min(nonempty) and r.nonempty_blocks==sum(nonempty) and r.valid_draws==valid.sum();close([r.difference,r.low95,r.high95],[total/nobs,lo,hi]);assert r.direction==('unsupported' if not supported else 'negative' if hi<0 else 'positive' if lo>0 else 'cross_zero')
     if (d,offset)==(7,0):close([lo,hi],[c['original_low95'],c['original_high95']])
    print('RESAMPLING_AUDIT',n,d,offset,flush=True)
 for c in contrasts:
  row=summary[summary.id==c['id']].iloc[0];g=df[df.contrast_id==c['id']];s=g[g.supported]
  for key,value in c.items():
   if isinstance(value,(int,float)):close(row[key],value)
   else:assert row[key]==value
  assert row.supported_settings==len(s) and row.negative_settings==(g.direction=='negative').sum() and row.positive_settings==(g.direction=='positive').sum() and row.cross_zero_settings==(g.direction=='cross_zero').sum() and row.unsupported_settings==(g.direction=='unsupported').sum();close([row.minimum_low95,row.maximum_high95],[s.low95.min(),s.high95.max()]);assert row.all_supported_negative==bool(len(s) and (s.high95<0).all()) and row.all_supported_positive==bool(len(s) and (s.low95>0).all())
 for file,h in read(OUT/'source_hashes.json').items():assert sha(file)==h,file
 result=dict(status='PASS',contrast_fold_vectors=countvectors,bootstrap_arrays=176,replicates_per_array=2000,original_seven_day_comparisons=22,design_rows=64,summary_rows=22,parents=len(recipe['parents']),source_files_unchanged=len(read(OUT/'source_hashes.json')),frozen_identities=len(read(OUT/'identity.json')));(OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':run()

"""Round48 stage A: fixed-cohort marginal/dependence diagnostic, no fitting."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib
import numpy as np
import pandas as pd
from model import ART
from seasonal_probability_core import FOLDS
from window_error_attribution_core import summarize,observation_events,fixed_cohort,selected_bounds,selected_bootstrap
OUT=ART/'window_error_attribution_v1';BASE=ART.parent.parent;SEEDS=[20241101,20241102,20241103];PARENT=ART/'missing_window_bounds_v1';PLAN=BASE/'研究笔记/高概率窗口误判归因与定向改进方案_2026-11-02.md'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def parents():
 counts={}
 for name in ['seasonal_rolling_v1','missing_window_bounds_v1','rolling_copula_v1']:
  m=read(ART/name/'final_manifest.json')
  for file,h in m.items():assert sha(file)==h,file
  counts[name]=len(m)
 return counts
def sources(fold,seed):
 b=next(x for x in read(ART/'seasonal_rolling_v1/inventory.json') if (x['fold'],x['model'],x['method'],x['seed'])==(fold,'gated_graph_atoms','rolling28',seed))
 j=next(x for x in read(ART/'rolling_copula_v1/inventory.json') if (x['fold'],x['model'],x['method'],x['seed'])==(fold,'graph_rolling28','rolling28',seed))
 assert b['source']==j['source'];assert sha(b['source'])==b['source_sha256'];assert sha(b['ranks_source'])==b['ranks_sha256']
 for r in j['ranks']:assert sha(r['source'])==r['sha256']
 return b,j
def rank_set(j):
 ranks=[];which=np.empty(j['origins'],int)
 for n,r in enumerate(j['ranks']):ranks.append(np.load(r['source'])['ranks']);which[r['positions']]=n
 return ranks,which

def diagnose():
 assert not OUT.exists(),'new diagnostic must not overwrite existing evidence'
 OUT.mkdir();counts=parents();(OUT/'protocol_frozen.md').write_bytes(PLAN.read_bytes());identity={str(p):sha(p) for p in [Path(__file__),PLAN,ART.parent/'window_error_attribution_core.py',ART.parent/'tests/test_window_error_attribution.py',ART/'data.npz']}
 js(OUT/'diagnostic_identity.json',identity);js(OUT/'recipe.json',dict(folds=list(FOLDS),seeds=SEEDS,threshold=10,high_probability=.8,cohort='mean of baseline seeds >= .8, common for all methods',bootstrap_seed=20261130,bootstrap_draws=2000,parent_counts=counts,start_utc=datetime.now(timezone.utc).isoformat()))
 inventory=[];rows=[];rng=np.random.default_rng(20261130)
 for fold in FOLDS:
  dest=OUT/fold;dest.mkdir();pop=dict(np.load(PARENT/fold/'population.npz'));np.savez_compressed(dest/'population.npz',**pop);y=np.load(ART/'seasonal_probability_v1'/fold/'test_truth.npz')['y'];draws=rng.integers(0,int(pop['blocks'].max())+1,(2000,int(pop['blocks'].max())+1));np.savez_compressed(dest/'draws.npz',draws=draws)
  observation=[observation_events(yi) for yi in y];obs={k:np.stack([r[k] for r in observation]) for k in observation[0]};np.savez_compressed(dest/'observation_counts.npz',**obs)
  for seed in SEEDS:
   b,j=sources(fold,seed);q=np.load(b['source'])['q'];r0=np.load(b['ranks_source'])['ranks'];rr,which=rank_set(j);base=[];joint=[]
   for pos,qi in enumerate(q):base.append(summarize(qi,r0));joint.append(summarize(qi,rr[which[pos]]))
   for case,data in [('B',base),('J',joint)]:np.savez_compressed(dest/f'diagnostic_{case}_{seed}.npz',**{k:np.stack([r[k] for r in data]) for k in data[0]})
   z=np.load(PARENT/fold/f'probabilities_graph_rolling28_{seed}.npz')
   for kind in ['single','group']:np.testing.assert_allclose(np.stack([r[kind] for r in base]),z[kind],rtol=0,atol=1e-12)
   for k in base[0]:
    if k not in ['single','group']:np.testing.assert_array_equal(np.stack([r[k] for r in base]),np.stack([r[k] for r in joint]))
   inventory.append(dict(fold=fold,seed=seed,base=b,joint=j));print('DIAG',fold,seed,len(q),flush=True)
  masks={kind:fixed_cohort(np.stack([np.load(dest/f'diagnostic_B_{s}.npz')[kind] for s in SEEDS])) for kind in ['single','group']};np.savez_compressed(dest/'fixed_cohorts.npz',**masks)
  for kind in ['single','group']:
   packs=[np.load(dest/f'diagnostic_B_{s}.npz') for s in SEEDS];p=np.mean([z[kind] for z in packs],0);lo=np.mean([z[kind+'_lower'] for z in packs],0);hi=np.mean([z[kind+'_upper'] for z in packs],0);ind=np.mean([z[kind+'_independent'] for z in packs],0);bad=np.mean([z[kind+'_expected_bad'] for z in packs],0);labels=pop[kind+'_labels'];inv=pop[kind+'_inverse'];count=17 if kind=='single' else 3
   for cohort,mask in [('all',np.ones_like(p,dtype=bool)),('baseline_high',masks[kind])]:
    for offset in [-1,0,24,48,66]:
     m=mask.copy()
     if offset!=-1:m&=(np.repeat([0,24,48,66],count)==offset)[None,:]
     n=int(m.sum());success=int((m&(labels==1)).sum());unknown=int((m&np.isnan(labels)).sum());sat=m&(lo>=1-1e-12)
     row=dict(fold=fold,kind=kind,cohort=cohort,offset=offset,n=n,known_success=success,known_failure=int((m&(labels==0)).sum()),unknown=unknown,mean_probability=float(p[m].mean()) if n else np.nan,mean_lower=float(lo[m].mean()) if n else np.nan,mean_upper=float(hi[m].mean()) if n else np.nan,mean_independent=float(ind[m].mean()) if n else np.nan,observed_lower=success/n if n else np.nan,observed_upper=(success+unknown)/n if n else np.nan,dependence_room=float((p-lo)[m].mean()) if n else np.nan,expected_bad=float(bad[m].mean()) if n else np.nan,observed_bad_lower=float(obs[kind+'_bad_lower'][m].mean()) if n else np.nan,observed_bad_upper=float(obs[kind+'_bad_upper'][m].mean()) if n else np.nan,saturated_n=int(sat.sum()),saturated_known_failures=int((sat&(labels==0)).sum()))
     if offset==-1:
      bound=selected_bounds(lo,-np.ones_like(lo),labels,inv,m);boot=selected_bootstrap(lo,-np.ones_like(lo),labels,inv,m,pop['blocks'],draws);valid=np.isfinite(boot['lower'])&np.isfinite(boot['upper']);row.update(marginal_gap_lower=bound['lower'],marginal_gap_upper=bound['upper'],outer_low=float(np.quantile(boot['lower'][valid],.025)) if valid.sum()>=1900 else np.nan,outer_high=float(np.quantile(boot['upper'][valid],.975)) if valid.sum()>=1900 else np.nan)
      np.savez_compressed(dest/f'diagnostic_bounds_{kind}_{cohort}.npz',lower=boot['lower'],upper=boot['upper'],n=boot['n'])
     rows.append(row)
 js(OUT/'source_inventory.json',inventory);pd.DataFrame(rows).to_csv(OUT/'diagnosis.csv',index=False,encoding='utf-8-sig');parents()
 for p,h in identity.items():assert sha(p)==h
 js(OUT/'diagnosis_complete.json',dict(status='COMPLETE',flows=24,seed_origins=2640,unique_origins=880,diagnostic_rows=len(rows),future_labels_fitted=False,utc=datetime.now(timezone.utc).isoformat()))
 print('DIAG COMPLETE',len(rows),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['diagnose','improve'],required=True);a=p.parse_args()
 if a.stage=='diagnose':diagnose()
 else:
  from research_threshold_shift import improve
  improve()

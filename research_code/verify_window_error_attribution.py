"""Independent saved-result audit for round48. Never calls producers or fit_shift."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from model import ART
OUT=ART/'window_error_attribution_v1';FOLDS=['M05','M07','M09','M11'];SEEDS=[20241101,20241102,20241103];OFFSETS=[0,24,48,66];GROUPS=[[0,1,2],[8,9,10],[13,15,16]];BINS=[(0,6),(6,24),(24,48),(48,72)]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,tol=3e-12):np.testing.assert_allclose(a,b,rtol=0,atol=tol,equal_nan=True)

def reference_bound(c,d,y,ix,mask,blocks,draws):
 # Accumulate shared unknown-event coefficients independently with add.at.
 k=int(ix.max())+1;unique=np.full(k,np.nan)
 for value in [0,1]:unique[np.unique(ix[y==value])]=value
 assert not np.intersect1d(np.unique(ix[y==0]),np.unique(ix[y==1])).size
 m=mask.astype(float);dd=np.zeros(k);np.add.at(dd,ix.ravel(),(d*m).ravel());constant=float((c*m).sum());n=int(mask.sum());known=np.isfinite(unique)
 def calc(cc,coef,count):
  return ((cc+coef[...,known]@unique[known]+np.minimum(coef[...,~known],0).sum(-1))/count,(cc+coef[...,known]@unique[known]+np.maximum(coef[...,~known],0).sum(-1))/count)
 lo,hi=calc(constant,dd,n) if n else (np.nan,np.nan)
 K=int(blocks.max())+1;D=np.zeros((K,k));C=np.zeros(K);N=np.zeros(K)
 for i in range(len(c)):
  block=int(blocks[i]);np.add.at(D[block],ix[i],d[i]*m[i]);C[block]+=(c[i]*m[i]).sum();N[block]+=m[i].sum()
 # Draw by directly selecting block matrices, batched to bound memory.
 lows=[];highs=[];counts=[]
 for start in range(0,len(draws),40):
  sample=draws[start:start+40];cc=C[sample].sum(1);coef=D[sample].sum(1);count=N[sample].sum(1);valid=count>0;ll=np.full(len(sample),np.nan);hh=ll.copy()
  if valid.any():ll[valid],hh[valid]=calc(cc[valid],coef[valid],count[valid])
  lows.extend(ll);highs.extend(hh);counts.extend(count)
 return lo,hi,np.array(lows),np.array(highs),np.array(counts),unique

def verify():
 assert read(OUT/'evaluation_complete.json')['status']=='COMPLETE'
 identities={}
 for name in ['diagnostic_identity.json','improvement_identity.json','evaluation_identity.json']:
  for p,h in read(OUT/name).items():assert sha(p)==h,p;identities[p]=h
 parents={}
 for name in ['seasonal_rolling_v1','missing_window_bounds_v1','rolling_copula_v1']:
  for p,h in read(ART/name/'final_manifest.json').items():assert sha(p)==h,p;parents[p]=h
 audit=pd.read_csv(OUT/'shift_audit.csv');inventory=read(OUT/'source_inventory.json');raw=np.load(ART/'data.npz')['raw'];cases={};pops={};masks={};draws={};refits=0;paths=0;marginal_cells=0
 for fold in FOLDS:
  pop=dict(np.load(OUT/fold/'population.npz'));pops[fold]=pop;ids=pop['ids'];truth=np.stack([raw[i+1:i+73] for i in ids]);close(truth,np.load(ART/'seasonal_probability_v1'/fold/'test_truth.npz')['y'],0);draws[fold]=np.load(OUT/fold/'draws.npz')['draws'];masks[fold]=dict(np.load(OUT/fold/'fixed_cohorts.npz'));obs=np.load(OUT/fold/'observation_counts.npz')
  sy=[];gy=[];sl=[];su=[];gl=[];gu=[]
  for off in OFFSETS:
   y=truth[:,off:off+6];bad=(np.isfinite(y)&(y>10)).sum(1);unknown=(~np.isfinite(y)).sum(1);sy.append(np.where(bad>0,0.,np.where(unknown==0,1.,np.nan)));sl.append(bad);su.append(bad+unknown)
   for nodes in GROUPS:
    bb=bad[:,nodes].sum(1);uu=unknown[:,nodes].sum(1);gy.append(np.where(bb>0,0.,np.where(uu==0,1.,np.nan)));gl.append(bb);gu.append(bb+uu)
  for kind,lab,low,up in [('single',np.concatenate(sy,1),np.concatenate(sl,1),np.concatenate(su,1)),('group',np.stack(gy,1),np.stack(gl,1),np.stack(gu,1))]:close(pop[kind+'_labels'],lab,0);close(obs[kind+'_bad_lower'],low,0);close(obs[kind+'_bad_upper'],up,0)
  for seed in SEEDS:
   src=next(r for r in inventory if (r['fold'],r['seed'])==(fold,seed));b=src['base'];j=src['joint'];assert sha(b['source'])==b['source_sha256'];q=np.load(b['source'])['q'].astype(float);changed=q.copy();r0=np.load(b['ranks_source'])['ranks'];rr={r['block']:np.load(r['source'])['ranks'] for r in j['ranks']}
   for block in np.unique(pop['blocks']):
    pos=np.flatnonzero(pop['blocks']==block);issue=int(ids[pos[0]]);saved=np.load(OUT/fold/f'shift_{seed}_w{block}.npz');scale=float(np.load(ART/'seasonal_probability_v1'/fold/'data.npz')['scale'])
    for g,(a,bn) in enumerate(BINS):
     t=ids[:,None,None]+np.arange(a+1,bn+1)[None,:,None];valid=np.isfinite(truth[:,a:bn])&(t<issue)&(t>=issue-672);counts=valid.sum((1,2));use=counts>0;n=int(use.sum());nlabels=int(valid.sum());row=audit[(audit.fold==fold)&(audit.seed==seed)&(audit.block==block)&(audit.lead_start==a+1)].iloc[0];assert row.origins==n and row.labels==nlabels
     if n<20 or nlabels<100:assert saved['delta'][g]==0 and np.isnan(saved['objectives'][g]).all()
     else:
      losses=[]
      for delta in saved['grid']:
       pp=(np.maximum(0,q[:,a:bn]+delta)<=10).mean(-1);loss=(pp-(truth[:,a:bn]<=10))**2;perorigin=np.where(valid,loss,0).sum((1,2))[use]/counts[use];losses.append(perorigin.mean()+.002*(delta/scale)**2)
      close(losses,saved['objectives'][g]);best=np.flatnonzero(np.array(losses)<=min(losses)+1e-12);chosen=min(best,key=lambda i:(abs(saved['grid'][i]),saved['grid'][i]));assert saved['delta'][g]==saved['grid'][chosen]
     changed[pos,a:bn]=np.maximum(0,q[pos,a:bn]+saved['delta'][g]);refits+=1
   for case,qq in [('B',q),('M',changed),('J',q),('MJ',changed)]:
    stored=dict(np.load(OUT/fold/f'diagnostic_{case}_{seed}.npz'));cases[fold,seed,case]=stored;safe_count=(qq<=10).sum(-1);close(stored['hourly'],safe_count/19,0)
    for i in range(len(q)):
     ranks=r0 if case in ['B','M'] else rr[int(pop['blocks'][i])];safe=(ranks//3)<safe_count[i][None,:,:];single=[];group=[];lo_s=[];hi_s=[];lo_g=[];hi_g=[]
     for off in OFFSETS:
      single.extend(safe[:,off:off+6].all(1).sum(0)/57);k=safe_count[i,off:off+6];lo_s.extend(np.maximum(0,k.sum(0)-5*19)/19);hi_s.extend(k.min(0)/19)
      for nodes in GROUPS:group.append(safe[:,off:off+6][:,:,nodes].all((1,2)).sum()/57);kk=k[:,nodes];lo_g.append(max(0,int(kk.sum())-17*19)/19);hi_g.append(kk.min()/19)
     for key,val in [('single',single),('group',group),('single_lower',lo_s),('single_upper',hi_s),('group_lower',lo_g),('group_upper',hi_g)]:close(stored[key][i],val)
     paths+=1
    if case in ['B','M']:
     saved=np.load(OUT/fold/f'hourly_scores_{case}_{seed}.npz');error=truth[...,None]-qq;tau=(np.arange(19)+.5)/19;crps=2*np.mean((tau-(error<0))*error,axis=-1);hourly=np.where(np.isfinite(truth),((safe_count/19)-(truth<=10))**2,np.nan);close(saved['crps'],crps);close(saved['brier'],hourly);marginal_cells+=int(np.isfinite(truth).sum())
   for left,right in [('B','J'),('M','MJ')]:
    for key in cases[fold,seed,left]:
     if key not in ['single','group']:close(cases[fold,seed,left][key],cases[fold,seed,right][key],0)
  for kind in ['single','group']:np.testing.assert_array_equal(masks[fold][kind],np.mean([cases[fold,s,'B'][kind] for s in SEEDS],0)>=.8)
  print('AUDIT SOURCES',fold,refits,paths,flush=True)
 # Reconstruct all scopes without importing evaluation population() or bound code.
 refs={};bound_rows=0;bootstrap_draws=0
 for scope in ['ALL',*FOLDS]:
  fs=FOLDS if scope=='ALL' else [scope]
  for kind in ['single','group']:
   y=np.concatenate([pops[f][kind+'_labels'] for f in fs]);keys=np.concatenate([pops[f][kind+'_keys'][pops[f][kind+'_inverse']].reshape(-1,2) for f in fs]);_,ix=np.unique(keys,axis=0,return_inverse=True);ix=ix.reshape(y.shape);blocks=np.concatenate([pops[f]['blocks']+9*i for i,f in enumerate(fs)]);dd=np.concatenate([draws[f]+9*i for i,f in enumerate(fs)],axis=1);high=np.concatenate([masks[f][kind] for f in fs]);pp={c:np.stack([np.concatenate([cases[f,s,c][kind] for f in fs]) for s in SEEDS]) for c in ['B','M','J','MJ']};refs[scope,kind]=(y,ix,blocks,dd,high,pp)
 allrows=pd.concat([pd.read_csv(OUT/'absolute_bounds.csv'),pd.read_csv(OUT/'paired_bounds.csv')],ignore_index=True)
 for row in allrows.itertuples():
  y,ix,blocks,dd,high,pp=refs[row.scope,row.kind];pa=pp[row.candidate];pb=None if row.baseline=='absolute' else pp[row.baseline];c=np.mean(pa**2,0) if pb is None else np.mean(pa**2-pb**2,0);d=1-2*np.mean(pa,0) if pb is None else 2*np.mean(pb-pa,0);mask=high if row.cohort=='baseline_high' else np.ones_like(y,bool);lo,hi,bl,bu,bn,unique=reference_bound(c,d,y,ix,mask,blocks,dd);close([lo,hi],[row.lower,row.upper]);assert row.n==mask.sum();saved=np.load(OUT/f'bounds_{row.scope}_{row.kind}_{row.cohort}_{row.candidate}_{row.baseline}.npz');close(saved['bootstrap_lower'],bl);close(saved['bootstrap_upper'],bu);close(saved['bootstrap_n'],bn,0);valid=np.isfinite(bl)&np.isfinite(bu);assert valid.sum()==row.valid_draws
  if row.interval_supported:close([np.quantile(bl[valid],.025),np.quantile(bu[valid],.975)],[row.outer_low,row.outer_high])
  for side,val in [('lower',lo),('upper',hi)]:
   w=saved['witness_'+side];close(w[np.isfinite(unique)],unique[np.isfinite(unique)],0);close(np.mean((c+d*w[ix])[mask]),val)
  bound_rows+=1;bootstrap_draws+=len(dd)
  if bound_rows%20==0:print('AUDIT BOUNDS',bound_rows,flush=True)
 # All reliability rows on fixed and own-selected cohorts.
 reli=0
 for file in ['fixed_cohort_reliability.csv','offset_reliability.csv','candidate_high_coverage.csv']:
  for row in pd.read_csv(OUT/file).itertuples():
   y,ix,blocks,dd,high,pp=refs[row.scope,row.kind];p=pp[row.case].mean(0);mask=(p>=.8) if file=='candidate_high_coverage.csv' else (high.copy() if row.cohort=='baseline_high' else np.ones_like(y,bool))
   if file=='offset_reliability.csv':mask&=(np.repeat(OFFSETS,17 if row.kind=='single' else 3)==row.offset)[None,:]
   n=int(mask.sum());assert n==row.n
   if n:close([p[mask].mean(),(y[mask]==1).sum()/n,((y[mask]==1).sum()+np.isnan(y[mask]).sum())/n],[row.mean_probability,row.observed_lower,row.observed_upper])
   reli+=1
 for p,h in identities.items():assert sha(p)==h
 result=dict(status='PASS',source_identities=len(identities),parent_frozen_files=len(parents),refit_lead_groups=refits,scenario_forecasts_reconstructed=paths,marginal_observed_cells_recomputed=marginal_cells,paired_and_absolute_bound_rows=bound_rows,bootstrap_draws_recomputed=bootstrap_draws,reliability_rows=reli,known_truth_from_raw=True,all_scope_unknowns_shared_across_folds=True,method='independent integer-rank event reconstruction; origin-mean grid refits; midpoint pinball CRPS; add.at shared-event bounds and selected-block resampling',utc=datetime.now(timezone.utc).isoformat())
 assert paths==10560 and refits==432 and bound_rows==140
 (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':verify()

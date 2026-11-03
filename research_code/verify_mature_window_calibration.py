"""Independent sklearn fits, raw-window populations and sparse Brier-bound audit."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from model import ART
from verify_missing_window_bounds import population,reference_bounds
from mature_window_calibration_core import fit_mature_map
from research_crps_distribution import guard
OUT=ART/'mature_window_calibration_v2';PARENT=ART/'missing_window_bounds_v1';FOLDS=['M05','M07','M09','M11'];SEEDS=[20241101,20241102,20241103];CASES=['base','calibrated'];PAIRS=[('calibrated','base')]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,tol=3e-12):np.testing.assert_allclose(a,b,rtol=0,atol=tol,equal_nan=True)
def table(file,rows,keys):
 a=pd.read_csv(OUT/file,float_precision='round_trip').sort_values(keys).reset_index(drop=True);b=pd.DataFrame(rows).sort_values(keys).reset_index(drop=True);assert len(a)==len(b)
 for k in b:
  if b[k].dtype.kind in 'biufc':close(a[k],b[k])
  else:assert a[k].fillna('').tolist()==b[k].fillna('').tolist(),(file,k)
 return len(a)
def reference_fit(ps,pg,ys,yg,ids,update):
 rows=[]
 for i,issue in enumerate(ids):
  selected=[]
  for kind,(prob,label,nodes) in enumerate([(ps,ys,17),(pg,yg,3)]):
   entries=[]
   for j in range(4*nodes):
    end=int(issue)+[6,30,54,72][j//nodes]
    if update-672<=end<update and np.isfinite(label[i,j]):entries.append((j,end))
   if entries:selected.append((kind,prob,label,entries))
  for kind,prob,label,entries in selected:
   for j,end in entries:rows.append([i,kind,j,end,min(1,max(0,prob[i,j])),label[i,j],1/len(selected)/len(entries)])
 t=np.asarray(rows,float).reshape(-1,7);n=len(set(t[:,0]));fallback=n<20
 if fallback:x=np.array([0.,1.]);v=x.copy()
 else:
  anchors=np.arange(21)/20;xx=np.r_[t[:,4],anchors];yy=np.r_[t[:,5],anchors];ww=np.r_[t[:,6],np.full(21,20/21)];model=IsotonicRegression(out_of_bounds='clip').fit(xx,yy,sample_weight=ww);x=np.unique(xx);v=model.predict(x)
 return x,v,t,dict(update=int(update),valid_origins=n,valid_labels=len(t),training_weight=float(t[:,6].sum()),earliest_end=int(t[:,3].min()) if len(t) else None,latest_end=int(t[:,3].max()) if len(t) else None,fallback=fallback,prior_weight=20.,lookback=672.,minimum_origins=20)
def run():
 guard();assert read(OUT/'prediction_complete.json')['status']==read(OUT/'evaluation_complete.json')['status']=='COMPLETE';identity=read(OUT/'identity.json');parent=read(PARENT/'final_manifest.json')
 for file,h in {**parent,**identity}.items():assert sha(file)==h,file
 archive=read(ART/'mature_window_calibration_v1/attempt1_archive_manifest.json')
 for file,h in archive.items():assert sha(file)==h,file
 recipe=read(OUT/'recipe.json');assert recipe['source_case']=='graph_rolling28' and recipe['lookback_hours']==672 and recipe['minimum_origins']==recipe['prior_weight']==20 and recipe['bootstrap_seed']==20261129;raw=np.load(ART/'data.npz')['raw'];inventory=read(OUT/'inventory.json');assert len(inventory)==24;probs={};pops={};updates=[];multiplicity={};rng=np.random.default_rng(20261129);counts=dict(populations=0,sklearn_map_refits=0,future_pollution_checks=0,output_arrays=0,forecast_origins=0,shared_witnesses=0,bootstrap_arrays=0,brute_force_assignments=0);high=[]
 for fold in FOLDS:
  savedpop=np.load(OUT/fold/'population.npz');assert sha(OUT/fold/'population.npz')==sha(PARENT/fold/'population.npz');ids=savedpop['ids'];blocks=savedpop['blocks'];k=int(blocks.max())+1;draw=rng.integers(0,k,(2000,k));np.testing.assert_array_equal(draw,np.load(OUT/fold/'bootstrap_draws.npz')['draws']);multiplicity[fold]=np.array([np.bincount(row,minlength=k)[blocks] for row in draw])
  for kind in ['single','group']:
   pop=population(raw,ids,kind);pop['blocks']=blocks;pops[fold,kind]=pop
   for name in ['keys','inverse','labels','complete','unique_labels']:np.testing.assert_array_equal(pop[name],savedpop[kind+'_'+name])
   counts['populations']+=1
  ys=np.where(pops[fold,'single']['complete'],pops[fold,'single']['labels'],np.nan);yg=np.where(pops[fold,'group']['complete'],pops[fold,'group']['labels'],np.nan)
  for seed in SEEDS:
   source=PARENT/fold/f'probabilities_graph_rolling28_{seed}.npz';assert sha(source)==sha(OUT/fold/f'probabilities_base_{seed}.npz');z=np.load(source);ps=np.rint(z['single']*57)/57;pg=np.rint(z['group']*57)/57;close(ps,z['single'],1e-12);close(pg,z['group'],1e-12);out_s=np.empty_like(ps);out_g=np.empty_like(pg)
   for block in np.unique(blocks):
    guard();positions=np.flatnonzero(blocks==block);u=int(ids[positions[0]]);x,v,t,audit=reference_fit(ps,pg,ys,yg,ids,u);path=OUT/fold/f'map_{seed}_w{int(block)}';saved=np.load(str(path)+'.npz');close(saved['training'],t,3e-14)
    # np.linspace and arange/20 anchors may differ by one ULP; compare functions.
    close(np.interp(saved['knots'],x,v),saved['values'],3e-12);assert np.all(np.diff(saved['values'])>=-1e-14);counts['sklearn_map_refits']+=1
    a=np.interp(ps[positions],x,v);b0=np.interp(pg[positions],x,v);b=b0.copy();triplets=[[0,1,2],[8,9,10],[13,15,16]]
    for offset in range(4):
     for j,nodes in enumerate(triplets):
      single=a[:,np.array(nodes)+offset*17];lo=np.maximum(0,single.sum(1)-2);hi=single.min(1);b[:,offset*3+j]=np.maximum(lo,np.minimum(hi,b[:,offset*3+j]));assert (b[:,offset*3+j]>=lo-1e-12).all() and (b[:,offset*3+j]<=hi+1e-12).all()
    out_s[positions]=a;out_g[positions]=b;delta=abs(b-b0);record=dict(fold=fold,seed=seed,block=int(block),applied_origins=len(positions),**audit,projected=int((delta>1e-12).sum()),maximum_shift=float(delta.max()),absolute_shift_sum=float(delta.sum()));jsonrow=read(str(path)+'.json')
    for field,value in record.items():
     if isinstance(value,(float,int,bool)) and value is not None:close(jsonrow[field],value)
     else:assert jsonrow[field]==value
    updates.append(record)
    sy=ys.copy();gy=yg.copy();sp=ps.copy();gp=pg.copy();sy[ids[:,None]+np.repeat([6,30,54,72],17)>=u]=9;gy[ids[:,None]+np.repeat([6,30,54,72],3)>=u]=9;sp[ids>=u]=np.nan;gp[ids>=u]=np.nan;poison=fit_mature_map(sp,gp,sy,gy,ids,u)
    for key in ['knots','values','training']:np.testing.assert_array_equal(poison[key],saved[key])
    counts['future_pollution_checks']+=1
   pred=np.load(OUT/fold/f'probabilities_calibrated_{seed}.npz');close(pred['single'],out_s);close(pred['group'],out_g);counts['output_arrays']+=2
   for case in CASES:
    item=next(r for r in inventory if (r['fold'],r['case'],r['seed'])==(fold,case,seed));assert item['origins']==len(ids) and sha(item['probability_source'])==item['probability_sha256'] and sha(item['base_source'])==item['base_sha256']==sha(source);counts['forecast_origins']+=len(ids)
  for case in CASES:
   packs=[np.load(OUT/fold/f'probabilities_{case}_{seed}.npz') for seed in SEEDS]
   for kind in ['single','group']:
    p=np.stack([z[kind] for z in packs]);probs[fold,case,kind]=p;label=np.broadcast_to(pops[fold,kind]['labels'],p.shape);selected=p>=.8;n=int(selected.sum());success=int((selected&(label==1)).sum());unknown=int((selected&np.isnan(label)).sum());high.append(dict(fold=fold,case=case,metric=kind,n=n,total=p.size,known_successes=success,known_failures=int((selected&(label==0)).sum()),unknown=unknown,probability_sum=float(p[selected].sum()),mean_probability=float(p[selected].mean()) if n else np.nan,observed_lower=success/n if n else np.nan,observed_upper=(success+unknown)/n if n else np.nan))
  print('MAP_AUDIT',fold,flush=True)
 counts['update_rows']=table('update_audit.csv',updates,['fold','seed','block']);assert len(updates)==108
 for case in CASES:
  for kind in ['single','group']:
   g=[r for r in high if r['case']==case and r['metric']==kind and r['fold']!='ALL'];n=sum(r['n'] for r in g);success=sum(r['known_successes'] for r in g);unknown=sum(r['unknown'] for r in g);psum=sum(r['probability_sum'] for r in g);high.append(dict(fold='ALL',case=case,metric=kind,n=n,total=sum(r['total'] for r in g),known_successes=success,known_failures=sum(r['known_failures'] for r in g),unknown=unknown,probability_sum=psum,mean_probability=psum/n if n else np.nan,observed_lower=success/n if n else np.nan,observed_upper=(success+unknown)/n if n else np.nan))
 counts['high_rows']=table('high_confidence_bounds.csv',high,['fold','case','metric']);absolute=[];paired=[]
 for candidate,baseline in [('base',None),('calibrated',None),('calibrated','base')]:
  for kind in ['single','group']:
   results={}
   for fold in FOLDS:
    guard();pop=pops[fold,kind];pa=probs[fold,candidate,kind];pb=probs[fold,baseline,kind] if baseline else None;r=reference_bounds(pa,pb,pop,multiplicity[fold]);results[fold]=r;z=np.load(OUT/fold/f'bounds_{candidate}__{baseline or "absolute"}__{kind}.npz');close(z['c'],r['c'],3e-14);close(z['d'],r['d'],3e-14)
    for side in ['lower','upper']:
     close(z['bootstrap_'+side],r['bootstrap_'+side]);w=z['witness_'+side];assert np.isin(w,[0,1]).all();known=np.isfinite(pop['unique_labels']);np.testing.assert_array_equal(w[known],pop['unique_labels'][known]);direct=((pa-w[pop['inverse']][None])**2).mean()
     if pb is not None:direct-=((pb-w[pop['inverse']][None])**2).mean()
     close(direct,r[side]);counts['shared_witnesses']+=1;counts['bootstrap_arrays']+=1
    np.testing.assert_array_equal(z['bootstrap_n'],r['bootstrap_n'])
    if baseline:
     unknown=np.flatnonzero(np.isnan(pop['unique_labels']))[:8];base=z['witness_lower'].copy();values=[]
     for assignment in itertools.product([0.,1.],repeat=len(unknown)):
      w=base.copy();w[unknown]=assignment;values.append(float((r['c']+r['d']*w[pop['inverse']]).mean()));counts['brute_force_assignments']+=1
     close(min(values),r['lower']);close(max(values),r['lower']+np.abs(r['D'][unknown]).sum()/r['n'])
   for scope in ['ALL']+FOLDS:
    selected=FOLDS if scope=='ALL' else [scope];n=sum(results[f]['n'] for f in selected);den=sum(results[f]['bootstrap_n'] for f in selected);lb=sum(results[f]['bootstrap_lower']*results[f]['bootstrap_n'] for f in selected)/den;ub=sum(results[f]['bootstrap_upper']*results[f]['bootstrap_n'] for f in selected)/den;valid=np.isfinite(lb)&np.isfinite(ub);ks=[len(np.unique(pops[f,kind]['blocks'])) for f in selected];supported=min(ks)>=4 and valid.sum()>=1900;lq=np.quantile(lb[valid],[.025,.975]) if supported else [np.nan,np.nan];uq=np.quantile(ub[valid],[.025,.975]) if supported else [np.nan,np.nan];cn=sum(results[f]['complete_case_n'] for f in selected)
    row=dict(scope=scope,candidate=candidate,baseline=baseline or 'absolute',metric=kind,primary=bool(baseline and scope=='ALL'),**{key:sum(results[f][key]*results[f]['n'] for f in selected)/n for key in ['lower','upper','pointwise_lower','pointwise_upper']},n=n,**{key:sum(results[f][key] for f in selected) for key in ['known_targets','unknown_targets','physical_events','unknown_physical_events']},complete_case_value=sum(results[f]['complete_case_sum'] for f in selected)/cn,complete_case_n=cn,lower_percentile_low=lq[0],lower_percentile_high=lq[1],upper_percentile_low=uq[0],upper_percentile_high=uq[1],outer_low=lq[0],outer_high=uq[1],valid_draws=int(valid.sum()),minimum_blocks_in_fold=min(ks),resampling_supported=supported);(paired if baseline else absolute).append(row)
   print('BOUND_AUDIT',candidate,baseline,kind,flush=True)
 counts['absolute_rows']=table('absolute_bounds.csv',absolute,['scope','candidate','metric']);counts['paired_rows']=table('paired_bounds.csv',paired,['scope','candidate','baseline','metric']);assert sum(r['primary'] for r in paired)==2;rel=[]
 for scope in ['ALL']+FOLDS:
  folds=FOLDS if scope=='ALL' else [scope]
  for case in CASES:
   for kind in ['single','group']:
    pp=[];yy=[]
    for fold in folds:
     a=probs[fold,case,kind];y=np.broadcast_to(pops[fold,kind]['labels'],a.shape);known=np.isfinite(y);pp.extend(a[known]);yy.extend(y[known])
    pp=np.array(pp);yy=np.array(yy);bins=np.searchsorted(np.arange(1,10)/10,pp,side='right')
    for b in range(10):use=bins==b;rel.append(dict(scope=scope,case=case,metric=kind,bin=b,n=int(use.sum()),mean_probability=float(pp[use].mean()) if use.any() else np.nan,observed_rate=float(yy[use].mean()) if use.any() else np.nan))
 counts['reliability_rows']=table('reliability.csv',rel,['scope','case','metric','bin']);assert counts['forecast_origins']==5280 and counts['output_arrays']==24 and counts['sklearn_map_refits']==counts['future_pollution_checks']==108
 for file,h in {**parent,**identity}.items():assert sha(file)==h,file
 counts.update(status='PASS',parent_manifest_files=len(parent),frozen_identities=len(identity),scope='causal event calibration and shared-label diagnostic; not a complete joint trajectory model');(OUT/'verification.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(json.dumps(counts,indent=2),flush=True)
if __name__=='__main__':run()

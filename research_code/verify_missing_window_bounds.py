"""Independent shared-event reconstruction, sparse resampling and exhaustive witnesses."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from model import ART
from research_crps_distribution import guard
OUT=ART/'missing_window_bounds_v1';SRC=ART/'seasonal_probability_v1';PARENT=ART/'window_pooling_v1';ROLLING=ART/'seasonal_rolling_v1';POOL=ART/'probability_pooling_v1'
FOLDS=['M05','M07','M09','M11'];SEEDS=[20241101,20241102,20241103];CASES=['graph_raw','graph_frozen','graph_rolling28','own_rolling28','graph_online','graph_crps_exact','graph_window','own_window'];PAIRS=[('graph_window','graph_crps_exact'),('graph_window','graph_rolling28'),('graph_online','graph_rolling28'),('graph_rolling28','graph_frozen'),('graph_window','own_window'),('graph_rolling28','own_rolling28')]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=3e-12):np.testing.assert_allclose(a,b,rtol=0,atol=tol,equal_nan=True)
def table(file,rows,keys):
 a=pd.read_csv(OUT/file,float_precision='round_trip').sort_values(keys).reset_index(drop=True);b=pd.DataFrame(rows).sort_values(keys).reset_index(drop=True);assert len(a)==len(b)
 for k in b:
  if b[k].dtype.kind in 'biufc':close(a[k],b[k])
  else:assert a[k].tolist()==b[k].tolist(),(file,k)
 return len(a)
def population(raw,ids,kind):
 nodes=[[i] for i in range(17)] if kind=='single' else [[0,1,2],[8,9,10],[13,15,16]];keys=[];labels=[];complete=[]
 for issue in ids:
  kk=[];yy=[];cc=[]
  for offset in [0,24,48,66]:
   for j,group in enumerate(nodes):
    obs=raw[issue+offset+1:issue+offset+7][:,group];known=obs[np.isfinite(obs)];full=len(known)==obs.size;label=0. if (known>10).any() else 1. if full else np.nan;kk.append((int(issue+offset+1),j));yy.append(label);cc.append(full)
  keys.append(kk);labels.append(yy);complete.append(cc)
 unique=sorted(set(k for row in keys for k in row));lookup={k:i for i,k in enumerate(unique)};inverse=np.array([[lookup[k] for k in row] for row in keys]);labels=np.array(labels);uy=np.empty(len(unique))
 for j in range(len(unique)):
  v=labels[inverse==j];assert ((np.isnan(v)&np.isnan(v[0]))|(v==v[0])).all();uy[j]=v[0]
 starts=np.unique(np.asarray(unique)[:,0]);assert np.diff(starts).min()>=6;assert len(set(sum(nodes,[])))==sum(map(len,nodes));return dict(keys=np.array(unique),inverse=inverse,labels=labels,complete=np.array(complete),unique_labels=uy)
def probability(item):
 r=item['source_item'];ranks=np.load(r['ranks_source'])['ranks'];assert sha(Path(r['ranks_source']))==r['ranks_sha256'];mixture='experts' in r;qs=[]
 for source in r['experts'] if mixture else [r]:assert sha(Path(source['source']))==source['source_sha256'];qs.append(np.load(source['source'])[source['source_key']])
 weights=np.load(r['weights_source'])['weights'] if mixture else np.ones((len(qs[0]),1))
 if mixture:assert sha(Path(r['weights_source']))==r['weights_sha256']
 out={'single':[],'group':[]}
 for i in range(len(qs[0])):
  # Direct order-statistic indexing, independent of couple_atoms/weighted_paths.
  paths=np.concatenate([q[i][np.arange(72)[None,:,None],np.arange(17)[None,None,:],ranks//3] for q in qs]);w=np.repeat(weights[i]/57,57);w=w/w.sum();single=[];groups=[]
  for start in [0,24,48,66]:
   condition=np.logical_and.reduce(paths[:,start:start+6]<=10,axis=1);single.extend(np.dot(w,condition).tolist())
   for nodes in [[0,1,2],[8,9,10],[13,15,16]]:groups.append(float(w@np.logical_and.reduce(condition[:,nodes],axis=1)))
  out['single'].append(single);out['group'].append(groups)
 return {k:np.array(v) for k,v in out.items()}
def reference_bounds(pa,pb,pop,origin_multiplicities):
 # Brier loss at y=0 and y=1; no production coefficient helpers.
 zero=(pa**2).mean(0);one=((pa-1)**2).mean(0)
 if pb is not None:zero-=(pb**2).mean(0);one-=((pb-1)**2).mean(0)
 delta=one-zero;labels=pop['labels'];inv=pop['inverse'];uy=pop['unique_labels'];known=np.isfinite(uy);n=zero.size;group=coo_matrix((delta.ravel(),(inv.ravel(),np.repeat(np.arange(len(zero)),zero.shape[1]))),shape=(len(uy),len(zero))).tocsr();D=np.asarray(group.sum(1)).ravel();constant=zero.sum();fixed=constant+np.dot(D[known],uy[known]);lower=(fixed+np.minimum(D[~known],0).sum())/n;upper=(fixed+np.maximum(D[~known],0).sum())/n;pl=np.where(np.isfinite(labels),zero+delta*np.nan_to_num(labels),np.minimum(zero,one)).mean();pu=np.where(np.isfinite(labels),zero+delta*np.nan_to_num(labels),np.maximum(zero,one)).mean();Dboot=group.dot(origin_multiplicities.T).T;N=origin_multiplicities.sum(1)*zero.shape[1];C=origin_multiplicities@zero.sum(1);fixedboot=C+Dboot[:,known]@uy[known];lb=(fixedboot+np.minimum(Dboot[:,~known],0).sum(1))/N;ub=(fixedboot+np.maximum(Dboot[:,~known],0).sum(1))/N;complete=pop['complete'];return dict(lower=lower,upper=upper,pointwise_lower=pl,pointwise_upper=pu,n=n,known_targets=int(np.isfinite(labels).sum()),unknown_targets=int(np.isnan(labels).sum()),physical_events=len(uy),unknown_physical_events=int(np.isnan(uy).sum()),complete_case_sum=float((zero+delta*np.nan_to_num(labels))[complete].sum()),complete_case_n=int(complete.sum()),bootstrap_lower=lb,bootstrap_upper=ub,bootstrap_n=N,c=zero,d=delta,D=D)
def run():
 guard();assert read(OUT/'evaluation_complete.json')['status']=='COMPLETE';parents={}
 for root in [PARENT,POOL,ROLLING]:
  manifest=read(root/'final_manifest.json')
  for file,h in manifest.items():assert sha(Path(file))==h,file
  parents[root.name]=len(manifest)
 for file,h in read(OUT/'identity.json').items():assert sha(Path(file))==h,file
 for file,h in read(OUT/'previous_hashes.json').items():assert sha(ART/file)==h,file
 raw=np.load(ART/'data.npz')['raw'];inventory=read(OUT/'inventory.json');assert len(inventory)==96;pops={};probs={};coverage=[];rng=np.random.default_rng(20261118);multiplicity={};counts=dict(populations=0,unconditional_probability_arrays=0,forecast_cases=0,shared_witnesses=0,bootstrap_arrays=0,brute_force_assignments=0);high=[]
 for name in FOLDS:
  ids=np.load(SRC/name/'data.npz')['test_ids'];saved=np.load(OUT/name/'population.npz');np.testing.assert_array_equal(saved['ids'],ids);np.testing.assert_array_equal(saved['blocks'],np.load(ROLLING/name/'archive_data.npz')['blocks']);blocks=saved['blocks'];k=int(blocks.max())+1;draw=rng.integers(0,k,(2000,k));np.testing.assert_array_equal(draw,np.load(OUT/name/'bootstrap_draws.npz')['draws']);multiplicity[name]=np.array([np.bincount(x,minlength=k)[blocks] for x in draw])
  for kind in ['single','group']:
   pop=population(raw,ids,kind);pop['blocks']=blocks;pops[name,kind]=pop
   for key in ['keys','inverse','labels','complete','unique_labels']:np.testing.assert_array_equal(saved[kind+'_'+key],pop[key])
   y=pop['labels'];complete=pop['complete'];uy=pop['unique_labels'];coverage.append(dict(fold=name,metric=kind,origins=len(ids),targets=y.size,complete_targets=int(complete.sum()),partial_known_failures=int((~complete&np.isfinite(y)).sum()),unknown_targets=int(np.isnan(y).sum()),known_successes=int((y==1).sum()),known_failures=int((y==0).sum()),physical_events=len(uy),unknown_physical_events=int(np.isnan(uy).sum()),repeated_predictions=y.size-len(uy)));counts['populations']+=1
  for case in CASES:
   seedpacks=[]
   for seed in SEEDS:
    item=next(r for r in inventory if (r['fold'],r['case'],r['seed'])==(name,case,seed));assert sha(Path(item['probability_source']))==item['probability_sha256'];reference=probability(item);savedp=np.load(item['probability_source']);old=np.load(item['source_item']['score_source'])
    for kind,oldkey in [('single','prob6'),('group','group_probability')]:
     close(reference[kind],savedp[kind],3e-13);previous=old[oldkey].reshape(len(ids),-1);valid=np.isfinite(previous);close(reference[kind][valid],previous[valid],3e-13);assert np.isfinite(savedp[kind]).all();counts['unconditional_probability_arrays']+=1
    seedpacks.append(dict(savedp));counts['forecast_cases']+=len(ids)
   for kind in ['single','group']:
    p=np.stack([r[kind] for r in seedpacks]);probs[name,case,kind]=p;label=np.broadcast_to(pops[name,kind]['labels'],p.shape);select=p>=.8;n=int(select.sum());success=int((select&(label==1)).sum());unknown=int((select&np.isnan(label)).sum());high.append(dict(fold=name,case=case,metric=kind,n=n,total=p.size,known_successes=success,known_failures=int((select&(label==0)).sum()),unknown=unknown,probability_sum=float(p[select].sum()),mean_probability=float(p[select].mean()) if n else np.nan,observed_lower=success/n if n else np.nan,observed_upper=(success+unknown)/n if n else np.nan))
   print('PROBABILITY_AUDIT',name,case,flush=True)
 for case in CASES:
  for kind in ['single','group']:
   g=[r for r in high if r['case']==case and r['metric']==kind and r['fold']!='ALL'];n=sum(r['n'] for r in g);success=sum(r['known_successes'] for r in g);unknown=sum(r['unknown'] for r in g);psum=sum(r['probability_sum'] for r in g);high.append(dict(fold='ALL',case=case,metric=kind,n=n,total=sum(r['total'] for r in g),known_successes=success,known_failures=sum(r['known_failures'] for r in g),unknown=unknown,probability_sum=psum,mean_probability=psum/n if n else np.nan,observed_lower=success/n if n else np.nan,observed_upper=(success+unknown)/n if n else np.nan))
 counts['coverage_rows']=table('coverage.csv',coverage,['fold','metric']);counts['high_confidence_rows']=table('high_confidence_bounds.csv',high,['fold','case','metric']);absolute=[];paired=[]
 for candidate,baseline in [(c,None) for c in CASES]+PAIRS:
  for kind in ['single','group']:
   results={}
   for name in FOLDS:
    guard();pop=pops[name,kind];pa=probs[name,candidate,kind];pb=probs[name,baseline,kind] if baseline else None;r=reference_bounds(pa,pb,pop,multiplicity[name]);results[name]=r;prefix=f'{candidate}__{baseline or "absolute"}__{kind}';z=np.load(OUT/name/f'bounds_{prefix}.npz');close(z['c'],r['c'],3e-14);close(z['d'],r['d'],3e-14)
    for side in ['lower','upper']:
     close(z['bootstrap_'+side],r['bootstrap_'+side]);w=z['witness_'+side];assert np.isin(w,[0,1]).all();known=np.isfinite(pop['unique_labels']);np.testing.assert_array_equal(w[known],pop['unique_labels'][known]);direct=((pa-w[pop['inverse']][None])**2).mean()
     if pb is not None:direct-=((pb-w[pop['inverse']][None])**2).mean()
     close(direct,r[side]);counts['shared_witnesses']+=1;counts['bootstrap_arrays']+=1
    np.testing.assert_array_equal(z['bootstrap_n'],r['bootstrap_n'])
    if (candidate,baseline)==PAIRS[0]:
     unknown=np.flatnonzero(np.isnan(pop['unique_labels']))[:8];base=z['witness_lower'].copy();values=[]
     for assignment in itertools.product([0.,1.],repeat=len(unknown)):
      witness=base.copy();witness[unknown]=assignment;values.append(float((r['c']+r['d']*witness[pop['inverse']]).mean()));counts['brute_force_assignments']+=1
     close(min(values),r['lower']);expected=r['lower']+np.abs(r['D'][unknown]).sum()/r['n'];close(max(values),expected)
   for scope in ['ALL']+FOLDS:
    selected=FOLDS if scope=='ALL' else [scope];n=sum(results[f]['n'] for f in selected);den=sum(results[f]['bootstrap_n'] for f in selected);lb=sum(results[f]['bootstrap_lower']*results[f]['bootstrap_n'] for f in selected)/den;ub=sum(results[f]['bootstrap_upper']*results[f]['bootstrap_n'] for f in selected)/den;valid=np.isfinite(lb)&np.isfinite(ub);blocks=[len(np.unique(pops[f,kind]['blocks'])) for f in selected];supported=min(blocks)>=4 and valid.sum()>=1900;lq=np.quantile(lb[valid],[.025,.975]) if supported else [np.nan,np.nan];uq=np.quantile(ub[valid],[.025,.975]) if supported else [np.nan,np.nan];cn=sum(results[f]['complete_case_n'] for f in selected);row=dict(scope=scope,candidate=candidate,baseline=baseline or 'absolute',metric=kind,primary=bool(baseline and scope=='ALL'),**{k:sum(results[f][k]*results[f]['n'] for f in selected)/n for k in ['lower','upper','pointwise_lower','pointwise_upper']},n=n,**{k:sum(results[f][k] for f in selected) for k in ['known_targets','unknown_targets','physical_events','unknown_physical_events']},complete_case_value=sum(results[f]['complete_case_sum'] for f in selected)/cn,complete_case_n=cn,lower_percentile_low=lq[0],lower_percentile_high=lq[1],upper_percentile_low=uq[0],upper_percentile_high=uq[1],outer_low=lq[0],outer_high=uq[1],valid_draws=int(valid.sum()),minimum_blocks_in_fold=min(blocks),resampling_supported=supported);(paired if baseline else absolute).append(row)
   print('BOUND_AUDIT',candidate,baseline,kind,flush=True)
 counts['absolute_rows']=table('absolute_bounds.csv',absolute,['scope','candidate','metric']);counts['paired_rows']=table('paired_bounds.csv',paired,['scope','candidate','baseline','metric']);assert sum(r['primary'] for r in paired)==12;assert counts['forecast_cases']==21120;counts.update(status='PASS',parent_manifest_files=parents,frozen_identities=len(read(OUT/'identity.json')),previous_unchanged=len(read(OUT/'previous_hashes.json')));(OUT/'verification.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(json.dumps(counts,indent=2),flush=True)
if __name__=='__main__':run()

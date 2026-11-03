"""Round31: shared physical-window outcomes, no imputation and no forecast changes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from model import ART
from seasonal_probability_core import FOLDS
from research_crps_distribution import guard,SEEDS
from crps_distribution_core import couple_atoms
from probability_pooling_core import weighted_paths
from missing_window_bounds_core import window_labels,physical_mapping,collapse_labels,coefficients,sharp_bounds,bootstrap_bounds,OFFSETS,GROUPS
OUT=ART/'missing_window_bounds_v1';SRC=ART/'seasonal_probability_v1';ROLLING=ART/'seasonal_rolling_v1';PARENT=ART/'window_pooling_v1';POOL=ART/'probability_pooling_v1';PLAN=ART.parent.parent/'研究笔记/缺测窗口共同标签误差界方案_2026-11-02.md'
G='gated_graph_atoms';O='gated_temporal_atoms'
CASES={'graph_raw':(G,'raw'),'graph_frozen':(G,'frozen'),'graph_rolling28':(G,'rolling28'),'own_rolling28':(O,'rolling28'),'graph_online':(G,'online_pool28'),'graph_crps_exact':(G,'crps_exact'),'graph_window':(G,'window_exact1'),'own_window':(O,'window_exact1')}
PAIRS=[('graph_window','graph_crps_exact'),('graph_window','graph_rolling28'),('graph_online','graph_rolling28'),('graph_rolling28','graph_frozen'),('graph_window','own_window'),('graph_rolling28','own_rolling28')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def js(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def parents():
 result={}
 for root in [PARENT,POOL,ROLLING]:
  manifest=read(root/'final_manifest.json')
  for file,h in manifest.items():assert sha(Path(file))==h,file
  result[root.name]=len(manifest)
 return result
def prepare():
 guard();parents();OUT.mkdir(exist_ok=True);protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0];p=OUT/'protocol_frozen.md'
 if p.exists():assert p.read_text(encoding='utf-8')==protocol
 else:p.write_text(protocol,encoding='utf-8')
 files=[p,Path(__file__)]+[Path(__file__).parent/n for n in ['missing_window_bounds_core.py','probability_pooling_core.py','crps_distribution_core.py','tests/test_missing_window_bounds.py']]+[root/'final_manifest.json' for root in [PARENT,POOL,ROLLING]]+[ART/'data.npz'];identity={str(f):sha(f) for f in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==identity
 else:js(OUT/'identity.json',identity)
 active=[OUT,ART/'multiscale_context_v1',ART/'distributional_schedule_v1',ART/'reliability_boundary_v1',ART/'rolling_copula_v1']
 if not (OUT/'previous_hashes.json').exists():js(OUT/'previous_hashes.json',{str(f.relative_to(ART)):sha(f) for f in ART.rglob('*') if f.is_file() and not any(r in f.parents for r in active) and f.suffix in ['.npz','.pt','.csv','.joblib']})
 js(OUT/'recipe.json',dict(cases=CASES,pairs=PAIRS,bootstrap_seed=20261118,bootstrap_replicates=2000,probability_cutoff=.8,outcome_threshold=10,concurrently_active_snapshot_exclusion=[str(p) for p in active[1:]],brute_force_audit='first eight unknown physical windows per fold and metric; other unknown windows held at lower-bound witness; enumerate 256 assignments',start_utc=datetime.now(timezone.utc).isoformat()))
def forecasts():
 inventory=read(PARENT/'inventory.json');raw=np.load(ART/'data.npz')['raw'];rows=[];coverage=[]
 for name in FOLDS:
  guard();path=OUT/name;path.mkdir(exist_ok=True);ids=np.load(SRC/name/'data.npz')['test_ids'];single,group=window_labels(raw,ids);labels={'single':single,'group':group};population=dict(ids=ids,blocks=np.load(ROLLING/name/'archive_data.npz')['blocks'])
  old=np.load(ROLLING/name/f'scores_{G}_{SEEDS[0]}_raw.npz')
  for kind in ['single','group']:
   keys,inv=physical_mapping(ids,kind);label=labels[kind];complete=np.isfinite(old['label6'].reshape(len(ids),-1) if kind=='single' else old['group_label']);unique=collapse_labels(label,inv);population.update({kind+'_labels':label,kind+'_complete':complete,kind+'_keys':keys,kind+'_inverse':inv,kind+'_unique_labels':unique});assert np.isfinite(label[complete]).all();coverage.append(dict(fold=name,metric=kind,origins=len(ids),targets=label.size,complete_targets=int(complete.sum()),partial_known_failures=int((~complete&np.isfinite(label)).sum()),unknown_targets=int(np.isnan(label).sum()),known_successes=int((label==1).sum()),known_failures=int((label==0).sum()),physical_events=len(unique),unknown_physical_events=int(np.isnan(unique).sum()),repeated_predictions=label.size-len(unique)))
  np.savez_compressed(path/'population.npz',**population)
  for case,(model,method) in CASES.items():
   for seed in SEEDS:
    guard();item=next(r for r in inventory if (r['fold'],r['model'],r['method'],r['seed'])==(name,model,method,seed));dest=path/f'probabilities_{case}_{seed}.npz';ranks=np.load(item['ranks_source'])['ranks'];assert sha(Path(item['ranks_source']))==item['ranks_sha256'];mixture='experts' in item
    if mixture:
     qs=[]
     for r in item['experts']:assert sha(Path(r['source']))==r['source_sha256'];qs.append(np.load(r['source'])[r['source_key']])
     q=np.stack(qs);weights=np.load(item['weights_source'])['weights'];assert sha(Path(item['weights_source']))==item['weights_sha256']
    else:assert sha(Path(item['source']))==item['source_sha256'];q=np.load(item['source'])[item['source_key']]
    ps=[];pg=[]
    for i in range(len(ids)):
     if mixture:sc,w=weighted_paths(q[:,i],ranks,weights[i]);w=w/w.sum()
     else:sc=couple_atoms(q[i],ranks);w=np.full(len(sc),1/len(sc))
     a=[];b=[]
     for offset in OFFSETS:
      a.append(np.sum(w[:,None]*(sc[:,offset:offset+6]<=10).all(1),axis=0))
      for nodes in GROUPS:b.append(float(w@(sc[:,offset:offset+6][:,:,nodes]<=10).all((1,2))))
     ps.append(np.concatenate(a));pg.append(b)
    ps=np.asarray(ps);pg=np.asarray(pg);oldscore=np.load(item['score_source']);oldsingle=oldscore['prob6'].reshape(len(ids),-1);oldgroup=oldscore['group_probability']
    for computed,previous in [(ps,oldsingle),(pg,oldgroup)]:ok=np.isfinite(previous);np.testing.assert_allclose(computed[ok],previous[ok],atol=3e-13,rtol=0)
    np.savez_compressed(dest,single=ps,group=pg);rows.append(dict(fold=name,case=case,model=model,seed=seed,method=method,source_item=item,probability_source=str(dest),probability_sha256=sha(dest),origins=len(ids),mixture=mixture));print('UNCONDITIONAL',name,case,seed,flush=True)
 assert len(rows)==96;js(OUT/'inventory.json',rows);csv('coverage.csv',coverage);js(OUT/'probabilities_complete.json',dict(status='COMPLETE',flows=96,forecast_origins=21120,unique_origins=880,utc=datetime.now(timezone.utc).isoformat()))
def analyze():
 rng=np.random.default_rng(20261118);pops={};draws={};probs={};high=[]
 for name in FOLDS:
  path=OUT/name;pops[name]=dict(np.load(path/'population.npz'));k=int(pops[name]['blocks'].max())+1;draws[name]=rng.integers(0,k,(2000,k));np.savez_compressed(path/'bootstrap_draws.npz',draws=draws[name])
  for case in CASES:
   packs=[np.load(path/f'probabilities_{case}_{seed}.npz') for seed in SEEDS]
   for kind in ['single','group']:
    a=np.stack([p[kind] for p in packs]);probs[name,case,kind]=a;label=np.broadcast_to(pops[name][kind+'_labels'],a.shape);select=a>=.8;n=int(select.sum());success=int((select&(label==1)).sum());unknown=int((select&np.isnan(label)).sum());high.append(dict(fold=name,case=case,metric=kind,n=n,total=a.size,known_successes=success,known_failures=int((select&(label==0)).sum()),unknown=unknown,probability_sum=float(a[select].sum()),mean_probability=float(a[select].mean()) if n else np.nan,observed_lower=success/n if n else np.nan,observed_upper=(success+unknown)/n if n else np.nan))
 for case in CASES:
  for kind in ['single','group']:
   r=[r for r in high if r['case']==case and r['metric']==kind and r['fold']!='ALL'];n=sum(x['n'] for x in r);success=sum(x['known_successes'] for x in r);unknown=sum(x['unknown'] for x in r);ps=sum(x['probability_sum'] for x in r);high.append(dict(fold='ALL',case=case,metric=kind,n=n,total=sum(x['total'] for x in r),known_successes=success,known_failures=sum(x['known_failures'] for x in r),unknown=unknown,probability_sum=ps,mean_probability=ps/n if n else np.nan,observed_lower=success/n if n else np.nan,observed_upper=(success+unknown)/n if n else np.nan))
 csv('high_confidence_bounds.csv',high);absolute=[];paired=[]
 for candidate,baseline in [(c,None) for c in CASES]+PAIRS:
  for kind in ['single','group']:
   values={};samples={};complete_sums={};complete_counts={}
   for name in FOLDS:
    guard();pop=pops[name];pa=probs[name,candidate,kind];pb=probs[name,baseline,kind] if baseline else None;c,d=coefficients(pa,pb);label=pop[kind+'_labels'];inv=pop[kind+'_inverse'];value=sharp_bounds(c,d,label,inv);sample=bootstrap_bounds(c,d,label,inv,pop['blocks'],draws[name]);values[name]=value;samples[name]=sample;complete=pop[kind+'_complete'];complete_sums[name]=float((c+d*np.where(np.isfinite(label),label,0))[complete].sum());complete_counts[name]=int(complete.sum());prefix=f'{candidate}__{baseline or "absolute"}__{kind}';np.savez_compressed(OUT/name/f'bounds_{prefix}.npz',c=c,d=d,witness_lower=value['witness_lower'],witness_upper=value['witness_upper'],bootstrap_lower=sample['lower'],bootstrap_upper=sample['upper'],bootstrap_n=sample['n'])
    for side in ['lower','upper']:np.testing.assert_allclose(np.mean(c+d*value['witness_'+side][inv]),value[side],rtol=0,atol=3e-13)
    assert value['pointwise_lower']<=value['lower']+1e-13 and value['upper']<=value['pointwise_upper']+1e-13
   for scope in ['ALL']+list(FOLDS):
    selected=list(FOLDS) if scope=='ALL' else [scope];n=sum(values[f]['n'] for f in selected);lower=sum(values[f]['lower']*values[f]['n'] for f in selected)/n;upper=sum(values[f]['upper']*values[f]['n'] for f in selected)/n;den=sum(samples[f]['n'] for f in selected);lb=sum(samples[f]['lower']*samples[f]['n'] for f in selected)/den;ub=sum(samples[f]['upper']*samples[f]['n'] for f in selected)/den;valid=np.isfinite(lb)&np.isfinite(ub);blocks=[len(np.unique(pops[f]['blocks'])) for f in selected];supported=min(blocks)>=4 and valid.sum()>=1900;lq=np.quantile(lb[valid],[.025,.975]) if supported else [np.nan,np.nan];uq=np.quantile(ub[valid],[.025,.975]) if supported else [np.nan,np.nan];cn=sum(complete_counts[f] for f in selected);row=dict(scope=scope,candidate=candidate,baseline=baseline or 'absolute',metric=kind,primary=bool(baseline and scope=='ALL'),lower=lower,upper=upper,pointwise_lower=sum(values[f]['pointwise_lower']*values[f]['n'] for f in selected)/n,pointwise_upper=sum(values[f]['pointwise_upper']*values[f]['n'] for f in selected)/n,n=int(n),known_targets=int(sum(values[f]['known_targets'] for f in selected)),unknown_targets=int(sum(values[f]['unknown_targets'] for f in selected)),physical_events=sum(values[f]['physical_events'] for f in selected),unknown_physical_events=sum(values[f]['unknown_physical_events'] for f in selected),complete_case_value=sum(complete_sums[f] for f in selected)/cn if cn else np.nan,complete_case_n=cn,lower_percentile_low=float(lq[0]),lower_percentile_high=float(lq[1]),upper_percentile_low=float(uq[0]),upper_percentile_high=float(uq[1]),outer_low=float(lq[0]),outer_high=float(uq[1]),valid_draws=int(valid.sum()),minimum_blocks_in_fold=min(blocks),resampling_supported=supported)
    (paired if baseline else absolute).append(row)
   print('BOUNDS',candidate,baseline,kind,flush=True)
 assert len(absolute)==80 and len(paired)==60 and sum(r['primary'] for r in paired)==12;csv('absolute_bounds.csv',absolute);csv('paired_bounds.csv',paired);parents();js(OUT/'evaluation_complete.json',dict(status='COMPLETE',absolute_rows=80,paired_rows=60,primary_comparisons=12,utc=datetime.now(timezone.utc).isoformat()))
if __name__=='__main__':prepare();forecasts();analyze()

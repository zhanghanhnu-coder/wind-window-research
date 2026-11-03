"""Round26: fixed marginal distributions, partially observed dependence archives."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
from partial_copula_core import fit_dependence,repair_correlation,make_ranks,assert_matured
from crps_distribution_core import couple_atoms
from probability_core import score_scenarios,break_dependence,BINS
from research_crps_distribution import guard,SEEDS,METRICS
from seasonal_probability_core import FOLDS
OUT=ART/'partial_copula_v1';SRC=ART/'seasonal_probability_v1';PLAN=ART.parent.parent/'研究笔记/不完整历史轨迹的联合情景研究方案_2026-11-02.md';MODELS=['gated_graph_atoms','gated_temporal_atoms','gated_mean_atoms'];COMPARE=['brier6','length_crps72','first_cdf_brier','group_brier','temporal_variogram','spatial_variogram']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def js(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def historical_bank(model,seed):return ART/('spatial_outage_controls_v1' if model=='gated_mean_atoms' else 'crps_distribution_v1')/f'bank_{model}_{seed}.npz'
def verify_parent():
 manifest=read(SRC/'final_manifest.json')
 for p,h in manifest.items():assert sha(Path(p))==h,p
 return len(manifest)

def prepare():
 guard();verify_parent();OUT.mkdir(exist_ok=True);frozen=OUT/'protocol_frozen.md';protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0]
 if frozen.exists():assert frozen.read_text(encoding='utf-8')==protocol
 else:frozen.write_text(protocol,encoding='utf-8')
 files=[frozen,Path(__file__)]+[Path(__file__).with_name(n) for n in ['partial_copula_core.py','model.py','crps_distribution_core.py','direct_window_core.py','probability_core.py','research_crps_distribution.py','seasonal_probability_core.py']]+[SRC/'final_manifest.json',ART/'data.npz',ART/'metadata.json',ART/'crps_distribution_v1/evaluation_data.npz']+[historical_bank(m,s) for m in MODELS for s in SEEDS];identity={str(p):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==identity
 else:js(OUT/'identity.json',identity)
 if not (OUT/'previous_hashes.json').exists():js(OUT/'previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 if not (OUT/'recipe.json').exists():js(OUT/'recipe.json',dict(models=MODELS,seeds=SEEDS,folds=list(FOLDS),scenarios=57,atoms=19,main_shrink=.05,graph_secondary_shrink=[0.,.1],minimum_coordinate_support=20,minimum_pair_support=100,eigenvalue_floor=1e-6,noise_seed=20261114,bootstrap_seed=20261114,bootstrap_replicates=2000,start_utc=datetime.now(timezone.utc).isoformat()))

def fit_templates():
 inventory=[];repairs=[];support=[];ev=np.load(ART/'crps_distribution_v1/evaluation_data.npz')
 for fold_index,name in enumerate(FOLDS):
  path=OUT/name;path.mkdir(exist_ok=True);f=np.load(SRC/name/'data.npz');y=np.load(SRC/name/'cal_truth.npz')['y'];assert_matured(f['cal_ids'],int(min(f['test_ids'])))
  for seed in SEEDS:
   base=np.random.default_rng(np.random.SeedSequence([20261114,fold_index,seed])).normal(size=(57,72,17));np.savez_compressed(path/f'noise_{seed}.npz',base=base)
   for model in MODELS:
    guard();q=np.load(SRC/name/f'predictions_{model}_{seed}_budget40.npz')['cal'];dep=fit_dependence(q,y,float(f['scale']),.05);np.savez_compressed(path/f'dependence_{model}_{seed}.npz',**{k:v for k,v in dep.items() if not k.endswith('_info')});details=[]
    for axis in ['time','space']:
     info=dep[axis+'_info'];details.append(dict(axis=axis,**info));repairs.append(dict(fold=name,model=model,seed=seed,method='partial05',axis=axis,**info))
    js(path/f'dependence_{model}_{seed}.json',details);support.append(dict(fold=name,model=model,seed=seed,origins=len(y),min_coordinate=int(np.isfinite(y).sum(0).min()),min_time_pair=int(dep['time_counts'].min()),min_space_pair=int(dep['space_counts'].min()),full_all_node_paths=int(np.isfinite(y).all((1,2)).sum()),latest_cal_label=int(max(f['cal_ids']+72)),first_test_issue=int(min(f['test_ids']))));templates={'partial05':make_ranks(base,dep['time_correlation'],dep['space_correlation'])}
    if model=='gated_graph_atoms':
     templates['time_broken']=break_dependence(templates['partial05'],'time');templates['space_broken']=break_dependence(templates['partial05'],'space')
     for method,shrink in [('partial00',0.),('partial10',.1)]:
      mats={}
      for axis in ['time','space']:
       mats[axis],info=repair_correlation(dep[axis+'_raw'],shrink);repairs.append(dict(fold=name,model=model,seed=seed,method=method,axis=axis,**info))
      templates[method]=make_ranks(base,mats['time'],mats['space'])
    if name=='M11':
     bank=np.load(historical_bank(model,seed));templates['historical_ecc']=bank['ranks'];idx=ev['complete_positions'][ev['selected57']];residual=y[idx]-q[idx,...,9];expected=np.argsort(np.argsort(residual,axis=0,kind='stable'),axis=0,kind='stable');np.testing.assert_array_equal(templates['historical_ecc'],expected);np.testing.assert_array_equal(f['cal_ids'][idx],ev['selected57_ids'])
    for method,ranks in templates.items():
     np.testing.assert_array_equal(np.sort(ranks,axis=0),np.broadcast_to(np.arange(57)[:,None,None],ranks.shape));dest=path/f'ranks_{model}_{seed}_{method}.npz';np.savez_compressed(dest,ranks=ranks.astype(np.int16));inventory.append(dict(fold=name,model=model,seed=seed,method=method,rank_sha256=sha(dest),marginal_sha256=sha(SRC/name/f'calibrated_{model}_{seed}.npz'),origins=int(len(f['test_ids']))))
    print('FITTED',name,model,seed,flush=True)
 assert len(inventory)==93;js(OUT/'inventory.json',inventory);csv('dependence_support.csv',support);csv('matrix_repairs.csv',repairs);verify_parent();js(OUT/'prediction_complete.json',dict(flows=93,dependence_fits=36,forecast_origins=sum(i['origins'] for i in inventory),utc=datetime.now(timezone.utc).isoformat(),representation='frozen 25 marginal arrays plus complete rank templates; no test truth read in fit phase'))

def evaluate():
 assert read(OUT/'prediction_complete.json')['flows']==93;d=np.load(ART/'data.npz');times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');rows=[];months=[];leads=[];rel=[];high=[];conservation=[]
 for item in read(OUT/'inventory.json'):
  guard();name=item['fold'];model=item['model'];seed=item['seed'];method=item['method'];path=OUT/name;f=np.load(SRC/name/'data.npz');q=np.load(SRC/name/f'calibrated_{model}_{seed}.npz')['q'];y=np.load(SRC/name/'test_truth.npz')['y'];ranks=np.load(path/f'ranks_{model}_{seed}_{method}.npz')['ranks'];assert sha(path/f'ranks_{model}_{seed}_{method}.npz')==item['rank_sha256'] and sha(SRC/name/f'calibrated_{model}_{seed}.npz')==item['marginal_sha256'];dest=path/f'scores_{model}_{seed}_{method}.npz';ts=times[f['test_ids']]
  if dest.exists():s=dict(np.load(dest))
  else:
   records=[]
   for i,(qi,yi) in enumerate(zip(q,y)):
    sc=couple_atoms(qi,ranks);np.testing.assert_array_equal(np.sort(sc,axis=0),np.repeat(qi,3,axis=-1).transpose(2,0,1));record=score_scenarios(sc,yi,train_scale=float(f['scale']));gp=[];gl=[]
    for offset in [0,24,48,66]:
     for group in [[0,1,2],[8,9,10],[13,15,16]]:
      truth=yi[offset:offset+6][:,group];ok=np.isfinite(truth).all();gp.append(float((sc[:,offset:offset+6][:,:,group]<=10).all((1,2)).mean()) if ok else np.nan);gl.append(float((truth<=10).all()) if ok else np.nan)
    record['group_probability']=np.array(gp);record['group_label']=np.array(gl);np.testing.assert_allclose(record['group_brier'],(record['group_probability']-record['group_label'])**2,atol=1e-12,rtol=0,equal_nan=True);records.append(record)
    if i%80==0:guard();print('SCORING',name,model,seed,method,i,flush=True)
   s={k:np.stack([r[k] for r in records]) for k in records[0]};np.savez_compressed(dest,**s)
  previous=np.load(SRC/name/f'scores_{model}_{seed}_smooth_affine.npz');maximum=0.
  for metric in ['crps','coverage','width','interval_score','low_miss','high_miss']:
   np.testing.assert_allclose(s[metric],previous[metric],atol=3e-12,rtol=0,equal_nan=True);maximum=max(maximum,float(np.nanmax(abs(s[metric]-previous[metric]))))
  if method=='space_broken':
   full=np.load(path/f'scores_{model}_{seed}_partial05.npz')
   for metric in ['brier3','brier6','brier12','length_crps24','length_crps48','length_crps72','any6_brier24','any6_brier48','any6_brier72','first_cdf_brier']:np.testing.assert_allclose(s[metric],full[metric],atol=3e-12,rtol=0,equal_nan=True)
  conservation.append(dict(fold=name,model=model,seed=seed,method=method,max_marginal_difference=maximum,space_single_node_conserved=method=='space_broken'));common=dict(fold=name,model=model,seed=seed,method=method)
  for metric in METRICS:
   z=s[metric];rows.append(dict(**common,metric=metric,value=float(np.nanmean(z)),n=int(np.isfinite(z).sum())))
   for month in sorted(set(ts.month)):
    a=z[ts.month==month];months.append(dict(**common,issue_month=int(month),metric=metric,value=float(np.nanmean(a)),n=int(np.isfinite(a).sum())))
  for a,b in BINS:
   for metric in ['crps','coverage','width','interval_score','low_miss','high_miss']:
    z=s[metric][:,a:b];leads.append(dict(**common,lead_start=a+1,lead_end=b,metric=metric,value=float(np.nanmean(z)),n=int(np.isfinite(z).sum())))
  for target,pk,yk in [('single_6h','prob6','label6'),('group_6h','group_probability','group_label')]:
   p=s[pk].ravel();label=s[yk].ravel();valid=np.isfinite(label);p=p[valid];label=label[valid];selected=p>=.8;high.append(dict(**common,target=target,n=int(selected.sum()),total=int(len(p)),mean_probability=float(p[selected].mean()) if selected.any() else np.nan,observed_rate=float(label[selected].mean()) if selected.any() else np.nan))
   for b in range(10):
    selected=(p>=b/10)&(p<(b+1)/10 if b<9 else p<=1);rel.append(dict(**common,target=target,bin=b,n=int(selected.sum()),mean_probability=float(p[selected].mean()) if selected.any() else np.nan,observed_rate=float(label[selected].mean()) if selected.any() else np.nan))
  print('RESULT',name,model,seed,method,'Brier6',np.nanmean(s['brier6']),'group',np.nanmean(s['group_brier']),flush=True)
 df=pd.DataFrame(rows)
 for key,g in df.groupby(['model','seed','method','metric'],sort=False):
  if g.fold.nunique()==4:rows.append(dict(fold='ALL',**dict(zip(['model','seed','method','metric'],key)),value=float(np.sum(g.value*g.n)/g.n.sum()),n=int(g.n.sum())))
 csv('metrics.csv',rows);csv('month_metrics.csv',months);csv('lead_metrics.csv',leads);csv('high_confidence.csv',high);csv('reliability.csv',rel);csv('marginal_conservation.csv',conservation)

def compare():
 d=np.load(ART/'data.npz');times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');rng=np.random.default_rng(20261114);weeks={};draws={};fold_ids={}
 for name,cfg in FOLDS.items():
  ids=np.load(SRC/name/'data.npz')['test_ids'];fold_ids[name]=ids;weeks[name]=((times[ids]-pd.Timestamp(cfg['test_start'],tz=times.tz)).days//7).to_numpy();k=int(weeks[name].max())+1;draws[name]=rng.integers(0,k,(2000,k))
 pairs=[('gated_graph_atoms','partial05','gated_graph_atoms',method) for method in ['time_broken','space_broken']]+[('gated_graph_atoms','partial05',base,'partial05') for base in MODELS[1:]]+[('gated_graph_atoms',method,'gated_graph_atoms','partial05') for method in ['partial00','partial10']]+[(model,'partial05',model,'historical_ecc') for model in MODELS];rows=[]
 for model,method,base,bmethod in pairs:
  available=['M11'] if bmethod=='historical_ecc' else list(FOLDS)
  for metric in COMPARE:
   sums={};counts={}
   for name in available:
    differences=[]
    for seed in SEEDS:
     a=np.load(OUT/name/f'scores_{model}_{seed}_{method}.npz')[metric];b=np.load(OUT/name/f'scores_{base}_{seed}_{bmethod}.npz')[metric];np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b));differences.append(a-b)
    z=np.mean(differences,0);sums[name]=np.array([np.nansum(z[weeks[name]==j]) for j in range(draws[name].shape[1])]);counts[name]=np.array([np.isfinite(z[weeks[name]==j]).sum() for j in range(draws[name].shape[1])])
   for scope in (['ALL']+available if len(available)==4 else available):
    chosen=available if scope=='ALL' else [scope];num=np.zeros(2000);den=np.zeros(2000);total=0.;n=0;blocks=[]
    for name in chosen:
     num+=sums[name][draws[name]].sum(1);den+=counts[name][draws[name]].sum(1);total+=sums[name].sum();n+=counts[name].sum();blocks.append(int((counts[name]>0).sum()))
    valid=den>0;supported=min(blocks)>=4 and valid.sum()>=1900;lo,hi=np.quantile(num[valid]/den[valid],[.025,.975]) if supported else [np.nan,np.nan];primary=scope=='ALL' and model=='gated_graph_atoms' and method=='partial05' and ((bmethod=='time_broken' and metric in ['brier6','first_cdf_brier']) or (bmethod=='space_broken' and metric=='group_brier') or (base=='gated_temporal_atoms' and metric in ['brier6','group_brier']))
    rows.append(dict(scope=scope,candidate=model,method=method,baseline=base,baseline_method=bmethod,metric=metric,primary=primary,difference=float(total/n),low95=float(lo),high95=float(hi),n=int(n),origins=sum(len(fold_ids[x]) for x in chosen),nonempty_blocks=sum(blocks),minimum_blocks_in_fold=min(blocks),valid_draws=int(valid.sum()),interval_supported=bool(supported)))
 assert len(rows)==198 and sum(r['primary'] for r in rows)==5;csv('paired_comparisons.csv',rows)

if __name__=='__main__':
 prepare();fit_templates();evaluate();compare();verify_parent();print('COMPLETE',flush=True)

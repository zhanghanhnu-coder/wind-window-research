"""Round35: change joint rank dependence while conserving every marginal."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
from seasonal_probability_core import FOLDS
from research_crps_distribution import guard,SEEDS
from rolling_copula_core import update_dependence,blend_correlations
from partial_copula_core import make_ranks
from probability_core import break_dependence,score_scenarios
from crps_distribution_core import couple_atoms
OUT=ART/'rolling_copula_v1';SRC=ART/'seasonal_probability_v1';COP=ART/'partial_copula_v1';ROLLING=ART/'seasonal_rolling_v1';PLAN=ART.parent.parent/'研究笔记/因果更新联合相关结构方案_2026-11-02.md';NETWORKS=['gated_graph_atoms','gated_temporal_atoms'];CASES={'graph_raw':(NETWORKS[0],'raw',['frozen','rolling28']),'graph_rolling28':(NETWORKS[0],'rolling28',['frozen','rolling28','rolling56','blend28','time_broken','space_broken']),'own_rolling28':(NETWORKS[1],'rolling28',['frozen','rolling28'])}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def js(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def parents():
 result={}
 for root in [SRC,COP,ROLLING]:
  manifest=read(root/'final_manifest.json')
  for file,h in manifest.items():assert sha(Path(file))==h,file
  result[root.name]=len(manifest)
 return result

def prepare():
 guard();parents();OUT.mkdir(exist_ok=True);protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0];p=OUT/'protocol_frozen.md'
 if p.exists():assert p.read_text(encoding='utf-8')==protocol
 else:p.write_text(protocol,encoding='utf-8')
 files=[p,Path(__file__)]+[Path(__file__).parent/n for n in ['rolling_copula_core.py','rolling_copula_tables.py','partial_copula_core.py','probability_core.py','crps_distribution_core.py','reliability_boundaries_core.py','tests/test_rolling_copula.py']]+[root/'final_manifest.json' for root in [SRC,COP,ROLLING]]+[ART/'data.npz'];identity={str(p):sha(p) for p in files}
 if (OUT/'identity.json').exists():assert read(OUT/'identity.json')==identity
 else:js(OUT/'identity.json',identity)
 active=[OUT]+[ART/n for n in ['window_pooling_v1','missing_window_bounds_v1','multiscale_context_v1','distributional_schedule_v1','reliability_boundary_v1']]
 if not (OUT/'previous_hashes.json').exists():js(OUT/'previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and not any(r in p.parents for r in active) and p.suffix in ['.npz','.pt','.csv','.joblib']})
 if not (OUT/'recipe.json').exists():js(OUT/'recipe.json',dict(cases=CASES,lookbacks=[28,56],shrink=.05,min_coordinate=20,min_pair=100,bootstrap_seed=20261123,bootstrap_replicates=2000,active_snapshot_exclusions=[str(p) for p in active[1:]],start_utc=datetime.now(timezone.utc).isoformat()))

def templates():
 updates=[];repairs=[];source_inventory=read(ROLLING/'inventory.json');inventory=[]
 for name in FOLDS:
  path=OUT/name;path.mkdir(exist_ok=True);f=np.load(SRC/name/'data.npz');archive=np.load(ROLLING/name/'archive_data.npz');schedule=read(ROLLING/name/'schedule.json');scale=float(f['scale'])
  for network in NETWORKS:
   for seed in SEEDS:
    guard();q=np.load(ROLLING/name/f'archive_predictions_{network}_{seed}.npz')['q'];base=np.load(COP/name/f'noise_{seed}.npz')['base'];frozen=dict(np.load(COP/name/f'dependence_{network}_{seed}.npz'));frozenrank=np.load(COP/name/f'ranks_{network}_{seed}_partial05.npz')['ranks'];previous={days:frozen for days in [28,56]};methods={m:[] for m in ['frozen','rolling28','rolling56','blend28','time_broken','space_broken']}
    for entry in schedule:
     block=entry['block'];issue=entry['issue_id'];current={}
     for days in [28,56]:
      guard();up=path/f'update_{network}_{seed}_{days}_w{block}.npz';uj=up.with_suffix('.json')
      if block==0:r={k:frozen[k] for k in ['time_correlation','space_correlation']};audit=dict(issue_id=issue,days=days,initial=True,reused_previous=False,reason='',valid_labels=None,valid_origins=None,source_positions=[],source_ids=[],earliest_label_id=None,latest_label_id=int(max(f['cal_ids']+72)),min_coordinate=int(np.isfinite(np.load(SRC/name/'cal_truth.npz')['y']).sum(0).min()));ranks=frozenrank;np.savez_compressed(up,**r);js(uj,dict(audit=audit,repair=[]))
      elif up.exists() and uj.exists():r=dict(np.load(up));saved=read(uj);audit=saved['audit'];ranks=make_ranks(base,r['time_correlation'],r['space_correlation'])
      else:
       r=update_dependence(q,archive['truth'],archive['ids'],issue,days,scale,previous[days]);audit=dict(initial=False,**r.pop('audit'));details=[dict(axis=axis,**r.pop(axis+'_info')) for axis in ['time','space']] if not audit['reused_previous'] else [];np.savez_compressed(up,**r);js(uj,dict(audit=audit,repair=details));ranks=make_ranks(base,r['time_correlation'],r['space_correlation'])
      assert audit['latest_label_id'] is None or audit['latest_label_id']<issue;updates.append(dict(fold=name,network=network,seed=seed,block=block,applied_origins=len(entry['positions']),**audit));repairs.extend([dict(fold=name,network=network,seed=seed,block=block,days=days,**info) for info in read(uj)['repair']]);previous[days]=r;current[days]=(r,ranks)
     mats=blend_correlations(frozen,current[28][0]);template={'frozen':frozenrank,'rolling28':current[28][1],'rolling56':current[56][1],'blend28':make_ranks(base,mats['time_correlation'],mats['space_correlation']),'time_broken':break_dependence(current[28][1],'time'),'space_broken':break_dependence(current[28][1],'space')}
     for method,ranks in template.items():
      rp=path/f'ranks_{network}_{seed}_{method}_w{block}.npz';np.savez_compressed(rp,ranks=ranks.astype(np.int16));methods[method].append(dict(block=block,issue_id=issue,positions=entry['positions'],source=str(rp),sha256=sha(rp)))
     print('TEMPLATE',name,network,seed,block,flush=True)
    for case,(net,margin,choices) in CASES.items():
     if network!=net:continue
     source=next(r for r in source_inventory if (r['fold'],r['model'],r['seed'],r['method'])==(name,network,seed,margin))
     for method in choices:inventory.append(dict(fold=name,model=case,network=network,seed=seed,method=method,marginal_method=margin,source=source['source'],source_key=source['source_key'],source_sha256=source['source_sha256'],ranks=methods[method],baseline_score_source=source['score_source'],baseline_score_sha256=sha(Path(source['score_source'])),new=method!='frozen',origins=len(f['test_ids']),score_source=str(path/f'scores_{case}_{seed}_{method}.npz') if method!='frozen' else source['score_source']))
 assert len(inventory)==120 and sum(r['new'] for r in inventory)==84 and len(updates)==432;csv('update_audit.csv',updates);csv('matrix_repairs.csv',repairs);js(OUT/'inventory.json',inventory);parents();js(OUT/'prediction_complete.json',dict(status='COMPLETE',flows=120,new_flows=84,forecast_origins=sum(r['origins'] for r in inventory),updates=432,noninitial_updates=384,support_fallbacks=sum(r['reused_previous'] for r in updates),matrix_repair_rows=len(repairs),utc=datetime.now(timezone.utc).isoformat()))

def evaluate():
 assert read(OUT/'prediction_complete.json')['status']=='COMPLETE';equiv=[]
 for item in read(OUT/'inventory.json'):
  guard();name=item['fold'];case=item['model'];seed=item['seed'];method=item['method'];f=np.load(SRC/name/'data.npz');q=np.load(item['source'])[item['source_key']];y=np.load(SRC/name/'test_truth.npz')['y'];old=dict(np.load(item['baseline_score_source']));dest=Path(item['score_source']);assert sha(Path(item['source']))==item['source_sha256'] and sha(Path(item['baseline_score_source']))==item['baseline_score_sha256']
  if not item['new']:continue
  ranks=[];block=np.empty(len(y),int)
  for j,r in enumerate(item['ranks']):assert sha(Path(r['source']))==r['sha256'];ranks.append(np.load(r['source'])['ranks']);block[r['positions']]=j
  if dest.exists():s=dict(np.load(dest))
  else:
   records=[]
   for i,(qi,yi) in enumerate(zip(q,y)):
    sc=couple_atoms(qi.astype(float),ranks[block[i]]);np.testing.assert_array_equal(np.sort(sc,axis=0),np.repeat(qi.astype(float),3,axis=-1).transpose(2,0,1));r=score_scenarios(sc,yi,train_scale=float(f['scale']));gp=[];gl=[]
    for offset in [0,24,48,66]:
     for group in [[0,1,2],[8,9,10],[13,15,16]]:
      obs=yi[offset:offset+6][:,group];valid=np.isfinite(obs).all();gp.append(float((sc[:,offset:offset+6][:,:,group]<=10).all((1,2)).mean()) if valid else np.nan);gl.append(float((obs<=10).all()) if valid else np.nan)
    r['group_probability']=np.array(gp);r['group_label']=np.array(gl);records.append(r)
    if i%80==0:print('SCORE',name,case,seed,method,i,flush=True);guard()
   s={k:np.stack([r[k] for r in records]) for k in records[0]};s.update({k:old[k] for k in ['probability','label','brier_hourly','mae']});assert len(s)==35;np.savez_compressed(dest,**s)
  maximum=0.
  for key in ['crps','coverage','width','interval_score','low_miss','high_miss','probability','label','brier_hourly','mae']:np.testing.assert_allclose(s[key],old[key],rtol=0,atol=3e-12,equal_nan=True);maximum=max(maximum,float(np.nanmax(abs(s[key]-old[key]))))
  if method=='space_broken':
   full=np.load(OUT/name/f'scores_{case}_{seed}_rolling28.npz')
   for key in ['brier3','brier6','brier12','length_crps24','length_crps48','length_crps72','any6_brier24','any6_brier48','any6_brier72','first_cdf_brier']:np.testing.assert_allclose(s[key],full[key],rtol=0,atol=3e-12,equal_nan=True)
  if method in ['rolling28','rolling56','blend28']:
   first=item['ranks'][0]['positions']
   for key in s:np.testing.assert_array_equal(s[key][first],old[key][first])
  equiv.append(dict(fold=name,model=case,seed=seed,method=method,max_marginal_difference=maximum,space_single_node_conserved=method=='space_broken',first_week_conserved=method in ['rolling28','rolling56','blend28']));print('RESULT',name,case,seed,method,'B6',np.nanmean(s['brier6']),flush=True)
 csv('marginal_conservation.csv',equiv)
 from rolling_copula_tables import tables,compare
 tables();compare();parents();js(OUT/'evaluation_complete.json',dict(status='COMPLETE',flows=120,new_flows=84,forecast_origins=26400,utc=datetime.now(timezone.utc).isoformat()));print('COMPLETE',flush=True)

if __name__=='__main__':prepare();templates();evaluate()

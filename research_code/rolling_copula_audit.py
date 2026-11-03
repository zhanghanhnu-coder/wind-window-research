"""Independent mature-coordinate correlation and rank audit for round35."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from model import ART
from research_crps_distribution import guard,SEEDS
from verify_partial_copula import independent_normal_scores,independent_corr,repair,get_ranks
OUT=ART/'rolling_copula_v1';SRC=ART/'seasonal_probability_v1';COP=ART/'partial_copula_v1';ROLLING=ART/'seasonal_rolling_v1';FOLDS=['M05','M07','M09','M11'];NETWORKS=['gated_graph_atoms','gated_temporal_atoms']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=3e-12):np.testing.assert_allclose(a,b,atol=tol,rtol=0,equal_nan=True)
def table(file,rows,keys):
 a=pd.read_csv(OUT/file).sort_values(keys).reset_index(drop=True);b=pd.DataFrame(rows).sort_values(keys).reset_index(drop=True);assert len(a)==len(b)
 for key in b:
  if b[key].dtype.kind in 'biufc':close(a[key],b[key],3e-11)
  else:
   norm=lambda x:'' if x is None or (isinstance(x,float) and np.isnan(x)) else str(x)
   assert a[key].map(norm).tolist()==b[key].map(norm).tolist(),(file,key)
 return len(a)

def audit_templates():
 counts=dict(update_records=0,maturity_masks=0,independent_dependence_fits=0,support_fallbacks=0,correlation_pair_entries=0,rank_templates=0,future_label_perturbations=0);updates=[];repairs=[]
 for fi,name in enumerate(FOLDS):
  f=np.load(SRC/name/'data.npz');a=np.load(ROLLING/name/'archive_data.npz');ids=a['ids'];target=ids[:,None]+np.arange(1,73);schedule=read(ROLLING/name/'schedule.json');scale=float(f['scale']);path=OUT/name
  for network in NETWORKS:
   for seed in SEEDS:
    guard();q=np.load(ROLLING/name/f'archive_predictions_{network}_{seed}.npz')['q'].astype(float);base=np.random.default_rng(np.random.SeedSequence([20261114,fi,seed])).normal(size=(57,72,17));np.testing.assert_array_equal(base,np.load(COP/name/f'noise_{seed}.npz')['base']);frozen=dict(np.load(COP/name/f'dependence_{network}_{seed}.npz'));frozenrank=np.load(COP/name/f'ranks_{network}_{seed}_partial05.npz')['ranks'];previous={d:frozen for d in [28,56]}
    for entry in schedule:
     block=entry['block'];issue=entry['issue_id'];current={}
     for days in [28,56]:
      guard();saved=dict(np.load(path/f'update_{network}_{seed}_{days}_w{block}.npz'));record=read(path/f'update_{network}_{seed}_{days}_w{block}.json');audit=record['audit'];assert audit['initial']==(block==0) and audit['issue_id']==issue and audit['days']==days
      if block==0:
       for k in ['time_correlation','space_correlation']:np.testing.assert_array_equal(saved[k],frozen[k])
       assert record['repair']==[] and not audit['reused_previous'] and audit['latest_label_id']==max(f['cal_ids']+72)<issue and audit['min_coordinate']==np.isfinite(np.load(SRC/name/'cal_truth.npz')['y']).sum(0).min()
      else:
       mask=np.zeros(a['truth'].shape,bool)
       for j,origin in enumerate(ids):
        for h in range(72):
         if origin<issue and issue-days*24<=origin+h+1<issue:mask[j,h]=np.isfinite(a['truth'][j,h])
       np.testing.assert_array_equal(mask,saved['mask']);pos=np.flatnonzero(mask.any((1,2)));known=target[mask.any(-1)];assert audit['valid_labels']==mask.sum() and audit['valid_origins']==len(pos) and audit['source_positions']==pos.tolist() and audit['source_ids']==ids[pos].tolist() and audit['earliest_label_id']==(int(known.min()) if len(known) else None) and audit['latest_label_id']==(int(known.max()) if len(known) else None) and audit['min_coordinate']==mask.sum(0).min();counts['maturity_masks']+=1
       reason='';corrs={};zs=None
       if mask.sum(0).min()<20:reason='insufficient observed calibration residuals in a coordinate'
       else:
        residual=(np.where(mask,a['truth'].astype(float),np.nan)-q[...,9])/np.maximum(q[...,17]-q[...,1],.1*scale);zs=independent_normal_scores(residual)
        for axis,matrix in [('space',zs.reshape(-1,17)),('time',zs.transpose(0,2,1).reshape(-1,72))]:
         valid=np.isfinite(matrix);counts_matrix=valid.astype(int).T@valid.astype(int)
         if counts_matrix.min()<100:reason='insufficient jointly observed pairs';break
         try:corr,pc=independent_corr(matrix)
         except AssertionError:reason='zero variance in a jointly observed pair';break
         corrs[axis]=(corr,pc)
       assert audit['reused_previous']==bool(reason) and audit['reason']==reason
       if reason:
        for k in ['time_correlation','space_correlation']:np.testing.assert_array_equal(saved[k],previous[days][k])
        assert record['repair']==[];counts['support_fallbacks']+=1
       else:
        close(zs,saved['normal_scores'],1e-14);np.testing.assert_array_equal(residual,saved['residual'])
        for axis in ['time','space']:
         corr,pc=corrs[axis];close(corr,saved[axis+'_raw']);np.testing.assert_array_equal(pc,saved[axis+'_counts']);fixed,info=repair(saved[axis+'_raw'],.05);close(fixed,saved[axis+'_correlation']);assert np.linalg.eigvalsh(fixed).min()>0;original=next(r for r in record['repair'] if r['axis']==axis)
         for k,val in info.items():close(original[k],val)
         counts['correlation_pair_entries']+=pc.size;repairs.append(dict(fold=name,network=network,seed=seed,block=block,days=days,axis=axis,**info))
        counts['independent_dependence_fits']+=1
       if seed==SEEDS[0] and block==4:
        from rolling_copula_core import update_dependence
        polluted=a['truth'].copy();polluted[np.broadcast_to((target>=issue)[:,:,None],polluted.shape)]=123456.;r=update_dependence(q,polluted,ids,issue,days,scale,previous[days])
        for k in ['time_correlation','space_correlation','mask']:np.testing.assert_array_equal(r[k],saved[k])
        assert r['audit']=={k:v for k,v in audit.items() if k!='initial'};counts['future_label_perturbations']+=1
      updates.append(dict(fold=name,network=network,seed=seed,block=block,applied_origins=len(entry['positions']),**audit));previous[days]=saved;current[days]=saved;counts['update_records']+=1
     full=get_ranks(base,current[28]['time_correlation'],current[28]['space_correlation']);tb=full.copy();sb=full.copy();rng=np.random.default_rng(20261101)
     for h in range(72):tb[:,h]=full[rng.permutation(57),h]
     rng=np.random.default_rng(20261101)
     for node in range(17):sb[:,:,node]=full[rng.permutation(57),:,node]
     templates={'frozen':frozenrank,'rolling28':full,'rolling56':get_ranks(base,current[56]['time_correlation'],current[56]['space_correlation']),'blend28':get_ranks(base,(frozen['time_correlation']+current[28]['time_correlation'])/2,(frozen['space_correlation']+current[28]['space_correlation'])/2),'time_broken':tb,'space_broken':sb}
     for method,ranks in templates.items():
      actual=np.load(path/f'ranks_{network}_{seed}_{method}_w{block}.npz')['ranks'];np.testing.assert_array_equal(actual,ranks);np.testing.assert_array_equal(np.sort(actual,axis=0),np.broadcast_to(np.arange(57)[:,None,None],actual.shape));counts['rank_templates']+=1
     print('DEPENDENCE_AUDIT',name,network,seed,block,flush=True)
 counts['update_table_rows']=table('update_audit.csv',updates,['fold','network','seed','block','days']);counts['repair_table_rows']=table('matrix_repairs.csv',repairs,['fold','network','seed','block','days','axis']);assert counts['update_records']==432 and counts['maturity_masks']==384 and counts['future_label_perturbations']==16 and counts['rank_templates']==1296;assert counts['independent_dependence_fits']+counts['support_fallbacks']==384;return counts

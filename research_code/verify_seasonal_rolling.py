"""Round27 independent maturity, calibration, distribution and block-score audit."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from model import ART
from crps_distribution_core import AtomNet,predict_atoms
from verify_seasonal_probability import reference_scores,crps
from verify_partial_copula import windows_reference,reconstruct,longest_direct
from research_crps_distribution import guard,SEEDS,METRICS as JOINT
OUT=ART/'seasonal_rolling_v1';SRC=ART/'seasonal_probability_v1';COP=ART/'partial_copula_v1'
FOLDS={'M05':('2024-04-01','2024-05-01'),'M07':('2024-06-01','2024-07-01'),'M09':('2024-08-01','2024-09-01'),'M11':('2024-10-01','2024-11-01')}
MODELS=['gated_graph_atoms','gated_temporal_atoms','gated_mean_atoms'];BINS=[(0,6),(6,24),(24,48),(48,72)];METRICS=JOINT+['brier_hourly','mae'];COMPARE=['crps','brier_hourly','brier6','group_brier','length_crps72','first_cdf_brier','spatial_variogram','temporal_variogram']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def close(a,b,tol=2e-11):np.testing.assert_allclose(a,b,atol=tol,rtol=0,equal_nan=True)
def compare_table(name,rows,keys):
 a=pd.read_csv(OUT/name);b=pd.DataFrame(rows);assert len(a)==len(b),(name,len(a),len(b));a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
 for k in b:
  if b[k].dtype.kind in 'biufc':close(a[k],b[k])
  else:assert a[k].fillna('').tolist()==b[k].fillna('').tolist(),(name,k)
 return len(a)

def transform(q,theta,scale):
 q=np.asarray(q,float);med=q[...,9,None];hours=np.arange(1,73);delta=np.interp(hours,[3.5,15.5,36.5,60.5],theta[:,0]);gamma=np.interp(hours,[3.5,15.5,36.5,60.5],theta[:,1]);return np.maximum(0,med+delta[None,:,None,None]*scale+gamma[None,:,None,None]*(q-med))

def objective(theta,q,y):
 # Pinball identity and its explicit subgradient, independent of pairwise CRPS code.
 delta,gamma=theta;med=q[:,9,None];shift=q-med;unclipped=med+delta+gamma*shift;v=np.maximum(unclipped,0);e=y[:,None]-v;tau=(np.arange(19)+.5)/19;loss=2*np.maximum(tau*e,(tau-1)*e).mean()+.001*(delta**2+(gamma-1)**2);dv=2*((v>y[:,None]).astype(float)+.5*(v==y[:,None])-tau)*(unclipped>0)/v.size;gradient=np.array([dv.sum()+.002*delta,(dv*shift).sum()+.002*(gamma-1)]);return float(loss),gradient

def archive_and_updates(times,raw):
 archive_rows=[];audits=[];detailrows=[];initial=[];counts=dict(archives=0,new_prediction_reloads=0,future_input_perturbations=0,updates=0,maturity_masks=0,calibration_objectives=0,independent_refitted_groups=0,future_label_perturbations=0,applied_prediction_flows=0);refits=[]
 for name,(calstart,teststart) in FOLDS.items():
  path=OUT/name;f=dict(np.load(SRC/name/'data.npz'));a=dict(np.load(path/'archive_data.npz'));ids=np.array([i for i in range(23,int(max(f['test_ids']))+1,6) if times[i]>=pd.Timestamp(calstart,tz=times.tz) and f['x'][i-23:i+1,:,-1].mean()>=.65]);np.testing.assert_array_equal(ids,a['ids']);positions={int(v):i for i,v in enumerate(ids)};y=np.stack([raw[i+1:i+73] for i in ids]);np.testing.assert_array_equal(y,a['truth']);np.testing.assert_array_equal(a['test_ids'],f['test_ids']);np.testing.assert_array_equal(a['test_positions'],[positions[int(i)] for i in f['test_ids']]);blocks=((times[f['test_ids']]-pd.Timestamp(teststart,tz=times.tz)).days//7).to_numpy();np.testing.assert_array_equal(blocks,a['blocks']);schedule=read(path/'schedule.json');assert len(schedule)==9;np.testing.assert_array_equal(a['update_ids'],[r['issue_id'] for r in schedule]);kinds=np.full(len(ids),'new',dtype='<U4')
  for split in ['cal','test']:
   kinds[[positions[int(i)] for i in f[split+'_ids']]]=split
  np.testing.assert_array_equal(kinds,a['source_kind']);scale=float(f['scale'])
  for r in schedule:
   pos=np.flatnonzero(blocks==r['block']);np.testing.assert_array_equal(pos,r['positions']);assert r['issue_id']==f['test_ids'][pos[0]] and r['issue_time']==str(times[r['issue_id']])
  for model in MODELS:
   for seed in SEEDS:
    guard();dest=path/f'archive_predictions_{model}_{seed}.npz';q=np.load(dest)['q'];source=dict(np.load(SRC/name/f'predictions_{model}_{seed}_budget40.npz'));ckpath=SRC/name/f'{model}_{seed}_budget40.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False);net=AtomNet(ck['adj'],ck['scale'],hidden=24,gated=True);net.load_state_dict(ck['state']);new=np.flatnonzero(kinds=='new')
    for split in ['cal','test']:np.testing.assert_array_equal(q[[positions[int(i)] for i in f[split+'_ids']]],source[split])
    if len(new):np.testing.assert_array_equal(q[new],predict_atoms(net,f['x'],ids[new]));counts['new_prediction_reloads']+=1
    # Perturb every later feature; chosen archived forecast remains unchanged.
    at=int(ids[new[0] if len(new) else 0]);changed=f['x'].copy();changed[at+1:]=12345.;np.testing.assert_array_equal(predict_atoms(net,changed,np.array([at])),predict_atoms(net,f['x'],np.array([at])));counts['future_input_perturbations']+=1
    archive_rows.append(dict(fold=name,model=model,seed=seed,archive_origins=len(ids),new_origins=len(new),cal_reused=int(sum(kinds=='cal')),test_reused=int(sum(kinds=='test')),first_issue=str(times[ids[0]]),last_issue=str(times[ids[-1]]),checkpoint_sha256=sha(ckpath),archive_sha256=sha(dest)));counts['archives']+=1;theta0=np.load(SRC/name/f'calibration_{model}_{seed}.npz')['theta'];initial_details=read(SRC/name/f'calibration_{model}_{seed}.json');frozen=np.load(SRC/name/f'calibrated_{model}_{seed}.npz')['q'];caly=np.load(SRC/name/'cal_truth.npz')['y']
    for method,days in [('rolling28',28),('rolling56',56)]:
     pred=np.load(path/f'predictions_{model}_{seed}_{method}.npz')['q'];previous=theta0.copy()
     for entry in schedule:
      guard();block=entry['block'];issue=entry['issue_id'];pos=np.array(entry['positions']);base=f'update_{model}_{seed}_{method}_w{block}';z=np.load(path/(base+'.npz'));theta=z['theta'];record=read(path/(base+'.json'));audit=record['audit'];details=record['details'];assert audit['issue_id']==issue and audit['lookback_days']==days
      if block==0:
       np.testing.assert_array_equal(theta,theta0);np.testing.assert_array_equal(pred[pos],frozen[pos]);assert record['source_calibration']==str(SRC/name/f'calibration_{model}_{seed}.npz') and details==initial_details and audit['initial_frozen'] and not audit['reused_previous'] and audit['fitted_groups']==0 and audit['optimizer_failures']==0;assert audit['valid_origins']==np.isfinite(caly).any((1,2)).sum() and audit['valid_labels']==np.isfinite(caly).sum();assert audit['group_counts']==[int(r['valid_targets']) for r in initial_details];assert audit['earliest_label_id']==min(f['cal_ids'])+1 and audit['latest_label_id']==max(f['cal_ids'])+72<issue
      else:
       target=ids[:,None]+np.arange(1,73);allowed=(ids[:,None]<issue)&(target<issue)&(target>=issue-days*24);mask=np.isfinite(y)&allowed[:,:,None];np.testing.assert_array_equal(z['mask'],mask);counts['maturity_masks']+=1;selected=np.flatnonzero(mask.any((1,2)));groupcounts=[int(mask[:,lo:hi].sum()) for lo,hi in BINS];support=len(selected)>=20 and min(groupcounts)>=100;known=target[mask.any(-1)];assert audit['source_positions']==selected.tolist() and audit['source_ids']==ids[selected].tolist();assert audit['valid_origins']==len(selected) and audit['valid_labels']==mask.sum() and audit['group_counts']==groupcounts;assert audit['earliest_label_id']==(int(known.min()) if len(known) else None) and audit['latest_label_id']==(int(known.max()) if len(known) else None);assert not audit['initial_frozen'] and audit['reused_previous']==(not support) and audit['fitted_groups']==(4 if support else 0)
       if not support:np.testing.assert_array_equal(theta,previous)
       for g,(lo,hi) in enumerate(BINS):
        info=details[g];assert info['lead_start']==lo+1 and info['lead_end']==hi and info['valid_targets']==groupcounts[g];close([info['delta'],info['gamma']],theta[g],1e-14)
        if support:
         qq=q[:,lo:hi][mask[:,lo:hi]].astype(float)/scale;yy=y[:,lo:hi][mask[:,lo:hi]].astype(float)/scale;obj=objective(theta[g],qq,yy)[0];close(obj,info['objective_used'],1e-10);counts['calibration_objectives']+=1;assert -2<=theta[g,0]<=2 and .25<=theta[g,1]<=4
         if info['fallback']:np.testing.assert_array_equal(theta[g],[0.,1.]);assert not info['success']
         else:assert info['success'];close(info['objective_candidate'],obj,1e-10)
         # Preselected first seed, week4, all folds/models/lookbacks: 96 independent fits.
         if seed==SEEDS[0] and block==4:
          fit=minimize(objective,[0.,1.],args=(qq,yy),jac=True,method='L-BFGS-B',bounds=[(-2,2),(.25,4)],options={'maxiter':200,'ftol':1e-10,'gtol':1e-7});used=fit.x if fit.success and np.isfinite(fit.fun) else np.array([0.,1.]);our=objective(used,qq,yy)[0];assert abs(our-obj)<2e-6,(name,model,method,g,our,obj);counts['independent_refitted_groups']+=1;refits.append(dict(fold=name,model=model,seed=seed,method=method,block=block,group=g,independent_success=bool(fit.success),independent_objective=our,saved_objective=obj,max_parameter_difference=float(max(abs(used-theta[g])))))
        else:assert info['objective_used'] is None and info['fallback'] and not info['success']
        detailrows.append(dict(fold=name,model=model,seed=seed,method=method,block=block,update_id=issue,reused_previous=audit['reused_previous'],**info))
       assert audit['optimizer_failures']==(sum(r['fallback'] for r in details) if support else 0)
       if seed==SEEDS[0] and model==MODELS[0] and block==4:
        from seasonal_rolling_core import calibrate_at
        pollution=y.copy();pollution[np.broadcast_to((target>=issue)[:,:,None],y.shape)]=98765.;rr=calibrate_at(q,pollution,ids,issue,days,scale,previous);np.testing.assert_array_equal(rr['theta'],theta);np.testing.assert_array_equal(rr['mask'],mask);assert rr['audit']=={k:v for k,v in audit.items() if k!='initial_frozen'};counts['future_label_perturbations']+=1
      np.testing.assert_array_equal(pred[pos],transform(source['test'][pos],theta,scale));previous=theta.copy();counts['updates']+=1;audits.append(dict(fold=name,model=model,seed=seed,method=method,block=block,update_id=issue,update_time=entry['issue_time'],applied_origins=len(pos),initial_frozen=audit['initial_frozen'],reused_previous=audit['reused_previous'],fitted_groups=audit['fitted_groups'],optimizer_failures=audit['optimizer_failures'],valid_origins=audit['valid_origins'],valid_labels=audit['valid_labels'],earliest_label_id=audit['earliest_label_id'],latest_label_id=audit['latest_label_id'],oldest_observation_age_hours=issue-audit['earliest_label_id'] if audit['earliest_label_id'] is not None else None))
     initial.append(dict(fold=name,model=model,seed=seed,method=method,initial_origins=int(sum(blocks==0)),initial_exact=True));counts['applied_prediction_flows']+=1
    print('ARCHIVE_AND_UPDATES',name,model,seed,flush=True)
 keys=['fold','model','seed'];counts['archive_rows']=compare_table('archive_summary.csv',archive_rows,keys);counts['update_rows']=compare_table('update_audit.csv',audits,keys+['method','block']);counts['calibration_rows']=compare_table('calibration.csv',detailrows,keys+['method','block','lead_start']);counts['initial_rows']=compare_table('initial_week_equivalence.csv',initial,keys+['method']);pd.DataFrame(refits).to_csv(OUT/'independent_refits.csv',index=False,encoding='utf-8-sig');return counts

def scores_and_tables(times):
 inventory=read(OUT/'inventory.json');assert len(inventory)==144;cache={};rows=[];months=[];weeks=[];leads=[];high=[];rel=[];equiv=[];counts=dict(forecast_cases=0,score_arrays=0,independent_run_length_cases=0)
 for item in inventory:
  guard();name=item['fold'];model=item['model'];seed=item['seed'];method=item['method'];assert sha(Path(item['source']))==item['source_sha256'] and sha(Path(item['ranks_source']))==item['ranks_sha256'];q=np.load(item['source'])[item['source_key']];ranks=np.load(item['ranks_source'])['ranks'];np.testing.assert_array_equal(ranks,np.load(COP/name/f'ranks_{model}_{seed}_partial05.npz')['ranks']);y=np.load(SRC/name/'test_truth.npz')['y'];s=dict(np.load(OUT/name/f'scores_{model}_{seed}_{method}.npz'));f=np.load(SRC/name/'data.npz');ts=times[f['test_ids']];blocks=((ts-pd.Timestamp(FOLDS[name][1],tz=times.tz)).days//7).to_numpy();assert len(q)==item['origins'];marginal=reference_scores(q,y)
  for k,v in marginal.items():close(s[k],v,3e-12)
  for start in range(0,len(q),8):
   stop=min(len(q),start+8);sc=reconstruct(q[start:stop].astype(float),ranks);np.testing.assert_array_equal(np.sort(sc,axis=1),np.repeat(q[start:stop],3,axis=-1).transpose(0,3,1,2));ref=windows_reference(sc,y[start:stop],float(f['scale']))
   for k,v in ref.items():close(s[k][start:stop],v,3e-12)
   counts['forecast_cases']+=stop-start
  counts['score_arrays']+=len(set(marginal)|set(ref))
  for pos in [0,len(q)//2,len(q)-1]:
   sc=reconstruct(q[pos:pos+1],ranks)[0]
   for node in [0,8,16]:
    obs=y[pos,:,node];lengths=np.array([longest_direct(v<=10) for v in sc[:,:,node]]);actual=longest_direct(obs<=10);score=np.mean(abs(lengths-actual))-.5*np.mean(abs(lengths[:,None]-lengths[None,:]));close(s['length_crps72'][pos,node],score if np.isfinite(obs).all() else np.nan,3e-12);counts['independent_run_length_cases']+=1
  if method=='frozen':
   for k,v in dict(np.load(COP/name/f'scores_{model}_{seed}_partial05.npz')).items():np.testing.assert_array_equal(s[k],v)
   old=np.load(SRC/name/f'scores_{model}_{seed}_smooth_affine.npz')
   for k in ['probability','label','brier_hourly','mae']:np.testing.assert_array_equal(s[k],old[k])
   equiv.append(dict(fold=name,model=model,seed=seed,joint_exact=True,hourly_exact=True))
  cache[(name,model,seed,method)]={k:s[k] for k in COMPARE};common=dict(fold=name,model=model,seed=seed,method=method)
  for metric in METRICS:
   z=s[metric];rows.append(dict(**common,metric=metric,value=float(np.nanmean(z)),n=int(np.isfinite(z).sum())))
   for month in sorted(set(ts.month)):
    v=z[ts.month==month];months.append(dict(**common,issue_month=int(month),metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
   for week in sorted(set(blocks)):
    v=z[blocks==week];weeks.append(dict(**common,week_block=int(week),metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
  for lo,hi in BINS:
   for metric in ['crps','brier_hourly','mae','coverage','width','interval_score','low_miss','high_miss']:
    v=s[metric][:,lo:hi];leads.append(dict(**common,lead_start=lo+1,lead_end=hi,metric=metric,value=float(np.nanmean(v)),n=int(np.isfinite(v).sum())))
  for target,pk,yk in [('hourly','probability','label'),('single_6h','prob6','label6'),('group_6h','group_probability','group_label')]:
   ok=np.isfinite(s[yk]);p=s[pk][ok];labels=s[yk][ok];select=p>=.8;high.append(dict(**common,target=target,n=int(select.sum()),total=len(p),mean_probability=float(p[select].mean()) if select.any() else np.nan,observed_rate=float(labels[select].mean()) if select.any() else np.nan))
   for b in range(10):
    select=(p>=b/10)&(p<(b+1)/10 if b<9 else p<=1);rel.append(dict(**common,target=target,bin=b,n=int(select.sum()),mean_probability=float(p[select].mean()) if select.any() else np.nan,observed_rate=float(labels[select].mean()) if select.any() else np.nan))
  print('SCORES',name,model,seed,method,flush=True)
 df=pd.DataFrame(rows)
 for key,g in df.groupby(['model','seed','method','metric'],sort=False):rows.append(dict(fold='ALL',**dict(zip(['model','seed','method','metric'],key)),value=float(np.dot(g.value,g.n)/g.n.sum()),n=int(g.n.sum())))
 tablecounts={};keys=['fold','model','seed','method']
 for file,values,extra in [('metrics.csv',rows,['metric']),('month_metrics.csv',months,['metric','issue_month']),('weekly_metrics.csv',weeks,['metric','week_block']),('lead_metrics.csv',leads,['metric','lead_start']),('high_confidence.csv',high,['target']),('reliability.csv',rel,['target','bin'])]:tablecounts[file]=compare_table(file,values,keys+extra)
 tablecounts['frozen_baseline_equivalence.csv']=compare_table('frozen_baseline_equivalence.csv',equiv,['fold','model','seed']);counts['table_rows']=tablecounts;return cache,counts

def paired(cache,times):
 rng=np.random.default_rng(20261115);weights={};weeks={};sizes={}
 for name,(_,start) in FOLDS.items():
  ids=np.load(SRC/name/'data.npz')['test_ids'];w=((times[ids]-pd.Timestamp(start,tz=times.tz)).days//7).to_numpy();weeks[name]=w;k=int(w.max())+1;draw=rng.integers(0,k,(2000,k));weights[name]=np.stack([np.bincount(row,minlength=k)[w] for row in draw]);sizes[name]=len(ids)
 df=pd.read_csv(OUT/'paired_comparisons.csv');assert len(df)==320 and df.primary.sum()==6
 for r in df.itertuples():
  selected=list(FOLDS) if r.scope=='ALL' else [r.scope];num=np.zeros(2000);den=np.zeros(2000);ss=0.;nn=0;blocks=[]
  for name in selected:
   a=np.stack([cache[(name,r.candidate,seed,r.method)][r.metric] for seed in SEEDS]);b=np.stack([cache[(name,r.baseline,seed,r.baseline_method)][r.metric] for seed in SEEDS]);np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b));diff=(a-b).mean(0);flat=diff.reshape(len(diff),-1);sums=np.nansum(flat,1);counts=np.isfinite(flat).sum(1);num+=weights[name]@sums;den+=weights[name]@counts;ss+=sums.sum();nn+=counts.sum();blocks.append(len(np.unique(weeks[name][counts>0])))
  valid=den>0;supported=min(blocks)>=4 and valid.sum()>=1900;lo,hi=np.quantile(num[valid]/den[valid],[.025,.975]) if supported else [np.nan,np.nan];close(r.difference,ss/nn);close(r.low95,lo);close(r.high95,hi);assert r.n==nn and r.origins==sum(sizes[n] for n in selected) and r.nonempty_blocks==sum(blocks) and r.minimum_blocks_in_fold==min(blocks) and r.valid_draws==valid.sum() and r.interval_supported==supported;primary=r.scope=='ALL' and r.candidate==MODELS[0] and r.method=='rolling28' and ((r.baseline==MODELS[0] and r.baseline_method=='frozen' and r.metric in ['crps','brier_hourly','brier6','group_brier']) or (r.baseline==MODELS[1] and r.baseline_method=='rolling28' and r.metric in ['brier6','group_brier']));assert r.primary==primary
 return len(df)

def run():
 guard();previous=read(OUT/'previous_hashes.json');identity=read(OUT/'identity.json');parents={}
 correction=read(OUT/'resume_precision_identity.json');correction_result=read(OUT/'resume_precision_verification.json');assert correction_result['status']=='PASS' and correction_result['preserved_files']==len(correction['preserved'])
 for mapping in [correction['sources'],correction['preserved']]:
  for file,h in mapping.items():assert sha(Path(file))==h,file
 for file,h in previous.items():assert sha(ART/file)==h,file
 for file,h in identity.items():assert sha(Path(file))==h,file
 for parent in [SRC,COP]:
  manifest=read(parent/'final_manifest.json')
  for file,h in manifest.items():assert sha(Path(file))==h,file
  parents[parent.name]=len(manifest)
 d=dict(np.load(ART/'data.npz'));times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');counts=archive_and_updates(times,d['raw']);cache,sc=scores_and_tables(times);counts.update(sc);phase=read(OUT/'prediction_complete.json');assert phase['flows']==144 and phase['forecast_origins']==counts['forecast_cases'] and phase['weekly_update_calls']==576 and phase['initial_frozen_updates']==72;updates=pd.read_csv(OUT/'update_audit.csv');assert phase['fitted_groups']==updates.fitted_groups.sum() and phase['support_fallbacks']==updates.reused_previous.sum() and phase['optimizer_failures']==updates.optimizer_failures.sum();counts.update(status='PASS',previous_unchanged=len(previous),frozen_identities=len(identity),parent_manifest_files=parents,precision_preserved_files=len(correction['preserved']),paired_comparisons=paired(cache,times));(OUT/'verification.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(json.dumps(counts,indent=2),flush=True)
if __name__=='__main__':run()

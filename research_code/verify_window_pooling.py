"""Independent event maturity, quadratic certificates and full distribution audit."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from model import ART
from pooling_reference import simplex_reference,marginal_reference,reconstruct_mixture,joint_reference,mixture_probability_reference
from research_crps_distribution import guard,SEEDS,METRICS as JOINT
OUT=ART/'window_pooling_v1';SRC=ART/'seasonal_probability_v1';PARENT=ART/'probability_pooling_v1';ROLLING=ART/'seasonal_rolling_v1'
FOLDS={'M05':'2024-05-01','M07':'2024-07-01','M09':'2024-09-01','M11':'2024-11-01'};MODELS=['gated_graph_atoms','gated_temporal_atoms','gated_mean_atoms'];EXPERTS=['raw','frozen','rolling28','rolling56'];METHODS={'crps_exact':0,'window_exact1':1,'window_exact025':.25,'window_exact4':4};BINS=[(0,6),(6,24),(24,48),(48,72)];METRICS=JOINT+['brier_hourly','mae'];COMPARE=['crps','brier_hourly','brier6','group_brier','length_crps72','first_cdf_brier','spatial_variogram','temporal_variogram']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,tol=3e-11):np.testing.assert_allclose(a,b,atol=tol,rtol=0,equal_nan=True)
def compare_table(name,rows,keys):
 a=pd.read_csv(OUT/name);b=pd.DataFrame(rows);assert len(a)==len(b),(name,len(a),len(b));a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
 for k in b:
  if b[k].dtype.kind in 'biufc':close(a[k],b[k])
  else:assert a[k].fillna('').tolist()==b[k].fillna('').tolist(),(name,k)
 return len(a)
def direct_event_labels(y):
 single=[];group=[]
 for start in [0,24,48,66]:
  yy=y[:,start:start+6];single.append(np.where(np.isfinite(yy).all(1),(yy<=10).all(1),np.nan))
  for nodes in [[0,1,2],[8,9,10],[13,15,16]]:
   z=yy[:,:,nodes];group.append(np.where(np.isfinite(z).all((1,2)),(z<=10).all((1,2)),np.nan))
 return np.stack(single,1),np.stack(group,1)
def stats_reference(prob,label,ids,issue,kind):
 ends=ids[:,None,None]+np.array([6,30,54,72])[None,:,None] if kind=='single' else ids[:,None]+np.repeat([6,30,54,72],3)[None];ends=np.broadcast_to(ends,label.shape);source=np.broadcast_to(ids.reshape((-1,)+(1,)*(label.ndim-1)),label.shape);mask=np.isfinite(label)&(source<issue)&(ends<issue)&(ends>=issue-672);p=prob[:,mask];y=label[mask];n=len(y);C=np.zeros((4,4));d=np.zeros(4)
 if n:
  for i in range(4):
   d[i]=np.mean(p[i]*y)
   for j in range(4):C[i,j]=np.mean(p[i]*p[j])
 sourcepos=np.flatnonzero(mask.reshape(len(ids),-1).any(1));known=ends[mask];audit=dict(valid_events=n,valid_origins=len(sourcepos),source_positions=sourcepos.tolist(),source_ids=ids[sourcepos].tolist(),earliest_end_id=int(known.min()) if n else None,latest_end_id=int(known.max()) if n else None)
 return dict(mask=mask,C=C,d=d,e=float(np.mean(y*y)) if n else 0.,audit=audit)
def objective(w,a,b,constant=0):return float(a@w-.5*w@b@w+.001*np.sum((w-.25)**2)+constant),a-b@w+.002*(w-.25)
def audit_weights(times):
 inventory=read(OUT/'inventory.json');parent_inventory=read(ROLLING/'inventory.json');rows=[];optima=[];refits=[];counts=dict(weight_records=0,hour_moment_references=0,event_masks=0,event_quadratics=0,global_optima=0,independent_slsqp_refits=0,future_perturbations=0,applied_weight_flows=0)
 for name,start in FOLDS.items():
  f=np.load(SRC/name/'data.npz');ids=f['test_ids'];y=np.load(SRC/name/'test_truth.npz')['y'];sl,gl=direct_event_labels(y);blocks=((times[ids]-pd.Timestamp(start,tz=times.tz)).days//7).to_numpy();schedule=read(ROLLING/name/'schedule.json');target=ids[:,None]+np.arange(1,73)
  for model in MODELS:
   for seed in SEEDS:
    guard();items=[next(r for r in parent_inventory if r['fold']==name and r['model']==model and r['seed']==seed and r['method']==m) for m in EXPERTS];old=[dict(np.load(ROLLING/name/f'scores_{model}_{seed}_{m}.npz')) for m in EXPERTS];sp=np.stack([z['prob6'] for z in old]);gp=np.stack([z['group_probability'] for z in old])
    for z in old:np.testing.assert_array_equal(z['label6'],sl);np.testing.assert_array_equal(z['group_label'],gl)
    for method,lam in METHODS.items():
     if model!=MODELS[0] and method in ['window_exact025','window_exact4']:continue
     inv=next(r for r in inventory if (r['fold'],r['model'],r['seed'],r['method'])==(name,model,seed,method));assert inv['experts']==items and inv['new'];assert inv['origins']==len(ids) and inv['lambda_value']==lam;assert inv['ranks_source']==items[0]['ranks_source'] and inv['ranks_sha256']==items[0]['ranks_sha256'];assert sha(Path(inv['weights_source']))==inv['weights_sha256'];assert sha(Path(inv['ranks_source']))==inv['ranks_sha256'];applied=np.load(inv['weights_source'])['weights'];previous=np.full(4,.25)
     for entry in schedule:
      block=entry['block'];issue=entry['issue_id'];pos=np.flatnonzero(blocks==block);np.testing.assert_array_equal(pos,entry['positions']);assert issue==ids[pos[0]];base=f'weights_{model}_{seed}_{method}_w{block}';z=np.load(OUT/name/(base+'.npz'));record=read(OUT/name/(base+'.json'));w=z['weights'];au=record['audit'];detail=record['details'];assert w.shape==(4,) and (w>=0).all();close(w.sum(),1,1e-14);assert au['issue_id']==issue and au['initial']==(block==0) and au['lambda_value']==lam
      if block==0:
       np.testing.assert_array_equal(w,np.full(4,.25));assert not au['optimizer_called'] and not au['insufficient_support'];assert au['hour_origins']==0 and au['hour_labels']==0 and au['hour_group_counts']==[0]*4;assert au['single']['valid_events']==0 and au['group']['valid_events']==0;assert detail['success'] is None and not detail['fallback']
      else:
       source=PARENT/name/f'weights_{model}_{seed}_online_pool28_w{block}.npz';assert record['hour_moments_source']==str(source) and record['hour_moments_sha256']==sha(source);hour=np.load(source);hm=np.isfinite(y)&((ids[:,None]<issue)&(target<issue)&(target>=issue-672))[:,:,None];np.testing.assert_array_equal(hm,hour['mask']);counts['hour_moment_references']+=1;stats={}
       for kind,p,label in [('single',sp,sl),('group',gp,gl)]:
        st=stats_reference(p,label,ids,issue,kind);stats[kind]=st;assert au[kind]==st['audit'];np.testing.assert_array_equal(z[kind+'_mask'],st['mask']);close(z[kind+'_C'],st['C'],2e-12);close(z[kind+'_d'],st['d'],2e-12);close(z[kind+'_e'],st['e']);counts['event_masks']+=1;counts['event_quadratics']+=1
       s=stats['single'];g=stats['group'];a=hour['a']-lam*(s['d']+g['d']);b=hour['b']-lam*(s['C']+g['C']);constant=lam/2*(s['e']+g['e']);close(z['a'],a,3e-12);close(z['b'],b,3e-12);close(detail['constant'],constant);hour_origins=int(hm.any((1,2)).sum());groupcounts=[int(hm[:,lo:hi].sum()) for lo,hi in BINS];hours_enough=hour_origins>=20 and min(groupcounts)>=100;events_enough=s['audit']['valid_events']>=100 and g['audit']['valid_events']>=50;support=hours_enough and (lam==0 or events_enough);assert au['hour_origins']==hour_origins and au['hour_labels']==hm.sum() and au['hour_group_counts']==groupcounts;assert au['hour_support']==hours_enough and au['event_support']==events_enough and au['insufficient_support']==(not support) and au['optimizer_called']==support
       if support:
        # Independent global face enumeration has no production tie/filter logic.
        best,bestw=simplex_reference(a,b);used,gradient=objective(w,a,b);gap=float(gradient@w-gradient.min());tangent=np.vstack([np.eye(3),-np.ones(3)]);mineig=float(np.linalg.eigvalsh(tangent.T@(-b+.002*np.eye(4))@tangent).min());close(detail['tangent_min_eigenvalue'],mineig);close(detail['used_objective'],used);close(detail['full_used_objective'],used+constant);assert detail['fallback']==(not detail['success']);assert len(detail['faces'])==15
        if detail['success']:
         assert gap<=1.01e-8 and mineig>=-1e-9;close(detail['candidate_gap'],gap);close(detail['candidate_objective'],used);close(detail['candidate_gradient'],gradient);close(detail['candidate_weights'],w);assert -1e-10<=used-best<=1.01e-8;eligible=[r for r in detail['faces'] if r['solved'] and r['feasible'] and r['gap']<=1e-8];assert eligible and any(r['face']==detail['selected_face'] for r in eligible)
        else:np.testing.assert_array_equal(w,previous)
        optima.append(dict(fold=name,model=model,seed=seed,method=method,block=block,used_objective=used+constant,global_objective=best+constant,objective_gap=used-best,frank_wolfe_gap=gap,optimizer_success=detail['success'],weight_distance=float(np.max(abs(w-bestw)))));counts['global_optima']+=1
        if seed==SEEDS[0] and block in [1,8]:
         result=minimize(lambda x:objective(x,a,b),np.full(4,.25),method='SLSQP',jac=True,bounds=[(0,1)]*4,constraints=[dict(type='eq',fun=lambda x:x.sum()-1,jac=lambda x:np.ones(4))],options=dict(maxiter=500,ftol=1e-12));delta=objective(result.x,a,b)[0]-best;assert abs(result.x.sum()-1)<1e-8 and result.x.min()>=-1e-8 and -1e-10<=delta<=1e-6;refits.append(dict(fold=name,model=model,seed=seed,method=method,block=block,slsqp_success=bool(result.success),status=int(result.status),iterations=int(result.nit),objective_gap=float(delta),weight_difference=float(np.max(abs(result.x-w)))));counts['independent_slsqp_refits']+=1
       else:np.testing.assert_array_equal(w,previous);assert detail['fallback'] and detail['used_objective'] is None and detail['full_used_objective'] is None
       if model==MODELS[0] and seed==SEEDS[0] and method in ['window_exact1','window_exact4'] and block in [1,8]:
        from window_pooling_core import fit_window_at
        ps=sp.copy();ls=sl.copy();pg=gp.copy();lg=gl.copy();future_single=np.broadcast_to((ids[:,None,None]+np.array([6,30,54,72])[None,:,None]>=issue),sl.shape);future_group=ids[:,None]+np.repeat([6,30,54,72],3)[None]>=issue;ps[:,future_single]=999;ls[future_single]=999;pg[:,future_group]=-999;lg[future_group]=-999;changed=fit_window_at(hour['a'],hour['b'],hm,ids,issue,previous,ps,ls,pg,lg,lam);np.testing.assert_array_equal(changed['weights'],w);assert changed['audit']=={k:v for k,v in au.items() if k!='initial'};assert changed['details']==detail;counts['future_perturbations']+=1
      np.testing.assert_array_equal(applied[pos],np.broadcast_to(w,(len(pos),4)));previous=w.copy();counts['weight_records']+=1;rows.append(dict(fold=name,model=model,seed=seed,method=method,lambda_value=lam,block=block,update_id=issue,update_time=str(times[issue]),applied_origins=len(pos),initial=au['initial'],optimizer_called=au['optimizer_called'],insufficient_support=au['insufficient_support'],optimizer_failed=bool(au['optimizer_called'] and detail['fallback']),hour_origins=au['hour_origins'],hour_labels=au['hour_labels'],single_events=au['single']['valid_events'],group_events=au['group']['valid_events'],**{f'weight_{k}':float(w[j]) for j,k in enumerate(EXPERTS)},gap=detail.get('candidate_gap'),objective=detail.get('full_used_objective')))
     counts['applied_weight_flows']+=1
    print('WEIGHT_AUDIT',name,model,seed,flush=True)
 assert counts['applied_weight_flows']==96 and counts['weight_records']==864 and counts['hour_moment_references']==768;counts['weight_table_rows']=compare_table('weight_audit.csv',rows,['fold','model','seed','method','block']);pd.DataFrame(optima).to_csv(OUT/'independent_weight_optima.csv',index=False,encoding='utf-8-sig');pd.DataFrame(refits).to_csv(OUT/'independent_slsqp_refits.csv',index=False,encoding='utf-8-sig');return counts

# The all-score/table and block-comparison routines are appended by the audited
# preparation script before this verifier is run, never by a running experiment.

def scores_and_tables(times):
 inventory=read(OUT/'inventory.json');assert len(inventory)==312 and sum(r['new'] for r in inventory)==96;rows=[];months=[];weeks=[];leads=[];rel=[];high=[];cache={};counts=dict(new_forecast_cases=0,new_score_arrays=0,baseline_arrays_unchanged=0,mixture_probability_conservations=0)
 for item in inventory:
  guard();name=item['fold'];model=item['model'];seed=item['seed'];method=item['method'];key=name,model,seed,method;s=dict(np.load(item['score_source']));f=np.load(SRC/name/'data.npz');y=np.load(SRC/name/'test_truth.npz')['y'];ts=times[f['test_ids']];blocks=((ts-pd.Timestamp(FOLDS[name],tz=times.tz)).days//7).to_numpy();assert len(s)==35
  if item['new']:
   q=np.stack([np.load(r['source'])[r['source_key']] for r in item['experts']]);w=np.load(item['weights_source'])['weights'];ranks=np.load(item['ranks_source'])['ranks'];old=[dict(np.load(ROLLING/name/f'scores_{model}_{seed}_{method0}.npz')) for method0 in EXPERTS]
   for start in range(0,len(y),4):
    stop=min(start+4,len(y));sc=reconstruct_mixture(q[:,start:stop],ranks);ref={**marginal_reference(q[:,start:stop],y[start:stop],w[start:stop]),**joint_reference(sc,y[start:stop],w[start:stop],float(f['scale']))};assert set(ref)==set(s)
    for k,value in ref.items():close(s[k][start:stop],value,3e-11)
    counts['new_forecast_cases']+=stop-start
   for metric in ['probability','prob3','prob6','prob12','group_probability','any6_prob24','any6_prob48','any6_prob72']:
    valid=np.isfinite(y) if metric=='probability' else np.isfinite(y[:,:int(metric.removeprefix('any6_prob'))]).all(1) if metric.startswith('any6_prob') else np.isfinite(s['group_label']) if metric=='group_probability' else np.isfinite(s['label'+metric.removeprefix('prob')]);weighted=mixture_probability_reference(np.stack([z[metric] for z in old]),w,valid);close(weighted,s[metric],3e-13);counts['mixture_probability_conservations']+=1
   counts['new_score_arrays']+=len(s)
  else:
   parent_hashes={**read(PARENT/'final_manifest.json'),**read(ROLLING/'final_manifest.json')};assert sha(Path(item['score_source']))==parent_hashes[str(Path(item['score_source']))];counts['baseline_arrays_unchanged']+=len(s)
  cache[key]={k:s[k] for k in COMPARE};common=dict(fold=name,model=model,seed=seed,method=method)
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
  print('SCORES_AUDIT',name,model,seed,method,flush=True)
 df=pd.DataFrame(rows)
 for key,g in df.groupby(['model','seed','method','metric'],sort=False):rows.append(dict(fold='ALL',**dict(zip(['model','seed','method','metric'],key)),value=float(np.dot(g.value,g.n)/g.n.sum()),n=int(g.n.sum())))
 tablecounts={};keys=['fold','model','seed','method']
 for file,values,extra in [('metrics.csv',rows,['metric']),('month_metrics.csv',months,['metric','issue_month']),('weekly_metrics.csv',weeks,['metric','week_block']),('lead_metrics.csv',leads,['metric','lead_start']),('high_confidence.csv',high,['target']),('reliability.csv',rel,['target','bin'])]:tablecounts[file]=compare_table(file,values,keys+extra)
 counts['table_rows']=tablecounts;return cache,counts

def paired(cache,times):
 rng=np.random.default_rng(20261117);weights={};weeks={};sizes={}
 for name,start in FOLDS.items():
  ids=np.load(SRC/name/'data.npz')['test_ids'];w=((times[ids]-pd.Timestamp(start,tz=times.tz)).days//7).to_numpy();weeks[name]=w;k=int(w.max())+1;draw=rng.integers(0,k,(2000,k));weights[name]=np.stack([np.bincount(row,minlength=k)[w] for row in draw]);sizes[name]=len(ids)
 df=pd.read_csv(OUT/'paired_comparisons.csv');assert len(df)==560 and df.primary.sum()==8;g=MODELS[0];expected=[(g,'window_exact1',g,m) for m in ['crps_exact','online_pool28','rolling28','equal_pool','raw','frozen','rolling56']]+[(g,'window_exact1',m,'window_exact1') for m in MODELS[1:]]+[(MODELS[1],'window_exact1',MODELS[1],'crps_exact')]+[(g,m,g,'window_exact1') for m in ['window_exact025','window_exact4']]+[(g,'crps_exact',g,'online_pool28'),(g,'window_exact4',g,'crps_exact')];assert set(map(tuple,df[['candidate','method','baseline','baseline_method']].drop_duplicates().values))==set(expected)
 for r in df.itertuples():
  selected=list(FOLDS) if r.scope=='ALL' else [r.scope];num=np.zeros(2000);den=np.zeros(2000);ss=0.;nn=0;blocks=[]
  for name in selected:
   a=np.stack([cache[(name,r.candidate,seed,r.method)][r.metric] for seed in SEEDS]);b=np.stack([cache[(name,r.baseline,seed,r.baseline_method)][r.metric] for seed in SEEDS]);np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b));diff=(a-b).mean(0);flat=diff.reshape(len(diff),-1);sums=np.nansum(flat,1);counts=np.isfinite(flat).sum(1);num+=weights[name]@sums;den+=weights[name]@counts;ss+=sums.sum();nn+=counts.sum();blocks.append(len(np.unique(weeks[name][counts>0])))
  valid=den>0;supported=min(blocks)>=4 and valid.sum()>=1900;lo,hi=np.quantile(num[valid]/den[valid],[.025,.975]) if supported else [np.nan,np.nan];close(r.difference,ss/nn);close(r.low95,lo);close(r.high95,hi);assert r.n==nn and r.origins==sum(sizes[n] for n in selected) and r.nonempty_blocks==sum(blocks) and r.minimum_blocks_in_fold==min(blocks) and r.valid_draws==valid.sum() and r.interval_supported==supported;main=r.scope=='ALL' and r.candidate==g and ((r.method=='window_exact1' and ((r.baseline==g and r.baseline_method=='crps_exact' and r.metric in ['crps','brier6','group_brier']) or (r.baseline==g and r.baseline_method=='rolling28' and r.metric in ['brier6','group_brier']) or (r.baseline==MODELS[1] and r.baseline_method=='window_exact1' and r.metric in ['brier6','group_brier']))) or (r.method=='crps_exact' and r.baseline==g and r.baseline_method=='online_pool28' and r.metric=='brier6'));assert r.primary==main
 return len(df)

def structural_audit():
 inventory=read(OUT/'inventory.json');old=read(PARENT/'inventory.json');baseline=[r for r in inventory if not r['new']];expected=[]
 for r in old:
  r=dict(r);r['new']=False;expected.append(r)
 assert baseline==expected;checks=0;initial_arrays=0
 for item in inventory:
  if not item['new']:continue
  name=item['fold'];model=item['model'];seed=item['seed'];s=np.load(item['score_source']);equal=np.load(PARENT/name/f'scores_{model}_{seed}_equal_pool.npz');first=np.load(ROLLING/name/'archive_data.npz')['blocks']==0
  for key in s.files:np.testing.assert_array_equal(s[key][first],equal[key][first]);initial_arrays+=1
  for short,long in [(3,6),(6,12)]:
   p=s[f'prob{short}'][:,:3];q=s[f'prob{long}'][:,:3];ok=np.isfinite(p)&np.isfinite(q);assert (p[ok]+1e-12>=q[ok]).all();checks+=int(ok.sum())
  groups=[[0,1,2],[8,9,10],[13,15,16]];joint=s['group_probability'].reshape(-1,4,3)
  for j,nodes in enumerate(groups):
   for node in nodes:
    marginal=s['prob6'][:,:,node];ok=np.isfinite(joint[:,:,j]);assert np.isfinite(marginal[ok]).all();assert (joint[:,:,j][ok]<=marginal[ok]+1e-12).all();checks+=int(ok.sum())
 return dict(baseline_inventory_exact=len(baseline),first_week_arrays_exact=initial_arrays,event_inclusion_cells=checks)
def run():
 guard();assert read(OUT/'evaluation_complete.json')['status']=='COMPLETE';previous=read(OUT/'previous_hashes.json');identity=read(OUT/'identity.json');parents={}
 for file,h in previous.items():assert sha(ART/file)==h,file
 for file,h in identity.items():assert sha(Path(file))==h,file
 for root in [PARENT,ROLLING]:
  manifest=read(root/'final_manifest.json')
  for file,h in manifest.items():assert sha(Path(file))==h,file
  parents[root.name]=len(manifest)
 d=np.load(ART/'data.npz');times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');counts=audit_weights(times);cache,sc=scores_and_tables(times);counts.update(sc);counts.update(structural_audit());phase=read(OUT/'weights_complete.json');audit=pd.read_csv(OUT/'weight_audit.csv');assert phase['new_forecast_origins']==counts['new_forecast_cases']==21120;assert phase['all_forecast_origins']==68640;assert phase['optimizer_calls']==audit.optimizer_called.sum() and phase['optimizer_failures']==audit.optimizer_failed.sum() and phase['support_fallbacks']==audit.insufficient_support.sum();counts.update(status='PASS',previous_unchanged=len(previous),frozen_identities=len(identity),parent_manifest_files=parents,paired_comparisons=paired(cache,times));(OUT/'verification.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(json.dumps(counts,indent=2),flush=True)
if __name__=='__main__':run()

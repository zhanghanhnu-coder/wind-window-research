"""Independent partial-observation dependence, rank and all-window score audit."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.special import ndtri
from model import ART
from research_crps_distribution import guard,SEEDS,METRICS
OUT=ART/'partial_copula_v1';SRC=ART/'seasonal_probability_v1';MODELS=['gated_graph_atoms','gated_temporal_atoms','gated_mean_atoms'];FOLDS=['M05','M07','M09','M11'];STARTS=['2024-05-01','2024-07-01','2024-09-01','2024-11-01'];COMPARE=['brier6','length_crps72','first_cdf_brier','group_brier','temporal_variogram','spatial_variogram'];BINS=[(0,6),(6,24),(24,48),(48,72)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def close(a,b,tol=2e-11):np.testing.assert_allclose(a,b,atol=tol,rtol=0,equal_nan=True)
def compare_table(name,rows,keys):
 a=pd.read_csv(OUT/name);b=pd.DataFrame(rows);assert len(a)==len(b),(name,len(a),len(b));a=a.sort_values(keys).reset_index(drop=True);b=b.sort_values(keys).reset_index(drop=True)
 for k in b:
  if b[k].dtype.kind in 'biufc':close(a[k],b[k])
  else:assert a[k].fillna('').tolist()==b[k].fillna('').tolist(),(name,k)
 return len(a)

def independent_normal_scores(x):
 output=np.full(x.shape,np.nan)
 for h in range(72):
  for node in range(17):
   good=np.isfinite(x[:,h,node]);values=x[good,h,node];assert len(values)>=20;order=np.argsort(values,kind='stable');_,counts=np.unique(values[order],return_counts=True);average=np.cumsum(counts)-(counts-1)/2;ranks=np.empty(len(values));ranks[order]=np.repeat(average,counts);output[good,h,node]=ndtri((ranks-.5)/len(values))
 return output
def independent_corr(x):
 k=x.shape[1];a=np.eye(k);count=np.zeros((k,k),int)
 for i in range(k):
  for j in range(i,k):
   m=np.isfinite(x[:,i])&np.isfinite(x[:,j]);count[i,j]=count[j,i]=m.sum();assert m.sum()>=100;xi=x[m,i]-x[m,i].mean();xj=x[m,j]-x[m,j].mean();den=np.sqrt(np.sum(xi*xi)*np.sum(xj*xj));assert den>0;a[i,j]=a[j,i]=np.dot(xi,xj)/den
 np.fill_diagonal(a,1.);return a,count
def repair(raw,shrink):
 r=(raw+raw.T)/2;ev,u=np.linalg.eigh(r);projected=(u*np.maximum(ev,1e-6))@u.T;diagonal=np.sqrt(np.diag(projected));projected/=diagonal[:,None]*diagonal[None,:];projected=(projected+projected.T)/2;np.fill_diagonal(projected,1);result=(1-shrink)*projected+shrink*np.eye(len(r));info=dict(shrink=shrink,minimum_eigenvalue_before=float(ev.min()),minimum_eigenvalue_repaired=float(np.linalg.eigvalsh(projected).min()),minimum_eigenvalue_final=float(np.linalg.eigvalsh(result).min()),repair_frobenius=float(np.linalg.norm(projected-r)),total_change_frobenius=float(np.linalg.norm(result-r)));return result,info
def rank_inverse(z):
 order=np.argsort(z,axis=0,kind='stable');ranks=np.empty(order.shape,int);np.put_along_axis(ranks,order,np.broadcast_to(np.arange(len(z))[:,None,None],z.shape),axis=0);return ranks
def get_ranks(base,t,s):
 # Explicit contraction, independent of production's broadcast matrix product.
 z=np.einsum('hi,sij,nj->shn',np.linalg.cholesky(t),base,np.linalg.cholesky(s),optimize=True);return rank_inverse(z)
def reconstruct(q,ranks):return np.take_along_axis(q,(ranks//3).transpose(1,2,0)[None],axis=-1).transpose(0,3,1,2)

def dependence_audit():
 support=[];repairs=[];templates={};fits=0;pair_entries=0;ev=np.load(ART/'crps_distribution_v1/evaluation_data.npz')
 for fold_index,name in enumerate(FOLDS):
  path=OUT/name;f=np.load(SRC/name/'data.npz');y=np.load(SRC/name/'cal_truth.npz')['y'].astype(float);assert max(f['cal_ids']+72)<min(f['test_ids'])
  for seed in SEEDS:
   base=np.random.default_rng(np.random.SeedSequence([20261114,fold_index,seed])).normal(size=(57,72,17));np.testing.assert_array_equal(base,np.load(path/f'noise_{seed}.npz')['base'])
   for model in MODELS:
    guard();q=np.load(SRC/name/f'predictions_{model}_{seed}_budget40.npz')['cal'].astype(float);r=(y-q[...,9])/np.maximum(q[...,17]-q[...,1],.1*float(f['scale']));z=independent_normal_scores(r);dep=np.load(path/f'dependence_{model}_{seed}.npz');np.testing.assert_array_equal(np.isfinite(z),np.isfinite(y));close(z,dep['normal_scores'],1e-14);np.testing.assert_array_equal(r,dep['residual']);corrs={};main={}
    for axis,mat in [('space',z.reshape(-1,17)),('time',z.transpose(0,2,1).reshape(-1,72))]:
     corr,counts=independent_corr(mat);close(corr,dep[axis+'_raw'],3e-12);np.testing.assert_array_equal(counts,dep[axis+'_counts']);pair_entries+=counts.size;corrs[axis]=dep[axis+'_raw'];fixed,info=repair(corrs[axis],.05);close(fixed,dep[axis+'_correlation'],3e-12);assert np.linalg.eigvalsh(fixed).min()>0;main[axis]=fixed;repairs.append(dict(fold=name,model=model,seed=seed,method='partial05',axis=axis,**info))
     original=next(a for a in read(path/f'dependence_{model}_{seed}.json') if a['axis']==axis)
     for k,value in info.items():close(original[k],value,3e-12)
    full=get_ranks(base,main['time'],main['space']);local={'partial05':full}
    if model=='gated_graph_atoms':
     tb=full.copy();sb=full.copy();rng=np.random.default_rng(20261101)
     for h in range(72):tb[:,h]=full[rng.permutation(57),h]
     rng=np.random.default_rng(20261101)
     for j in range(17):sb[:,:,j]=full[rng.permutation(57),:,j]
     local.update(time_broken=tb,space_broken=sb)
     for method,lam in [('partial00',0.),('partial10',.1)]:
      mats={}
      for axis in ['time','space']:
       mats[axis],info=repair(corrs[axis],lam);repairs.append(dict(fold=name,model=model,seed=seed,method=method,axis=axis,**info))
      local[method]=get_ranks(base,mats['time'],mats['space'])
    if name=='M11':
     chosen=ev['complete_positions'][ev['selected57']]
     # The frozen historical bank used float32 subtraction and stable ties.
     residual=y[chosen].astype(np.float32)-q[chosen,...,9].astype(np.float32);local['historical_ecc']=rank_inverse(residual);np.testing.assert_array_equal(f['cal_ids'][chosen],ev['selected57_ids']);bank=ART/('spatial_outage_controls_v1' if model=='gated_mean_atoms' else 'crps_distribution_v1')/f'bank_{model}_{seed}.npz';np.testing.assert_array_equal(local['historical_ecc'],np.load(bank)['ranks'])
    for method,ranks in local.items():
     saved=np.load(path/f'ranks_{model}_{seed}_{method}.npz')['ranks'];np.testing.assert_array_equal(ranks,saved);np.testing.assert_array_equal(np.sort(saved,axis=0),np.broadcast_to(np.arange(57)[:,None,None],saved.shape));templates[(name,model,seed,method)]=saved
    support.append(dict(fold=name,model=model,seed=seed,origins=len(y),min_coordinate=int(np.isfinite(y).sum(0).min()),min_time_pair=int(dep['time_counts'].min()),min_space_pair=int(dep['space_counts'].min()),full_all_node_paths=int(np.isfinite(y).all((1,2)).sum()),latest_cal_label=int(max(f['cal_ids']+72)),first_test_issue=int(min(f['test_ids']))));fits+=1;print('DEPENDENCE',name,model,seed,flush=True)
 compare_table('dependence_support.csv',support,['fold','model','seed']);compare_table('matrix_repairs.csv',repairs,['fold','model','seed','method','axis']);return templates,dict(dependence_fits=fits,correlation_pair_entries=pair_entries,rank_templates=len(templates),matrix_repair_rows=len(repairs))

def empirical(values,truth):
 ordered=np.sort(values,axis=1);k=values.shape[1];e=truth[:,None]-ordered;tau=(np.arange(k)+.5).reshape(1,k,1)/k;return 2*np.maximum(tau*e,(tau-1)*e).mean(1)
def windows_reference(sc,y,scale):
 good=sc<=10;out={};valid=np.isfinite(y)
 for duration in [3,6,12]:
  probs=[];labels=[]
  for start in [0,24,48,72-duration]:
   mask=valid[:,start:start+duration].all(1);p=good[:,:,start:start+duration].all(2).mean(1);label=(y[:,start:start+duration]<=10).all(1);probs.append(np.where(mask,p,np.nan));labels.append(np.where(mask,label,np.nan))
  out[f'prob{duration}']=np.stack(probs,1);out[f'label{duration}']=np.stack(labels,1);out[f'brier{duration}']=(out[f'prob{duration}']-out[f'label{duration}'])**2
 for h in [24,48,72]:
  positions=np.arange(1,h+1);last=np.maximum.accumulate(np.where(~good[:,:,:h],positions[None,None,:,None],0),axis=2);lengths=(positions[None,None,:,None]-last).max(2);observed=(positions[None,:,None]-np.maximum.accumulate(np.where(~(y[:,:h]<=10),positions[None,:,None],0),axis=1)).max(1);mask=valid[:,:h].all(1);out[f'length_crps{h}']=np.where(mask,empirical(lengths,observed),np.nan);p=(lengths>=6).mean(1);out[f'any6_prob{h}']=np.where(mask,p,np.nan);out[f'any6_brier{h}']=np.where(mask,(p-(observed>=6))**2,np.nan)
 has=np.lib.stride_tricks.sliding_window_view(good,6,axis=2).all(-1);cdf=np.maximum.accumulate(has,axis=2).mean(1);observed=np.lib.stride_tricks.sliding_window_view(y<=10,6,axis=1).all(-1);labelcdf=np.maximum.accumulate(observed,axis=1);out['first_cdf_brier']=np.where(valid.all(1),((cdf-labelcdf)**2).mean(1),np.nan);gp=[];gl=[]
 for offset in [0,24,48,66]:
  for nodes in [[0,1,2],[8,9,10],[13,15,16]]:
   mask=valid[:,offset:offset+6][:,:,nodes].all((1,2));p=good[:,:,offset:offset+6][:,:,:,nodes].all((2,3)).mean(1);label=(y[:,offset:offset+6][:,:,nodes]<=10).all((1,2));gp.append(np.where(mask,p,np.nan));gl.append(np.where(mask,label,np.nan))
 out['group_probability']=np.stack(gp,1);out['group_label']=np.stack(gl,1);out['group_brier']=(out['group_probability']-out['group_label'])**2
 pairs=[(h,i,h,j) for h in [5,23,47,71] for i in range(17) for j in range(i+1,17)];hours=[0,5,11,23,47,71];tpairs=[(h,n,h2,n) for pos,h in enumerate(hours) for h2 in hours[pos+1:] for n in range(17)]
 for key,items in [('spatial_variogram',pairs),('temporal_variogram',tpairs)]:
  a,i,b,j=np.array(items).T;pred=np.sqrt(abs(sc[:,:,a,i]-sc[:,:,b,j])/scale).mean(1)
  # Preserve the frozen score's float32 observation subtraction/square root;
  # forecast expectations are float64. All event labels are exact comparisons.
  obs32=y.astype(np.float32);actual=np.sqrt(abs(obs32[:,a,i]-obs32[:,b,j])/scale);out[key]=(pred-actual)**2
 out['ess']=np.full((len(y),1),57.);return out
def longest_direct(v):
 groups=np.diff(np.r_[-1,np.flatnonzero(~v),len(v)]);return int(groups.max()-1)

def scores_and_tables(templates,times):
 inventory=read(OUT/'inventory.json');assert len(inventory)==93;cache={};rows=[];months=[];leads=[];rel=[];high=[];conservation=[];cases=0;arrays=0;brute=0
 for item in inventory:
  guard();name=item['fold'];model=item['model'];seed=item['seed'];method=item['method'];path=OUT/name;key=(name,model,seed,method);ranks=templates[key];assert sha(path/f'ranks_{model}_{seed}_{method}.npz')==item['rank_sha256'] and sha(SRC/name/f'calibrated_{model}_{seed}.npz')==item['marginal_sha256'];f=np.load(SRC/name/'data.npz');q=np.load(SRC/name/f'calibrated_{model}_{seed}.npz')['q'];y=np.load(SRC/name/'test_truth.npz')['y'].astype(float);s=dict(np.load(path/f'scores_{model}_{seed}_{method}.npz'));previous=np.load(SRC/name/f'scores_{model}_{seed}_smooth_affine.npz');assert len(y)==item['origins'];maximum=0.
  for metric in ['crps','coverage','width','interval_score','low_miss','high_miss']:close(s[metric],previous[metric],3e-12);maximum=max(maximum,float(np.nanmax(abs(s[metric]-previous[metric]))));arrays+=1
  for start in range(0,len(q),8):
   stop=min(start+8,len(q));sc=reconstruct(q[start:stop],ranks);np.testing.assert_array_equal(np.sort(sc,axis=1),np.repeat(q[start:stop],3,axis=-1).transpose(0,3,1,2));ref=windows_reference(sc,y[start:stop],float(f['scale']))
   for k,v in ref.items():close(s[k][start:stop],v,3e-12)
   cases+=stop-start
  arrays+=len(ref)
  for pos in [0,len(q)//2,len(q)-1]:
   sc=reconstruct(q[pos:pos+1],ranks)[0]
   for node in [0,8,16]:
    obs=y[pos,:,node];valid=np.isfinite(obs).all();lengths=np.array([longest_direct(row<=10) for row in sc[:,:,node]]);truth=longest_direct(obs<=10);score=np.mean(abs(lengths-truth))-.5*np.mean(abs(lengths[:,None]-lengths[None,:]));close(s['length_crps72'][pos,node],score if valid else np.nan,3e-12);brute+=1
  if method=='space_broken':
   full=np.load(path/f'scores_{model}_{seed}_partial05.npz')
   for metric in ['brier3','brier6','brier12','length_crps24','length_crps48','length_crps72','any6_brier24','any6_brier48','any6_brier72','first_cdf_brier']:close(s[metric],full[metric],3e-12)
  conservation.append(dict(fold=name,model=model,seed=seed,method=method,max_marginal_difference=maximum,space_single_node_conserved=method=='space_broken'));cache[key]={k:s[k] for k in COMPARE};common=dict(fold=name,model=model,seed=seed,method=method);ts=times[f['test_ids']]
  for metric in METRICS:
   z=s[metric];rows.append(dict(**common,metric=metric,value=float(np.nanmean(z)),n=int(np.isfinite(z).sum())))
   for month in sorted(set(ts.month)):
    a=z[ts.month==month];months.append(dict(**common,issue_month=int(month),metric=metric,value=float(np.nanmean(a)),n=int(np.isfinite(a).sum())))
  for a,b in BINS:
   for metric in ['crps','coverage','width','interval_score','low_miss','high_miss']:
    z=s[metric][:,a:b];leads.append(dict(**common,lead_start=a+1,lead_end=b,metric=metric,value=float(np.nanmean(z)),n=int(np.isfinite(z).sum())))
  for target,pk,yk in [('single_6h','prob6','label6'),('group_6h','group_probability','group_label')]:
   p=s[pk].ravel();label=s[yk].ravel();ok=np.isfinite(label);p=p[ok];label=label[ok];select=p>=.8;high.append(dict(**common,target=target,n=int(select.sum()),total=int(len(p)),mean_probability=float(p[select].mean()) if select.any() else np.nan,observed_rate=float(label[select].mean()) if select.any() else np.nan))
   for b in range(10):
    select=(p>=b/10)&(p<(b+1)/10 if b<9 else p<=1);rel.append(dict(**common,target=target,bin=b,n=int(select.sum()),mean_probability=float(p[select].mean()) if select.any() else np.nan,observed_rate=float(label[select].mean()) if select.any() else np.nan))
  print('SCORES',name,model,seed,method,flush=True)
 df=pd.DataFrame(rows)
 for key,g in df.groupby(['model','seed','method','metric'],sort=False):
  if g.fold.nunique()==4:rows.append(dict(fold='ALL',**dict(zip(['model','seed','method','metric'],key)),value=float(np.dot(g.value,g.n)/g.n.sum()),n=int(g.n.sum())))
 keys=['fold','model','seed','method'];tablecounts={}
 for name,values,extra in [('metrics.csv',rows,['metric']),('month_metrics.csv',months,['metric','issue_month']),('lead_metrics.csv',leads,['metric','lead_start']),('high_confidence.csv',high,['target']),('reliability.csv',rel,['target','bin']),('marginal_conservation.csv',conservation,[])]:tablecounts[name]=compare_table(name,values,keys+extra)
 return cache,dict(forecast_cases=cases,complete_score_arrays=arrays,independent_run_length_cases=brute,table_rows=tablecounts)

def comparison_audit(cache,times):
 rng=np.random.default_rng(20261114);weights={};weeks={};sizes={}
 for name,start in zip(FOLDS,STARTS):
  ids=np.load(SRC/name/'data.npz')['test_ids'];w=((times[ids]-pd.Timestamp(start,tz=times.tz)).days//7).to_numpy();weeks[name]=w;k=int(w.max())+1;draw=rng.integers(0,k,(2000,k));weights[name]=np.stack([np.bincount(row,minlength=k)[w] for row in draw]);sizes[name]=len(ids)
 df=pd.read_csv(OUT/'paired_comparisons.csv');assert len(df)==198 and df.primary.sum()==5
 for r in df.itertuples():
  selected=FOLDS if r.scope=='ALL' else [r.scope];num=np.zeros(2000);den=np.zeros(2000);ss=0.;nn=0;blocks=[]
  for name in selected:
   a=np.stack([cache[(name,r.candidate,seed,r.method)][r.metric] for seed in SEEDS]);b=np.stack([cache[(name,r.baseline,seed,r.baseline_method)][r.metric] for seed in SEEDS]);np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b));diff=(a-b).mean(0);flat=diff.reshape(len(diff),-1);sums=np.nansum(flat,1);counts=np.isfinite(flat).sum(1);num+=weights[name]@sums;den+=weights[name]@counts;ss+=sums.sum();nn+=counts.sum();blocks.append(len(np.unique(weeks[name][counts>0])))
  valid=den>0;supported=min(blocks)>=4 and valid.sum()>=1900;lo,hi=np.quantile(num[valid]/den[valid],[.025,.975]) if supported else [np.nan,np.nan];close(r.difference,ss/nn);close(r.low95,lo);close(r.high95,hi);assert r.n==nn and r.origins==sum(sizes[name] for name in selected) and r.nonempty_blocks==sum(blocks) and r.minimum_blocks_in_fold==min(blocks) and r.valid_draws==valid.sum() and r.interval_supported==supported;primary=r.scope=='ALL' and r.candidate=='gated_graph_atoms' and r.method=='partial05' and ((r.baseline_method=='time_broken' and r.metric in ['brier6','first_cdf_brier']) or (r.baseline_method=='space_broken' and r.metric=='group_brier') or (r.baseline=='gated_temporal_atoms' and r.metric in ['brier6','group_brier']));assert r.primary==primary
 return len(df)

def run():
 guard();previous=read(OUT/'previous_hashes.json');identity=read(OUT/'identity.json');parent=read(SRC/'final_manifest.json')
 for path,value in previous.items():assert sha(ART/path)==value,path
 for source in [identity,parent]:
  for path,value in source.items():assert sha(Path(path))==value,path
 phase=read(OUT/'prediction_complete.json');assert phase['flows']==93 and phase['dependence_fits']==36;d=np.load(ART/'data.npz');times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');templates,counts=dependence_audit();cache,scorecounts=scores_and_tables(templates,times);assert phase['forecast_origins']==scorecounts['forecast_cases'];counts.update(scorecounts);counts.update(status='PASS',previous_unchanged=len(previous),frozen_identities=len(identity),parent_manifest_files=len(parent),paired_comparisons=comparison_audit(cache,times));(OUT/'verification.json').write_text(json.dumps(counts,indent=2),encoding='utf-8');print(json.dumps(counts,indent=2),flush=True)
if __name__=='__main__':run()

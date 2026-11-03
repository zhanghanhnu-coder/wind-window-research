"""Frozen propagation-graph experiment; new artifacts only, raw files read-only."""
from pathlib import Path
import json,hashlib,copy,argparse,itertools,concurrent.futures
import numpy as np
import pandas as pd
import torch
from model import ART,origins,batches,pinball,predict
from propagation_graph import difference,lag_profile,best_lag,dtw,lag_adjacency,directed_pair_edges,LagNet

ROOT=Path(__file__).resolve().parent.parent
OUT=ART/'propagation_v1';OUT.mkdir(exist_ok=True)
RESULT=ROOT/'核心结果/传播图研究';RESULT.mkdir(parents=True,exist_ok=True)
SEEDS=[20241101,20241102,20241103]
NAMES=['lagcorr','lagmi','dtw','fusion','fusion_zero']

def save_csv(rows,name):
 pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')

def data():
 d=np.load(ART/'data.npz');meta=json.loads((ART/'metadata.json').read_text(encoding='utf-8'))
 times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai')
 return d,meta,times

def extract():
 d,meta,times=data();nodes=meta['nodes'];files=sorted((ROOT/'xx风电场 2024.01.01-2024.12.31 (2)').rglob('*.csv'))
 def one(path):
  if '资产' in path.name:return None
  head=pd.read_csv(path,nrows=1);node=str(head.k_device.iloc[0])[-2:]
  if node not in nodes:return None
  cols=[c for c in ['k_ts','feature_A','feature_B'] if c in head]
  f=pd.read_csv(path,usecols=cols);ts=pd.to_numeric(f.k_ts,errors='coerce');bad=ts.isna()
  if bad.any():ts.loc[bad]=pd.to_datetime(f.loc[bad,'k_ts']).dt.tz_localize('Asia/Shanghai').astype('int64')/1e6
  index=pd.DatetimeIndex(pd.to_datetime(ts,unit='ms',utc=True)).tz_convert('Asia/Shanghai')
  if index.duplicated().any():raise ValueError('duplicate timestamps')
  v=pd.to_numeric(f['feature_A'],errors='coerce');v=v.mask(v.abs().between(1e-45,1e-30)|(v<0)|(v>60))
  a=pd.to_numeric(f.get('feature_B',pd.Series(np.nan,index=f.index)),errors='coerce');a=a.mask(a.abs().between(1e-45,1e-30)|(a<0)|(a>360))
  return node,pd.DataFrame({'wind':v.to_numpy(),'direction':a.to_numpy()},index=index),{'file':path.name,'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
 groups={};manifest=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for item in pool.map(one,files):
   if item is not None:node,frame,info=item;groups.setdefault(node,[]).append(frame);manifest.append(info)
 grid=pd.date_range('2024-01-01','2025-01-01',freq='5min',inclusive='left',tz='Asia/Shanghai')
 wind={};directions={};checks=[]
 for node in nodes:
  frame=pd.concat(groups[node]).sort_index()
  if frame.index.duplicated().any():raise ValueError('overlapping source exports')
  wind[node]=frame.wind.reindex(grid);directions[node]=frame.direction.reindex(grid)
  h=frame.wind.resample('h',closed='left',label='right').agg(['mean','count']).reindex(times)
  hh=h['mean'].where(h['count']==12).to_numpy();old=d['mean'][:,nodes.index(node)]
  if not np.allclose(hh,old,equal_nan=True,atol=1e-6):raise ValueError('five minute extraction differs from canonical hourly cache')
  checks.append({'node':node,'valid_5min_wind':int(wind[node].notna().sum()),'valid_5min_direction':int(directions[node].notna().sum()),'hourly_matches_canonical':True})
 w=pd.DataFrame(wind);a=pd.DataFrame(directions);rad=np.deg2rad(a)
 sn=np.sin(rad).resample('h',closed='left',label='right').mean();cs=np.cos(rad).resample('h',closed='left',label='right').mean();count=a.resample('h',closed='left',label='right').count()
 strength=np.sqrt(sn**2+cs**2);direction=(np.rad2deg(np.arctan2(sn,cs))%360).where((count==12)&(strength>=.5)).reindex(times)
 np.savez_compressed(OUT/'fine_data.npz',wind=w.to_numpy(dtype='float32'),direction=a.to_numpy(dtype='float32'),timestamps=np.asarray(grid.astype(str),dtype=str),hourly_direction=direction.to_numpy(dtype='float32'))
 save_csv(checks,'data_audit.csv');save_csv(manifest,'raw_read_manifest.csv');print('EXTRACTION_COMPLETE',len(manifest),flush=True)

def construct():
 d,meta,times=data();nodes=meta['nodes'];n=len(nodes);fit=times<pd.Timestamp('2024-09-01',tz=times.tz);v=d['mean'][fit].astype(float);dv=difference(v)
 config={'version':'propagation_v1','training_graph_end_exclusive':'2024-09-01','lag_hours':6,'MI_bins':8,'DTW_block_hours':168,'DTW_band_hours':6,'top_k':4,'seeds':SEEDS,'epochs':12,'models':NAMES,'scope':'lead-lag association, not physical causal propagation','primary':'unchanged v2 Brier, 6h threshold10 offsets0/24/48/66','data_sha256':hashlib.sha256((ART/'data.npz').read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((ROOT/'研究笔记/传播图研究方案_2026-11-01.md').read_bytes()).hexdigest()}
 (OUT/'recipe.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
 pairs=[];profiles=[];months=[];dtw_rows=[];method_pairs={k:[] for k in NAMES[:3]}
 for i,j in itertools.combinations(range(n),2):
  for signal,values in [('level',v),('change',dv)]:
   for method in ['corr','mi']:
    ls,ss,count=lag_profile(values[:,i],values[:,j],6,method,100);lag,score,_=best_lag(values[:,i],values[:,j],6,method,100)
    row={'node_i':nodes[i],'node_j':nodes[j],'signal':signal,'method':method,'lag_hours':lag,'score':score,'n_common':count,'zero_score':float(ss[6]),'gain_over_zero':float(score-ss[6])};pairs.append(row)
    profiles.extend(dict(row,lag_hours=int(l),score=float(s)) for l,s in zip(ls,ss))
    if signal=='change':method_pairs['lagcorr' if method=='corr' else 'lagmi'].append((i,j,lag,score))
  for month in range(1,9):
   vv=difference(v[times[fit].month==month])
   for method in ['corr','mi']:
    lag,score,count=best_lag(vv[:,i],vv[:,j],6,method,100)
    months.append({'node_i':nodes[i],'node_j':nodes[j],'month':month,'method':method,'lag_hours':lag,'score':score,'n_common':count})
  costs=[];lags=[]
  for start in range(0,len(v)-167,168):
   a=v[start:start+168,i];b=v[start:start+168,j]
   if not np.isfinite(a).all() or not np.isfinite(b).all() or a.std()<1e-8 or b.std()<1e-8:continue
   cost,lag,path=dtw((a-a.mean())/a.std(),(b-b.mean())/b.std(),6)
   costs.append(cost);lags.append(lag);dtw_rows.append({'node_i':nodes[i],'node_j':nodes[j],'block_start':str(times[fit][start]),'rms_cost':cost,'path_median_lag':lag,'path_length':len(path)})
  # At least three complete blocks; median path lag rounded to an integer hour.
  lag=int(np.rint(np.median(lags))) if len(lags)>=3 else 0
  score=1/(1+float(np.mean(costs))) if len(lags)>=3 else 0.
  method_pairs['dtw'].append((i,j,lag,score))
  pairs.append({'node_i':nodes[i],'node_j':nodes[j],'signal':'standardized_week','method':'dtw','lag_hours':lag,'score':score,'n_common':len(lags),'zero_score':np.nan,'gain_over_zero':np.nan})
 graphs={};edges=[]
 for name,rows in method_pairs.items():
  es=[]
  for i,j,lag,score in rows:es.extend(directed_pair_edges(i,j,lag,score))
  graphs[name]=lag_adjacency(n,es)
 # Pre-fixed equal-method fusion; a tie in lag votes prefers zero/shorter lag.
 maxima={name:max([r[3] for r in rows]+[1e-12]) for name,rows in method_pairs.items()};fes=[]
 for k,(i,j) in enumerate(itertools.combinations(range(n),2)):
  votes={};total=0.
  for name,rows in method_pairs.items():
   _,_,lag,score=rows[k];weight=score/maxima[name];votes[lag]=votes.get(lag,0)+weight;total+=weight
  lag=min(votes,key=lambda l:(-votes[l],abs(l),l));fes.extend(directed_pair_edges(i,j,lag,total/3))
 graphs['fusion']=lag_adjacency(n,fes);graphs['fusion_zero']=graphs['fusion'].sum(0,keepdims=True)
 for name,a in graphs.items():
  for lag,receiver,source in zip(*np.where(a>0)):
   edges.append({'model':name,'source':nodes[source],'receiver':nodes[receiver],'lag_hours':int(lag),'weight':float(a[lag,receiver,source])})
 np.savez_compressed(OUT/'graphs.npz',**graphs);save_csv(edges,'graph_edges.csv');save_csv(pairs,'pair_summary.csv');save_csv(profiles,'hourly_lag_profiles.csv');save_csv(months,'monthly_lags.csv');save_csv(dtw_rows,'dtw_blocks.csv')
 # Diagnostic only: train-period 5min resolution and angle-sector conditioning.
 f=np.load(OUT/'fine_data.npz');ft=pd.DatetimeIndex(f['timestamps']);five=f['wind'][ft<pd.Timestamp('2024-09-01',tz='Asia/Shanghai')].astype(float);fine=[]
 for signal,values in [('level',five),('change',difference(five))]:
  for i,j in itertools.combinations(range(n),2):
   lag,score,count=best_lag(values[:,i],values[:,j],12,'corr',1000)
   fine.append({'node_i':nodes[i],'node_j':nodes[j],'signal':signal,'lag_minutes':lag*5,'score':score,'n_common':count})
 save_csv(fine,'five_minute_lags.csv')
 a=f['hourly_direction'][fit];z=np.exp(1j*np.deg2rad(a));cnt=np.isfinite(a).sum(1);mean=np.nansum(z,axis=1)/np.maximum(cnt,1);angle=np.rad2deg(np.angle(mean))%360;valid=(cnt>=10)&(abs(mean)>=.5);sector=(angle//90).astype(int);sector_rows=[]
 for sector_id in range(4):
  keep=valid&(sector==sector_id);masked=dv.copy();masked[~keep]=np.nan
  for i,j in itertools.combinations(range(n),2):
   lag,score,count=best_lag(masked[:,i],dv[:,j],6,'corr',100)
   sector_rows.append({'node_i':nodes[i],'node_j':nodes[j],'angle_bin':f'{sector_id*90}-{(sector_id+1)*90}','lag_hours':lag,'score':score,'n_common':count,'note':'unconfirmed absolute direction convention; diagnostic only'})
 save_csv(sector_rows,'direction_sector_lags.csv')
 summary={name:{'edges':int((a>0).sum()),'nonzero_lag_edges':int((a[1:]>0).sum()),'receivers_without_neighbor':int((a.sum((0,2))==0).sum())} for name,a in graphs.items()}
 summary['direction']={'valid_training_hours':int(valid.sum()),'total_training_hours':len(valid),'sector_hours':[int((valid&(sector==i)).sum()) for i in range(4)]}
 (OUT/'graph_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('GRAPH_COMPLETE',summary,flush=True)

def train():
 d,meta,times=data();graphs=np.load(OUT/'graphs.npz');x=d['x'];raw=d['raw'];center=meta['center'];scale=meta['scale'];y=((raw-center)/scale).astype('float32')
 tr=origins(times,'2024-01-12','2024-09-01',3);va=origins(times,'2024-09-01','2024-10-01',6)
 tr=np.array([i for i in tr if x[i-23:i+1,:,-1].mean()>=.65]);va=np.array([i for i in va if x[i-23:i+1,:,-1].mean()>=.65]);te=d['test_ids'];cal=d['cal_ids']
 truth=np.stack([raw[i+1:i+73] for i in te]);ct=np.stack([raw[i+1:i+73] for i in cal]);vy=np.stack([y[i+1:i+73] for i in va]);selected=np.random.default_rng(20241101).choice(len(cal),32,replace=False)
 records=[];windows=[];losses={};history={}
 for seed in SEEDS:
  for name in NAMES:
   ckpath=OUT/f'{name}_{seed}.pt';errpath=OUT/f'{name}_{seed}_scores.npz';recordpath=OUT/f'{name}_{seed}_record.json'
   if ckpath.exists() and errpath.exists() and recordpath.exists():
    record=json.loads(recordpath.read_text());errors=np.load(errpath)['errors'];print('REUSE',name,seed,flush=True)
   else:
    torch.manual_seed(seed);net=LagNet(graphs[name]);opt=torch.optim.Adam(net.parameters(),lr=.001);best=float('inf');hist=[]
    for epoch in range(1,13):
     net.train()
     for xx,yy in batches(x,y,tr,32,True,epoch):
      opt.zero_grad();loss=pinball(net(xx),yy);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
     q=predict(net,x,va);vl=float(pinball(torch.from_numpy(q),torch.from_numpy(vy)));hist.append(vl)
     if vl<best:best=vl;state=copy.deepcopy(net.state_dict());be=epoch
    net.load_state_dict(state);med=predict(net,x,te)[...,1]*scale+center;cm=predict(net,x,cal)[...,1]*scale+center;residual=ct-cm;sc=np.maximum(0,med[:,None]+residual[selected][None]);errors=[]
    for offset in [0,24,48,66]:
     obs=truth[:,offset:offset+6];valid=np.isfinite(obs).all(1);label=(obs<=10).all(1);prob=(sc[:,:,offset:offset+6]<=10).all(2).mean(1);errors.append(np.where(valid,(prob-label)**2,np.nan))
    errors=np.stack(errors,axis=1);record={'model':name,'seed':seed,'selected_epoch':be,'validation_pinball':best,'mae_all':float(np.nanmean(abs(med-truth))),'brier_primary':float(np.nanmean(errors)),'brier_first6':float(np.nanmean(errors[:,0]))}
    for h in [1,6,24,72]:record[f'mae_{h}h']=float(np.nanmean(abs(med[:,h-1]-truth[:,h-1])))
    torch.save({'state':state,'lag_adj':graphs[name],'center':center,'scale':scale,'best_epoch':be,'dilation':5,'seed':seed,'scope':'experimental statistical lead-lag graph'},ckpath)
    np.savez_compressed(errpath,errors=errors,median=med,residual=residual,cal_ids=cal,test_ids=te)
    recordpath.write_text(json.dumps(record,indent=2),encoding='utf-8');(OUT/f'{name}_{seed}_history.json').write_text(json.dumps(hist),encoding='utf-8')
   records.append(record);losses.setdefault(name,[]).append(errors)
   for month in [0,11,12]:
    mask=np.ones(len(te),bool) if month==0 else times[te].month==month
    for k,offset in enumerate([0,24,48,66]):windows.append({'model':name,'seed':seed,'month':month,'offset':offset,'brier':float(np.nanmean(errors[mask,k])),'n':int(np.isfinite(errors[mask,k]).sum())})
   save_csv(records,'training_metrics.csv');save_csv(windows,'window_metrics.csv');print('TRAINED',json.dumps(record),flush=True)
 np.savez_compressed(OUT/'paired_losses.npz',**{k:np.stack(v) for k,v in losses.items()},test_ids=te)
 old=np.load(ART/'v2/paired_losses.npz');assert np.array_equal(old['test_ids'],te)
 for name in ['graph24','temporal24','mean24','random24']:losses[name]=old[name]
 previous=pd.read_csv(ART/'v2/ablation_metrics.csv');save_csv(pd.concat([pd.DataFrame(records),previous],ignore_index=True),'all_model_metrics.csv')
 weeks=((times[te]-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();keys=np.unique(weeks);draw=np.random.default_rng(20241101).integers(0,len(keys),(2000,len(keys)));comparisons=[]
 comparisons_to_run=[(name,base) for name in NAMES for base in ['graph24','temporal24']]+[('fusion','fusion_zero')]
 for name,base in comparisons_to_run:
  a=np.asarray(losses[name]);b=np.asarray(losses[base]);assert np.array_equal(np.isfinite(a),np.isfinite(b));delta=np.mean(a-b,axis=0)
  sums=np.array([np.nansum(delta[weeks==k]) for k in keys]);counts=np.array([np.isfinite(delta[weeks==k]).sum() for k in keys]);boot=sums[draw].sum(1)/counts[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975])
  comparisons.append({'candidate':name,'baseline':base,'difference':float(np.nanmean(delta)),'low95':lo,'high95':hi,'blocks':len(keys),'replicates':2000,'exploratory_not_multiplicity_adjusted':True})
 save_csv(comparisons,'paired_comparisons.csv');print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('step',choices=['extract','construct','train']);args=parser.parse_args();globals()[args.step]()

from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json,hashlib,argparse,concurrent.futures,itertools
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from conditional_core import design,changes,error,formula_error,shift
from event_graph import state_at,KINDS
from propagation_graph import lag_profile

ROOT=Path(__file__).resolve().parent.parent;ART=ROOT/'风电运维Demo/artifacts';OUT=ART/'conditional_v1';OUT.mkdir(exist_ok=True);RESULT=ROOT/'核心结果/测点与条件预测研究';RESULT.mkdir(parents=True,exist_ok=True)
PAIRS=[('10','01'),('09','17'),('09','19'),('14','19')];PAIRS=PAIRS+[(b,a) for a,b in PAIRS]
CODES={'wind':'feature_A','north':'feature_B','heading':'feature_C','relative':'feature_D'}

def dump(x,name):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(x,name):pd.DataFrame(x).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def load():
 fine=np.load(ART/'propagation_v1/fine_data.npz');times=pd.DatetimeIndex(fine['timestamps']);nodes=json.loads((ART/'metadata.json').read_text(encoding='utf-8'))['nodes']
 return fine,times,nodes

def extract():
 fine,times,nodes=load();selected=sorted(set(sum(([a,b] for a,b in PAIRS),[])))
 if not (OUT/'previous_hashes.json').exists():dump({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ART.rglob('*') if p.is_file() and 'conditional_v1' not in p.parts and p.suffix in ['.pt','.npz','.csv','.joblib']},'previous_hashes.json')
 protocol=(ROOT/'研究笔记/测点与条件预测研究方案_2026-11-01.md').read_text(encoding='utf-8').split('## 执行记录')[0]
 if (OUT/'protocol_frozen.md').exists():assert (OUT/'protocol_frozen.md').read_text(encoding='utf-8')==protocol
 else:(OUT/'protocol_frozen.md').write_text(protocol,encoding='utf-8')
 files=sorted((ROOT/'xx风电场 2024.01.01-2024.12.31 (2)').rglob('*.csv'))
 def one(path):
  if '资产' in path.name:return None
  head=pd.read_csv(path,nrows=1);node=str(head.k_device.iloc[0])[-2:]
  if node not in selected:return None
  df=pd.read_csv(path,usecols=[c for c in ['k_ts']+list(CODES.values()) if c in head]);ts=pd.to_numeric(df.k_ts,errors='coerce');bad=ts.isna()
  if bad.any():ts.loc[bad]=pd.to_datetime(df.loc[bad,'k_ts']).dt.tz_localize('Asia/Shanghai').astype('int64')/1e6
  index=pd.DatetimeIndex(pd.to_datetime(ts,unit='ms',utc=True)).tz_convert('Asia/Shanghai');z={};quality=[]
  for name,code in CODES.items():
   v=pd.to_numeric(df.get(code,pd.Series(np.nan,index=df.index)),errors='coerce');tiny=v.abs().between(1e-45,1e-30)
   bounds={'wind':(0,60),'north':(0,360),'heading':(-360,360),'relative':(-180,180)}[name];valid=v.between(*bounds)&~tiny
   quality.append({'file':path.name,'node':node,'channel':name,'rows':len(v),'finite':int(v.notna().sum()),'tiny':int(tiny.sum()),'valid':int(valid.sum()),'negative_valid':int((valid&(v<0)).sum())});z[name]=v.where(valid).to_numpy()
  frame=pd.DataFrame(z,index=index);frame['heading_original']=frame.heading;frame.heading=frame.heading%360
  return node,frame,quality,{'file':path.name,'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
 groups={};quality=[];manifest=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for item in pool.map(one,files):
   if item is not None:node,z,q,m=item;groups.setdefault(node,[]).append(z);quality.extend(q);manifest.append(m)
 arrays={name:np.full((len(times),len(selected)),np.nan,np.float32) for name in ['wind','north','heading','relative']};clocks=[]
 for j,node in enumerate(selected):
  z=pd.concat(groups[node]).sort_index();delta=np.diff(z.index.asi8)/(60*1e9);off=(z.index.minute%5!=0)|(z.index.second!=0)|(z.index.microsecond!=0)
  clocks.append({'node':node,'rows':len(z),'duplicate_times':int(z.index.duplicated().sum()),'off_5min_grid':int(off.sum()),'gap_intervals_over5min':int((delta>5).sum()),'missing_grid_rows':len(times)-int(times.isin(z.index).sum()),'first_time':str(z.index.min()),'last_time':str(z.index.max())})
  assert not z.index.duplicated().any();assert not off.any();z=z.reindex(times)
  for name in arrays:arrays[name][:,j]=z[name].to_numpy(dtype=np.float32)
  assert np.allclose(arrays['wind'][:,j],fine['wind'][:,nodes.index(node)],equal_nan=True)
  assert np.allclose(arrays['north'][:,j],fine['direction'][:,nodes.index(node)],equal_nan=True)
 np.savez_compressed(OUT/'sensor_data.npz',nodes=np.array(selected),**arrays);csv(clocks,'timestamp_audit.csv');csv(quality,'raw_channel_quality.csv');csv(manifest,'raw_read_manifest.csv')
 rows=[];formula=[];choice=[];fit=times<pd.Timestamp('2024-09-01',tz=times.tz)
 for j,node in enumerate(selected):
  for name,arr in arrays.items():
   vals=arr[:,j].astype(float);dv=changes(vals,name!='wind');acc=changes(dv);valid2=np.isfinite(dv);valid3=np.isfinite(acc)&(abs(dv)>.01)&(abs(shift(dv,1))>.01)
   for month in range(1,13):
    m=times.month==month;v2=m&valid2;v3=m&valid3
    rows.append({'node':node,'channel':name,'month':month,'valid':int(np.isfinite(vals[m]).sum()),'calendar_rows':int(m.sum()),'consecutive_valid':int(v2.sum()),'repeat_fraction':float(np.mean(abs(dv[v2])<=1e-6)) if v2.any() else None,'moving_triplets':int(v3.sum()),'near_linear_fraction':float(np.mean(abs(acc[v3])<=1e-4)) if v3.any() else None})
  north=arrays['north'][:,j];heading=arrays['heading'][:,j];relative=arrays['relative'][:,j];options=[]
  for sign,offset in itertools.product([1,-1],[0,180]):
   e=formula_error(north,heading,relative,sign,offset);options.append((float(np.nanmean(e[fit])),sign,offset))
   for period,months in [('train',range(1,9)),('sep_oct',[9,10]),('nov_dec',[11,12])]:
    m=np.isin(times.month,months)&np.isfinite(e);formula.append({'node':node,'sign':sign,'offset':offset,'period':period,'n':int(m.sum()),'circular_MAE':float(np.mean(e[m])) if m.any() else None,'median_error':float(np.median(e[m])) if m.any() else None,'within1degree':float(np.mean(e[m]<=1)) if m.any() else None})
  best=min(options);choice.append({'node':node,'sign':best[1],'offset':best[2],'train_MAE':best[0]})
 csv(rows,'channel_monthly_patterns.csv');csv(formula,'angle_formula_checks.csv');csv(choice,'training_selected_angle_formula.csv')
 print('EXTRACTED',len(manifest),'files',clocks,flush=True);print('FORMULA',choice,flush=True)

def identity():
 paths=[OUT/'protocol_frozen.md',OUT/'sensor_data.npz',ART/'propagation_v1/fine_data.npz',ART/'event_propagation_v1/events.json',Path(__file__),ROOT/'风电运维Demo/conditional_core.py']
 return hashlib.sha256(''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in paths).encode()).hexdigest()

def masks(times,valid):
 out={};end=times+pd.Timedelta(minutes=5)
 for name,a,b in [('train','2024-01-01','2024-09-01'),('val','2024-09-01','2024-10-01'),('oct','2024-10-01','2024-11-01'),('test','2024-11-01','2025-01-01')]:
  out[name]=valid&(times>=pd.Timestamp(a,tz=times.tz))&(end<pd.Timestamp(b,tz=times.tz))
 return out

def fit_select(features,target,current,columns,m,signal):
 best=None;trials=[]
 for alpha in [.1,1.,10.,100.]:
  model=make_pipeline(StandardScaler(),Ridge(alpha=alpha));model.fit(features[m['train']][:,columns],target[m['train']]);pred=model.predict(features[m['val']][:,columns]);score=float(np.mean(error(pred,target[m['val']],signal,current[m['val']])))
  trials.append({'alpha':alpha,'validation_MAE':score})
  if best is None or score<best[0]:best=(score,alpha,model)
 return best[2],best[1],trials

def run():
 fine,times,nodes=load();sensors=np.load(OUT/'sensor_data.npz');events=json.loads((ART/'event_propagation_v1/events.json').read_text(encoding='utf-8'));states=np.array([KINDS.index(state_at(events,int(i)+1)) for i in range(len(times))],np.int8)
 token=identity();dump({'identity':token,'pairs':PAIRS,'signal_targets':['wind','direction'],'alphas':[.1,1,10,100],'prediction':'next recorded 5min sample; publication delay unconfirmed','scope':'conditional linear predictive information; not causal identification'},'recipe.json')
 wind=fine['wind'].astype(float);direction=fine['direction'].astype(float);metrics=[];counts=[];comparisons=[];partials=[];tuning=[];sensitivity=[];eventsignals=np.load(ART/'event_propagation_v1/event_signals.npz')
 for source,target in PAIRS:
  si=nodes.index(source);ti=nodes.index(target)
  for signal in ['direction','wind']:
   z=design(wind,direction,si,ti,signal);features=z['features'];yy=z['target'];cur=z['current'];valid=np.isfinite(features).all(1)&np.isfinite(yy);split=masks(times,valid)
   collections=[('main',features,z['columns'],split)]
   if signal=='direction':
    k=list(sensors['nodes']).index(target);extra=np.column_stack([changes(sensors['heading'][:,k],True),changes(sensors['relative'][:,k],True)]);aug=np.column_stack([features,extra]);sm=masks(times,valid&np.isfinite(extra).all(1))
    if sm['train'].sum()>=2000 and sm['val'].sum()>=200 and sm['test'].sum()>=200:collections.append(('sensor',aug,{'common':list(range(23)),'source5':list(range(23))+[23],'common_sensor':list(range(23))+[26,27],'sensor_source5':list(range(23))+[23,26,27]},sm))
    else:counts.append({'source':source,'target':target,'signal':signal,'collection':'sensor_skipped','train':int(sm['train'].sum()),'val':int(sm['val'].sum()),'oct':int(sm['oct'].sum()),'test':int(sm['test'].sum())})
   for collection,xx,sets,m in collections:
    counts.append({'source':source,'target':target,'signal':signal,'collection':collection,**{name:int(mask.sum()) for name,mask in m.items()}})
    if min(m['train'].sum(),m['val'].sum(),m['test'].sum())<100:raise ValueError('Insufficient primary cases')
    ids=np.flatnonzero(m['oct']|m['test']);truth=yy[ids];current=cur[ids];predictions={};errors={};models={}
    for name,cols in sets.items():
     model,alpha,trials=fit_select(xx,yy,cur,cols,m,signal);models[name]=model;pred=model.predict(xx[ids][:,cols]);predictions[name]=pred;errors[name]=error(pred,truth,signal,current)
     tuning.extend({'source':source,'target':target,'signal':signal,'collection':collection,'model':name,**t} for t in trials)
     path=OUT/f'{collection}_{signal}_{source}_{target}_{name}.joblib';joblib.dump({'model':model,'columns':cols,'alpha':alpha,'identity':token},path)
    predictions['persistence']=np.zeros(len(ids));errors['persistence']=error(predictions['persistence'],truth,signal,current)
    groups={'oct':times[ids].month==10,'test_all':times[ids].month>=11,'nov':times[ids].month==11,'dec':times[ids].month==12}
    for k,kind in enumerate(KINDS):groups['test_'+kind]=(times[ids].month>=11)&(states[ids]==k)
    for name,e in errors.items():
     for scope,mask in groups.items():metrics.append({'source':source,'target':target,'signal':signal,'collection':collection,'model':name,'scope':scope,'n':int(mask.sum()),'MAE':float(np.mean(e[mask])) if mask.any() else None})
    pairs=[('source5','common')]
    if collection=='main':pairs += [('source5','source30'),('source5','source_week'),('common','own')]
    else:pairs += [('sensor_source5','common_sensor'),('common_sensor','common')]
    for candidate,baseline in pairs:
     delta=errors[candidate]-errors[baseline]
     for scope in ['test_all','test_turn']:
      use=groups[scope];weeks=((times[ids]-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();keys=np.unique(weeks[use]);sums=np.array([delta[use&(weeks==k)].sum() for k in keys]);ns=np.array([(use&(weeks==k)).sum() for k in keys])
      if len(keys)<2:continue
      draw=np.random.default_rng(20241101).integers(0,len(keys),(2000,len(keys)));boot=sums[draw].sum(1)/ns[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975])
      comparisons.append({'source':source,'target':target,'signal':signal,'collection':collection,'candidate':candidate,'baseline':baseline,'scope':scope,'n':int(use.sum()),'blocks':len(keys),'difference':float(delta[use].mean()),'low95':lo,'high95':hi,'unadjusted_exploratory':True})
    if collection=='main':
     source_model=make_pipeline(StandardScaler(),Ridge(alpha=10));source_model.fit(xx[m['train'],:23],xx[m['train'],23]);source_res=xx[ids,23]-source_model.predict(xx[ids,:23]);target_res=truth-predictions['common'];raw_source=xx[ids,23]
     joblib.dump({'model':source_model,'identity':token},OUT/f'residual_{signal}_{source}_{target}.joblib')
     for scope,mask in groups.items():
      corr=lambda a,b:float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>1e-10 and np.std(b)>1e-10 else None
      partials.append({'source':source,'target':target,'signal':signal,'scope':scope,'n':int(mask.sum()),'raw_lagged_corr':corr(raw_source[mask],truth[mask]),'controlled_residual_corr':corr(source_res[mask],target_res[mask])})
    np.savez_compressed(OUT/f'{collection}_{signal}_{source}_{target}_predictions.npz',indices=ids,truth=truth,current=current,state=states[ids],identity=np.array(token),**{'pred_'+name:p for name,p in predictions.items()},**{'error_'+name:e for name,e in errors.items()})
    print('DONE',source,target,signal,collection,{name:int(mask.sum()) for name,mask in m.items()},flush=True)
   # Primary candidate direction sign is fixed, no event-wise lag re-selection here.
   if signal=='direction' and (source,target) in PAIRS[:4]:
    for event in events:
     if event['kind']!='turn' or not event['window_eligible']:continue
     ix=event['index'];entry={'source':source,'target':target,'event_id':event['event_id'],'month':event['month'],'time':event['time']}
     for label,values in [('raw',changes(direction,True)),('smooth30',eventsignals['turn_signal'])]:
      v=values[ix-24:ix+49];ls,ss,count=lag_profile(v[:,si],v[:,ti],12,'corr',36);entry[label+'_n']=count;entry[label+'_gain5']=float(ss[13]-ss[12]) if count>=36 else None;entry[label+'_corr5']=float(ss[13]) if count>=36 else None
     sensitivity.append(entry)
  csv(metrics,'prediction_metrics.csv');csv(counts,'sample_counts.csv');csv(comparisons,'paired_comparisons.csv');csv(partials,'conditional_correlations.csv');csv(tuning,'validation_tuning.csv');csv(sensitivity,'smoothing_sensitivity.csv')
 dump({'model_files':len(list(OUT.glob('*.joblib'))),'prediction_groups':len(list(OUT.glob('*_predictions.npz'))),'identity':token},'run_complete.json');print('COMPLETE',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('step',choices=['extract','run']);args=parser.parse_args();globals()[args.step]()

"""Round14: fixed calibration design, frozen models, same October information."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from model import ART
from static_window_core import OFFSETS,select_primary,matched_positions,fit_calibration,apply_calibration,frequency_baseline
from research_direct_window import CONFIGS,SEEDS,DEADLINE
SRC=ART/'direct_window_v1';OUT=ART/'static_window_v1';PLAN=ART.parent.parent/'研究笔记/直接窗口静态校准方案_2026-11-01.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')

def prepare():
 assert datetime.now(timezone.utc)<DEADLINE,'authorized run deadline reached'
 OUT.mkdir(exist_ok=True);f=OUT/'protocol_frozen.md';protocol=PLAN.read_text(encoding='utf-8').split('## 执行记录')[0]
 if f.exists():assert f.read_text(encoding='utf-8')==protocol
 else:f.write_text(protocol,encoding='utf-8')
 if not (OUT/'previous_hashes.json').exists():js('previous_hashes.json',{str(p.relative_to(ART)):sha(p) for p in ART.rglob('*') if p.is_file() and OUT not in p.parents and p.suffix in ['.npz','.pt','.csv','.joblib']})
 files=[f,Path(__file__),Path(__file__).with_name('static_window_core.py'),Path(__file__).with_name('rolling_core.py'),ART/'data.npz',SRC/'split_data.npz',SRC/'identity.json']+sorted(SRC.glob('predictions_*budget25.npz'))+[SRC/'predictions_logistic_single.npz']+sorted(SRC.glob('predictions_scenario32_*.npz'))
 ident={str(p):sha(p) for p in files};path=OUT/'identity.json'
 if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==ident
 else:js(path.name,ident)

def fit_and_predict():
 data=np.load(ART/'data.npz');split=np.load(SRC/'split_data.npz');selected=matched_positions(split['cal_ids'],data['cal_ids']);caly=split['cal_labels'][:,OFFSETS];testy=split['test_labels'][:,OFFSETS]
 assert len(selected)==58 and np.isfinite(caly[selected]).all()
 assert np.max(split['cal_ids']+72)<=np.min(split['test_ids'])-1
 np.savez_compressed(OUT/'evaluation_data.npz',cal_ids=split['cal_ids'],selected_positions=selected,test_ids=split['test_ids'],cal_labels=caly,test_labels=testy,offsets=OFFSETS)
 fits=[];inventory=[]
 for model in [*CONFIGS,'logistic']:
  for seed in SEEDS if model in CONFIGS else [0]:
   source=SRC/f'predictions_{model}_{seed}_budget25.npz' if seed else SRC/'predictions_logistic_single.npz';raw=np.load(source);cp=select_primary(raw['cal'],raw['offsets']);tp=select_primary(raw['test'],raw['offsets'])
   for method in ['raw','shared58','shared112','byoffset58']:
    if method=='raw':pred=tp;theta=np.array([[1.,0.]]);info=[]
    else:
     ix=np.arange(len(caly)) if method=='shared112' else selected;theta,info=fit_calibration(cp[ix],caly[ix],by_offset=method=='byoffset58');pred=apply_calibration(tp,theta)
    ident=f'{model}__{method}__{seed}';np.savez_compressed(OUT/f'predictions_{ident}.npz',test=pred,theta=theta,offsets=OFFSETS)
    inventory.append(dict(id=ident,model=model,method=method,seed=seed))
    for j,r in enumerate(info):fits.append(dict(model=model,method=method,seed=seed,group=j,a=theta[j,0],b=theta[j,1],**r))
 for seed in SEEDS:
  old=np.load(SRC/f'predictions_scenario32_{seed}.npz');ident=f'scenario32__frozen__{seed}';np.savez_compressed(OUT/f'predictions_{ident}.npz',test=select_primary(old['test'],old['offsets']),offsets=OFFSETS);inventory.append(dict(id=ident,model='scenario32',method='frozen',seed=seed))
 freq=frequency_baseline(caly[selected]);ident='oct_frequency58__frequency__0';np.savez_compressed(OUT/f'predictions_{ident}.npz',test=np.broadcast_to(freq,testy.shape),offsets=OFFSETS);inventory.append(dict(id=ident,model='oct_frequency58',method='frequency',seed=0))
 csv('fits.csv',fits);js('inventory.json',inventory);print('fitted',len(fits),'calibration groups; predictions',len(inventory),flush=True)

def evaluate():
 dat=np.load(OUT/'evaluation_data.npz');y=dat['test_labels'];raw=np.load(ART/'data.npz');times=pd.DatetimeIndex(pd.to_datetime(raw['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai')[dat['test_ids']];rows=[];reliability=[];high=[]
 for item in json.loads((OUT/'inventory.json').read_text(encoding='utf-8')):
  p=np.load(OUT/f'predictions_{item["id"]}.npz')['test'];assert p.shape==y.shape and np.isfinite(p).all() and np.all((p>=0)&(p<=1));err=(p-y)**2;np.savez_compressed(OUT/f'errors_{item["id"]}.npz',errors=err)
  for month in [0,11,12]:
   mask=np.ones(len(y),bool) if month==0 else times.month==month
   for metric,v in [('brier_primary',err)]+[(f'brier_offset{a}',err[:,j]) for j,a in enumerate(OFFSETS)]:rows.append(dict(**item,month=month,metric=metric,value=float(np.nanmean(v[mask])),n=int(np.isfinite(v[mask]).sum())))
   pp=p[mask].ravel();yy=y[mask].ravel();valid=np.isfinite(yy);pp=pp[valid];yy=yy[valid]
   for b in range(10):
    mm=(pp>=b/10)&(pp<(b+1)/10 if b<9 else pp<=1);reliability.append(dict(**item,month=month,bin=b,n=int(mm.sum()),mean_probability=float(pp[mm].mean()) if mm.any() else np.nan,observed_rate=float(yy[mm].mean()) if mm.any() else np.nan))
   mm=pp>=.8;high.append(dict(**item,month=month,n=int(mm.sum()),fraction=float(mm.mean()),mean_probability=float(pp[mm].mean()) if mm.any() else np.nan,observed_rate=float(yy[mm].mean()) if mm.any() else np.nan))
 csv('metrics.csv',rows);csv('reliability.csv',reliability);csv('high_confidence.csv',high)
 return times

def compare(times):
 def loss(model,method):
  seeds=SEEDS if model in CONFIGS or model=='scenario32' else [0]
  return np.mean([np.load(OUT/f'errors_{model}__{method}__{s}.npz')['errors'] for s in seeds],axis=0)
 pairs=[]
 for n in [*CONFIGS,'logistic']:
  for method in ['shared58','shared112','byoffset58']:pairs.append((n,method,n,'raw',False))
  pairs.append((n,'shared58','scenario32','frozen',n=='gated_graph_bce'))
 pairs.extend([('gated_graph_bce','shared58','gated_temporal_bce','shared58',True),('gated_graph_bce','byoffset58','gated_graph_bce','shared58',False),('gated_graph_bce','shared112','gated_graph_bce','shared58',False),('oct_frequency58','frequency','scenario32','frozen',False)])
 week=((times-pd.Timestamp('2024-11-01',tz=times.tz)).days//7).to_numpy();keys=np.unique(week);draw=np.random.default_rng(20261104).integers(0,len(keys),(2000,len(keys)));rows=[]
 for m,c,b,d,primary in pairs:
  delta=loss(m,c)-loss(b,d);ss=np.array([np.nansum(delta[week==k]) for k in keys]);nn=np.array([np.isfinite(delta[week==k]).sum() for k in keys]);boot=ss[draw].sum(1)/nn[draw].sum(1);lo,hi=np.quantile(boot,[.025,.975]);rows.append(dict(candidate=m,method=c,baseline=b,baseline_method=d,difference=float(np.nanmean(delta)),low95=lo,high95=hi,blocks=len(keys),primary=primary))
 csv('paired_comparisons.csv',rows)

if __name__=='__main__':
 prepare();fit_and_predict();compare(evaluate());print('COMPLETE',flush=True)

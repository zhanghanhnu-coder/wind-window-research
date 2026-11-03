from functools import lru_cache
import json
import numpy as np
import pandas as pd
import torch
from model import Net,ART
from window_analysis import summarize_windows

class Forecaster:
 def __init__(self):
  self.meta=json.loads((ART/'metadata.json').read_text(encoding='utf-8'));d=np.load(ART/'data.npz')
  self.x=d['x'];self.raw=d['raw'];self.mean=d['mean'];self.adj=d['adj'];self.times=pd.DatetimeIndex(pd.to_datetime(d['timestamps'].tolist(),utc=True)).tz_convert('Asia/Shanghai');self.nodes=self.meta['nodes'];self.models={};self.banks={}
  for name in ['graph','temporal','persistence']:
   if name!='persistence':
    ck=torch.load(ART/f'{name}.pt',map_location='cpu',weights_only=False);net=Net(ck['adj']);net.load_state_dict(ck['state']);net.eval();self.models[name]=net
   bank=np.load(ART/f'{name}_residuals.npz');self.banks[name]=(bank['residual'],bank['origins'])
  for name in ['temporal24','graph24','mean24','random24']:
   if not (ART/'v2'/f'{name}.pt').exists():continue
   ck=torch.load(ART/'v2'/f'{name}.pt',map_location='cpu',weights_only=False);net=Net(ck['adj'],dilation=ck['dilation']);net.load_state_dict(ck['state']);net.eval();self.models[name]=net
   bank=np.load(ART/'v2'/f'{name}_residuals.npz');self.banks[name]=(bank['residual'],bank['origins'])
 def locate(self,issue):
  t=pd.Timestamp(issue)
  if t.tzinfo is None:t=t.tz_localize('Asia/Shanghai')
  t=t.tz_convert('Asia/Shanghai')
  if t<pd.Timestamp('2024-11-01',tz=t.tz) or t>pd.Timestamp('2024-12-28 23:00',tz=t.tz):raise ValueError('演示起点限于 2024-11-01 至 2024-12-28 23:00')
  if t not in self.times:raise ValueError('请选择整点时间')
  return int(self.times.get_loc(t))
 @lru_cache(maxsize=256)
 def scenarios(self,issue,model='temporal'):
  if model not in self.banks:raise ValueError('未知预测模型')
  i=self.locate(issue);past=self.raw[i-23:i+1];mask=self.x[i-23:i+1,:,-1]
  if mask.mean()<.65:raise ValueError('过去 24 小时可用数据不足 65%，本次停止自动排程')
  if model=='persistence':
   last=np.array([v[np.isfinite(v)][-1] if np.isfinite(v).any() else self.meta['center'] for v in past.T]);median=np.broadcast_to(last,(72,len(self.nodes))).copy()
  else:
   with torch.no_grad():median=self.models[model](torch.from_numpy(self.x[i-23:i+1][None]))[0,...,1].numpy()*self.meta['scale']+self.meta['center']
  bank,origins=self.banks[model]
  if any(pd.Timestamp(str(t))+pd.Timedelta(hours=72)>self.times[i] for t in origins):raise RuntimeError('残差库含有预测时点之后的信息')
  selected=np.random.default_rng(20241101).choice(len(bank),32,replace=len(bank)<32)
  return np.maximum(0,median[None]+bank[selected]),median,selected
 def quality(self,issue):
  i=self.locate(issue);mask=self.x[i-23:i+1,:,-1]>0;out=[]
  for j,node in enumerate(self.nodes):
   valid=int(mask[:,j].sum());where=np.flatnonzero(mask[:,j]);age=int(23-where[-1]) if len(where) else None
   out.append({'node':node,'valid_hours':valid,'total_hours':24,'latest_valid_age_hours':age,'eligible':bool(valid>=16 and age is not None and age<=3),'rule':'至少16/24有效小时，最近有效记录距起点不超过3小时；工程门控，非安全保证'})
  return out
 def payload(self,issue,model='temporal',threshold=10,duration=6):
  if not np.isfinite(threshold) or not 1<=threshold<=60 or int(duration)!=duration or not 1<=duration<=24:raise ValueError('阈值或作业长度不合法')
  duration=int(duration);sc,median,sel=self.scenarios(issue,model);i=self.locate(issue);qs=np.quantile(sc,[.1,.5,.9],axis=0)
  offsets=list(range(0,73-duration,3));prob=np.stack([(sc[:,s:s+duration]<=threshold).all(1).mean(0) for s in offsets],axis=1)
  truth=self.raw[i+1:i+73];history=self.raw[i-23:i+1]
  def clean(a):return np.where(np.isfinite(a),a,None).tolist()
  adjacency=self.models[model].adj.numpy() if model in self.models else np.eye(len(self.nodes))
  return {'version':'2.0','issue':str(self.times[i]),'model':model,'nodes':self.nodes,'offsets':offsets,'duration':duration,'threshold':threshold,'q10':clean(qs[0]),'q50':clean(qs[1]),'q90':clean(qs[2]),'point_median':clean(median),'probabilities':prob.tolist(),'history':clean(history),'observed_after_issue':clean(truth),'input_valid_fraction':float(self.x[i-23:i+1,:,-1].mean()),'scenario_origins':self.banks[model][1][sel].tolist(),'adjacency':adjacency.tolist(),'quality':self.quality(issue),'window_analysis':summarize_windows(sc,self.nodes,threshold,duration),'warning':'事后观测仅供对照，不进入预测或当前调度；情景比例不是安全保证。'}

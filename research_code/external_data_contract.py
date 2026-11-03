"""As-of forecast feature preparation only; never trains or changes frozen models."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import argparse,csv,hashlib,json,math

FIELDS=['source','run_id','product_type','forecast_issued_at','available_at','valid_at','node','variable','value','unit','statistic','interval_hours']
UNITS={'wind_speed':'m/s','significant_wave_height':'m','wave_period':'s','wind_direction':'degree'}
def instant(value):
 if not isinstance(value,str):raise ValueError('时间须为带时区的ISO8601文本')
 try:d=datetime.fromisoformat(value.replace('Z','+00:00'))
 except (ValueError,TypeError) as e:raise ValueError('时间格式不合法') from e
 if d.tzinfo is None or d.utcoffset() is None:raise ValueError('时间必须明确时区，不能猜本地时间')
 return d.astimezone(timezone.utc)
def finite(value):
 if isinstance(value,bool):raise ValueError('布尔值不是测量数值')
 try:x=float(value)
 except (ValueError,TypeError) as e:raise ValueError('数值格式不合法') from e
 if not math.isfinite(x):raise ValueError('未知值请留空，不能用NaN或无穷数')
 return x
def definition(variable,record):
 if variable not in UNITS or not isinstance(record,dict) or set(record)!={'unit','statistic','interval_hours'}:raise ValueError('未知变量或定义字段不完整')
 unit=record['unit'];stat=record['statistic'];hours=finite(record['interval_hours'])
 if unit!=UNITS[variable] or stat not in ['instant','mean','max'] or (hours!=0 if stat=='instant' else hours<=0):raise ValueError('单位、统计类型或统计区间不合法')
 return dict(unit=unit,statistic=stat,interval_hours=hours)
def build_feature_block(records,*,source,issue_at,nodes,definitions,horizon=72):
 issue=instant(issue_at)
 if issue.minute or issue.second or issue.microsecond:raise ValueError('当前接口要求整点起报')
 if not isinstance(source,str) or not source.strip():raise ValueError('必须指定来源')
 if not isinstance(nodes,list) or not nodes or any(not isinstance(n,str) or not n.strip() for n in nodes) or len(set(nodes))!=len(nodes):raise ValueError('机组清单须非空且不重复')
 if isinstance(horizon,bool) or not isinstance(horizon,int) or not 1<=horizon<=72:raise ValueError('预测长度须为1至72整数小时')
 if not isinstance(definitions,dict) or not definitions:raise ValueError('须指定每个变量的定义')
 specs={v:definition(v,d) for v,d in definitions.items()};variables=sorted(specs);eligible=[]
 for record in records:
  if not isinstance(record,dict) or set(record)!=set(FIELDS):raise ValueError('预报字段不完整或包含未知字段')
  if record['source']!=source:continue
  issued=instant(record['forecast_issued_at']);available=instant(record['available_at'])
  if issued>available:raise ValueError('可用时刻不能早于预报发行时刻')
  # No value, statistic or valid-time inspection for unavailable forecasts.
  if issued>issue or available>issue:continue
  if not isinstance(record['run_id'],str) or not record['run_id'].strip():raise ValueError('预报批次编号不能为空')
  eligible.append((issued,record))
 if not eligible:raise ValueError('起报当时没有已发布且可用的指定来源预报')
 latest=max(d for d,_ in eligible);selected=[r for d,r in eligible if d==latest];runs={r['run_id'] for r in selected}
 if len(runs)!=1:raise ValueError('同一发行时刻有多个批次，必须先明确来源/批次定义')
 run=next(iter(runs));valid_times=[issue+timedelta(hours=k) for k in range(1,horizon+1)];time_index={d:i for i,d in enumerate(valid_times)};node_index={n:i for i,n in enumerate(nodes)};variable_index={v:i for i,v in enumerate(variables)};values=[[[None for _ in variables] for _ in nodes] for _ in valid_times];availability=[[[None for _ in variables] for _ in nodes] for _ in valid_times];seen=set()
 for r in selected:
  if r['product_type']!='forecast':raise ValueError('这里只接受历史发布预报，不能把再分析或观测当预报')
  valid=instant(r['valid_at'])
  if valid<latest:raise ValueError('预报有效时刻不能早于发行时刻')
  n=r['node'];v=r['variable']
  if n not in node_index or v not in variable_index:raise ValueError('选中批次含未约定的机组或变量')
  got=definition(v,{k:r[k] for k in ['unit','statistic','interval_hours']})
  if got!=specs[v]:raise ValueError('预报变量定义与请求不一致；不能混用瞬时、平均和最大值')
  key=(n,v,valid)
  if key in seen:raise ValueError('同一节点变量有效时刻重复，需提供者先核对')
  seen.add(key);x=None if r['value'] is None or r['value']=='' else finite(r['value'])
  if x is not None and (x<0 or (v=='wave_period' and x<=0) or (v=='wind_direction' and x>=360)):raise ValueError('预报数值超出变量定义范围')
  if valid in time_index:
   values[time_index[valid]][node_index[n]][variable_index[v]]=x;availability[time_index[valid]][node_index[n]][variable_index[v]]=instant(r['available_at']).isoformat()
 known=[[[v is not None for v in node] for node in lead] for lead in values];count=sum(x for lead in known for node in lead for x in node)
 result=dict(kind='external_forecast_features_only',source=source,run_id=run,forecast_issued_at=latest.isoformat(),issue_at=issue.isoformat(),valid_at=[d.isoformat() for d in valid_times],nodes=nodes.copy(),variables=variables,definitions=specs,values=values,available_at=availability,known=known,known_fraction=count/(horizon*len(nodes)*len(variables)),scope='Only as-of available forecast features; missing cells remain unknown. No model update, window probability or safety decision is produced.')
 result['snapshot_id']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest();return result
def read_forecast_csv(path):
 with Path(path).open('r',encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f)
  if reader.fieldnames is None or len(set(reader.fieldnames))!=len(reader.fieldnames) or set(reader.fieldnames)!=set(FIELDS):raise ValueError('CSV表头须与模板一致，不得重复')
  return list(reader)
def write_new_json(path,value):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 parser=argparse.ArgumentParser(description='仅准备起报当时可用的外部天气特征，不更新模型');parser.add_argument('--input',required=True);parser.add_argument('--request',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();request=json.loads(Path(args.request).read_text(encoding='utf-8-sig'));block=build_feature_block(read_forecast_csv(args.input),**request);write_new_json(args.output,block);print(json.dumps(dict(status='PASS',kind=block['kind'],run_id=block['run_id'],known_fraction=block['known_fraction'],snapshot_id=block['snapshot_id']),ensure_ascii=False))
if __name__=='__main__':main()

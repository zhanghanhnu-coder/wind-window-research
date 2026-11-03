"""Synthetic-only independent as-of reference and CSV/CLI verification."""
from pathlib import Path
import csv,hashlib,json,os,subprocess,sys
from datetime import datetime,timedelta,timezone
import pandas as pd
from model import ART
from external_data_contract import FIELDS,build_feature_block
OUT=ART/'external_contract_v1';BASE=ART.parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def run():
 assert not OUT.exists(),'never overwrite prior checks';OUT.mkdir();nodes=['01','04','19'];defs={'wind_speed':dict(unit='m/s',statistic='instant',interval_hours=0),'significant_wave_height':dict(unit='m',statistic='instant',interval_hours=0)};zero=datetime(2024,5,1,tzinfo=timezone.utc);rows=[]
 for run_hour in [0,6,12]:
  issued=zero+timedelta(hours=run_hour)
  for lead in range(1,25):
   for ni,node in enumerate(nodes):
    for vi,var in enumerate(defs):
     if (lead+ni+vi)%7==0:continue
     value=(run_hour+lead+ni)/10+vi
     rows.append(dict(source='synthetic_only',run_id=f'synthetic_{run_hour}',product_type='forecast',forecast_issued_at=issued.isoformat(),available_at=(issued+timedelta(hours=2)).isoformat(),valid_at=(issued+timedelta(hours=lead)).isoformat(),node=node,variable=var,value=str(value),unit=defs[var]['unit'],statistic='instant',interval_hours='0'))
 csvpath=OUT/'synthetic_only_forecasts.csv'
 with csvpath.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
 sources=[Path(__file__),ART.parent/'external_data_contract.py',ART.parent/'tests/test_external_data_contract.py',BASE/'研究笔记/历史发布天气数据接入与时间隔离方案_2026-11-02.md'];identity={str(p.resolve()):sha(p) for p in sources};js(OUT/'identity.json',identity)
 df=pd.DataFrame(rows);df['issued']=pd.to_datetime(df.forecast_issued_at,utc=True);df['available']=pd.to_datetime(df.available_at,utc=True);df['valid']=pd.to_datetime(df.valid_at,utc=True);checks=[];poisoned_values=0;cell_checks=0
 for hour in range(2,25,2):
  issue=zero+timedelta(hours=hour);req=dict(source='synthetic_only',issue_at=issue.isoformat(),nodes=nodes,definitions=defs,horizon=12);got=build_feature_block(rows,**req);eligible=df[(df.issued<=issue)&(df.available<=issue)];selected=eligible[eligible.issued==eligible.issued.max()];assert got['run_id']==selected.run_id.iloc[0];lookup={(r.node,r.variable,r.valid):r for r in selected.itertuples()}
  for k in range(12):
   for n,node in enumerate(nodes):
    for v,var in enumerate(sorted(defs)):
     ref=lookup.get((node,var,issue+timedelta(hours=k+1)));expected=float(ref.value) if ref is not None else None;assert got['values'][k][n][v]==expected;assert got['known'][k][n][v]==(ref is not None);assert got['available_at'][k][n][v]==(ref.available.isoformat() if ref is not None else None);cell_checks+=1
  poison=[]
  for r in rows:
   q=r.copy()
   if pd.Timestamp(r['forecast_issued_at'])>issue or pd.Timestamp(r['available_at'])>issue:q.update(value='future_value_must_not_be_read',valid_at='future_valid_time_must_not_be_read');poisoned_values+=1
   poison.append(q)
  assert build_feature_block(poison,**req)==got;assert build_feature_block(list(reversed(rows)),**req)==got;js(OUT/f'synthetic_block_{hour:02d}.json',got);checks.append(dict(issue_at=issue.isoformat(),run_id=got['run_id'],known_fraction=got['known_fraction'],snapshot_id=got['snapshot_id']))
 req=dict(source='synthetic_only',issue_at=(zero+timedelta(hours=8)).isoformat(),nodes=nodes,definitions=defs,horizon=12);js(OUT/'synthetic_only_request.json',req);target=OUT/'synthetic_only_cli_result.json';env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';cmd=[sys.executable,str(ART.parent/'external_data_contract.py'),'--input',str(csvpath),'--request',str(OUT/'synthetic_only_request.json'),'--output',str(target)];proc=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',env=env,timeout=30);(OUT/'cli_stdout.log').write_text(proc.stdout,encoding='utf-8');(OUT/'cli_stderr.log').write_text(proc.stderr,encoding='utf-8');assert proc.returncode==0,proc.stderr;assert json.loads(target.read_text(encoding='utf-8'))==build_feature_block(rows,**req);before=sha(target);second=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',env=env,timeout=30);assert second.returncode!=0 and 'FileExistsError' in second.stderr and sha(target)==before;(OUT/'cli_overwrite_rejected.log').write_text(second.stderr,encoding='utf-8')
 for p,h in identity.items():assert sha(p)==h,p
 proof=dict(status='PASS',synthetic_only=True,actual_external_weather_records=0,input_rows=len(rows),asof_requests=len(checks),independent_cell_checks=cell_checks,future_poison_tests=len(checks),future_values_poisoned=poisoned_values,row_order_invariance_tests=len(checks),csv_cli_match=True,output_overwrite_rejected=True,source_files_unchanged=len(identity),checks=checks,scope='Preparation interface only: no real NWP/waves, no model fitting, no accuracy/safety claims.');js(OUT/'verification.json',proof);print(json.dumps(proof,ensure_ascii=False,indent=2))
if __name__=='__main__':run()

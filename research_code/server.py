"""Offline localhost research demo. Start: python server.py"""
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlsplit
from pathlib import Path
import json,threading,argparse,traceback
import numpy as np
import pandas as pd
from forecast import Forecaster
from scheduler import plan
from replay import replay
from planning import make_plan,compare_plans
from model import ROOT,ART

F=None;LOCK=threading.Lock()

def rows(name):
 p=ART/name
 return json.loads(pd.read_csv(p).to_json(orient='records')) if p.exists() else []

class Handler(BaseHTTPRequestHandler):
 def send(self,status,body,kind='application/json; charset=utf-8'):
  if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False,allow_nan=False).encode('utf-8')
  self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
 def do_GET(self):
  path=urlsplit(self.path).path
  if path=='/api/meta':
   self.send(200,{'version':'2.0','length_metrics':rows('v2/window_length_metrics.csv'),'available_models':list(F.banks),'ablation_metrics':rows('v2/ablation_metrics.csv'),'paired_comparisons':rows('v2/paired_comparisons.csv'),'replay_v2':rows('v2/replay_summary.csv'),'context_comparison':rows('v2/context_comparison.csv'),'metadata':F.meta,'forecast_metrics':rows('forecast_metrics.csv'),'window_metrics':rows('window_metrics.csv'),'interval_metrics':rows('interval_metrics.csv'),'replay_metrics':rows('replay_summary.csv'),'sensitivity':rows('sensitivity.csv'),'graph_audit':rows('graph_seed_audit.csv'),'nodes':F.nodes,'default_issue':'2024-11-02T00:00','scope':['仅有2024年数据，所有结论属于同一年内的探索性检验。','17台同型号机组；小时数据必须有12条有效的5分钟记录。','图边来自训练期的统计关系；圆形布局不是地理地图。','缺少海浪、真实工单、船舶能力和坐标；路途与维修任务为模拟。','72小时情景来自10月58段历史误差；有重叠，不承诺概率覆盖或现场安全。','风速阈值沿用原始数值刻度；实际单位和传感器定义仍需厂方确认。']});return
  assets={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/research.js':('research.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8')}
  if path not in assets:self.send(404,{'error':'页面不存在'});return
  file,kind=assets[path];self.send(200,(ROOT/'static'/file).read_bytes(),kind)
 def do_POST(self):
  try:
   host=self.headers.get('Host','');origin=self.headers.get('Origin')
   if origin and origin not in [f'http://{host}']:raise ValueError('仅接受本地页面的请求')
   length=int(self.headers.get('Content-Length','0'))
   if not 0<length<=100000:raise ValueError('请求内容为空或过大')
   d=json.loads(self.rfile.read(length));path=urlsplit(self.path).path
   if not isinstance(d,dict):raise ValueError('请求必须是对象')
   with LOCK:
    if path=='/api/forecast':r=F.payload(d['issue'],d.get('model','graph'),d.get('threshold',10),d.get('duration',6))
    elif path in ['/api/plan','/api/compare']:
     fn=make_plan if path=='/api/plan' else compare_plans
     r=fn(F,d['issue'],d['tasks'],**{k:d[k] for k in ['model','crews','travel','policy','threshold','minimum_probability','risk_weight'] if k in d})
    elif path=='/api/replay':r=replay(F,**{k:d[k] for k in ['start','days','policy','model','crews','travel','threshold','minimum_probability','risk_weight','arrivals_per_day'] if k in d})
    else:self.send(404,{'error':'接口不存在'});return
   self.send(200,r)
  except (ValueError,KeyError,TypeError,OverflowError) as e:self.send(400,{'error':str(e)})
  except Exception:
   traceback.print_exc();self.send(500,{'error':'计算失败，详细记录见服务器日志'})
 def log_message(self,format,*args):print(self.log_date_time_string(),format%args,flush=True)

def main():
 global F
 parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);a=parser.parse_args();F=Forecaster()
 class LocalServer(ThreadingHTTPServer):allow_reuse_address=False
 server=LocalServer(('127.0.0.1',a.port),Handler);print(f'Wind research demo: http://127.0.0.1:{a.port}',flush=True)
 server.serve_forever()
if __name__=='__main__':main()

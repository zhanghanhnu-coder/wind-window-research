"""Serve only bundled public assets, bound to localhost."""
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
from pathlib import Path
import json,sys
from core import snapshot
ROOT=Path(getattr(sys,'_MEIPASS',Path(__file__).parent))/'static'
class Handler(SimpleHTTPRequestHandler):
 extensions_map={**SimpleHTTPRequestHandler.extensions_map,'.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.svg':'image/svg+xml'}
 def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT),**kw)
 def do_GET(self):
  u=urlparse(self.path)
  if u.path=='/api/scenario':
   try:
    q=parse_qs(u.query);w=float(q.get('wind',['8'])[0]);c=int(q.get('crews',['2'])[0])
    if not 0<=w<=25 or not 0<=c<=4:raise ValueError()
    payload=json.dumps(snapshot(w,crews=c),ensure_ascii=False).encode();self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.end_headers();self.wfile.write(payload)
   except (ValueError,TypeError):self.send_error(400,'Invalid synthetic inputs')
  else:super().do_GET()
if __name__=='__main__':
 print('Synthetic xx wind farm demo: http://127.0.0.1:8788 (Ctrl+C to stop)')
 ThreadingHTTPServer(('127.0.0.1',8788),Handler).serve_forever()

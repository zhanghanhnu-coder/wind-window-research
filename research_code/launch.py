"""Start the localhost app once and open its UI. No external service."""
from pathlib import Path
import sys,json,time,subprocess,webbrowser
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parent;URL='http://127.0.0.1:8765'

def ready():
 try:
  with urlopen(URL+'/api/meta',timeout=1) as response:d=json.load(response)
  expected=json.loads((ROOT/'artifacts/metadata.json').read_text(encoding='utf-8'))
  if d.get('metadata',{}).get('data_sha256')!=expected['data_sha256']:raise RuntimeError('8765端口不是本项目，请关闭冲突程序后重试。')
  return True
 except (ConnectionError,TimeoutError,OSError):return False

def main():
 if not ready():
  out=(ROOT/'artifacts/launcher_server.log').open('ab')
  flags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
  process=subprocess.Popen([sys.executable,'-u',str(ROOT/'server.py')],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,creationflags=flags)
  (ROOT/'artifacts/launcher_server.pid').write_text(str(process.pid),encoding='ascii')
  for _ in range(60):
   if ready():break
   if process.poll() is not None:raise RuntimeError('服务未启动，请查看 artifacts/launcher_server.log')
   time.sleep(.5)
  else:raise RuntimeError('启动超时，请查看 artifacts/launcher_server.log')
 webbrowser.open(URL);print('Demo is ready: '+URL)

if __name__=='__main__':
 try:main()
 except Exception as e:print(str(e));sys.exit(1)

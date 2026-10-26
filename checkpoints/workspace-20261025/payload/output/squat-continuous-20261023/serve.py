from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import json
O=Path(__file__).resolve().parent
VIDEOS={r['id']:r['path'] for r in json.loads((O/'full-frame-manifest.json').read_text())}
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(O),**kwargs)
 def translate_path(self,path):
  if path.startswith('/video/'):
   ident=path.split('?',1)[0].removeprefix('/video/');return VIDEOS.get(ident,str(O/'not-found'))
  return super().translate_path(path)
ThreadingHTTPServer(('127.0.0.1',8795),Handler).serve_forever()

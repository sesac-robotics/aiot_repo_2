"""Local-only browser controls and annotated live frames for ESP32-CAM."""
from pathlib import Path
import argparse,json,threading,webbrowser
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from .engine import LiveEngine,EXERCISES,DEFAULT_SOURCE
WEB=Path(__file__).resolve().parent/'web'

class LiveHTTPServer(ThreadingHTTPServer):
    daemon_threads=True
    def server_close(self):
        try:self.engine.stop()
        finally:super().server_close()

def make_server(port=8780,default_source=DEFAULT_SOURCE,default_exercise='squats'):
    engine=LiveEngine()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,code,data,content_type='application/json; charset=utf-8'):
            payload=json.dumps(data,ensure_ascii=False,allow_nan=False).encode() if not isinstance(data,bytes) else data
            self.send_response(code);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
            try:self.wfile.write(payload)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/api/state':self.reply(200,engine.snapshot())
            elif path=='/api/config':self.reply(200,dict(default_source=default_source,default_exercise=default_exercise,exercises=EXERCISES))
            elif path=='/api/frame.jpg':self.reply(200,engine.image(),'image/jpeg')
            elif path in ['/','/app.js','/styles.css']:
                file=WEB/('index.html' if path=='/' else path[1:]);mime={'html':'text/html; charset=utf-8','js':'text/javascript; charset=utf-8','css':'text/css; charset=utf-8'}[file.suffix[1:]];self.reply(200,file.read_bytes(),mime)
            else:self.reply(404,dict(error='Not found'))
        def do_POST(self):
            origin=self.headers.get('Origin');host=self.headers.get('Host','')
            if origin and origin!=f'http://{host}':self.reply(403,dict(error='이 화면에서만 카메라를 제어할 수 있습니다.'));return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<=length<=8192:raise ValueError('요청 크기가 올바르지 않습니다.')
                body=json.loads(self.rfile.read(length) or b'{}')
                if not isinstance(body,dict):raise ValueError('JSON 객체가 필요합니다.')
                if self.path=='/api/start':
                    record=body.get('record',False)
                    if not isinstance(record,bool):raise ValueError('로그 저장 옵션이 올바르지 않습니다.')
                    engine.start(body.get('source',default_source),body.get('exercise',default_exercise),record)
                elif self.path=='/api/stop':engine.stop()
                elif self.path=='/api/reset':engine.reset()
                else:self.reply(404,dict(error='Not found'));return
                self.reply(200,engine.snapshot())
            except (ValueError,TypeError,RuntimeError) as e:self.reply(400,dict(error=str(e)))
    server=LiveHTTPServer(('127.0.0.1',port),Handler);server.engine=engine
    return server

def main():
    p=argparse.ArgumentParser(description='ESP32-CAM 실시간 자세 오류 테스트');p.add_argument('--port',type=int,default=8780);p.add_argument('--source',default=DEFAULT_SOURCE);p.add_argument('--exercise',choices=list(EXERCISES),default='squats');p.add_argument('--no-browser',action='store_true');a=p.parse_args()
    with make_server(a.port,a.source,a.exercise) as server:
        url=f'http://127.0.0.1:{server.server_port}/';print('\nESP32-CAM 실시간 자세 오류 테스트',flush=True);print('화면 주소: '+url,flush=True);print('브라우저에서 카메라 주소와 종목을 확인하고 [실시간 시작]을 누르세요.',flush=True);print('종료: 브라우저 [중지], 서버 종료: 터미널 Ctrl+C',flush=True)
        if not a.no_browser:threading.Thread(target=webbrowser.open,args=(url,),daemon=True).start()
        try:server.serve_forever()
        except KeyboardInterrupt:pass

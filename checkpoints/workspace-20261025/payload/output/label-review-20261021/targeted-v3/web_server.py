"""기존 자세 점수 웹에 실험 모델 항목 분석 화면을 새 파일로 연결한다.

기존 소스는 수정하지 않는다. 127.0.0.1:8766에서만 제공하며 원본은 보존한다.
학습 모델은 통과한 항목만 예측하고 전체 100점 점수를 생성하지 않는다.
"""
from http.server import ThreadingHTTPServer
from pathlib import Path
import json
import hashlib
import math
import re
import subprocess
import sys
import threading
import uuid
from urllib.parse import urlparse,parse_qs
ROOT=Path(__file__).resolve().parent
BASE=Path('/home/jayden/SeSACPAI-Mid')
sys.path.insert(0,str(BASE/'posture-score-pilot'))
import server as original
JOBS=ROOT/'web_jobs'


def catalog():
    """기존 저장 영상과 직접 검토한 검증 예제의 허용 목록을 반환한다.

    Returns:
        dict: 원본 경로·운동·기본 구간을 담은 내부 목록. 원본 경로는 API에서 숨긴다.
    Raises:
        OSError: 저장 자료를 읽을 수 없는 경우.
    """
    import cv2
    values={}
    for path in sorted((BASE/'posture-score-pilot/jobs').glob('*/result.json')):
        source=path.parent/'input.video'
        if not source.is_file():continue
        r=json.loads(path.read_text());ex={'pushup':'pushups','squat':'squats'}.get(r.get('exercise'))
        if ex is None:continue
        cap=cv2.VideoCapture(str(source));fps=cap.get(5);length=cap.get(7)/fps if fps>0 else 0;cap.release()
        if length<1.2:continue
        ident='saved_'+path.parent.name;values[ident]=dict(id=ident,title='저장 영상 · '+('푸시업' if ex=='pushups' else '스쿼트')+' · '+path.parent.name[:6],exercise=ex,source=str(source),duration=length,start=0,end=min(5,length),source_job=path.parent.name)
    rows=json.loads((ROOT/'labels.json').read_text());valid={r['id'] for r in json.loads((ROOT/'feature_report.json').read_text()) if r['valid']}
    for r in rows:
        if r['split']!='validation' or r['id'] not in valid:continue
        ident='example_'+r['id'];seg=r['segment'];values[ident]=dict(id=ident,title='검증 예제 · '+('푸시업' if r['exercise']=='pushups' else '스쿼트')+' · '+r['id'],exercise=r['exercise'],source=r['video_path'],duration=seg['end_frame']/seg['fps']+1/seg['fps'],start=seg['start_frame']/seg['fps'],end=seg['end_frame']/seg['fps'],source_job=r['id'])
    return values


def run_model(server,ident):
    """분석 프로세스를 한 개 실행하고 상태를 저장한다.

    Args:
        server: 상태 저장소가 있는 로컬 서버.
        ident: 서버가 생성한 작업 UUID.
    Returns:
        None: 완료·실패 상태와 작업 파일을 저장한다.
    """
    folder=JOBS/ident
    try:
        with (folder/'worker.log').open('w') as log:
            process=subprocess.run([str(BASE/'mediapipe-pilot/run_nvidia.sh'),str(ROOT/'worker.py'),str(folder/'request.json')],stdout=log,stderr=subprocess.STDOUT,timeout=120,env=__import__('os').environ|{'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        if process.returncode:raise RuntimeError('모델 분석을 완료하지 못했습니다. 관절 관측·선택 구간 또는 온도를 확인하세요.')
        state=dict(state='done',result=json.loads((folder/'result.json').read_text()))
    except Exception as error:state=dict(state='failed',error=str(error))
    finally:original.LOCK.release()
    (folder/'status.json').write_text(json.dumps(state,ensure_ascii=False));server.model_states[ident]=state


class ModelHandler(original.Handler):
    """기존 HTTP 기능에 실험 모델 화면과 제한된 로컬 분석 API를 추가한다."""

    def do_GET(self):
        """화면·허용 목록·작업 상태·관절 영상만 제공한다. 알 수 없는 경로는 404다."""
        parsed=urlparse(self.path);path=parsed.path
        if path in ['/','/index.html']:
            page=(original.ROOT/'index.html').read_text();addition='<section><b>학습 모델 항목 분석 · 실험</b><p>항목별 모델의 검증 상태와 관절 표시 영상을 확인할 수 있습니다. 검증 기준 미달 항목과 종합 100점 평가는 보류합니다.</p><a href="/model/">학습 모델 분석 화면 열기 →</a><p><small>아래 기존 분석은 규칙 기반입니다. 두 결과를 같은 점수로 해석하지 마세요.</small></p></section>'
            return self.send(page.replace('</header>','</header>'+addition,1).encode(),'text/html; charset=utf-8')
        if path in ['/model/','/model/app.js']:
            name='index.html' if path=='/model/' else 'app.js';kind='text/html; charset=utf-8' if name.endswith('html') else 'text/javascript; charset=utf-8';return self.send((ROOT/'web'/name).read_bytes(),kind)
        if path=='/model/metrics':
            config=json.loads((ROOT/'active_model.json').read_text()) if (ROOT/'active_model.json').exists() else dict(report='training_report.json')
            return self.send((ROOT/config['report']).read_bytes())
        if path=='/model/catalog':
            self.server.model_catalog=catalog()
            return self.send(json.dumps([{k:v for k,v in row.items() if k!='source'} for row in self.server.model_catalog.values()],ensure_ascii=False).encode())
        if path=='/model/status':
            ident=parse_qs(parsed.query).get('id',[''])[0]
            if not re.fullmatch('[a-f0-9]{32}',ident):return self.send(b'{}',status=400)
            state=self.server.model_states.get(ident);p=JOBS/ident/'status.json'
            if state is None and p.exists():state=json.loads(p.read_text())
            if state and state.get('state')=='done':
                config=json.loads((ROOT/'active_model.json').read_text()) if (ROOT/'active_model.json').exists() else dict(file='item_models.joblib')
                current_hash=hashlib.sha256((ROOT/config['file']).read_bytes()).hexdigest()
                if state['result'].get('model_sha256')!=current_hash:state=dict(state='stale',error='이전 모델의 결과입니다. 현재 모델로 다시 분석하세요.')
            return self.send(json.dumps(state or dict(state='unknown'),ensure_ascii=False).encode(),status=200 if state else 404)
        match=re.fullmatch('/model/jobs/([a-f0-9]{32})/(overlay.mp4|result.json)',path)
        if match:
            file=JOBS/match[1]/match[2]
            if not file.is_file():return self.send(b'{}',status=404)
            if match[2]=='result.json':return self.send(file.read_bytes())
            size=file.stat().st_size;start=0;end=size-1;requested=self.headers.get('Range')
            if requested:
                m=re.fullmatch(r'bytes=(\d+)-(\d*)',requested)
                if not m:return self.send(b'',status=416)
                start=int(m[1]);end=min(end,int(m[2]) if m[2] else end)
            if not 0<=start<=end<size:return self.send(b'',status=416)
            self.send_response(206 if requested else 200);self.send_header('Content-Type','video/mp4');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
            if requested:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            with file.open('rb') as f:f.seek(start);self.wfile.write(f.read(end-start+1))
            return
        if path.startswith('/model/'):return self.send(b'{}',status=404)
        return super().do_GET()

    def do_POST(self):
        """확인한 한 반복 구간만 비동기 모델 분석으로 접수한다. 나머지는 기존 API다."""
        if urlparse(self.path).path!='/model/analyze':return super().do_POST()
        if self.headers.get('Origin') not in [None,'http://127.0.0.1:8766','http://localhost:8766']:return self.send(b'{}',status=403)
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=4096:raise ValueError('요청 크기 오류')
            self.connection.settimeout(15);payload=json.loads(self.rfile.read(size))
            if not isinstance(payload,dict):raise ValueError('요청 형식 오류')
            source=self.server.model_catalog.get(payload.get('source_id'))
            if source is None or payload.get('acknowledged') is not True:raise ValueError('영상과 한 반복 구간 확인란을 선택하세요.')
            start=float(payload['start']);end=float(payload['end'])
            if not math.isfinite(start+end) or not 0<=start<end<=source['duration'] or not 1.2<=end-start<=15:raise ValueError('영상 안에서 1.2~15초의 한 반복 구간을 지정하세요.')
        except (ValueError,TypeError,KeyError) as error:return self.send(json.dumps(dict(error=str(error)),ensure_ascii=False).encode(),status=400)
        if not original.LOCK.acquire(blocking=False):return self.send(json.dumps(dict(error='다른 영상 분석이 진행 중입니다.'),ensure_ascii=False).encode(),status=409)
        ident=uuid.uuid4().hex;folder=JOBS/ident;folder.mkdir(parents=True)
        (folder/'request.json').write_text(json.dumps(dict(source=source['source'],source_job=source['source_job'],exercise=source['exercise'],start=start,end=end,diagnostics=payload.get('diagnostics') is True),ensure_ascii=False));self.server.model_states[ident]=dict(state='processing')
        threading.Thread(target=run_model,args=(self.server,ident),daemon=True).start();return self.send(json.dumps(dict(id=ident)).encode(),status=202)


def serve():
    """기존 화면과 실험 모델 화면을 8766 루프백 포트에 연다.

    Returns:
        None: 종료 신호까지 요청을 처리한다.
    Raises:
        OSError: 포트를 사용 중이거나 파일을 읽을 수 없는 경우.
    """
    JOBS.mkdir(exist_ok=True);web=ThreadingHTTPServer(('127.0.0.1',8766),ModelHandler);web.source_library=original.SourceLibrary(original.ROOT);web.model_states={};web.model_catalog=catalog();print('http://127.0.0.1:8766/model/',flush=True);web.serve_forever()


if __name__=='__main__':serve()

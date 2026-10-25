"""One camera reader, newest-frame inference, no raw video recording."""
from collections import Counter,deque
from pathlib import Path
import copy,json,threading,time
from urllib.parse import urlparse,urlunparse
from urllib.request import urlopen
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[1]
EXERCISES={'squats':'스쿼트','pushups':'푸시업','elbow_plank':'엘보 플랭크','ytw':'YTW','shoulder_press':'숄더 프레스'}
DEFAULT_SOURCE='http://192.168.15.208:81/stream'

class TemporalGate:
    """Never carry a prediction across a missing person, reconnect, or large gap."""
    def __init__(self,length=32):self.length=length;self.points=deque(maxlen=length);self.last=None;self.aspect=None;self.recent=deque(maxlen=3);self.invalid_run=0
    def clear(self):self.points.clear();self.recent.clear();self.last=None;self.aspect=None;self.invalid_run=0
    def append(self,p,stamp,aspect,required):
        if self.last is not None and (stamp<=self.last or stamp-self.last>1. or abs(aspect-self.aspect)>.01):self.clear()
        if p is None or not np.isfinite(p).all():
            self.clear();return 'pose_not_visible'
        self.last=stamp;self.aspect=aspect
        if np.min(p[required,3])<.3:
            self.invalid_run+=1;self.recent.clear()
            if self.invalid_run>=3:self.clear()
            else:self.points.append(p)
            return 'pose_not_visible'
        self.invalid_run=0;self.points.append(p)
        return 'ready' if len(self.points)==self.length else 'warming_up'
    def stabilize(self,result):
        self.recent.append(set(result.get('detected_errors',[])))
        counts=Counter(t for frame in self.recent for t in frame)
        return [t for t in result.get('detected_errors',[]) if counts[t]>=2]

class CameraReader:
    def __init__(self,source):
        self.source=source;self.stop_event=threading.Event();self.lock=threading.Lock();self.latest=None;self.seq=0;self.epoch=0;self.error='';self.connected=False;self.capture_fps=0.;self.reads=deque(maxlen=40)
        self.transport='stream';self.capture_url=None
        self.thread=threading.Thread(target=self.run,daemon=True)
    def start(self):self.thread.start()
    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None:self.thread.join(timeout=6)
    def snapshot(self):
        with self.lock:return self.seq,self.epoch,self.latest,self.connected,self.error,self.capture_fps
    def snapshots(self,url):
        self.transport='snapshot'
        while not self.stop_event.is_set():
            begin=time.monotonic()
            with urlopen(url,timeout=3) as response:payload=response.read(5_000_001)
            if len(payload)>5_000_000:raise ConnectionError('카메라 사진이 너무 큽니다.')
            frame=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR)
            if frame is None:raise ConnectionError('카메라 JPEG 사진을 읽을 수 없습니다.')
            now=time.monotonic();self.reads.append(now)
            with self.lock:
                self.latest=(frame,now);self.seq+=1;self.connected=True;self.error='';self.capture_fps=(len(self.reads)-1)/(self.reads[-1]-self.reads[0]) if len(self.reads)>1 else 0.
            self.stop_event.wait(max(0,1/6-(time.monotonic()-begin)))
    def run(self):
        local=Path(self.source).is_file();camera=self.source.isdigit();attempt=0
        while not self.stop_event.is_set():
            cap=None
            try:
                with self.lock:
                    if not self.error:self.error='카메라 연결 중…'
                    self.connected=False;self.latest=None;self.epoch+=1
                if self.capture_url or urlparse(self.source).path=='/capture':
                    self.snapshots(self.capture_url or self.source);break
                cap=cv2.VideoCapture(int(self.source)) if camera else cv2.VideoCapture(self.source,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1,cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,4000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,3000])
                if not cap.isOpened():
                    parsed=urlparse(self.source)
                    if parsed.scheme in ['http','https'] and parsed.port==81 and parsed.path=='/stream':
                        self.capture_url=urlunparse((parsed.scheme,parsed.hostname+':80','/capture','','',''))
                        raise ConnectionError('영상 스트림 응답 없음 · 사진 연속 수신으로 전환 중…')
                    raise ConnectionError('카메라에 연결할 수 없습니다. 전원·Wi-Fi·주소를 확인하세요.')
                self.reads.clear();decoded=0
                fps=cap.get(cv2.CAP_PROP_FPS);pace=1/max(fps,1) if local else 0.;next_frame=time.monotonic();attempt=0
                while not self.stop_event.is_set():
                    ok,frame=cap.read()
                    if not ok:
                        if local and decoded:
                            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                            with self.lock:self.epoch+=1;self.latest=None;self.connected=False
                            self.reads.clear();continue
                        raise ConnectionError('영상 수신이 끊겼습니다. 재연결 중…')
                    decoded+=1;now=time.monotonic();self.reads.append(now)
                    with self.lock:
                        self.latest=(frame,now);self.seq+=1;self.connected=True;self.error='';self.capture_fps=(len(self.reads)-1)/(self.reads[-1]-self.reads[0]) if len(self.reads)>1 else 0.
                    if local:
                        next_frame+=pace;self.stop_event.wait(max(0,next_frame-time.monotonic()))
            except Exception as e:
                with self.lock:self.connected=False;self.latest=None;self.error=str(e);self.epoch+=1
                attempt+=1
                if self.stop_event.wait(min(attempt,4)):break
            finally:
                if cap is not None:cap.release()
        with self.lock:self.connected=False;self.latest=None

class LiveEngine:
    def __init__(self):
        self.lock=threading.RLock();self.control=threading.Lock();self.stop_event=threading.Event();self.reset_event=threading.Event();self.thread=None;self.reader=None;self.frame_seq=0;self.jpeg=self.placeholder('Press Start to connect ESP32-CAM');self.state=self.idle()
    @staticmethod
    def idle():return dict(running=False,status='idle',message='카메라 주소와 종목을 선택한 뒤 시작하세요.',exercise='squats',source=DEFAULT_SOURCE,buffer_frames=0,required_frames=32,capture_fps=0.,processing_fps=0.,pose_visible=False,result=None,last_result=None,last_prediction_at=None,error='',mode='live',predictions=0)
    @staticmethod
    def placeholder(message):
        image=np.zeros((480,800,3),np.uint8);image[:]=(31,25,18)
        cv2.putText(image,message,(25,235),cv2.FONT_HERSHEY_SIMPLEX,.65,(210,220,235),1,cv2.LINE_AA)
        return cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,80])[1].tobytes()
    def snapshot(self):
        with self.lock:
            state=copy.deepcopy(self.state);state['frame_sequence']=self.frame_seq
            return state
    def image(self):
        with self.lock:return self.jpeg
    def publish(self,**values):
        with self.lock:self.state.update(values)
    def clear_view(self,message):
        with self.lock:self.jpeg=self.placeholder(message);self.frame_seq+=1
    def reset(self):
        if self.snapshot()['running']:
            self.reset_event.set();self.publish(result=None,buffer_frames=0,pose_visible=False,status='warming_up',message='자세 수집을 다시 시작합니다.')
    def start(self,source,exercise,record=False):
        source=str(source).strip()
        if exercise not in EXERCISES:raise ValueError('지원하지 않는 운동 종목입니다.')
        if not source:raise ValueError('카메라 주소를 입력하세요.')
        if not(source.isdigit() or Path(source).is_file() or urlparse(source).scheme in ['http','https']):raise ValueError('HTTP(S) 스트림 주소, 웹캠 번호, 또는 저장 영상 경로를 입력하세요.')
        with self.control:
            self._stop()
            self.stop_event=threading.Event();self.reset_event=threading.Event();self.reader=CameraReader(source)
            with self.lock:
                self.state=self.idle();self.state.update(running=True,status='loading_model',message='학습된 모델을 불러오는 중…',source=source,exercise=exercise,mode='video_replay' if Path(source).is_file() else 'live',record=bool(record))
            self.clear_view('Loading model...')
            self.thread=threading.Thread(target=self.run,args=(exercise,bool(record)),daemon=True);self.thread.start()
    def stop(self):
        with self.control:self._stop()
    def _stop(self):
        self.stop_event.set()
        if self.reader:self.reader.stop()
        if self.thread:
            self.thread.join(timeout=8)
            if self.thread.is_alive():raise RuntimeError('이전 작업을 종료하는 중입니다. 잠시 뒤 다시 시작하세요.')
        self.reader=None;self.thread=None
        self.publish(running=False,status='idle',message='테스트가 중지됐습니다.',result=None,buffer_frames=0,pose_visible=False,capture_fps=0.,processing_fps=0.)
        self.clear_view('Stopped - press Start to connect')
    def run(self,exercise,record):
        import mediapipe as mp
        from suite_inference import FiveExerciseClassifier
        model=None;pose=None;log=None
        try:
            model=FiveExerciseClassifier().model(exercise)
            required=[11,12,23,24,25,26,27,28] if exercise=='squats' else model.saved['required_joints']
            length=model.length if exercise=='squats' else model.saved['sequence_length'];fps=model.fps if exercise=='squats' else model.saved['sample_fps']
            gate=TemporalGate(length);pose=mp.solutions.pose.Pose(model_complexity=1,min_detection_confidence=.5,min_tracking_confidence=.5)
            if record:
                directory=ROOT/'live_sessions';directory.mkdir(exist_ok=True);filename=directory/(time.strftime('%Y%m%d_%H%M%S')+'_'+exercise+'_'+str(time.time_ns())[-6:]+'.jsonl');log=filename.open('w');self.publish(log_path=str(filename))
            self.publish(status='connecting',message='ESP32-CAM에 연결 중…',required_frames=length)
            self.reader.start();seen=-1;epoch=-1;last_tick=0.;processing_times=deque(maxlen=30)
            while not self.stop_event.is_set():
                seq,current_epoch,item,connected,error,capture_fps=self.reader.snapshot()
                if self.reset_event.is_set():gate.clear();self.reset_event.clear();pose.reset();last_tick=0.
                if current_epoch!=epoch:
                    epoch=current_epoch;gate.clear();pose.reset();last_tick=0.;self.publish(result=None,buffer_frames=0,pose_visible=False)
                if not connected or item is None:
                    gate.clear();self.publish(status='connecting',message=error or '영상 연결을 기다리는 중…',error=error,result=None,buffer_frames=0,pose_visible=False,capture_fps=0.,processing_fps=0.)
                    if seen!=-2:self.clear_view('Camera offline - check power / Wi-Fi / URL');seen=-2
                    self.stop_event.wait(.1);continue
                frame,stamp=item
                if time.monotonic()-stamp>1.:
                    gate.clear();self.publish(status='connecting',message='최신 영상 수신을 기다립니다.',result=None,buffer_frames=0,pose_visible=False);self.clear_view('Waiting for a fresh camera frame');self.stop_event.wait(.1);continue
                if seq==seen or stamp-last_tick+1e-8<1/fps:self.stop_event.wait(.01);continue
                seen=seq;last_tick=stamp;h,w=frame.shape[:2];aspect=w/h
                resized=cv2.resize(frame,(round(w*480/max(h,w)),round(h*480/max(h,w)))) if max(h,w)>480 else frame
                result=pose.process(cv2.cvtColor(resized,cv2.COLOR_BGR2RGB));raw=np.array([[l.x,l.y,l.z,l.visibility] for l in result.pose_landmarks.landmark],np.float32) if result.pose_landmarks else None
                stage=gate.append(raw,stamp,aspect,required);prediction=None;message='전신과 필요한 관절이 잘 보이도록 위치를 조정하세요.'
                if stage=='ready':
                    prediction=model.predict_landmarks(np.array(gate.points),aspect)
                    if prediction['status']=='ok':
                        stable=gate.stabilize(prediction);prediction['raw_detected_errors']=prediction.get('detected_errors',[]);prediction['detected_errors']=stable
                        mapping=dict(zip(prediction['raw_detected_errors'],prediction.get('feedbacks',[])));prediction['feedbacks']=[mapping[t] for t in stable if t in mapping]
                        prediction['display_smoothing']='At least 2 of the last 3 predictions, also present in current prediction'
                        stage='predicting';message='선택 오류 감지' if stable else '선택 오류 미검출 · 전체 정상 판정은 아닙니다.'
                        self.publish(predictions=self.snapshot()['predictions']+1)
                        if log:log.write(json.dumps(dict(time=time.time(),exercise=exercise,prediction=prediction),ensure_ascii=False)+'\n');log.flush()
                    else:stage='pose_not_visible';prediction=None;gate.clear()
                elif stage=='warming_up':message=f'자세 수집 중 {len(gate.points)}/{length} · 최초 약 {length/fps:.1f}초 필요'
                processing_times.append(time.monotonic());processing_fps=(len(processing_times)-1)/(processing_times[-1]-processing_times[0]) if len(processing_times)>1 else 0.
                annotated=frame.copy()
                if result.pose_landmarks:mp.solutions.drawing_utils.draw_landmarks(annotated,result.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS)
                if max(annotated.shape[:2])>960:annotated=cv2.resize(annotated,(round(w*960/max(h,w)),round(h*960/max(h,w))))
                cv2.putText(annotated,f'{exercise.upper()} | {stage} | {len(gate.points)}/{length}',(12,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(50,220,255),2,cv2.LINE_AA)
                jpeg=cv2.imencode('.jpg',annotated,[cv2.IMWRITE_JPEG_QUALITY,80])[1].tobytes()
                with self.lock:
                    if self.stop_event.is_set():break
                    if self.reset_event.is_set():continue
                    if prediction is not None:self.state.update(last_result=copy.deepcopy(prediction),last_prediction_at=time.time())
                    self.jpeg=jpeg;self.frame_seq+=1;self.state.update(status=stage,message=message,error='',result=prediction,buffer_frames=len(gate.points),pose_visible=raw is not None and stage!='pose_not_visible',transport=self.reader.transport,capture_fps=round(capture_fps,1),processing_fps=round(processing_fps,1),last_frame_at=time.time())
        except Exception as e:
            self.publish(running=False,status='error',error=str(e),message='실행 중 오류가 발생했습니다: '+str(e),result=None,buffer_frames=0,pose_visible=False);self.clear_view('Error - see the status message')
        finally:
            if self.reader:self.reader.stop()
            if pose:pose.close()
            if model:model.close()
            if log:log.close()

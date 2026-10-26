"""USB 실시간 관절 미리보기와 수동 구간 점수 확인용 로컬 어댑터.

팀 소스·학습 자료를 변경하지 않는다. 명시적 촬영 모드에서만 로컬 영상을 저장한다.
동작 시작·종료와 구도는 사용자가 확인하며 연속 프레임 점수를 주장하지 않는다.
"""
import sys
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import json
import hashlib
import time
import threading
import subprocess
import queue
import signal
from collections import deque
from movement import MovementTracker
from shoulder_candidate import ShoulderTracker
from live_observation import mask_frame_border, movement_observation, ObservationReadyGate, require_start_ready, VERSION as LIVE_OBSERVATION_VERSION
from recording import Recorder, RUBRIC
from front_policy import apply_front_policy, frontal_motion_observations
from front_candidate import predict as predict_front_candidate
from accepted_items import evaluate as evaluate_accepted_items
from method_comparison import compare_candidate
from rubric100 import assessment as assess100
from unified_assessment import apply as unify_assessment
from item_observation import VERSION as ITEM_OBSERVATION_VERSION
from squat_coach import LiveCoach, completed_feedback, SOURCES as COACH_SOURCES
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np
import mediapipe as mp

ROOT = Path('/home/jayden/SeSACPAI-Mid')
from runtime_snapshot import prepare_snapshot, read_source, digest, SOURCE
RUNTIME_ROOT, LOADED_DIGEST = prepare_snapshot()
sys.path.insert(0, str(RUNTIME_ROOT))
from app.scoring.engine import ScoringEngine
from app.scoring.sequence import PoseSequence

RECORDER = Recorder(Path(__file__).parent / "recordings")
LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()
COACH=LiveCoach()
def make_tracker(exercise, view='front'):
    """운동 전용 분절기를 선택하며 미지원 구도는 시작 전에 안내한다."""
    if exercise == 'shoulder':
        if view == 'side':
            raise ValueError('숄더 프레스는 정면 또는 정면 사선으로 구도를 선택하세요.')
        return ShoulderTracker(exercise, view)
    return MovementTracker(exercise, view)


TRACKER = make_tracker("shoulder")
REP_QUEUE = queue.Queue(maxsize=20)
STATE = dict(status='카메라 준비 중', jpeg=None, collecting=False, samples=[], result=None, tick=0, visible=0, exercise='shoulder', view='front', live_result=None, live_error=None, generation=0, started_at=None, movement=TRACKER.status(), repetitions=[], last_rep=None,item_observation_version=ITEM_OBSERVATION_VERSION,standing_state_version='initial-standing-state-v1',standing_state_model_sha256=hashlib.sha256(Path(__file__).with_name('standing-state').joinpath('model.joblib').read_bytes()).hexdigest())
SOURCE_COMMIT = subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
SOURCE_DIRTY = bool(subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--','app/scoring','artifacts/posture_score'],text=True).strip())
ENGINE = ScoringEngine()
for exercise_code in ("squats", "pushups", "shoulder", "plank", "ytw"):
    ENGINE.model(exercise_code)
STATE.update(source_commit=SOURCE_COMMIT,source_dirty=SOURCE_DIRTY,loaded_code_model_sha256=LOADED_DIGEST, model_version=ENGINE.manifest["version"], feature_version=ENGINE.manifest["feature_version"], update_available=False, source_check_error=None, extraction_delegate="CPU")
EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]

def mosaic_faces(frame, xy):
    """현재 분석 대상의 얼굴 관절 주변만 약한 모자이크로 표시한다.

    Args:
        frame: 표시용 BGR 영상. 얼굴 영역만 제자리 수정한다.
        xy: 원본 분석의 33×2 픽셀 관절. 배열은 변경하지 않는다.

    Returns:
        None: 얼굴 관절이 부족하거나 영역이 비정상이면 적용하지 않는다.
    """
    h, w = frame.shape[:2]
    head = xy[:11][np.isfinite(xy[:11]).all(axis=1)]
    if len(head) < 5:
        return
    lo, hi = head.min(axis=0), head.max(axis=0)
    span = hi[0] - lo[0]
    if span < 12 or span > w * .45:
        return
    cx = (lo[0] + hi[0]) / 2
    cy = (lo[1] + hi[1]) / 2 + span * .05
    bw, bh = span * 1.15, max(hi[1]-lo[1], span * 1.12)
    x0, y0 = max(0, int(cx-bw/2)), max(0, int(cy-bh/2))
    x1, y1 = min(w, int(cx+bw/2)), min(h, int(cy+bh/2))
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    tile = cv2.resize(roi, (max(8, (x1-x0)//6), max(8, (y1-y0)//6)), interpolation=cv2.INTER_AREA)
    frame[y0:y1, x0:x1] = cv2.resize(tile, (x1-x0,y1-y0), interpolation=cv2.INTER_NEAREST)

def capture():
    """카메라 한 대를 열고 10Hz 관측·관절 표시 JPEG를 메모리에 보관한다."""
    cap = cv2.VideoCapture('/dev/v4l/by-id/usb-ZHHWY_Sandberg_USB_Webcam_Pro-video-index0', cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(3,640); cap.set(4,480); cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    options = mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'mediapipe-pilot/models/heavy.task')), running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1)
    latest = dict(frame=None, captured_at=0., serial=0, error=None)
    frame_lock = threading.Lock()
    stopped = threading.Event()
    reader = None
    cadence = deque(maxlen=60)
    ready_gate = ObservationReadyGate()
    def read_latest():
        # 영상 수신이 Heavy 추론 시간을 기다리지 않도록 최신 실제 프레임만 보관한다.
        while not stopped.is_set():
            ok, frame = cap.read()
            with frame_lock:
                if not ok:
                    latest['error']='카메라 프레임 수신 실패'
                    return
                latest.update(frame=frame,captured_at=time.monotonic(),serial=latest['serial']+1)
    try:
        if not cap.isOpened(): raise RuntimeError('USB 웹캠 열기 실패')
        origin=time.monotonic(); last=-1; last_serial=-1
        reader=threading.Thread(target=read_latest,daemon=True);reader.start()
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as pose:
            while True:
                target = origin+(last+1)/10
                time.sleep(max(0.,target-time.monotonic()))
                with frame_lock:
                    frame = latest['frame']
                    captured_at = latest['captured_at']
                    serial = latest['serial']
                    error = latest['error']
                if error: raise RuntimeError(error)
                if frame is None:
                    time.sleep(.005);continue
                if time.monotonic()-captured_at>.35: raise RuntimeError('카메라 최신 프레임 지연')
                tick=max(0,round((captured_at-origin)*10))
                if serial==last_serial or tick<=last:
                    time.sleep(.005);continue
                last=tick;last_serial=serial
                frame=frame.copy();h,w=frame.shape[:2]
                cadence.append(tick)
                observed_fraction=len(cadence)/(cadence[-1]-cadence[0]+1)
                actual_hz=(len(cadence)-1)/max(.1,(cadence[-1]-cadence[0])/10)
                result=pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))),tick*100)
                xy=np.full((33,2),np.nan)
                if result.pose_landmarks:
                    for j,p in enumerate(result.pose_landmarks[0]):
                        if p.visibility>=.5 and p.presence>=.5: xy[j]=[p.x*w,p.y*h]
                with LOCK:
                    if RECORDER.active:
                        clean = frame.copy()
                        mosaic_faces(clean, xy)
                        RECORDER.push(clean, xy, tick, captured_at)
                for a,b in EDGES:
                    if np.isfinite(xy[[a,b]]).all(): cv2.line(frame,tuple(xy[a].astype(int)),tuple(xy[b].astype(int)),(60,220,80),2)
                for p in xy:
                    if np.isfinite(p).all(): cv2.circle(frame,tuple(p.astype(int)),3,(20,100,255),-1)
                mosaic_faces(frame, xy)
                with LOCK:
                    if STATE['collecting']:
                        STATE['samples'].append((tick,xy.copy()))
                        # Preserve the whole session for end-of-session pose export; live inference still uses [-60:].
                        segment = TRACKER.push(tick, xy, width=w, height=h)
                        STATE['movement'] = TRACKER.status()
                        # Frontal coaching is produced by the same assessment as
                        # the points; the old independent warning thresholds are
                        # retained in squat_coach.py, not run as a second authority.
                        if segment:
                            entry = dict(number=TRACKER.count, status='평가 중', duration=round((segment[-1][0]-segment[0][0])/10,1))
                            STATE['repetitions'].append(entry)
                            try:
                                REP_QUEUE.put_nowait((STATE['generation'], TRACKER.count, segment, w, h, STATE['exercise'], STATE['view'], TRACKER.standing_reference))
                            except queue.Full:
                                entry['status'] = '평가 대기 초과'
                    cv2.putText(frame,'LIVE USB / Heavy / segment scoring',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(50,220,70),1)
                    required_front = [11,12,23,24,25,26]
                    present = np.isfinite(xy[required_front]).all(1)
                    missing_names = [name for name,ok in zip(['왼쪽 어깨','오른쪽 어깨','왼쪽 골반','오른쪽 골반','왼쪽 무릎','오른쪽 무릎'],present) if not ok]
                    STATE.update(camera_at=time.monotonic(),sample_cadence=dict(actual_hz=round(actual_hz,2),observed_slot_fraction=round(observed_fraction,3),required_hz=10,minimum_observed_fraction=.8),status='실시간 관절 추출 중 · 얼굴 모자이크 적용',tick=tick,visible=int(np.isfinite(xy).all(1).sum()),width=w,height=h,front_squat_observation=dict(visible=int(present.sum()),required=6,missing=missing_names),jpeg=cv2.imencode('.jpg',frame)[1].tobytes())
                    STATE.update(live_observation_version=LIVE_OBSERVATION_VERSION,
                                 observation_options=ready_gate.update(tick,xy,w,h))
    except Exception as e:
        with LOCK:
            STATE['status']=str(e)
            RECORDER.stop('카메라 오류: '+str(e))
        preserve_interrupted_session('카메라 연결 중단: ' + str(e))
    finally:
        stopped.set()
        if reader is not None: reader.join(timeout=1)
        cap.release()

def safe(x):
    """배열·비유한 값을 JSON 허용 값으로 변환한다."""
    if isinstance(x,dict): return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)): return [safe(v) for v in x]
    if isinstance(x,np.generic): return safe(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x


def preserve_interrupted_session(reason):
    """관측 단절/정상 종료 신호 직전 자료를 별도 중단 기록으로 보존한다.

    중단을 실패 정답으로 만들지 않는다. 정상 종료 세션과 구분해 저장한다.
    """
    with LOCK:
        if not STATE['collecting']:
            return None
        STATE['collecting'] = False
        samples = list(STATE['samples'])
        folder = Path(__file__).with_name('interrupted-sessions') / str(time.time_ns())
        STATE.update(interruption_notice='측정이 중단됐습니다. 새 운동은 시작하지 않았습니다. ' + reason,
                     last_interrupted_session_folder=str(folder))
        data = safe({k:v for k,v in STATE.items() if k not in ('samples','jpeg')})
        data.update(interrupted=True, interruption_reason=reason,
                    finalizing=False, results_may_be_pending=True)
        width,height = STATE.get('width',640),STATE.get('height',480)
    folder.mkdir(parents=True,exist_ok=False)
    if samples:
        np.savez_compressed(folder/'pose.npz', image_xy=np.array([p for _,p in samples]),
                            frame_numbers=np.array([t for t,_ in samples]),fps=10.,width=width,height=height)
    (folder/'interrupted-state.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print('INTERRUPTED_SESSION ' + str(folder) + ' ' + reason,flush=True)
    return folder


def stop_on_signal(signum, frame):
    """SIGTERM/인터럽트 시 운동을 재개하지 않고 중단 자료를 보존한다."""
    preserve_interrupted_session('서버 종료 신호 ' + str(signum))
    raise SystemExit(0)

def score_samples(samples, width, height, exercise, view, complete=False, phase='complete',standing_reference=None,return_samples=None):
    """실제 표본의 누락을 NaN으로 유지해 동일 모델로 구간을 평가한다.

    Args:
        samples: 10Hz 슬롯 번호와 픽셀 관절 배열의 목록.
        width: 영상 너비, 픽셀.
        height: 영상 높이, 픽셀.
        exercise: 사용자가 선택한 운동 코드.
        view: 사용자가 확인한 촬영 방향.
        complete: 사용자가 완결 수행을 확인한 경우만 True.

    Returns:
        dict: 모델 결과. 완결 미확인 실시간 구간의 총점은 None이다.

    Raises:
        ValueError: 표본 부족·잘못된 구도 또는 입력 계약 불일치.
    """
    if len(samples) < 10:
        raise ValueError('최소 1초 이상 수집하세요')
    ticks = np.arange(samples[0][0], samples[-1][0]+1)
    xy = np.full((len(ticks),33,2), np.nan)
    for tick, points in samples:
        xy[tick-ticks[0]] = points
    xy = mask_frame_border(xy, width, height)
    return_xy=None
    if return_samples:
        return_xy=np.full((return_samples[-1][0]-return_samples[0][0]+1,33,2),np.nan)
        for tick,points in return_samples:return_xy[tick-return_samples[0][0]]=points
        return_xy = mask_frame_border(return_xy, width, height)
    seq = PoseSequence(xy,ticks-ticks[0],10.,width,height)
    with MODEL_LOCK:
        result=apply_front_policy(safe(ENGINE.score(exercise,seq,view=view,complete_attempt=complete)),exercise,view)
        if exercise=='squats' and view=='front':
            result['frontal_motion_observations']=frontal_motion_observations(xy)
            try: result['front_candidate']=predict_front_candidate(xy)
            except ValueError as error: result['front_candidate']={'error':str(error),'experimental':True}
            result['method_comparison']=compare_candidate(result.get('front_candidate'),xy,width=width,height=height,standing_reference=standing_reference,phase=phase,return_xy=return_xy)
            result['coaching']=completed_feedback(xy,result.get('front_candidate'))
            result['accepted_items']=evaluate_accepted_items(xy,width,height)
            for item in result['accepted_items']['items']:
                for check in result['coaching'].get('checks',[]):
                    if check['id']==item['id']:check['status']=item.get('guidance',item['status'])
                if item.get('reason') or item.get('disagreement'):
                    result['coaching']['messages'].append(item['guidance'])
        result['score_100']=assess100(result,exercise)
        result = unify_assessment(result,exercise,view,phase)
        if exercise == 'shoulder':
            titles = dict(overhead_reach='양팔 상단 도달', lower_return='하단 복귀',
                          bilateral_timing='양팔 타이밍', frontal_trunk_hold='몸통 좌우 안정')
            source = {item['id']:item for item in result['items']}
            reasons = {
                'required_joints_not_simultaneously_observed_in_80_percent':'필요 관절의 동시 관측이 동작 구간의 80% 미만입니다.',
                'wrists_not_visible_at_overhead_peak':'상단에서 양손목을 확인하지 못했습니다.',
                'no_observed_lift_motion':'팔을 올린 구간을 확인하지 못했습니다.',
                'bilateral_peak_not_fully_visible':'양팔 상단 구간의 관측이 부족합니다.',
                'required_measurement_unavailable':'관측이 부족해 필요한 측정값을 계산하지 못했습니다.',
                'unsupported_model_view_bank':'이 항목의 현재 구도 모델이 없습니다.',
                'frontal_trunk_requires_front_view':'몸통 좌우 항목은 정면 구도가 필요합니다.',
            }
            for item in result['score_100']['items']:
                item['title'] = titles.get(item['id'],item['title'])
                for reason in source.get(item['id'],{}).get('reasons',[]):
                    if reason in reasons and reasons[reason] not in item['reasons']:
                        item['reasons'].append(reasons[reason])
            result['score_100']['scope'] = '숄더 프레스 정면 관측의 항목별 실험 점수입니다. 미평가 항목을 채우거나 100점으로 환산하지 않습니다. 정상만 학습한 AI 항목의 오류 판별은 미검증입니다.'
        return result


def live_evaluation_loop():
    """운동 중 최근 6초의 실제 관측 항목을 약 1초마다 갱신한다.

    미완결 구간의 총점·반복 수를 생성하지 않는다. 세션이 바뀌면
    이전 계산 결과를 버리고 현재 관절 미인식 시 이전 평가를 숨긴다.
    """
    while True:
        time.sleep(.8)
        with LOCK:
            active = STATE['collecting']
            frontal = STATE['exercise']=='squats' and STATE['view']=='front'
            segmented = frontal or STATE['exercise']=='shoulder'
            # Rest and adjacent repetitions must not enter the active attempt.
            samples = list(TRACKER.buffer) if segmented and TRACKER.active else ([] if segmented else list(STATE['samples'][-60:]))
            standing_reference=TRACKER.standing_reference
            generation = STATE['generation']
            exercise, view = STATE['exercise'], STATE['view']
            width, height = STATE.get('width',640), STATE.get('height',480)
            visible = STATE['visible']
        if not active:
            continue
        if segmented and not samples:
            with LOCK:
                if STATE['generation']==generation:
                    STATE.update(live_result=None,live_error=None)
                    STATE['coaching']=(STATE.get('last_rep') or {}).get('coaching')
            continue
        try:
            if visible < 6:
                raise ValueError('필요한 관절을 확인할 수 없습니다. 카메라 구도를 조정하세요.')
            result = score_samples(samples,width,height,exercise,view,phase='active',standing_reference=standing_reference)
            result['provisional'] = True
            result['window_seconds'] = (samples[-1][0]-samples[0][0])/10
            result['updated_at'] = time.time()
            with LOCK:
                if STATE['generation']==generation and STATE['collecting'] and (not segmented or (TRACKER.active and TRACKER.buffer and TRACKER.buffer[0][0]==samples[0][0])):
                    STATE.update(live_result=result, live_error=None)
                    if frontal: STATE['coaching']=result.get('coaching')
        except Exception as error:
            with LOCK:
                if STATE['generation']==generation:
                    STATE.update(live_result=None,live_error=str(error))


def observed_points(result, method):
    """관측 항목의 득점·가능 배점만 합산하고 누락을 만점으로 채우지 않는다."""
    if (result.get('score_100') or {}).get('version')=='front-observation100-consistent-v1':
        return result[method]
    items = [i for i in result['items'] if i[method+'_grade'] is not None and not (method=='ai' and i.get('single_grade_training',False))]
    return dict(earned=round(sum(i['weight']*i[method+'_grade']/2 for i in items),1),
                maximum=sum(i['weight'] for i in items),
                full_score=len(items)==len(result['items']),
                excluded_unvalidated_items=[i['id'] for i in result['items'] if method=='ai' and i.get('single_grade_training',False)])


def finish_observation_window(samples,standing_reference,available):
    """Include up to .3 seconds of observed standing, never the next descent."""
    end=samples[-1][0];out=list(samples)
    for tick,points in available:
        if tick<=end:continue
        if tick>end+3:break
        if standing_reference is not None and np.isfinite(points[[23,24]]).all():
            hip=points[[23,24]].mean(0)
            drop=(hip[1]-standing_reference['points'][[23,24]].mean(0)[1])/standing_reference['scale']
            if drop>.09:break
        out.append((tick,points.copy()))
    return out


def repetition_evaluation_loop():
    """자동 검출한 반복별로 점수를 계산하고 원래 세션에만 반영한다."""
    while True:
        entry=REP_QUEUE.get()
        generation, number, samples, width, height, exercise, view = entry[:7]
        standing_reference=entry[7] if len(entry)>7 else None
        try:
            return_samples=None
            if exercise=='squats' and view=='front' and standing_reference is not None:
                deadline=time.monotonic()+.6
                while time.monotonic()<deadline:
                    with LOCK:
                        wait=STATE['generation']==generation and STATE['collecting'] and STATE['tick']<samples[-1][0]+3
                    if not wait:break
                    time.sleep(.03)
                with LOCK:
                    available=list(STATE['samples']) if STATE['generation']==generation else []
                return_samples=finish_observation_window(samples,standing_reference,available)
            result = score_samples(samples,width,height,exercise,view,False,standing_reference=standing_reference,return_samples=return_samples)
            row = dict(number=number, status='자동 구간 평가 · 실험',
                       duration=round((samples[-1][0]-samples[0][0])/10,1),
                       ai=observed_points(result,'ai'), rule=observed_points(result,'rule'),
                       items=result['items'],front_candidate=result.get('front_candidate'),coaching=result.get('coaching'),accepted_items=result.get('accepted_items'),method_comparison=result.get('method_comparison'),score_100=result.get('score_100'),legacy_score_100=result.get('legacy_score_100'),legacy_coaching=result.get('legacy_coaching'),legacy_method_totals=result.get('legacy_method_totals'))
        except Exception as error:
            row = dict(number=number,status='평가 보류',error=str(error))
        with LOCK:
            if STATE['generation']==generation:
                STATE['repetitions']=[row if r['number']==number else r for r in STATE['repetitions']]
                STATE['last_rep']=row
                if exercise=='squats' and view=='front' and not TRACKER.active:
                    STATE['coaching']=row.get('coaching')
        REP_QUEUE.task_done()


def finalize_session(folder, generation):
    """Persist final rows after queued repetition evaluations finish."""
    REP_QUEUE.join()
    with LOCK:
        if STATE['generation']!=generation:
            return
        data=safe({k:v for k,v in STATE.items() if k not in ('samples','jpeg')})
    data['finalizing']=False
    try:
        (folder/'final-state.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    finally:
        with LOCK:
            STATE['finalizing']=False


def source_watch_loop():
    """원본 갱신을 알리되 운동 중 코드·모델을 교체하지 않는다."""
    while True:
        try:
            current = digest(read_source())
            with LOCK: STATE.update(update_available=current!=LOADED_DIGEST, source_check_error=None)
        except Exception as error:
            with LOCK: STATE['source_check_error']=str(error)
        time.sleep(5)


def restart_runtime():
    """카메라 파일 핸들을 상속하지 않고 새 버전을 읽는 프로세스로 전환한다."""
    subprocess.Popen(
        [sys.executable, '-c',
         'import os,sys,time; time.sleep(.5); os.execv(sys.executable,[sys.executable,sys.argv[1]])',
         str(Path(__file__).resolve())], close_fds=True, start_new_session=True)
    os._exit(0)


class Handler(BaseHTTPRequestHandler):
    """로컬 영상·상태와 명시적 수집·평가 명령만 제공한다."""
    def log_message(self,*args): pass
    def send(self,data,kind='application/json',status=200):
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        self.path = self.path.split('?', 1)[0]
        if self.path=='/': return self.send(Path(__file__).with_name('index.html').read_bytes(),'text/html; charset=utf-8')
        if self.path=='/standing-state':
            return self.send(Path(__file__).with_name('standing-state').joinpath('replay-state.json').read_bytes(),'application/json')
        if self.path=='/squat-guide': return self.send(Path(__file__).with_name('squat-guide.html').read_bytes(),'text/html; charset=utf-8')
        if self.path=='/last-live-test': return self.send(Path(__file__).with_name('front-candidate').joinpath('last-live-test.html').read_bytes(),'text/html; charset=utf-8')
        if self.path=='/saved-coaching': return self.send(Path(__file__).with_name('front-candidate').joinpath('saved-coaching.html').read_bytes(),'text/html; charset=utf-8')
        if self.path=='/candidate/torso': return self.send(Path(__file__).with_name('front-candidate').joinpath('torso.html').read_bytes(),'text/html; charset=utf-8')
        if self.path=='/candidate': return self.send(Path(__file__).with_name('front-candidate').joinpath('index.html').read_bytes(),'text/html; charset=utf-8')
        if self.path=='/record': return self.send(Path(__file__).with_name('record.html').read_bytes(),'text/html; charset=utf-8')
        if self.path=='/rubric': return self.send(json.dumps(RUBRIC,ensure_ascii=False).encode())
        if self.path=='/record/state':
            with LOCK: data=RECORDER.status()
            return self.send(json.dumps(data,ensure_ascii=False).encode())
        if self.path=='/frame.jpg':
            with LOCK: data=STATE['jpeg']
            return self.send(data or b'','image/jpeg',200 if data else 503)
        if self.path=='/unified-squat':
            return self.send((Path(__file__).parent.parent/'squat-unified-20261023/index.html').read_bytes(),'text/html; charset=utf-8')
        if self.path=='/unified-state':
            return self.send((Path(__file__).parent.parent/'squat-unified-20261023/replay-state.json').read_bytes(),'application/json')
        if self.path=='/rubric100':
            return self.send((Path(__file__).parent.parent/'squat-unified-20261023/rubric.json').read_bytes(),'application/json')
        if self.path=='/common-comparison':
            return self.send((Path(__file__).parent.parent/'squat-common-comparison-20261023/index.html').read_bytes(),'text/html; charset=utf-8')
        if self.path=='/replay-cycles.png':
            return self.send((Path(__file__).parent.parent/'squat-replay-improvement-20261023/cycles.png').read_bytes(),'image/png')
        if self.path=='/saved-state':
            candidates=sorted(Path(__file__).with_name('sessions').glob('*/final-state.json'),key=lambda p:int(p.parent.name))
            if not candidates:return self.send(b'{}','application/json',404)
            data=json.loads(candidates[-1].read_text())
            for row in data.get('repetitions',[]):row['method_comparison']=compare_candidate(row.get('front_candidate'),evidence=(row.get('method_comparison') or {}).get('feature_evidence'))
            data['repetitions']=[unify_assessment(row,data.get('exercise','squats'),data.get('view','front')) for row in data.get('repetitions',[])]
            if data.get('last_rep'):data['last_rep']['method_comparison']=compare_candidate(data['last_rep'].get('front_candidate'),evidence=(data['last_rep'].get('method_comparison') or {}).get('feature_evidence'))
            if data.get('last_rep'):data['last_rep']=unify_assessment(data['last_rep'],data.get('exercise','squats'),data.get('view','front'))
            return self.send(json.dumps(safe(data),ensure_ascii=False).encode(),'application/json')
        if self.path=='/consistent-state':
            return self.send((Path(__file__).parent.parent/'squat-consistent-20261023/after-state.json').read_bytes())
        if self.path=='/consistent-report':
            return self.send((Path(__file__).parent.parent/'squat-consistent-20261023/index.html').read_bytes(),'text/html; charset=utf-8')
        if self.path=='/replay-state':
            data=json.loads(Path(__file__).with_name('replay-state.json').read_text())
            for row in data.get('repetitions',[]):row['score_100']=assess100(row,data.get('exercise','squats'))
            if data.get('last_rep'):data['last_rep']['score_100']=assess100(data['last_rep'],data.get('exercise','squats'))
            return self.send(json.dumps(safe(data),ensure_ascii=False).encode(),'application/json')
        if self.path=='/state':
            with LOCK: d={k:v for k,v in STATE.items() if k not in ('jpeg','samples')};d['sample_count']=len(STATE['samples']);d['elapsed_seconds']=round(time.monotonic()-STATE['started_at'],1) if STATE['collecting'] and STATE['started_at'] else (round((STATE['samples'][-1][0]-STATE['samples'][0][0])/10,1) if len(STATE['samples'])>1 else 0)
            return self.send(json.dumps(safe(d),ensure_ascii=False).encode())
        self.send(b'{}',status=404)
    def do_POST(self):
        global TRACKER, COACH
        try:
            d=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))) or b'{}')
            if self.path=='/record/start':
                with LOCK:
                    if STATE['collecting']: raise ValueError('기존 운동 인식을 먼저 종료하세요')
                    if STATE['jpeg'] is None: raise ValueError('카메라 연결을 확인하세요')
                    if time.monotonic()-STATE.get('camera_at',0)>1: raise ValueError('카메라 최신 영상 대기 중')
                    if STATE['update_available']: raise ValueError('최신 소스를 적용한 뒤 촬영하세요')
                    RECORDER.begin(dict(source_commit=SOURCE_COMMIT,code_model_sha256=LOADED_DIGEST,source_model_version=STATE['model_version'],extraction_delegate='CPU',extraction_model='MediaPipe Heavy',source_api='PoseSequence(image_xy,frame_numbers,fps,width,height)'))
                return self.send(b'{}')
            if self.path=='/record/stop':
                with LOCK: data=RECORDER.stop()
                return self.send(json.dumps(data,ensure_ascii=False).encode())
            if self.path=='/reload':
                with LOCK:
                    if STATE['collecting'] or RECORDER.active: raise ValueError('운동·촬영을 종료한 뒤 최신 버전을 적용하세요')
                self.send(b'{}')
                threading.Timer(.4,restart_runtime).start()
                return
            if self.path=='/begin':
                if d.get('exercise') not in ('shoulder','pushups','squats','plank','ytw') or d.get('view') not in ('front','side','front_oblique'):
                    raise ValueError('운동과 촬영 방향을 선택하세요')
                with LOCK:
                    if STATE['collecting'] or RECORDER.active or STATE.get('finalizing'): raise ValueError('이미 운동 인식·촬영 중이거나 이전 결과를 저장 중입니다.')
                    if STATE['jpeg'] is None: raise ValueError('웹캠 연결 후 시작하세요')
                    if STATE['update_available']: raise ValueError('새 점수 모듈이 있습니다. 최신 버전 적용 후 시작하세요')
                    require_start_ready(STATE,d['exercise'],d['view'],time.monotonic())
                    TRACKER = make_tracker(d['exercise'],d['view'])
                    COACH=LiveCoach(); STATE['coaching']=None
                    STATE.update(movement=TRACKER.status(),repetitions=[],last_rep=None,samples=[],collecting=True,result=None,live_result=None,live_error=None,exercise=d['exercise'],view=d['view'],generation=STATE['generation']+1,started_at=time.monotonic())
                return self.send(b'{}')
            if self.path=='/end':
                with LOCK:
                    STATE['collecting']=False; samples=list(STATE['samples']);w=STATE['width'];h=STATE['height']
                    STATE['finalizing']=True
                    folder=Path(__file__).parent/'sessions'/str(time.time_ns());folder.mkdir(parents=True)
                    if samples: np.savez_compressed(folder/'pose.npz',image_xy=np.array([p for _,p in samples]),frame_numbers=np.array([t for t,_ in samples]),fps=10.,width=w,height=h)
                    (folder/'state.json').write_text(json.dumps(safe({k:v for k,v in STATE.items() if k not in ('samples','jpeg')}),ensure_ascii=False,indent=2))
                    STATE['saved_session_folder']=str(folder)
                    generation=STATE['generation']
                with LOCK:
                    STATE['live_result'] = None
                    summary = dict(movement=STATE['movement'], repetitions=STATE['repetitions'],
                                   status='운동 인식 종료 · 미완료 동작은 횟수 제외')
                threading.Thread(target=finalize_session,args=(folder,generation),daemon=True).start()
                return self.send(json.dumps(safe(summary),ensure_ascii=False).encode())
            self.send(b'{}',status=404)
        except Exception as e: self.send(json.dumps({'error':str(e)},ensure_ascii=False).encode(),status=400)

def camera_loop():
    """USB 재연결 시 미리보기만 복구하고 운동은 자동 재개하지 않는다."""
    while True:
        capture()
        with LOCK:
            STATE.update(jpeg=None, collecting=False, samples=[], result=None, live_result=None, live_error=None, visible=0, generation=STATE['generation']+1)
        time.sleep(2)

if __name__=='__main__':
    STATE['interruption_notice'] = os.environ.get('LIVE_RECOVERY_NOTICE')
    signal.signal(signal.SIGTERM,stop_on_signal)
    signal.signal(signal.SIGINT,stop_on_signal)
    threading.Thread(target=source_watch_loop,daemon=True).start()
    threading.Thread(target=camera_loop,daemon=True).start()
    threading.Thread(target=live_evaluation_loop,daemon=True).start()
    threading.Thread(target=repetition_evaluation_loop,daemon=True).start()
    print('http://127.0.0.1:8791/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8791),Handler).serve_forever()

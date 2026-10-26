"""로컬 시각 언어 모델로 영상의 오류·점수 초안을 작성한다. 외부로 영상을 전송하지 않는다."""
import base64
import json
import math
import os
import subprocess
import time
import urllib.request
from pathlib import Path
import cv2
import numpy as np
from common import BASE, OUT, save, progress, digest

MODEL='qwen3-vl:4b-instruct'
URL='http://127.0.0.1:11435'
SERVER_PROCESS=None
ERRORS={
 'pushups':['hips_high','hips_sag','shallow','head_alignment','asymmetry','other'],
 'squats':['knees_in','heels_lift','shallow','trunk_alignment','asymmetry','other'],
 'plank':['hips_high','hips_sag','head_alignment','shoulder_alignment','other'],
 'ytw':['arms_low','hand_orientation','sequence_order','trunk_alignment','speed','other'],
 'shoulder':['back_arch','asymmetry','incomplete_range','trunk_momentum','head_alignment','other']}
NAMES={'pushups':'standard push-up','squats':'bodyweight squat','plank':'forearm plank','ytw':'Y-T-W raise','shoulder':'dumbbell overhead shoulder press'}


def request(endpoint, body=None, timeout=120):
    """로컬 HTTP API의 JSON 응답을 반환한다.

    Args:
        endpoint: 로컬 서버 API 경로.
        body: 요청 객체 또는 None.
        timeout: 응답 제한 시간, 초.
    Raises:
        OSError: 서버 접근 실패.
        ValueError: 응답이 JSON 형식이 아닌 경우.
    """
    req=urllib.request.Request(URL+endpoint,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.load(response)


def start_server():
    """작업 전용 로컬 서버와 모델을 준비하고 이 작업이 시작한 프로세스를 반환한다."""
    global SERVER_PROCESS
    runtime=BASE/'local-ai-runtime'
    env=os.environ.copy()
    for key in ['LD_PRELOAD','__EGL_VENDOR_LIBRARY_FILENAMES']:
        env.pop(key,None)
    env.update(OLLAMA_HOST='127.0.0.1:11435',OLLAMA_MODELS=str(runtime/'models'),OLLAMA_NUM_PARALLEL='1',OLLAMA_MAX_LOADED_MODELS='1',OLLAMA_CONTEXT_LENGTH='2048',OLLAMA_FLASH_ATTENTION='1',OLLAMA_KV_CACHE_TYPE='q8_0',CUDA_VISIBLE_DEVICES='0',OLLAMA_GPU_OVERHEAD='200000000',OLLAMA_NO_CLOUD='1')
    process=None
    try:request('/api/version',timeout=2)
    except OSError:
        log=(OUT/'ollama.log').open('a')
        process=subprocess.Popen([str(runtime/'bin/ollama'),'serve'],env=env,stdout=log,stderr=subprocess.STDOUT)
        log.close()
        for _ in range(60):
            if process.poll() is not None:raise RuntimeError('로컬 라벨 모델 서버 시작 실패')
            try:request('/api/version',timeout=2);break
            except OSError:time.sleep(1)
        else:raise RuntimeError('로컬 라벨 모델 서버 응답 없음')
    with (OUT/'model-pull.log').open('a') as log:
        subprocess.run([str(runtime/'bin/ollama'),'pull',MODEL],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
    save(OUT/'teacher_model.json',dict(model=MODEL,server=request('/api/version'),metadata=request('/api/show',{'model':MODEL}),source='https://ollama.com/library/qwen3-vl:4b-instruct',status='unreviewed_AI_teacher_not_expert'))
    if process is not None:SERVER_PROCESS=process
    return process


def montage(row):
    """영상의 균등한 6시점을 3×2 이미지로 저장해 경로를 반환한다.

    영상 전체를 연속 시청한 결과가 아니다. 각 칸에 초 단위 시각을 표시한다.
    영상 열기·프레임 읽기 실패 시 ValueError를 발생시킨다.
    """
    path=OUT/'montages_v3'/(row['id']+'.jpg')
    if path.exists():return path
    path.parent.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(row['path']);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    if count<12 or fps<=0:cap.release();raise ValueError('영상 메타데이터 오류')
    canvas=np.zeros((2*224,3*256,3),dtype=np.uint8)
    try:
        for i,frame_index in enumerate(np.linspace(int(row.get('start_frame',0)),min(count-1,int(row.get('end_frame',count-1))),6).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_index));ok,frame=cap.read()
            if not ok:raise ValueError('라벨링 프레임 읽기 실패')
            h,w=frame.shape[:2];scale=min(256/w,200/h);w2,h2=max(1,int(w*scale)),max(1,int(h*scale));frame=cv2.resize(frame,(w2,h2))
            y=(i//3)*224;x=(i%3)*256
            canvas[y:y+h2,x:x+w2]=frame
            cv2.putText(canvas,f"{i+1}: {(frame_index-int(row.get('start_frame',0)))/fps:.2f}s",(x+4,y+216),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    finally:cap.release()
    if not cv2.imwrite(str(path),canvas):raise OSError('이미지 저장 실패')
    return path


def acceptable(label):
    """AI 초안이 학습 입력으로 쓸 최소 형식을 갖췄는지 반환한다. 정확도를 보증하지 않는다."""
    score=label.get('score');confidence=label.get('confidence')
    return (label.get('assessable') is True and isinstance(score,(int,float)) and not isinstance(score,bool)
        and math.isfinite(score) and 0<=score<=100 and isinstance(confidence,(int,float))
        and math.isfinite(confidence) and .65<=confidence<=1 and bool(label.get('evidence_en')))


def label_one(row, gate=None):
    """원본 정답을 보여주지 않고 시각 모델의 독립 초안을 저장해 반환한다."""
    path=OUT/'score_labels'/(row['id']+'.json')
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('eligible_for_experimental_training') or previous.get('quality_version')=='20261020-v3-exercise-specific':return previous
        archived=OUT/'previous_exclusions_v2';archived.mkdir(exist_ok=True)
        archive=archived/path.name
        if not archive.exists():path.rename(archive)
        else:path.unlink()

    if gate is not None:
        visibility=gate.check(row)
        if not visibility['visible']:
            value=dict(quality_version='20261020-v3-exercise-specific',id=row['id'],exercise=row['exercise'],split=row['split'],person=row['person'],assessable=False,score=None,confidence=0,observed_errors=[],review_status='unreviewed',eligible_for_experimental_training=False,visibility=visibility,reason='필수 신체 부위 또는 채점 촬영 방향 조건 부족. 자세 점수 생성 안 함.')
            save(path,value);return value
    else:
        visibility=None
    attempts_path=OUT/'label_attempts'/(row['id']+'.json')
    attempts=json.loads(attempts_path.read_text()).get('starts',0) if attempts_path.exists() else 0
    if attempts>=2:
        value=dict(quality_version='20261020-v3-exercise-specific',id=row['id'],exercise=row['exercise'],split=row['split'],person=row['person'],assessable=False,score=None,confidence=0,observed_errors=[],review_status='unreviewed',eligible_for_experimental_training=False,reason='반복 중단된 자동 라벨 요청을 제외함. 임의 점수 생성 안 함.')
        save(path,value);return value
    save(attempts_path,dict(starts=attempts+1))
    image=montage(row)
    policy=json.loads((OUT/'label_policy.json').read_text())
    prompt=f'''You are annotating an educational exercise dataset. These 6 panels are evenly sampled frames from ONE video, left-to-right then top-to-bottom. Timestamps are seconds. Target exercise: {NAMES[row['exercise']]}.
Inspect the visible body, movement range, alignment and control. Do not diagnose injury. Do not infer hidden joints. A missing visible error is not proof of perfect form. If the target action, essential joints or sufficient motion are not visible, set assessable=false and score=null. Plank is static; movement range is not required. YTW requires observing relevant arm positions; do not guess temporal speed from sparsely sampled frames. For shoulder press, do not confuse dumbbells resting by the side with a completed press. Do not score background people.
Give a holistic 0-100 quality DRAFT based on what is visible, not an error probability. Rubric: {json.dumps(policy['anchors'],ensure_ascii=False)}. Do not calculate a score by an angle threshold, source class or count of errors. Express honest confidence in visibility and interpretation, not self-certainty.
Return JSON only. Required keys: assessable (boolean), score (number or null), confidence (0..1), evidence_en (one concise English sentence describing specific visible body alignment or motion evidence), observed_errors (array, codes restricted to {ERRORS[row['exercise']]}), limitations_en (one concise English sentence). Include whether a full relevant motion is seen. Do not use generic descriptions like a person is moving. Do not invent errors to fill the list. If uncertain, leave error list empty and explain limitations.'''
    failures=[]
    for attempt in range(2):
        try:
            response=request('/api/generate',dict(model=MODEL,prompt=prompt,images=[base64.b64encode(image.read_bytes()).decode()],format='json',stream=False,think=False,keep_alive='10m',options=dict(temperature=0,num_ctx=2048,num_predict=320,seed=37)))
            value=json.loads(response['response'])
            if not isinstance(value.get('assessable'),bool):raise ValueError('판정 가능 여부 형식 오류')
            if not isinstance(value.get('observed_errors'),list) or any(x not in ERRORS[row['exercise']] for x in value['observed_errors']):raise ValueError('오류 코드 형식 오류')
            if value.get('score') is not None and (not isinstance(value['score'],(float,int)) or not 0<=value['score']<=100):raise ValueError('점수 범위 오류')
            value.update(quality_version='20261020-v3-exercise-specific',label_version='20261020-v3',visibility=visibility,id=row['id'],exercise=row['exercise'],split=row['split'],person=row['person'],teacher=MODEL,review_status='unreviewed',label_source='local_VLM_visual_draft',unit='clip_quality_0_to_100',montage_sha256=digest(image),sampled_frames=6,model_inference_seconds=response.get('total_duration',0)/1e9)
            value['eligible_for_experimental_training']=acceptable(value)
            save(path,value);return value
        except Exception as exc:
            failures.append(repr(exc))
            # 모델의 HTTP 실패 후 같은 정체된 서버에 요청을 계속 쌓지 않는다.
            if isinstance(exc,(OSError,TimeoutError)) and SERVER_PROCESS is not None:
                SERVER_PROCESS.terminate()
                try:SERVER_PROCESS.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    SERVER_PROCESS.kill();SERVER_PROCESS.wait(timeout=5)
                start_server()
            time.sleep(2)
    value=dict(id=row['id'],exercise=row['exercise'],assessable=False,score=None,confidence=0,evidence_ko='',observed_errors=[],review_status='unreviewed',eligible_for_experimental_training=False,failures=failures)
    save(path,value);return value


def label_all(candidates):
    """후보별 초안을 작성하고 오류·점수 라벨링 진행을 남긴다."""
    from quality import VisibilityGate
    labels={}
    counts={}
    gate=VisibilityGate()
    quotas={'train':42,'validation':14,'test':14}
    try:
        # 1차는 42/14/14를 목표로 하고, 2차는 부족한 합계를 다른 분할의 추가 자료로 채운다.
        # 같은 ID를 다시 세거나 같은 참여자의 분할을 옮기지 않는다.
        for phase in [1,2]:
            for row in candidates:
                key=(row['exercise'],row['split'])
                total=sum(n for (ex,sp),n in counts.items() if ex==row['exercise'])
                if row['id'] in labels:continue
                if phase==1 and counts.get(key,0)>=quotas[row['split']]:continue
                if phase==2 and total>=70:continue
                value=label_one(row,gate)
                labels[row['id']]=value
                if value.get('eligible_for_experimental_training'):counts[key]=counts.get(key,0)+1
                progress('score_AI_labeling',selection_pass=phase,attempted=len(labels),candidate_pool=len(candidates),eligible=sum(counts.values()),target=350,per_group={ex+'_'+sp:n for (ex,sp),n in counts.items()})
            if all(sum(n for (e,sp),n in counts.items() if e==ex)>=70 for ex in NAMES):break
    finally:
        gate.close()
    result=list(labels.values())
    save(OUT/'AI_labels.json',result)
    return result

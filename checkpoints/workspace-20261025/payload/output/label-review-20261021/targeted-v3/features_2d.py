"""관절의 화면상 기하 특징을 추가 추출한다. 채점 라벨이나 점수는 생성하지 않는다."""
import hashlib
import json
from pathlib import Path
import sys
import cv2
import mediapipe as mp
import numpy as np
OUT=Path(__file__).resolve().parent
BASE=Path('/home/jayden/SeSACPAI-Mid')
sys.path.insert(0,str(OUT))
from train import load
from runtime_guard import temperature_check


def summarize(sequence):
    """화면 좌표에서 관절각·상대 높이의 시간 요약을 계산한다.

    Args:
        sequence: N시점×33관절×2의 픽셀 좌표. 미관측은 NaN이다.
    Returns:
        tuple: 각도(도)·몸통 길이 대비 거리의 특징 배열과 이름 목록.
    """
    seq=np.asarray(sequence);channels={}
    for name,joints in [('elbow_left',(11,13,15)),('elbow_right',(12,14,16)),('hip_left',(11,23,25)),('hip_right',(12,24,26)),('knee_left',(23,25,27)),('knee_right',(24,26,28)),('body_line_left',(11,23,27)),('body_line_right',(12,24,28))]:
        a,b,c=[seq[:,j] for j in joints];u=a-b;v=c-b;den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1);cosine=np.divide(np.sum(u*v,axis=1),den,out=np.full(len(seq),np.nan),where=den>1e-6);channels[name]=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    torso=(np.linalg.norm(seq[:,11]-seq[:,23],axis=1)+np.linalg.norm(seq[:,12]-seq[:,24],axis=1))/2
    torso=np.where(torso>1,torso,np.nan)
    for side,hip,knee,ankle,toe in [('left',23,25,27,31),('right',24,26,28,32)]:
        channels['hip_knee_height_'+side]=(seq[:,knee,1]-seq[:,hip,1])/torso
        channels['hip_ankle_height_'+side]=(seq[:,ankle,1]-seq[:,hip,1])/torso
        channels['knee_toe_height_'+side]=(seq[:,toe,1]-seq[:,knee,1])/torso
        channels['hip_ankle_distance_'+side]=np.linalg.norm(seq[:,hip]-seq[:,ankle],axis=1)/torso
    values=[];names=[]
    for name,channel in channels.items():
        good=channel[np.isfinite(channel)];stats=[*np.percentile(good,[5,25,50,75,95]),float(np.std(good))] if good.size else [np.nan]*6
        for suffix,value in zip(['p05','p25','p50','p75','p95','std','coverage'],stats+[len(good)/len(seq)]):names.append('image_'+name+'_'+suffix);values.append(value)
    return np.array(values),names


def extract_one(row):
    """정해진 라벨 구간을 10FPS로 읽어 2D 관절과 요약 특징을 반환한다.

    Args:
        row: video_path와 시작·종료 프레임이 포함된 직접 검토 라벨.
    Returns:
        tuple: 특징 배열, 특징명, 시점별 좌표, 평균 관측률.
    Raises:
        ValueError: 영상 구간을 읽을 수 없는 경우.
    """
    cap=cv2.VideoCapture(row['video_path']);fps=cap.get(5);first=row['segment']['start_frame'];last=row['segment']['end_frame'];cap.set(cv2.CAP_PROP_POS_FRAMES,first)
    options=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(BASE/'mediapipe-pilot/models/heavy.task'),delegate=mp.tasks.BaseOptions.Delegate.GPU),running_mode=mp.tasks.vision.RunningMode.VIDEO,min_pose_detection_confidence=.3,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
    sequence=[];next_ms=0
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
            for j in range(last-first+1):
                if not cap.grab():raise ValueError('구간 디코딩 중단')
                timestamp=j*1000/fps
                if timestamp+1e-6<next_ms:continue
                ok,frame=cap.retrieve()
                if not ok:raise ValueError('장면 디코딩 실패')
                h,w=frame.shape[:2];scale=min(1,640/max(h,w));frame=cv2.resize(frame,(round(w*scale),round(h*scale)));h,w=frame.shape[:2]
                result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))),round(timestamp));points=np.full((33,2),np.nan)
                if result.pose_landmarks:
                    for i,p in enumerate(result.pose_landmarks[0]):
                        if p.visibility>=.3 and p.presence>=.5 and 0<=p.x<=1 and 0<=p.y<=1:points[i]=[p.x*w,p.y*h]
                sequence.append(points);next_ms+=100
    finally:cap.release()
    if not sequence:raise ValueError('관절 시점 없음')
    seq=np.asarray(sequence);x,names=summarize(seq);return x,names,seq,float(np.isfinite(seq).mean())


def extract():
    """학습·검증 자료만 추출하고 재개 가능한 캐시와 진행 기록을 저장한다.

    Returns:
        None: image_features/*.npz와 image_feature_report.json을 저장한다.
    Raises:
        RuntimeError: 고온 또는 센서 오류로 분석을 중단한 경우.
    """
    folder=OUT/'image_features';folder.mkdir(exist_ok=True);rows=[];report=[]
    for ex in ['pushups','squats']:
        for split in ['train','validation']:rows.extend(load(ex,split))
    for i,item in enumerate(rows,1):
        r=item['label'];temperature_check();p=folder/(r['id']+'.npz');fingerprint=hashlib.sha256(json.dumps(dict(hash=r['video_sha256'],segment=r['segment']),sort_keys=True).encode()).hexdigest()
        if p.exists():
            with np.load(p) as z:assert str(z['fingerprint'])==fingerprint;coverage=float(z['coverage'])
        else:
            x,names,seq,coverage=extract_one(r);np.savez_compressed(p,X=x,names=names,sequence=seq,coverage=coverage,fingerprint=fingerprint)
        report.append(dict(id=r['id'],coverage=coverage));(OUT/'image_feature_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(i,'/',len(rows),r['id'],round(coverage,2),flush=True)
    print('화면 기하 특징 추출 완료',len(rows),flush=True)


if __name__=='__main__':extract()

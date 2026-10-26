"""오류 분류 특징과 점수용 연속 관절 특징을 RTX Heavy로 추출한다."""
import json
from pathlib import Path
import sys
import time
import cv2
import mediapipe as mp
import numpy as np
from common import BASE, OUT, progress
sys.path.insert(0,str(BASE/'mediapipe-pilot/qevd_coach'))
import pose_features


def load_feature(path):
    """NPZ에서 특징 벡터를 반환한다. 관측 부족으로 빈 벡터인 경우 None을 반환한다."""
    with np.load(path) as z:
        value=z['X']
        return value if value.size else None


def extract_errors(items):
    """기존 12시점 캐시를 재사용하고 나머지 오류 특징을 추출한다.

    Args:
        items: 원본 라벨과 내장 SSD 영상 경로가 포함된 목록.
    Returns:
        정상 완료한 클립 수. 관측 부족도 기록하고 임의 보충하지 않는다.
    Raises:
        RuntimeError: NVIDIA EGL 추론 초기화 실패.
    """
    folder=OUT/'error_features';folder.mkdir(exist_ok=True)
    existing=BASE/'mediapipe-pilot/qevd_coach/artifacts_v2/heavy/features'
    failures=[];reused=0
    with pose_features.make_detector('heavy') as detector:
        for i,row in enumerate(items,1):
            target=folder/(row['id']+'.npz')
            if not target.exists():
                previous=existing/(row['original_id']+'.npz')
                if row['label_source'].startswith('QEVD') and previous.exists():
                    import shutil
                    shutil.copy2(previous,target);reused+=1
                else:
                    try:
                        x,sequence,coverage=pose_features.extract_clip(detector,Path(row['path']))
                        temp=target.with_suffix('.tmp.npz')
                        np.savez_compressed(temp,X=[] if x is None else x,sequence=sequence,coverage=coverage)
                        temp.replace(target)
                    except Exception as exc:
                        failures.append(dict(id=row['id'],error=repr(exc)))
            if i%100==0 or i==len(items):
                progress('error_feature_extraction',done=i,total=len(items),reused_this_run=reused,failed=len(failures))
    from common import save
    save(OUT/'error_feature_failures.json',failures)
    return len(items)-len(failures)


def dense_clip(row):
    """클립을 10 FPS로 읽어 32시점의 정규화 관절 특징으로 변환한다.

    Args:
        row: 영상 경로와 ID가 포함된 객체.
    Returns:
        특징 벡터 또는 None, 표본 수, 관측 비율. 결측은 NaN.
    Raises:
        ValueError: 영상 메타데이터 또는 읽기 실패.
    """
    cap=cv2.VideoCapture(row['path']);fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0 or count<12:cap.release();raise ValueError('영상 메타데이터 오류')
    opts=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(BASE/'mediapipe-pilot/models/heavy.task'),delegate=mp.tasks.BaseOptions.Delegate.GPU),running_mode=mp.tasks.vision.RunningMode.VIDEO,min_pose_detection_confidence=.3,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
    first=int(row.get('start_frame',0));last=min(count-1,int(row.get('end_frame',count-1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES,first)
    sequences=[];next_ts=0
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(opts) as detector:
            for index in range(last-first+1):
                if not cap.grab():break
                ts=index*1000/fps
                if ts+1e-6<next_ts:continue
                ok,frame=cap.retrieve()
                if not ok:break
                h,w=frame.shape[:2];scale=min(1,640/max(h,w));frame=cv2.resize(frame,(int(w*scale),int(h*scale)))
                image=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
                result=detector.detect_for_video(image,round(ts));world=np.full((33,3),np.nan,dtype=np.float32)
                if result.pose_landmarks:
                    local=result.pose_landmarks[0];points=np.array([[p.x,p.y,p.z] for p in result.pose_world_landmarks[0]])
                    if all(local[k].visibility>=.3 and local[k].presence>=.5 for k in [11,12,23,24]):
                        length=(np.linalg.norm(points[11]-points[23])+np.linalg.norm(points[12]-points[24]))/2
                        if length>.01:
                            norm=(points-(points[23]+points[24])/2)/length
                            for k,p in enumerate(local):
                                if p.visibility>=.3 and p.presence>=.5:world[k]=norm[k]
                sequences.append(world[pose_features.BODY]);next_ts+=100
    finally:cap.release()
    seq=np.asarray(sequences);coverage=float(np.isfinite(seq).mean()) if seq.size else 0
    if len(seq)<12 or coverage<.35:return None,len(seq),coverage
    sampled=seq[np.linspace(0,len(seq)-1,32).astype(int)].reshape(32,-1)
    # 관측값으로만 통계를 계산하고 관측이 없는 채널은 NaN으로 유지한다.
    stats=[]
    for col in sampled.T:
        good=col[np.isfinite(col)]
        stats.extend([float(np.mean(good)),float(np.std(good)),float(np.min(good)),float(np.max(good))] if good.size else [np.nan]*4)
    return np.concatenate([sampled.ravel(),stats]),len(seq),coverage


def extract_scores(candidates, labels):
    """학습 가능한 AI 점수 초안에 연속 특징을 추가하고 재개 가능한 NPZ를 저장한다."""
    eligible={x['id'] for x in labels if x.get('eligible_for_experimental_training')}
    rows=[r for r in candidates if r['id'] in eligible]
    folder=OUT/'score_features';folder.mkdir(exist_ok=True)
    for i,row in enumerate(rows,1):
        target=folder/(row['id']+'.npz')
        if not target.exists():
            x,n,cov=dense_clip(row)
            temp=target.with_suffix('.tmp.npz');np.savez_compressed(temp,X=[] if x is None else x,sampled_frames=n,coverage=cov);temp.replace(target)
        progress('score_feature_extraction',done=i,total=len(rows))

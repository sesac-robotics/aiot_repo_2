"""선택한 한 반복 구간을 모델로 평가하고 관절 표시 영상을 저장한다."""
import json
import math
from pathlib import Path
import subprocess
import sys
import cv2
import mediapipe as mp
import numpy as np
from predict import predict
BASE=Path('/home/jayden/SeSACPAI-Mid')
sys.path.insert(0,str(BASE/'overnight-posture-pipeline'))
import features
import safety_guard
EDGES=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]


from runtime_guard import temperature_check


def overlay(source,folder,first,last,fps,result):
    """영상의 선택 구간에 추정 관절과 모델 항목 점수를 표시한다.

    Args:
        source: 로컬 원본 영상 경로.
        folder: 결과를 저장할 폴더.
        first: 시작 프레임 번호.
        last: 종료 프레임 번호.
        fps: 원본 초당 프레임 수.
        result: 종합 점수가 없는 항목별 모델 결과.
    Returns:
        None: 소리 없는 H.264 overlay.mp4를 저장한다.
    Raises:
        RuntimeError: 고온·디코딩·인코딩 오류가 발생한 경우.
    """
    cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_POS_FRAMES,first);stride=max(1,math.ceil(fps/10));writer=None;raw=folder/'overlay.avi'
    options=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(BASE/'mediapipe-pilot/models/heavy.task'),delegate=mp.tasks.BaseOptions.Delegate.GPU),running_mode=mp.tasks.vision.RunningMode.VIDEO)
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
            for j in range(last-first+1):
                ok,frame=cap.read()
                if not ok:raise RuntimeError('영상 구간 디코딩 실패')
                if j%stride:continue
                if j%(stride*30)==0:temperature_check()
                h,w=frame.shape[:2];scale=min(1,640/max(h,w));w=max(2,int(w*scale)//2*2);h=max(2,int(h*scale)//2*2);frame=cv2.resize(frame,(w,h))
                pose=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))),round(j*1000/fps));points={}
                if pose.pose_landmarks:
                    points={i:(round(p.x*w),round(p.y*h)) for i,p in enumerate(pose.pose_landmarks[0]) if p.visibility>=.3 and p.presence>=.5 and 0<=p.x<=1 and 0<=p.y<=1}
                for a,b in EDGES:
                    if a in points and b in points:cv2.line(frame,points[a],points[b],(80,230,70),2)
                for i in {n for e in EDGES for n in e}:
                    if i in points:cv2.circle(frame,points[i],3,(0,220,255),-1)
                text=['MODEL ITEM DRAFT - NOT TOTAL SCORE']+[f"{item['id']}: {item['predicted_points']:.1f}/{item['weight']}" if item['predicted_points'] is not None else (f"{item['id']}: DIAG {item['diagnostic_grade']:.2f}/2 - WITHHELD" if item.get('diagnostic_grade') is not None else f"{item['id']}: WITHHELD") for item in result['items']]
                for line,value in enumerate(text):cv2.putText(frame,value,(8,18+line*17),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
                if writer is None:writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'MJPG'),fps/stride,(w,h))
                if not writer.isOpened():raise RuntimeError('관절 영상 기록 실패')
                writer.write(frame)
    finally:
        cap.release()
        if writer is not None:writer.release()
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(raw),'-an','-c:v','libx264','-threads','2','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'overlay.mp4')],check=True,timeout=45)
    raw.unlink()


def run(request_path):
    """저장 요청의 한 반복 구간을 분석한다.

    Args:
        request_path: 원본·운동·초 단위 구간을 담은 JSON 파일 경로.
    Returns:
        None: result.json과 관절 표시 영상을 저장한다.
    Raises:
        ValueError: 구간이 영상 밖이거나 1.2~15초 범위를 벗어난 경우.
    """
    request_path=Path(request_path);r=json.loads(request_path.read_text());source=Path(r['source']);folder=request_path.parent;cap=cv2.VideoCapture(str(source));fps=cap.get(5);count=int(cap.get(7));cap.release()
    start=float(r['start']);end=float(r['end'])
    if not fps>0 or not math.isfinite(start+end) or not 0<=start<end<=count/fps or not 1.2<=end-start<=15:raise ValueError('영상 범위 안에서 1.2~15초의 한 반복 구간을 지정하세요.')
    first=round(start*fps);last=min(count-1,round(end*fps));reading=temperature_check();x,samples,coverage=features.dense_clip(dict(path=str(source),id=folder.name,start_frame=first,end_frame=last))
    if x is None:raise ValueError('관절 관측이 부족합니다. 전신이 잘 보이는 구간을 선택하세요.')
    image_vector=None
    if (folder.parent.parent/'active_model.json').exists():
        from features_2d import extract_one
        image_vector,_,_,_=extract_one(dict(video_path=str(source),segment=dict(start_frame=first,end_frame=last)))
    result=predict(x,r['exercise'],image_vector=image_vector,diagnostics=r.get('diagnostics',False));result.update(segment=dict(start=start,end=end),samples=samples,coverage=coverage,temperature_before=reading,source_job=r['source_job'])
    overlay(source,folder,first,last,fps,result);(folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('모델 항목 분석 완료',flush=True)


if __name__=='__main__':run(sys.argv[1])

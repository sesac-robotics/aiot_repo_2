"""VS Code F5: choose exercise and saved video/ESP32-CAM URL for trained-model inference."""
from pathlib import Path
import argparse,json,time
import cv2,numpy as np
from suite_inference import FiveExerciseClassifier

def main():
    p=argparse.ArgumentParser();p.add_argument('--exercise',choices=['squats','pushups','elbow_plank','ytw','shoulder_press'],required=True);p.add_argument('--source',required=True);p.add_argument('--no-window',action='store_true');p.add_argument('--max-frames',type=int,default=0);a=p.parse_args()
    cap=cv2.VideoCapture(int(a.source)) if a.source.isdigit() else cv2.VideoCapture(a.source,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    if not cap.isOpened():raise ValueError('Cannot open supplied source')
    dispatcher=FiveExerciseClassifier();model=dispatcher.model(a.exercise);fps=cap.get(cv2.CAP_PROP_FPS);is_file=Path(a.source).is_file();i=0;latest={}
    if is_file and fps<=0:raise ValueError('Invalid source FPS')
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            result=model.push_bgr_frame(frame,i/fps if is_file else time.monotonic());i+=1
            if result:
                latest=result
                if result['status']=='ok':print(json.dumps(result),flush=True)
            if not a.no_window:
                label=', '.join(latest.get('detected_errors',[])) or latest.get('status','warming_up')
                cv2.putText(frame,label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,200,255),2);cv2.imshow('Exercise errors - experimental',frame)
                if cv2.waitKey(1)&255==ord('q'):break
            if a.max_frames and i>=a.max_frames:break
        if is_file and latest.get('status')=='warming_up' and len(model.window)>=8:
            latest=model.predict_landmarks(np.array(model.window),model.aspect);latest['partial_window_at_file_end']=True
        print(json.dumps(dict(processed_frames=i,last_result=latest)),flush=True)
    finally:cap.release();dispatcher.close();cv2.destroyAllWindows()

if __name__=='__main__':main()

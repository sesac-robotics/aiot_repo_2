"""Classify a saved video, webcam, or explicit ESP32-CAM stream URL.
Example: predict.py --source /path/to/video.mp4 --no-window
"""
from pathlib import Path
import argparse,json,time
import cv2
import numpy as np
from inference import PostureErrorClassifier

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--no-window',action='store_true');parser.add_argument('--max-frames',type=int,default=0);args=parser.parse_args()
    cap=cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():raise RuntimeError('Cannot open supplied video source')
    classifier=PostureErrorClassifier();index=0;fps=cap.get(cv2.CAP_PROP_FPS);is_file=Path(args.source).is_file();latest={}
    if is_file and fps<=0:raise RuntimeError('Video file has invalid FPS')
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            stamp=index/fps if is_file else time.monotonic()
            result=classifier.push_bgr_frame(frame,stamp);index+=1
            if result:
                latest=result
                if result['status']=='ok':print(json.dumps(result),flush=True)
            if not args.no_window:
                cv2.putText(frame,', '.join(latest.get('detected_errors',[])) or latest.get('status','warming_up'),(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,200,255),2)
                cv2.imshow('Squat posture errors - experimental',frame)
                if cv2.waitKey(1)&255==ord('q'):break
            if args.max_frames and index>=args.max_frames:break
        if is_file and latest.get('status')=='warming_up' and len(classifier.window)>=8:
            latest=classifier.predict_landmarks(np.asarray(classifier.window),classifier.aspect)
            latest['partial_window_at_file_end']=True
            print(json.dumps(latest),flush=True)
        print(json.dumps(dict(processed_frames=index,last_result=latest)),flush=True)
    finally:cap.release();classifier.close();cv2.destroyAllWindows()

if __name__=='__main__':main()

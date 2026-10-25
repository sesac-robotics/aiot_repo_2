from pathlib import Path
import json,cv2,numpy as np,mediapipe as mp
O=Path(__file__).resolve().parent;B=O.parents[1];receipts=[]
for path in sorted((O/'cfrep').glob('*.mp4')):
 dest=O/(path.stem+'-heavy.npz')
 if dest.exists():continue
 cap=cv2.VideoCapture(str(path));fps=cap.get(5);w=int(cap.get(3));h=int(cap.get(4));N=int(cap.get(7));points=[];frames=[]
 opts=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(B/'mediapipe-pilot/models/heavy.task')),running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=1)
 with mp.tasks.vision.PoseLandmarker.create_from_options(opts) as pose:
  for f in range(N):
   ok,bgr=cap.read()
   if not ok:break
   if f%3:continue
   result=pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)),round(f/fps*1000));xy=np.full((33,2),np.nan)
   if result.pose_landmarks:
    for j,p in enumerate(result.pose_landmarks[0]):
     if p.visibility>=.5 and p.presence>=.5:xy[j]=[p.x*w,p.y*h]
   points.append(xy);frames.append(f)
 cap.release();np.savez_compressed(dest,image_xy=np.array(points),frame_numbers=frames,fps=fps,width=w,height=h);print(path.stem,len(frames),flush=True)

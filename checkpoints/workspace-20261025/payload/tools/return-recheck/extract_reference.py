import cv2,numpy as np,mediapipe as mp,json,sys
from pathlib import Path
B=Path(__file__).resolve().parents[2];O=B/'output/return-recheck-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'));from common import sha
p=B/'output/squat-return-20261023/cfrep/squat_front_m4_6_6.mp4';model=B/'mediapipe-pilot/models/heavy.task';opts=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_poses=1);xs=[];c=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);width=int(c.get(3));height=int(c.get(4))
with mp.tasks.vision.PoseLandmarker.create_from_options(opts) as detector:
 for f in range(3):
  ok,a=c.read();assert ok;r=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(a,cv2.COLOR_BGR2RGB)));xy=np.full((33,2),np.nan)
  if r.pose_landmarks:
   for j,l in enumerate(r.pose_landmarks[0]):
    if l.visibility>=.6 and l.presence>=.6 and 0<=l.x<1 and 0<=l.y<1:xy[j]=[l.x*width,l.y*height]
  xs.append(xy)
c.release();np.savez_compressed(O/'m4-front-initial-reference.npz',image_xy=np.array(xs),frame_numbers=np.arange(3),width=width,height=height);(O/'m4-front-reference-receipt.json').write_text(json.dumps(dict(source=str(p),source_sha256=sha(p),frames=[0,1,2],model_sha256=sha(model),delegate='CPU',visibility_min=.6,presence_min=.6,imputed=False,split='train'),indent=2));print('EXTRACTED3actual reference frames')

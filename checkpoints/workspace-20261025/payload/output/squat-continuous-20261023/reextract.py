from pathlib import Path
import json,cv2,numpy as np,mediapipe as mp
O=Path(__file__).resolve().parent;R=O.parents[1];S=O.parent/'squat-relabel-expanded-20261023'
opts=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(R/'mediapipe-pilot/models/heavy.task')),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_poses=1)
rows=json.loads((S/'source-manifest.json').read_text());out=[]
with mp.tasks.vision.PoseLandmarker.create_from_options(opts) as d:
 for ident in ['Vasil_01','Vasil_02','Vasil_08']:
  row=next(r for r in rows if r['id']==ident);z=np.load(S/'observations'/ident/'front.npz');cap=cv2.VideoCapture(row['views']['front']['path']);new=[]
  for fn,old in zip(z['frame_numbers'],z['image_xy']):
   cap.set(0 if False else cv2.CAP_PROP_POS_FRAMES,int(fn));ok,bgr=cap.read();assert ok
   pred=d.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
   xy=np.array([[p.x*int(z['width']),p.y*int(z['height'])] for p in pred.pose_landmarks[0]]) if pred.pose_landmarks else np.full((33,2),np.nan);new.append(xy)
  new=np.array(new);np.savez(O/(ident+'-image.npz'),image_xy=new,frame_numbers=z['frame_numbers'],width=z['width'],height=z['height']);delta=np.linalg.norm(new[:,[25,26]]-z['image_xy'][:,[25,26]],axis=2)
  out.append(dict(id=ident,n=len(new),knee_delta_median_px=float(np.nanmedian(delta)),knee_delta_p90_px=float(np.nanpercentile(delta,90)),note='검출 방식 간 차이이며 정답 대비 위치 오차는 아님'))
  cap.release()
(O/'extraction-comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(out)

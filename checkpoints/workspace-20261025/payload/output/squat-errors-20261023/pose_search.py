from pathlib import Path
import json,random,cv2,numpy as np,mediapipe as mp
O=Path(__file__).resolve().parent;R=O.parents[1];P=R/'output/squat-priority-20261023';r=json.loads((P/'remaining-review.json').read_text())['remaining'];seen={x['id'] for x in json.loads((O/'selected.json').read_text())};seen|={x['id'] for p in [P/'pose-screen.json',P/'new-people/pose-screen.json'] for x in json.loads(p.read_text())};protected=set(json.loads((R/'output/five-exercise-recovery-20261023/final_test_reservation.json').read_text())['reserved_people'])|{'qevd_'+str(x) for x in json.loads((R/'data/qevd-selected/manifests/official_test_workers.json').read_text())};r=[x for x in r if x['id'] not in seen and x['person'] not in protected and not any(t in x['provider_hints'] for t in ['squats - plie','squats - hold','squats - not visible'])];random.Random(1009).shuffle(r);chosen=[];counts={}
for x in r:
 if counts.get(x['person'],0)>=3:continue
 chosen.append(x);counts[x['person']]=counts.get(x['person'],0)+1
 if len(chosen)==512:break
opts=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(R/'mediapipe-pilot/models/lite.task')),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_poses=1);results=[]
with mp.tasks.vision.PoseLandmarker.create_from_options(opts) as detector:
 for j,row in enumerate(chosen):
  c=cv2.VideoCapture(row['path']);n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));w=c.get(cv2.CAP_PROP_FRAME_WIDTH);h=c.get(cv2.CAP_PROP_FRAME_HEIGHT);fps=c.get(cv2.CAP_PROP_FPS);frames=np.linspace(0,n-1,12,dtype=int).tolist();points=[]
  for f in frames:
   c.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=c.read();xy=np.full((33,2),np.nan)
   if ok:
    det=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
    if det.pose_landmarks:
     for k,l in enumerate(det.pose_landmarks[0]):
      if l.visibility>=.6 and l.presence>=.6 and 0<=l.x<1 and 0<=l.y<1:xy[k]=[l.x*w,l.y*h]
   points.append(xy)
  c.release();x=np.array(points);valid=np.isfinite(x[:,[11,12,23,24]]).all((1,2));knees=valid&np.isfinite(x[:,[25,26,27,28]]).all((1,2));out=dict(row,frame_count=n,fps=fps,review_frames=frames,torso_coverage=float(valid.mean()),ankle_knee_coverage=float(knees.mean()),not_a_label=True,trunk_signal=None,knee_signal=None,return_signal=None)
  if valid.sum()>=8:
   a=x[valid];s=a[:,[11,12]].mean(1);hip=a[:,[23,24]].mean(1);delta=s-hip;ang=np.degrees(np.arctan2(delta[:,0],-delta[:,1]));out['trunk_signal']=float(np.percentile(abs(ang-np.median(ang)),90));height=hip[:,1];baseline=np.min(height);span=np.ptp(height);out['return_signal']=float((height[-1]-baseline)/span) if span>10 else None
  if knees.sum()>=8:
   a=x[knees];gap=abs(a[:,25,0]-a[:,26,0])/np.maximum(10,abs(a[:,27,0]-a[:,28,0]));out['knee_signal']=float(np.percentile(gap,15))
  results.append(out)
  if (j+1)%32==0:(O/'pose-search.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(j+1,flush=True)
(O/'pose-search.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print('done',len(results),flush=True)

from pathlib import Path
import json,random,cv2,numpy as np,mediapipe as mp
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent;U=R/'output/squat-unified-20261023';old=json.loads((U/'labels.aligned.frozen.json').read_text())['rows'];allowed={r['person'] for r in old if r['split']=='train'}-{r['person'] for r in old if r['split']=='validation'};seen={r['id'] for r in json.loads((O/'preparation.json').read_text())['selected']}
rs=[r for r in json.loads((U/'review-queue.json').read_text()) if r['source']=='QEVD' and r.get('person') in allowed and r['id'] not in seen and not any('plie' in l or 'hold' in l or 'not visible' in l for l in r['provider_hints'])];random.Random(941).shuffle(rs)
# Diversity-first cap; rank different movements without assigning clinical/error labels.
chosen=[];counts={}
for r in rs:
 if counts.get(r['person'],0)>=8:continue
 chosen.append(r);counts[r['person']]=counts.get(r['person'],0)+1
 if len(chosen)>=192:break
options=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(R/'mediapipe-pilot/models/lite.task')),running_mode=mp.tasks.vision.RunningMode.IMAGE,num_poses=1)
output=[]
with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
 for j,r in enumerate(chosen):
  c=cv2.VideoCapture(r['path']);n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));fps=c.get(cv2.CAP_PROP_FPS);coords=[];frames=np.linspace(0,n-1,9,dtype=int).tolist()
  for f in frames:
   c.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=c.read();xy=np.full((33,2),np.nan)
   if ok:
    im=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));det=detector.detect(im)
    if det.pose_landmarks:
     for k,l in enumerate(det.pose_landmarks[0]):
      if l.visibility>=.5 and l.presence>=.5:xy[k]=[l.x,l.y]
   coords.append(xy)
  c.release();x=np.array(coords);torso=np.isfinite(x[:,[11,12,23,24]]).all((1,2));knee=torso&np.isfinite(x[:,[25,26]]).all((1,2));out=dict(r,frames=frames,frame_count=n,fps=fps,torso_fraction=float(torso.mean()),knee_fraction=float(knee.mean()),knee_gap_change=None,trunk_change=None,not_a_label=True)
  if torso.sum()>=5:
   a=x[torso];sh=a[:,[11,12]].mean(1);hip=a[:,[23,24]].mean(1);angle=np.degrees(np.arctan2(hip[:,0]-sh[:,0],hip[:,1]-sh[:,1]));out['trunk_change']=float(np.percentile(abs(angle-np.median(angle[:2])),90))
  if knee.sum()>=5:
   a=x[knee];ratio=abs(a[:,26,0]-a[:,25,0])/np.maximum(.005,abs(a[:,24,0]-a[:,23,0]));base=np.median(ratio[:2]);out['knee_gap_change']=float(max(0,1-np.percentile(ratio/max(.01,base),10)))
  output.append(out)
  if (j+1)%16==0:print(j+1,flush=True);(O/'pose-screen.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
(O/'pose-screen.json').write_text(json.dumps(output,ensure_ascii=False,indent=2));print('done',len(output),flush=True)

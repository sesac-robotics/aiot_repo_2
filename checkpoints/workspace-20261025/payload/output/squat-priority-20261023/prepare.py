from pathlib import Path
import json,cv2,numpy as np,collections,hashlib
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent;U=R/'output/squat-unified-20261023'
queue=[r for r in json.loads((U/'review-queue.json').read_text()) if r['source']=='QEVD'];old=json.loads((U/'labels.aligned.frozen.json').read_text())['rows'];allowed={r['person'] for r in old if r['split']=='train'}-{r['person'] for r in old if r['split']=='validation'}
buckets=collections.defaultdict(list)
for r in queue:
 if r.get('person') not in allowed:continue
 labels=r.get('provider_hints',[])
 if any('plie' in x or 'hold' in x or 'not visible' in x for x in labels):continue
 bucket='return' if any('insufficient' in x for x in labels) else 'trunk' if any('back not straight' in x for x in labels) else 'knee' if any('knees over toes' in x or 'narrow' in x for x in labels) else 'normal' if any('no obvious issue' in x for x in labels) else None
 if bucket:buckets[bucket].append(r)
# Pixel change is a source-selection heuristic only; no pose/model/rule scores used.
scanned=[];selected=[];used=set()
for bucket in ['return','trunk','knee','normal']:
 choices=[];perperson=collections.Counter()
 for r in buckets[bucket]:
  if perperson[r['person']]>=2:continue
  cap=cv2.VideoCapture(r['path']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);imgs=[]
  for f in np.linspace(0,max(0,n-1),9,dtype=int):
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(f));ok,im=cap.read()
   if ok:imgs.append(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(96,72)))
  cap.release()
  if len(imgs)<9:continue
  v=float(np.median([np.abs(a.astype(float)-b).mean() for a,b in zip(imgs,imgs[1:])]))
  entry=dict(r,bucket=bucket,frame_count=n,fps=fps,motion_proxy=v);scanned.append(entry);choices.append(entry);perperson[r['person']]+=1
  if len(choices)>=36:break
 people=set()
 for r in sorted(choices,key=lambda r:r['motion_proxy'],reverse=True):
  if r['person'] in people or r['id'] in used:continue
  selected.append(r);people.add(r['person']);used.add(r['id'])
  if len(people)>=8:break
for page in range((len(selected)+3)//4):
 canvas=Image.new('RGB',(8*190,4*240),'#edf3f5');draw=ImageDraw.Draw(canvas)
 for j,r in enumerate(selected[page*4:page*4+4]):
  c=cv2.VideoCapture(r['path']);frames=np.linspace(0,r['frame_count']-1,8,dtype=int).tolist();r['review_frames']=frames;r['sheet']=str(O/f'screen-{page:02}.jpg')
  for k,n in enumerate(frames):
   c.set(cv2.CAP_PROP_POS_FRAMES,n);ok,bgr=c.read()
   if not ok:continue
   im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((186,206));x=k*190;y=j*240;canvas.paste(im,(x,y+30));draw.text((x+3,y+3),r['id']+' f'+str(n),fill='black')
  c.release()
 canvas.save(O/f'screen-{page:02}.jpg')
(O/'preparation.json').write_text(json.dumps(dict(rubric_sha256=hashlib.sha256((U/'rubric.json').read_bytes()).hexdigest(),scanned=scanned,selected=selected,selection_uses_no_pose_score_or_target=True,people_role='existing development people only; no reserved final video opened'),ensure_ascii=False,indent=2));print(len(scanned),len(selected),flush=True)

from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
O=Path(__file__).parent/'new-people';data=json.loads((O/'pose-screen.json').read_text());chosen=[];seen=set()
for key in ['trunk_change','knee_gap_change']:
 people={}
 for r in sorted([r for r in data if r.get(key) is not None],key=lambda r:r[key],reverse=True):
  if r['id'] in seen or people.get(r['person'],0)>=2:continue
  chosen.append(dict(r,ranked_by=key));seen.add(r['id']);people[r['person']]=people.get(r['person'],0)+1
  if sum(x['ranked_by']==key for x in chosen)>=16:break
for page in range((len(chosen)+3)//4):
 canvas=Image.new('RGB',(8*190,4*240),'#edf3f5');d=ImageDraw.Draw(canvas)
 for j,r in enumerate(chosen[page*4:page*4+4]):
  cap=cv2.VideoCapture(r['path']);frames=np.linspace(0,r['frame_count']-1,8,dtype=int).tolist();r['review_frames']=frames;r['sheet']=str(O/f'ranked-{page:02}.jpg')
  for k,n in enumerate(frames):
   cap.set(cv2.CAP_PROP_POS_FRAMES,n);ok,bgr=cap.read()
   if not ok:continue
   im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((186,206));x=k*190;y=j*240;canvas.paste(im,(x,y+30));d.text((x+3,y+3),r['id']+' f'+str(n),fill='black')
  cap.release()
 canvas.save(O/f'ranked-{page:02}.jpg')
(O/'ranked-preparation.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2));print(len(data),len(chosen))

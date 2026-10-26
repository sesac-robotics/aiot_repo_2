from pathlib import Path
import json,random,cv2,numpy as np
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent;R=O.parents[1]
r=json.loads((R/'output/squat-priority-20261023/remaining-review.json').read_text())['remaining'];reserved=set(json.loads((R/'output/five-exercise-recovery-20261023/final_test_reservation.json').read_text())['reserved_people']);reserved|={'qevd_'+str(x) for x in json.loads((R/'data/qevd-selected/manifests/official_test_workers.json').read_text())};r=[x for x in r if x['person'] not in reserved and not any(t in x['provider_hints'] for t in ['squats - plie','squats - hold','squats - not visible'])];random.Random(301).shuffle(r);selected=[];seen=set();people={}
for tag in ['squats - insufficient','squats - back not straight','squats - narrow']:
 n=0
 for x in r:
  if tag not in x['provider_hints'] or x['id'] in seen or people.get(x['person'],0)>=2:continue
  selected.append(dict(x,selection_hint=tag));seen.add(x['id']);people[x['person']]=people.get(x['person'],0)+1;n+=1
  if n==32:break
for page in range((len(selected)+3)//4):
 canvas=Image.new('RGB',(8*200,4*250),'#eef3f5');d=ImageDraw.Draw(canvas)
 for j,x in enumerate(selected[page*4:page*4+4]):
  c=cv2.VideoCapture(x['path']);n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));x['frame_count']=n;x['fps']=c.get(cv2.CAP_PROP_FPS);x['review_frames']=np.linspace(0,n-1,8,dtype=int).tolist();x['sheet']=str(O/f'screen-{page:02}.jpg')
  for k,f in enumerate(x['review_frames']):
   c.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=c.read()
   if not ok:continue
   im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((196,218));px=k*200;py=j*250;canvas.paste(im,(px,py+28));d.text((px+3,py+3),x['id']+' f'+str(f),fill='black')
  c.release()
 canvas.save(O/f'screen-{page:02}.jpg')
(O/'selected.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2));print(len(selected))

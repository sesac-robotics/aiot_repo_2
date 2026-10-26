from pathlib import Path
import json,cv2,numpy as np,sys
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent;data=json.loads((O/'pose-search.json').read_text());chosen=[];seen=set();prefix=sys.argv[1] if len(sys.argv)>1 else 'ranked'
if prefix!='ranked':seen={r['id'] for r in json.loads((O/'ranked.json').read_text())}
for key in ['trunk_signal','knee_signal']:
 n=0;people={}
 for r in sorted([r for r in data if r.get(key) is not None],key=lambda r:r[key],reverse=key!='knee_signal'):
  if r['id'] in seen or people.get(r['person'],0)>=1:continue
  chosen.append(dict(r,ranked_by=key));seen.add(r['id']);people[r['person']]=people.get(r['person'],0)+1;n+=1
  if n==16:break
for page in range((len(chosen)+3)//4):
 canvas=Image.new('RGB',(8*200,4*250),'#eef3f5');d=ImageDraw.Draw(canvas)
 for j,r in enumerate(chosen[page*4:page*4+4]):
  c=cv2.VideoCapture(r['path']);frames=np.linspace(0,r['frame_count']-1,8,dtype=int).tolist();r['review_frames']=frames;r['sheet']=str(O/f'{prefix}-{page:02}.jpg')
  for k,f in enumerate(frames):
   c.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=c.read()
   if not ok:continue
   im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((196,218));x=k*200;y=j*250;canvas.paste(im,(x,y+28));d.text((x+3,y+3),r['id']+' f'+str(f),fill='black')
  c.release()
 canvas.save(O/f'{prefix}-{page:02}.jpg')
(O/f'{prefix}.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2));print(len(data),len(chosen))

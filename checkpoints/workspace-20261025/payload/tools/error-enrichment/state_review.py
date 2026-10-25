from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024';rows=[]
inv={r['id']:r for r in json.loads((B/'output/r2d2-review-20261024/inventory.json').read_text())};labs=json.loads((B/'data/r2d2-team-20261023/annotations/labels.frozen.json').read_text())['labels']
for vid in ['V01','V02','V06','V07','V10','V12','V13','V17','V18']:
 r=inv[vid];p=next(x['person'] for x in labs if x['source_id']==vid);rows.append(dict(id=vid,path=r['copy'],person='r2d2_'+p,split='validation' if p=='P04' else 'train',times=[t for t in np.arange(0,4.01,.5) if t<r['duration']-.05]))
prior=json.loads((B/'output/squat-return-20261023/labels.frozen.json').read_text())['labels'];seen=set()
for r in prior:
 if r['person'] in seen:continue
 seen.add(r['person']);rows.append(dict(id=Path(r['path']).stem,path=r['path'],person=r['person'],split=r['split'],times=np.arange(0,4.01,.5).tolist()))
for i,r in enumerate(rows):
 c=cv2.VideoCapture(r['path']);fps=c.get(5);r['fps']=fps
 if i%3==0:canvas=Image.new('RGB',(9*180,3*300),'#dddddd')
 for j,t in enumerate(r['times']):
  c.set(1,round(t*fps));ok,a=c.read()
  if not ok:raise ValueError(r['id'])
  im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));
  if im.width/im.height>1.6:
   # full source remains available, crop centered to show posture detail
   w=int(im.height);x=(im.width-w)//2;im=im.crop((x,0,x+w,im.height))
  im.thumbnail((180,265));x=j*180;y=(i%3)*300;canvas.paste(im,(x+(180-im.width)//2,y+30));ImageDraw.Draw(canvas).text((x+2,y+3),r['id']+' '+str(t),fill='black')
 c.release();r['sheet']=str(O/f'state-review-{i//3+1:02}.jpg');canvas.save(r['sheet'],quality=90)
(O/'state-review-manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(len(rows))

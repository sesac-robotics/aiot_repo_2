from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/label-recheck-20261024';rows=json.loads((O/'new-intake.json').read_text())
for r in rows:
 if r['exercise']!='ytw':continue
 c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);times=np.arange(0,r['duration']-.06,.25);canvas=Image.new('RGB',(6*180,((len(times)+5)//6)*250),'white')
 for k,t in enumerate(times):
  c.set(1,round(t*r['fps']));ok,a=c.read();assert ok
  h,w=a.shape[:2]
  if w>h*1.3:
   x=(w-h)//2;a=a[:,x:x+h]
  tile=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));tile.thumbnail((180,225));x=(k%6)*180;y=(k//6)*250;canvas.paste(tile,(x+(180-tile.width)//2,y+23));ImageDraw.Draw(canvas).text((x+2,y+3),r['review_id']+' '+str(round(t,2)),fill='black')
 c.release();canvas.save(O/(r['review_id']+'-dense.jpg'),quality=92)
print('NEW_DENSE_COMPLETE',flush=True)

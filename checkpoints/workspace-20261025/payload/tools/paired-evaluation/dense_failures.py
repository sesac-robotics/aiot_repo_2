import json,cv2,numpy as np
from PIL import Image,ImageDraw
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/paired-failure-mining-20261024';rows=json.loads((O/'intake-v2.frozen.json').read_text())['records']+json.loads((O/'upper-intake.frozen.json').read_text())
for r in rows:
 if r['exercise']=='shoulder':a,b=6.5,15.5
 else:a,b=0,r['duration']-.08
 times=np.arange(a,min(b,r['duration']-.05),.25);c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);canvas=Image.new('RGB',(8*190,((len(times)+7)//8)*190),'white')
 for k,t in enumerate(times):
  c.set(1,round(t*r['fps']));ok,im=c.read();assert ok;tile=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));tile.thumbnail((190,165));x=k%8*190;y=k//8*190;canvas.paste(tile,(x+(190-tile.width)//2,y+22));ImageDraw.Draw(canvas).text((x+3,y+3),r['review_id']+f' {t:.2f}s',fill='black')
 c.release();canvas.save(O/(r['review_id']+'-dense.jpg'),quality=92);print(r['review_id'],len(times),flush=True)

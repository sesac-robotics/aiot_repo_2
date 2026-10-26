import json,cv2,numpy as np
from PIL import Image,ImageDraw
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/label-verified-five-expanded-20261024';rows=json.loads((B/'output/label-recheck-20261024/new-intake.json').read_text());done=[]
for r in rows:
 p=O/'observations'/r['review_id']/'primary.npz'
 if not p.exists():continue
 z=np.load(p);xy=z['image_xy'];ns=z['frame_numbers'];c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);canvas=Image.new('RGB',(960,360),'white')
 for col,i in enumerate(np.linspace(0,len(ns)-1,3).round().astype(int)):
  c.set(1,int(ns[i]));ok,im=c.read();assert ok
  for point in xy[i]:
   if np.isfinite(point).all():cv2.circle(im,tuple(point.astype(int)),3,(0,255,0),-1)
  tile=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));tile.thumbnail((320,330));canvas.paste(tile,(col*320+(320-tile.width)//2,25));ImageDraw.Draw(canvas).text((col*320+5,5),f"{r['review_id']} frame {ns[i]}",fill='black')
 c.release();out=O/(r['review_id']+'-tracking.jpg');canvas.save(out,quality=88);done.append(str(out))
(O/'tracking-sheets.json').write_text(json.dumps(done,indent=2))
print('QA',len(done))

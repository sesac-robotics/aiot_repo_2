import json,cv2,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];O=B/'output/return-label-expansion-v3-20261024'
rows=json.loads((O/'intake.frozen.json').read_text())['records']
for r in rows:
 if r['review_id'] not in ['E06','E08','E10','E12']:continue
 frames=np.round(np.arange(0,min(2.8,(r['frame_count']-3)/r['fps']),.1)*r['fps']).astype(int)
 c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);canvas=Image.new('RGB',(1600,((len(frames)+3)//4)*280),'white');d=ImageDraw.Draw(canvas);current=-1
 for k,f in enumerate(frames):
  while current<f:ok=c.grab();current+=1;assert ok
  ok,a=c.retrieve();assert ok;im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((400,250));x=k%4*400;y=k//4*280;canvas.paste(im,(x+(400-im.width)//2,y+25));d.text((x,y),f"{r['review_id']} f{f} {f/r['fps']:.2f}s",fill='black')
 c.release();canvas.save(O/(r['review_id']+'-dense.jpg'),quality=95);print('DENSE',r['review_id'],flush=True)

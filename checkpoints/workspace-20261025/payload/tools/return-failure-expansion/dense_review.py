"""Chronological original-pixel review; no pose prediction is a target label."""
import json,cv2,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/return-failure-expansion-20261024';rows=json.loads((O/'intake.frozen.json').read_text())['records'];ids={'N001','N002','N003','N004','N005','N006','N008','N010','N012','N014','N016'}
for r in rows:
 if r['review_id'] not in ids:continue
 times=np.arange(0,r['duration']-.06,.125);frames=np.round(times*r['fps']).astype(int);cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);canvas=Image.new('RGB',(8*210,((len(frames)+7)//8)*205),'white');d=ImageDraw.Draw(canvas);current=-1
 for k,(t,f) in enumerate(zip(times,frames)):
  while current<f:ok=cap.grab();current+=1;assert ok
  ok,a=cap.retrieve();assert ok;im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((210,180));x=k%8*210;y=k//8*205;canvas.paste(im,(x+(210-im.width)//2,y+24));d.text((x+2,y+2),f"{r['review_id']} {f/r['fps']:.3f}s f{f}",fill='black')
 cap.release();canvas.save(O/(r['review_id']+'-dense.jpg'),quality=94);print('DENSE',r['review_id'],flush=True)

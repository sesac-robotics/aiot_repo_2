import json,cv2,numpy as np
from pathlib import Path
from fractions import Fraction
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/initial-pose-return-20261024';rs=json.loads((O/'source-intake.frozen.json').read_text())
for r in rs:
 ts=np.arange(r['review_start'],r['review_end'],.25);cap=cv2.VideoCapture(r['source'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);canvas=Image.new('RGB',(1520,((len(ts)+7)//8)*210),'white')
 first=round(ts[0]*float(Fraction(r['fps'])));cap.set(1,first);decoded=first-1
 for k,t in enumerate(ts):
  wanted=round(t*float(Fraction(r['fps'])))
  while decoded<wanted:
   assert cap.grab();decoded+=1
  ok,a=cap.retrieve();assert ok;im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((190,185));x=k%8*190;y=k//8*210;canvas.paste(im,(x+(190-im.width)//2,y+22));ImageDraw.Draw(canvas).text((x+3,y+3),f"{r['id']} {t:.2f}s",fill='black')
 cap.release();canvas.save(O/(r['id']+'-review.jpg'),quality=93);print(r['id'],len(ts),flush=True)

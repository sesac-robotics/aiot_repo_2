import json,math
from pathlib import Path
from fractions import Fraction
import cv2
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ytw-work-20261024/boundary-v2';I={r['id']:r for r in json.load(open(B/'output/r2d2-review-20261024/inventory.json'))}
starts={'V38':[0,3.9,7.4,11.3,23.5],'V40':[2.9,7.9,12.3,17.2,22.9,28.7],'V44':[1.5,6.4,9.9,13.2,18.5,23.5]}
for sid,ss in starts.items():
 r=I[sid];fps=float(Fraction(r['fps']));cap=cv2.VideoCapture(r['copy']);images={};requests={round((s+i*.1)*fps):round(s+i*.1,3) for s in ss for i in range(13)};n=0
 while True:
  ok,a=cap.read()
  if not ok:break
  if n in requests:
   im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((125,175));images[requests[n]]=(im.copy(),n)
  n+=1
 cap.release();canvas=Image.new('RGB',(1690,200*len(ss)),'white');d=ImageDraw.Draw(canvas)
 for row,s in enumerate(ss):
  for col in range(13):
   t=round(s+col*.1,3);im,frame=images[t];x=col*130;y=row*200;canvas.paste(im,(x+(130-im.width)//2,y+25));d.text((x+1,y+1),f'{t:.1f}s f{frame}',fill='black')
 canvas.save(O/f'{sid}-negative-detail.jpg',quality=95);print(sid,flush=True)

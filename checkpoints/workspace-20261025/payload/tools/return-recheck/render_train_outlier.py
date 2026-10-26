import json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];O=B/'output/return-recheck-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
src=B/'output/squat-return-20261023/cfrep/squat_diag_m2_8_6.mp4';c=cv2.VideoCapture(str(src),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);cases={'m2_r14':[649,655,661,667,673,679,685,691]};needed=sorted({i for a in cases.values() for i in a});ims={};current=-1
for f in needed:
 while current<f:ok=c.grab();current+=1;assert ok
 ok,a=c.retrieve();assert ok;cols=np.flatnonzero((a.max(2)>20).mean(0)>.1)
 if len(cols):a=a[:,cols[0]:cols[-1]+1]
 im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((300,440));ims[f]=im
c.release()
for name,frames in cases.items():
 canvas=Image.new('RGB',(1200,960),'white');d=ImageDraw.Draw(canvas)
 for j,f in enumerate(frames):
  im=ims[f];x=j%4*300;y=j//4*480;canvas.paste(im,(x+(300-im.width)//2,y+25));d.text((x,y),f'{name} f{f} {f/30:.2f}s',fill='black')
 canvas.save(O/f'{name}-dense.jpg',quality=97)
(O/'m2-review-intake.frozen.json').write_text(json.dumps(dict(source=str(src),source_sha256=sha(src),split='validation',fit_allowed=False,cases=cases,labels_not_changed_by_predictions=True),indent=2));print('RENDERED4case sheets')

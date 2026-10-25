import json,cv2,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];O=B/'output/return-recheck-20261024';paths=sorted((B/'output/squat-return-20261023/cfrep').glob('*.mp4'));manifest=[]
for k,p in enumerate(paths):
 if k%2==0:canvas=Image.new('RGB',(1500,720),'white')
 d=ImageDraw.Draw(canvas);c=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);frames=[0,3,6,9,12,15]
 for j,f in enumerate(frames):
  c.set(1,f);ok,a=c.read();assert ok;cols=np.flatnonzero((a.max(2)>20).mean(0)>.1)
  if len(cols):a=a[:,cols[0]:cols[-1]+1]
  im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((250,315));x=j*250;y=k%2*360;canvas.paste(im,(x+(250-im.width)//2,y+25));d.text((x,y),p.stem+f' f{f}',fill='black')
 c.release();manifest.append(dict(path=str(p),frames=frames))
 if k%2==1 or k==len(paths)-1:canvas.save(O/f'initial-{k//2+1:02}.jpg',quality=96)
(O/'initial-review-intake.json').write_text(json.dumps(manifest,indent=2));print('RENDERED',len(paths),'initial sequences')

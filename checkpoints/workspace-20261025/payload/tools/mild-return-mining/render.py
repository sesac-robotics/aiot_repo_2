import json,cv2,collections,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];O=B/'output/mild-return-mining-20261024';rs=json.loads((O/'intake.frozen.json').read_text());images={}
by=collections.defaultdict(list)
for r in rs:by[r['path']].append(r)
for path,rows in sorted(by.items(),key=lambda a:'r2d2-team' in a[0]):
 cap=cv2.VideoCapture(path,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);N=int(cap.get(7));requested=sorted({min(f,N-3) for r in rows for f in r['review_frames']});current=-1
 for f in requested:
  while current<f:
   ok=cap.grab();current+=1
   if not ok:break
  if not ok:print('MISSING_FRAME',Path(path).stem,f,flush=True);break
  ok,a=cap.retrieve();assert ok;cols=np.flatnonzero((a.max(2)>20).mean(0)>.1)
  if len(cols):a=a[:,cols[0]:cols[-1]+1]
  im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((220,248));images[path,f]=im.copy()
 cap.release();print('SOURCE',Path(path).stem,len(requested),flush=True)
for k,r in enumerate(rs):
 if k%4==0:canvas=Image.new('RGB',(1760,1200),'white')
 d=ImageDraw.Draw(canvas);y=k%4*300;d.text((2,y),r['review_id']+' '+r['id'],fill='black')
 for j,f in enumerate(r['review_frames']):
  available=[q for p,q in images if p==r['path']];f=min(f,max(available));im=images[r['path'],f];canvas.paste(im,(j*220+(220-im.width)//2,y+24));d.text((j*220,y+275),('REF ' if j==0 else '')+f'f{f} {f/r["fps"]:.2f}s',fill='black')
 if k%4==3:canvas.save(O/f'review-{k//4+1:02}.jpg',quality=94)
print('PREPARED',len(rs),'intervals',len(by),'sources',flush=True)

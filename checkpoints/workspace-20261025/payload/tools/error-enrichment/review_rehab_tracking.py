from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024';rs=[r for r in json.loads((O/'extraction.json').read_text()) if r['review_id'].startswith('R')];out=[]
for k,r in enumerate(rs):
 z=np.load(r['extraction']['cache']);xy=z['image_xy'];frames=z['frame_numbers'];indices=np.linspace(0,len(frames)-1,3).round().astype(int);c=cv2.VideoCapture(r['source'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
 if k%4==0:canvas=Image.new('RGB',(900,4*300),'#dddddd')
 for j,i in enumerate(indices):
  c.set(1,int(frames[i]));ok,a=c.read();assert ok
  h,w=a.shape[:2]
  for u,v in [(11,12),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]:
   if np.isfinite(xy[i,[u,v]]).all():cv2.line(a,tuple(xy[i,u].round().astype(int)),tuple(xy[i,v].round().astype(int)),(20,230,40),4)
  a=a[round(h*.3):round(h*.84),round(w*.2):round(w*.85)];im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((300,270));x=j*300;y=(k%4)*300;canvas.paste(im,(x+(300-im.width)//2,y+25));ImageDraw.Draw(canvas).text((x+2,y+3),r['review_id']+' f'+str(frames[i]),fill='black')
 c.release();p=O/f'rehab-tracking-{k//4+1:02}.jpg';canvas.save(p,quality=92);out.append(dict(review_id=r['review_id'],sheet=str(p),sampled_frames=frames[indices].tolist(),cache=r['extraction']['cache']))
(O/'rehab-tracking-manifest.json').write_text(json.dumps(out,indent=2));print('REHAB_TRACKING_READY',len(out))

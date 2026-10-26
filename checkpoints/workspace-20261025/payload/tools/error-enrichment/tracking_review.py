from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024';rs=json.loads((O/'extraction.json').read_text());out=[]
edges=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(27,31),(28,32)]
for k,r in enumerate(rs):
 z=np.load(r['extraction']['cache']);xy=z['image_xy'];frames=z['frame_numbers'];c=cv2.VideoCapture(r['source'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);ix=np.linspace(0,len(frames)-1,4).round().astype(int)
 if k%6==0:canvas=Image.new('RGB',(1200,6*230),'#dddddd')
 for j,i in enumerate(ix):
  c.set(1,int(frames[i]));ok,a=c.read()
  if not ok:raise ValueError(r['review_id'])
  for u,v in edges:
   if np.isfinite(xy[i,[u,v]]).all():cv2.line(a,tuple(xy[i,u].round().astype(int)),tuple(xy[i,v].round().astype(int)),(30,240,70),max(1,a.shape[0]//400))
  im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((300,205));x=j*300;y=(k%6)*230;canvas.paste(im,(x+(300-im.width)//2,y+22));ImageDraw.Draw(canvas).text((x+2,y+2),r['review_id']+' f'+str(frames[i]),fill='black')
 c.release();p=O/f'tracking-review-{k//6+1:02}.jpg';canvas.save(p,quality=90);out.append(dict(review_id=r['review_id'],sheet=str(p),cache_sha256=__import__('hashlib').file_digest(Path(r['extraction']['cache']).open('rb'),'sha256').hexdigest(),sampled_frames=frames[ix].tolist()))
(O/'tracking-review-manifest.json').write_text(json.dumps(out,indent=2));print('tracking sheets',1+k//6)

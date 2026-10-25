from pathlib import Path
import json,sys,hashlib,cv2
import numpy as np
from PIL import Image,ImageDraw
from anatomical_turn_features_v1 import extract
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-work-20261024/grade-support-v1-20261025'
rows=json.loads((B/'output/return-target-contract-v1-20261025/reviewed-events.frozen.json').read_text());inv={r['id']:r for r in json.loads((B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json').read_text())['sources']}
results=[]
for r in rows:
 a=r['annotation'];p=O/'observations'/a['id']/'primary.npz';z=np.load(p);m=json.loads(p.with_suffix('.json').read_text());s=inv[a['id']]
 assert m['config']['source_sha256']==s['source_sha256']==a['source_sha256'];assert str(z['fingerprint'])==m['fingerprint']
 fps=m['config']['segment']['fps'];times=z['time_seconds'];ticks=np.round((times-times[0])*10).astype(int)+round(times[0]*10)
 assert len(np.unique(ticks))==len(ticks)
 c=cv2.VideoCapture(s['path']);w,h=int(c.get(3)),int(c.get(4))
 f=extract(list(zip(ticks,z['image_xy'])),a,fps,w,h,native_frame_numbers=z['frame_numbers'])
 results.append(dict(id=a['id'],split=s['split'],annotation=a,features=f,cache=str(p),source_sha256=s['source_sha256'],cache_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 # Exact native-pixel overlay across peak, descent/turn, next ascent for direct QA.
 can=Image.new('RGB',(1400,840),'white');d=ImageDraw.Draw(can)
 targets=np.linspace(0,min(3.1,times[-1]),10)
 for i,t in enumerate(targets):
  j=int(np.argmin(abs(times-t)));c.set(1,int(z['frame_numbers'][j]));ok,fr=c.read();assert ok
  im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));im.thumbnail((275,385));xx=(i%5)*280+(280-im.width)//2;yy=(i//5)*420+20;can.paste(im,(xx,yy));scale=im.width/w
  for u,v in [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24)]:
   q=z['image_xy'][j,[u,v]]
   if np.isfinite(q).all():d.line([(xx+pt[0]*scale,yy+pt[1]*scale) for pt in q],fill='lime',width=2)
  d.text((i%5*280+2,i//5*420+3),f'f{z["frame_numbers"][j]} {times[j]:.3f}s',fill='black')
 c.release();can.save(O/(a['id']+'-chain.jpg'),quality=92)
(O/'validation-feature-diagnostics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
for r in results:print(r['id'],r['features'])

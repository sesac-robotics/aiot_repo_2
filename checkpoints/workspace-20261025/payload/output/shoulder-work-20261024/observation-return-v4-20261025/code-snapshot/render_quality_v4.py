from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-work-20261024/observation-return-v4-20261025'
inv={r['id']:r for r in json.loads((B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json').read_text())['sources']}
rows=json.loads((O/'replay.json').read_text());targets=[];seen=set()
for r in rows:
 q=r['features'].get('quality',{}).get('flags',[])
 if not q or r['source_id'] in seen:continue
 seen.add(r['source_id']);targets.append((r,q[0]['tick']))
for page in range((len(targets)+5)//6):
 can=Image.new('RGB',(1200,840),'white');d=ImageDraw.Draw(can)
 for i,(r,tick) in enumerate(targets[page*6:(page+1)*6]):
  z=np.load(r['cache']);ts=z['time_seconds'];ticks=np.round((ts-ts[0])*10).astype(int)+round(ts[0]*10);j=int(np.argmin(abs(ticks-tick)))
  c=cv2.VideoCapture(inv[r['source_id']]['path']);c.set(1,int(z['frame_numbers'][j]));ok,f=c.read();c.release();assert ok
  h,w=f.shape[:2];im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));im.thumbnail((390,365));xx=i%3*400+(400-im.width)//2;yy=i//3*420+30;can.paste(im,(xx,yy));scale=im.width/w
  for a,b in [(11,12),(11,23),(12,24),(23,24)]:
   pts=z['image_xy'][j,[a,b]]
   if np.isfinite(pts).all():d.line([(xx+p[0]*scale,yy+p[1]*scale) for p in pts],fill='lime',width=2)
  d.text((i%3*400+2,i//3*420+2),r['source_id']+' '+str(ts[j])[:5],fill='black')
 can.save(O/f'geometry-review-{page+1}.jpg',quality=90)

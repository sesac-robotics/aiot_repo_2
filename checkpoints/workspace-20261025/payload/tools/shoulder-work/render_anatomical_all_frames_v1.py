from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-work-20261024/grade-support-v1-20261025';inv={r['id']:r for r in json.loads((B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json').read_text())['sources']};receipts=[]
for r in json.loads((O/'validation-feature-diagnostics.json').read_text()):
 s=inv[r['id']];z=np.load(r['cache']);a=r['annotation'];last=a['next_ascent_frames'][1];selected=np.where(z['frame_numbers']<=last)[0];c=cv2.VideoCapture(s['path']);pages=[]
 for page in range((len(selected)+11)//12):
  can=Image.new('RGB',(1400,1260),'white');d=ImageDraw.Draw(can)
  for pos,j in enumerate(selected[page*12:(page+1)*12]):
   c.set(1,int(z['frame_numbers'][j]));ok,fr=c.read();assert ok;h,w=fr.shape[:2]
   im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));im.thumbnail((345,390));xx=pos%4*350+(350-im.width)//2;yy=pos//4*420+24;can.paste(im,(xx,yy));sc=im.width/w
   for u,v in [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24)]:
    xy=z['image_xy'][j,[u,v]]
    if np.isfinite(xy).all():d.line([(xx+q[0]*sc,yy+q[1]*sc) for q in xy],fill='lime',width=2)
   d.text((pos%4*350+2,pos//4*420+3),f'f{z["frame_numbers"][j]} {z["time_seconds"][j]:.3f}s',fill='black')
  path=O/f'{r["id"]}-all-chain-{page+1}.jpg';can.save(path,quality=92);pages.append(str(path))
 c.release();receipts.append(dict(id=r['id'],native_frames=z['frame_numbers'][selected].tolist(),pages=pages,review_status='pending_direct_visual_review'))
(O/'all-frame-overlay-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')

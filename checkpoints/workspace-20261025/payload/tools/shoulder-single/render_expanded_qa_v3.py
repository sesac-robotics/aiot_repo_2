from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-expanded-train-v3-20261025';rows=json.loads((O/'feature-candidates.json').read_text());receipts=[]
for r in rows:
 z=np.load(r['cache_path']);fs=z['frame_numbers'];a,b=r['window_seconds'];inds=np.where((fs/r['fps']>=a-.04)&(fs/r['fps']<=b+.04))[0];c=cv2.VideoCapture(r['path']);pages=[]
 for page in range((len(inds)+23)//24):
  can=Image.new('RGB',(1200,1484),'white');dr=ImageDraw.Draw(can);dr.text((2,3),r['id']+' every actual 10Hz context observation; raw joints, no interpolation',fill='black')
  for pos,i in enumerate(inds[page*24:(page+1)*24]):
   c.set(1,int(fs[i]));ok,fr=c.read();assert ok;im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));im.thumbnail((196,335));xx=pos%6*200+(200-im.width)//2;yy=pos//6*365+47;can.paste(im,(xx,yy));scale=im.width/fr.shape[1]
   for u,v in [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24)]:
    xy=z['image_xy'][i,[u,v]]
    if np.isfinite(xy).all():dr.line([(xx+p[0]*scale,yy+p[1]*scale) for p in xy],fill='red',width=2)
   dr.text((pos%6*200+2,pos//6*365+27),f'f{fs[i]} {fs[i]/r["fps"]:.2f}s',fill='black')
  name=r['id']+f'-qa-{page+1}.jpg';can.save(O/name,quality=94);pages.append(name)
 c.release();receipts.append(dict(id=r['id'],frames=fs[inds].tolist(),pages=pages))
(O/'qa-render-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
# Raw detail of a repeated partial lower-turn candidate. No pose-derived grades.
r=next(x for x in json.loads((O/'source-plan.frozen.json').read_text()) if x['id']=='youtube_vlFGTI5JzjI');frames=np.rint(np.arange(158,172,.25)*r['fps']).astype(int);c=cv2.VideoCapture(r['path'])
for page in range((len(frames)+23)//24):
 can=Image.new('RGB',(1280,860),'white');dr=ImageDraw.Draw(can);dr.text((3,2),'Colossus native 4Hz 158-172s: inspect actual turns, not red X',fill='black')
 for j,f in enumerate(frames[page*24:(page+1)*24]):
  c.set(1,int(f));ok,fr=c.read();assert ok;im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));im.thumbnail((316,170));xx=j%4*320;yy=j//4*140+30;im.thumbnail((316,110));can.paste(im,(xx,yy));dr.text((xx,yy-13),f'{f/r["fps"]:.3f}s f{f}',fill='black')
 can.save(O/f'colossus-partial-dense-{page+1}.jpg',quality=95)
c.release();print('rendered',sum(len(x['frames']) for x in receipts),'joint observations')

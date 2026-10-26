from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-review-train-v1-20261025';rows=json.loads((O/'review-plan.frozen.json').read_text());out=O/'review';receipts=[]
for n in [2,19,20,22,23,25,26,27,33,34,37,39,40,42,45,46,48,49,52,53,59,60]:
 r=rows[n];z=np.load(r['cache_path']);inds=r['cache_indices'];c=cv2.VideoCapture(r['source_path']);pages=[]
 for pg in range((len(inds)+23)//24):
  name=f'dense-{n:02d}-{pg:02d}.jpg';pages.append('review/'+name)
  if (out/name).exists():continue
  can=Image.new('RGB',(1200,1040),'white');d=ImageDraw.Draw(can);d.text((3,3),f'#{n} {r["id"]} ALL cached observations p{pg+1}',fill='black')
  for j,i in enumerate(inds[pg*24:pg*24+24]):
   f=int(z['frame_numbers'][i]);c.set(1,f);ok,fr=c.read();assert ok;im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));im.thumbnail((196,220));x=j%6*200+(200-im.width)//2;y=36+j//6*250;can.paste(im,(x,y));s=im.width/fr.shape[1]
   for u,v in [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]:
    p=z['image_xy'][i,[u,v]]
    if np.isfinite(p).all():d.line([(x+a*s,y+b*s) for a,b in p],fill=(255,80,50),width=1)
   d.text((j%6*200+3,23+j//6*250),f'{f/r["fps"]:.3f}s f{f}',fill='black')
  can.save(out/name,quality=95)
 c.release();receipts.append(dict(index=n,id=r['id'],frames=z['frame_numbers'][inds].tolist(),pages=pages));(O/'dense-render-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n');print(n,len(inds),flush=True)

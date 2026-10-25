from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-rebuild-v2-20261025';rows=json.loads((O/'event-feature-candidates.json').read_text());receipts=[]
for r in rows:
 if not r.get('event'):continue
 z=np.load(r['cache_path']);fs=z['frame_numbers'];a=r['event']['peak_frames'][0];b=r['event']['turn_frames'][1];inds=np.where((fs>=a)&(fs<=b))[0];c=cv2.VideoCapture(r['path']);pages=[]
 # Render EVERY actual context sample; no sparse render can approve unrendered frames.
 for page in range((len(inds)+11)//12):
  subset=inds[page*12:(page+1)*12];can=Image.new('RGB',(1200,1060),'white');dr=ImageDraw.Draw(can);dr.text((3,2),r['id']+f' lower={r["lower_status"]} body={r["body_status"]}',fill='black')
  for pos,i in enumerate(subset):
   c.set(1,int(fs[i]));ok,fr=c.read();assert ok;im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));im.thumbnail((295,315));xx=pos%4*300+(300-im.width)//2;yy=pos//4*345+35;can.paste(im,(xx,yy));scale=im.width/fr.shape[1]
   for u,v in [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24)]:
    xy=z['image_xy'][i,[u,v]]
    if np.isfinite(xy).all():dr.line([(xx+p[0]*scale,yy+p[1]*scale) for p in xy],fill='red',width=2)
   dr.text((pos%4*300+2,pos//4*345+20),f'f{fs[i]} {fs[i]/r["fps"]:.3f}s',fill='black')
  name=r['id']+f'-context-{page+1}.jpg';can.save(O/name,quality=92);pages.append(name)
 c.release();receipts.append(dict(id=r['id'],native_frames=fs[inds].tolist(),pages=pages,all_actual_context_samples_rendered=True))
(O/'all-context-render-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n');print('rendered',len(receipts),'events',sum(len(r['native_frames']) for r in receipts),'actual frames',sum(len(r['pages']) for r in receipts),'pages')

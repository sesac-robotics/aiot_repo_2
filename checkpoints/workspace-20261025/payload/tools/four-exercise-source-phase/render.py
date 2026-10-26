from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-source-phase-v2-20261025';rows=json.loads((O/'review-plan.frozen.json').read_text());out=O/'review';out.mkdir(exist_ok=True);receipts=[]
for pair in range((len(rows)+1)//2):
 if (out/f'pair-{pair:02d}.jpg').exists():
  for block,r in enumerate(rows[pair*2:pair*2+2]):
   z=np.load(r['cache_path']);inds=np.array(r['cache_indices']);inds=inds[np.unique(np.rint(np.linspace(0,len(inds)-1,12)).astype(int))];receipts.append(dict(id=r['id'],index=pair*2+block,source_frames=z['frame_numbers'][inds].tolist(),sampling='12_uniform_actual_cache_observations_not_every_video_frame',page=f'review/pair-{pair:02d}.jpg'))
  continue
 can=Image.new('RGB',(1200,1580),'white');d=ImageDraw.Draw(can)
 for block,r in enumerate(rows[pair*2:pair*2+2]):
  n=pair*2+block;z=np.load(r['cache_path']);inds=np.array(r['cache_indices']);inds=inds[np.unique(np.rint(np.linspace(0,len(inds)-1,12)).astype(int))];c=cv2.VideoCapture(r['source_path']);ybase=block*790;d.text((4,ybase+4),f"#{n:02d} {r['exercise']} {r['id']}   native sampled observations",fill='black');frames=[]
  for j,i in enumerate(inds):
   f=int(z['frame_numbers'][i]);c.set(1,f);ok,fr=c.read();assert ok;im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));im.thumbnail((296,225));x=j%4*300+(300-im.width)//2;y=ybase+40+j//4*250;can.paste(im,(x,y));s=im.width/fr.shape[1]
   for u,v in [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]:
    p=z['image_xy'][i,[u,v]]
    if np.isfinite(p).all():d.line([(x+a*s,y+b*s) for a,b in p],fill=(255,80,50),width=2)
   d.text((j%4*300+3,ybase+26+j//4*250),f'{f/r["fps"]:.3f}s f{f}',fill='black');frames.append(f)
  c.release();receipts.append(dict(id=r['id'],index=n,source_frames=frames,sampling='12_uniform_actual_cache_observations_not_every_video_frame',page=f'review/pair-{pair:02d}.jpg'))
 can.save(out/f'pair-{pair:02d}.jpg',quality=93);print(pair,flush=True)
(O/'render-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')

import json,math
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ytw-work-20261024/logo-v1/video-review';O.mkdir(exist_ok=True)
inv={r['id']:r for r in json.load(open(B/'output/r2d2-review-20261024/inventory.json'))}
manifest=[]
for sid in ['V37','V39','V41','V43']:
 r=inv[sid];cap=cv2.VideoCapture(r['copy']);fps=cap.get(cv2.CAP_PROP_FPS);duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps;times=[i*.25 for i in range(math.ceil(duration*4))];paths=[]
 for page in range(math.ceil(len(times)/40)):
  canvas=Image.new('RGB',(1000,1800),'white');d=ImageDraw.Draw(canvas)
  for j,t in enumerate(times[page*40:(page+1)*40]):
   cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,a=cap.read()
   if not ok:continue
   im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((195,195));x=(j%5)*200;y=(j//5)*225
   canvas.paste(im,(x+(200-im.width)//2,y+25));d.text((x+5,y+5),f'{sid} {t:.2f}s',fill='black')
  p=O/f'{sid}-{page+1:02d}.jpg';canvas.save(p,quality=90);paths.append(str(p))
 cap.release();manifest.append(dict(source_id=sid,source=r['copy'],fps=fps,duration=duration,interval_seconds=.25,sheets=paths))
(O/'manifest.json').write_text(json.dumps(manifest,indent=2))
print([(r['source_id'],len(r['sheets'])) for r in manifest])

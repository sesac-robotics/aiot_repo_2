from pathlib import Path
import json,cv2,numpy as np,collections
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/label-recheck-20261024';rows=json.loads((B/'data/error-enrichment-20261024/annotations/standing-states-final.frozen.json').read_text());manifest=[];last=None;c=None
for k,r in enumerate(rows):
 if last!=r['path']:
  if c is not None:c.release()
  c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);last=r['path']
 c.set(1,r['frame']);ok,a=c.read();assert ok,(r['id'],r['frame'])
 columns=np.where((a.max(2)>20).mean(0)>.1)[0]
 if len(columns):a=a[:,columns[0]:columns[-1]+1]
 h,w=a.shape[:2]
 if w>h*1.3:
  x=(w-h)//2;a=a[:,x:x+h]
 tile=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));tile.thumbnail((160,230))
 if k%30==0:canvas=Image.new('RGB',(6*160,5*255),'#dddddd')
 x=(k%6)*160;y=((k%30)//6)*255;canvas.paste(tile,(x+(160-tile.width)//2,y+24));ImageDraw.Draw(canvas).text((x+2,y+2),f'S{k:03} {r["review_id"][:14]} f{r["frame"]}',fill='black')
 sheet=O/f'state-audit-{k//30+1:02}.jpg';canvas.save(sheet,quality=92);manifest.append(dict(review_index=k,id=r['id'],source=r['path'],frame=r['frame'],sheet=str(sheet)))
if c:c.release()
(O/'state-audit-manifest.json').write_text(json.dumps(manifest,indent=2));print('STATE_ATLAS',len(rows),flush=True)

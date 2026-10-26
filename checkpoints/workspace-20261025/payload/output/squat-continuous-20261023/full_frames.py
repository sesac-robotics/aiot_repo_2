from pathlib import Path
import json,cv2,math,hashlib
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent;rs=json.loads((O.parent/'squat-errors-20261023/review-decisions.json').read_text())['records'];out=[]
for j,r in enumerate(rs):
 c=cv2.VideoCapture(r['path']);frames=[];fps=c.get(cv2.CAP_PROP_FPS);expected=int(c.get(cv2.CAP_PROP_FRAME_COUNT))
 while True:
  ok,bgr=c.read()
  if not ok:break
  frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
 c.release();pages=[]
 for pg in range(math.ceil(len(frames)/144)):
  subset=frames[pg*144:(pg+1)*144];canvas=Image.new('RGB',(12*160,math.ceil(len(subset)/12)*150),'#eaf0f4');d=ImageDraw.Draw(canvas)
  for k,frame in enumerate(subset):
   im=Image.fromarray(frame);im.thumbnail((158,130));x=k%12*160;y=k//12*150;canvas.paste(im,(x,y+19));d.text((x+2,y+2),f'{r["id"]} f{pg*144+k}',fill='black')
  path=O/'frames'/f'{j:03}-{r["id"]}-{pg}.jpg';path.parent.mkdir(exist_ok=True);canvas.save(path,quality=86);pages.append(str(path))
 out.append(dict(id=r['id'],person=r['person'],path=r['path'],source_sha256=r['source_sha256'],fps=fps,expected_frames=expected,decoded_frames=len(frames),full_decode=len(frames)==expected,frame_step=1,pages=pages))
 if (j+1)%16==0:print(j+1,flush=True)
(O/'full-frame-manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('done',len(out),sum(r['decoded_frames'] for r in out),flush=True)

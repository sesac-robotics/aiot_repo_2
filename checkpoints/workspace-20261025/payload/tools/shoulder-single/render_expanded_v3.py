from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-expanded-train-v3-20261025';rows=json.loads((O/'source-plan.frozen.json').read_text());receipts=[]
for r in rows:
 hz=4 if r['family']=='team' else 1;frames=np.unique(np.rint(np.arange(0,r['frame_count'],r['fps']/hz)).astype(int));frames=frames[frames<r['frame_count']];ncol=6;nrows=4 if r['family']=='team' else 6;cellh=365 if r['family']=='team' else 145;batch=ncol*nrows;c=cv2.VideoCapture(r['path']);pages=[]
 for page in range((len(frames)+batch-1)//batch):
  can=Image.new('RGB',(1200,cellh*nrows+24),'white');dr=ImageDraw.Draw(can);dr.text((2,3),r['id']+f' full-duration {hz}Hz source review; no labels inferred',fill='black')
  for j,f in enumerate(frames[page*batch:(page+1)*batch]):
   c.set(1,int(f));ok,fr=c.read();assert ok;im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));im.thumbnail((196,cellh-30));xx=j%ncol*200+(200-im.width)//2;yy=j//ncol*cellh+47;can.paste(im,(xx,yy));dr.text((j%ncol*200+2,j//ncol*cellh+27),f'f{f} {f/r["fps"]:.2f}s',fill='black')
  name=r['id']+f'-full-{page+1:02d}.jpg';can.save(O/name,quality=93);pages.append(name)
 c.release();receipts.append(dict(id=r['id'],sample_hz=hz,sampled_native_frames=frames.tolist(),pages=pages));(O/'full-review-preparation.json').write_text(json.dumps(receipts,indent=2)+'\n');print(r['id'],len(frames),len(pages),flush=True)

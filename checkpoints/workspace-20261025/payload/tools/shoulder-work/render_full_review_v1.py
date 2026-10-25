from pathlib import Path
import sys,json,math
import cv2
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-work-20261024/full-review-v1';manifest=json.loads((O/'inventory.frozen.json').read_text())['sources'];receipts=[]
for r in manifest:
 if r['owner']!='shoulder':continue
 cap=cv2.VideoCapture(r['path']);fps=cap.get(5);n=int(cap.get(7));duration=n/fps;frames=list(range(0,n,max(1,round(fps/2))));pages=[]
 for page in range(math.ceil(len(frames)/24)):
  target=O/'boards'/r['id']/f'timeline-{page+1:02d}.jpg';target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():
   canvas=Image.new('RGB',(1344,800),'#ffffff');draw=ImageDraw.Draw(canvas);draw.text((5,3),f"{r['id']} role={r['split']} PAGE {page+1} full timeline 2fps",fill='black')
   for slot,f in enumerate(frames[page*24:(page+1)*24]):
    cap.set(0 if False else cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read()
    if not ok:continue
    im=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));im.thumbnail((224,170));x=(slot%6)*224;y=(slot//6)*190+30;canvas.paste(im,(x+(224-im.width)//2,y));draw.text((x+3,y+172),f'f{f} t{f/fps:.2f}s',fill='black')
   canvas.save(target,quality=85)
  pages.append(str(target))
 cap.release();receipts.append(dict(id=r['id'],fps=fps,frames=n,duration=duration,width=int(img.shape[1]) if 'img' in locals() else None,height=int(img.shape[0]) if 'img' in locals() else None,review_times=[f/fps for f in frames],pages=pages,status='rendered_not_visually_reviewed'))
 (O/'render-receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
 print(r['id'],len(pages),flush=True)

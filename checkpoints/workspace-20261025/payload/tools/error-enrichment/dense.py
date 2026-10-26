from pathlib import Path
import json,cv2,numpy as np,sys
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024'
rows=json.loads((O/'intake.json').read_text());ids=set(sys.argv[1:]);receipt=[]
for r in rows:
 if r['review_id'] not in ids:continue
 start,end=(7,15) if r['exercise']=='shoulder' else (0,min(r['duration']-.07,6.5))
 c=cv2.VideoCapture(r['path']);fps=c.get(5);indices=np.rint(np.arange(start,end,.25)*fps).astype(int);canvas=Image.new('RGB',(8*200,((len(indices)+7)//8)*245),'white');d=ImageDraw.Draw(canvas)
 for k,f in enumerate(indices):
  c.set(1,int(f));ok,a=c.read()
  if not ok:raise ValueError(r['review_id'])
  im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB))
  if im.width/im.height>1.6:
   # Center detail is for review only; full-source overview remains available.
   w=int(im.height*.95);x=(im.width-w)//2;im=im.crop((x,0,x+w,im.height))
  im.thumbnail((200,220));x=(k%8)*200;y=(k//8)*245;canvas.paste(im,(x+(200-im.width)//2,y+22));d.text((x+2,y+3),f'{r["review_id"]} {f/fps:.2f}s',fill='black')
 c.release();p=O/(r['review_id']+'-dense.jpg');canvas.save(p,quality=90);receipt.append(dict(review_id=r['review_id'],start=start,end=end,frames=indices.tolist(),path=str(p)))
(O/'dense-manifest.json').write_text(json.dumps(receipt,indent=2));print('dense',len(receipt))

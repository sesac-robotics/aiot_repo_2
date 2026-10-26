from pathlib import Path
import json,math,cv2
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-work-20261024/full-review-v1';journal=json.loads((O/'shoulder-review-journal.json').read_text());inv={r['id']:r for r in json.loads((O/'inventory.frozen.json').read_text())['sources']};receipts=[]
for r in journal:
 cap=cv2.VideoCapture(inv[r['id']]['path']);fps=cap.get(5);indices=sorted({int(round(t*fps)) for a,b in r['dense_followup_windows'] for t in __import__('numpy').arange(a,b+.01,.2)})
 pages=[]
 for page in range(math.ceil(len(indices)/24)):
  target=O/'dense'/r['id']/f'dense-{page+1:02d}.jpg';target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():
   im=Image.new('RGB',(1536,930),'white');dr=ImageDraw.Draw(im);dr.text((3,3),r['id']+' DENSE5Hz '+str(r['dense_followup_windows']),fill='black')
   for i,f in enumerate(indices[page*24:(page+1)*24]):
    cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,fr=cap.read()
    if not ok:continue
    v=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));v.thumbnail((256,205));x=i%6*256;y=i//6*220+30;im.paste(v,(x+(256-v.width)//2,y));dr.text((x+3,y+205),f'f{f} t{f/fps:.3f}',fill='black')
   im.save(target,quality=90)
  pages.append(str(target))
 cap.release();receipts.append(dict(id=r['id'],indices=indices,pages=pages,method='5Hz_original_frame_chronological',status='rendered_not_reviewed'))
 (O/'dense-receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
 print(r['id'],len(pages),flush=True)

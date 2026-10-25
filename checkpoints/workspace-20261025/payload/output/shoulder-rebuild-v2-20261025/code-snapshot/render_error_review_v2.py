from pathlib import Path
import json,cv2,numpy as np,hashlib
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-rebuild-v2-20261025';inv={r['id']:r for r in json.loads((B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json').read_text())['sources']};spec=[(1,14),(6,15),(7,15),(10,13),(11,13),(13,20),(14,21),(16,10),(18,9),(19,17),(23,8),(26,23)];receipts=[]
for num,start in spec:
 s=inv[f'subject_{num:03d}_shoulder_bad_front'];assert s['split']=='train';assert hashlib.sha256(Path(s['path']).read_bytes()).hexdigest()==s['source_sha256'];c=cv2.VideoCapture(s['path']);fps=c.get(5);frames=list(range(round(start*fps),min(round((start+3)*fps)+1,int(c.get(7))),3));can=Image.new('RGB',(1200,350*((len(frames)+3)//4)+25),'white');dr=ImageDraw.Draw(can);dr.text((3,3),s['id']+' additional Train middle window; provider bad is NOT an item grade',fill='black')
 for j,f in enumerate(frames):
  c.set(1,f);ok,fr=c.read();assert ok;im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));im.thumbnail((295,315));can.paste(im,(j%4*300+(300-im.width)//2,j//4*350+50));dr.text((j%4*300+2,j//4*350+28),f'f{f} {f/fps:.2f}s',fill='black')
 c.release();path=O/(s['id']+'-additional-errors.jpg');can.save(path,quality=94);receipts.append(dict(source_id=s['id'],source_sha256=s['source_sha256'],split='train',native_frames=frames,times=[f/fps for f in frames],evidence=str(path),review_status='pending',grade=None))
(O/'additional-error-review-intake.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n');print(len(receipts),sum(len(r['native_frames']) for r in receipts))

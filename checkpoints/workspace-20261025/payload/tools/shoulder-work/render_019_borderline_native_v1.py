from pathlib import Path
import json,cv2
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-work-20261024/grade-support-v1-20261025';inv={r['id']:r for r in json.loads((B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json').read_text())['sources']}
for kind,a,b in [('bad',201,210),('good',36,42)]:
 s=inv[f'subject_019_shoulder_{kind}_front'];c=cv2.VideoCapture(s['path']);fps=c.get(5);can=Image.new('RGB',(1400,900),'white');d=ImageDraw.Draw(can)
 for i,f in enumerate(range(a,b+1)):
  c.set(1,f);ok,fr=c.read();assert ok;im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));im.thumbnail((275,420));xx=i%5*280+(280-im.width)//2;yy=i//5*450+24;can.paste(im,(xx,yy));d.text((i%5*280+3,i//5*450+3),f'f{f} {f/fps:.4f}s',fill='black')
 c.release();can.save(O/f'subject_019_{kind}-frame-by-frame.jpg',quality=95)

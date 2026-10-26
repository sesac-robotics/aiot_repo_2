from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-work-20261024/grade-support-v1-20261025';inv={r['id']:r for r in json.loads((B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json').read_text())['sources']}
for sid,a,b in [('subject_019_shoulder_bad_front',12.6,14.6),('subject_019_shoulder_good_front',.8,3.2),('subject_019_shoulder_bad_front',4.4,6.4)]:
 s=inv[sid];assert s['split']=='train';c=cv2.VideoCapture(s['path']);fps=c.get(5);frames=list(range(round(a*fps),round(b*fps)+1,3));can=Image.new('RGB',(1400,450*((len(frames)+4)//5)),'white');d=ImageDraw.Draw(can)
 for i,f in enumerate(frames):
  c.set(1,f);ok,fr=c.read();assert ok;im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));im.thumbnail((275,420));xx=i%5*280+(280-im.width)//2;yy=i//5*450+24;can.paste(im,(xx,yy));d.text((i%5*280+2,i//5*450+3),f'f{f} {f/fps:.3f}s',fill='black')
 c.release();can.save(O/f'{sid}-{a}-native.jpg',quality=94)

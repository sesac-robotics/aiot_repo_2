"""Dense original frames for the two recovered Train cycles; no new poses."""
from pathlib import Path
import json,cv2,hashlib
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];O=B/'output/squat-count-Train-profile-20261025'
inv=next(r for r in json.loads((B/'output/r2d2-review-20261024/inventory.json').read_text())if r['id']=='V10')
a,b=map(int,inv['fps'].split('/'));fps=a/b
windows=[(5.8,8.4),(11.1,13.9)];targets=[list(range(round(s*fps),round(e*fps)+1,round(fps/5)))for s,e in windows];wanted={f for fs in targets for f in fs};images={};cap=cv2.VideoCapture(inv['copy'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);start=min(wanted);cap.set(1,start)
for f in range(start,max(wanted)+1):
 ok,x=cap.read();assert ok
 if f in wanted:
  im=Image.fromarray(cv2.cvtColor(x,cv2.COLOR_BGR2RGB));im.thumbnail((220,390));images[f]=im
cap.release();receipts=[]
for i,fs in enumerate(targets):
 out=Image.new('RGB',(1100,420*((len(fs)+4)//5)),'white');d=ImageDraw.Draw(out)
 for j,f in enumerate(fs):
  im=images[f];xx=j%5*220;yy=j//5*420;out.paste(im,(xx+(220-im.width)//2,yy+26));d.text((xx+3,yy+4),f'V10 f{f} {f/fps:.3f}s',fill='black')
 p=O/f'V10-recovered-cycle-{i+1}-5Hz.jpg';assert not p.exists();out.save(p,quality=97);receipts.append(dict(path=str(p),sheet_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_sha256=inv['sha256'],split='train',frames=fs,times_seconds=[f/fps for f in fs]))
p=O/'recovered-cycle-panels.frozen.json';assert not p.exists();p.write_text(json.dumps(receipts,indent=2)+'\n');print('Dense recovered-cycle panels',len(receipts))

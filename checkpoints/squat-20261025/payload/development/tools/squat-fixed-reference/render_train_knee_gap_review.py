"""Original Train pixels around actual knee gaps, without re-extracting poses."""
import json,hashlib
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];O=B/'output/squat-count-Train-profile-20261025'
def main():
 inv={r['id']:r for r in json.loads((B/'output/r2d2-review-20261024/inventory.json').read_text())};raw=json.loads((O/'natural-knee-gaps.frozen.json').read_text());gaps=[r for r in raw['gaps']if r['bracketed_by_actual_knee']and 4<=r['slots']<=7];receipts=[]
 for sid in sorted({r['source_id']for r in gaps}):
  sub=[r for r in gaps if r['source_id']==sid];z=np.load(B/'output/integrated-five-20261024/team-observations'/sid/'side.npz');ticks=np.rint(z['time_seconds']*10).astype(int);cap=cv2.VideoCapture(inv[sid]['copy'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);out=Image.new('RGB',(1400,530*len(sub)),'white');d=ImageDraw.Draw(out);frames=[]
  for row,g in enumerate(sub):
   ts=[g['start_tick']-1,g['start_tick'],(g['start_tick']+g['end_tick'])//2,g['end_tick'],g['end_tick']+1]
   for col,t in enumerate(ts):
    ix=int(np.flatnonzero(ticks==t)[0]);f=int(z['frame_numbers'][ix]);cap.set(1,f);ok,a=cap.read();assert ok;im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((280,490));xx=col*280;yy=row*530;out.paste(im,(xx+(280-im.width)//2,yy+34));d.text((xx+3,yy+3),f'{sid} t{t/10:.1f}s f{f}',fill='black');d.text((xx+3,yy+17),'knee both missing'if g['start_tick']<=t<=g['end_tick']else 'actual knee seen',fill='black');frames.append(dict(gap_id=g['id'],tick=t,frame=f))
  cap.release();p=O/(sid+'-natural-knee-gaps.jpg');assert not p.exists();out.save(p,quality=97);receipts.append(dict(source_id=sid,source_sha256=inv[sid]['sha256'],path=str(p),sheet_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),samples=frames))
 p=O/'gap-review-panels.frozen.json';assert not p.exists();p.write_text(json.dumps(receipts,indent=2)+'\n');print('Train original-pixel review panels',len(receipts))
if __name__=='__main__':main()

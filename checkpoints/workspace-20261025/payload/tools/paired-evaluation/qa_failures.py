import json,cv2,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/paired-failure-mining-20261024';rows=[r for r in json.loads((O/'labels-v2.frozen.json').read_text())['reviewed'] if r['train_eligible']]
for k,r in enumerate(rows):
 z=np.load(O/'observations'/r['review_id']/'primary.npz');cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
 if k%4==0:sheet=Image.new('RGB',(1200,4*250),'white')
 indices=np.where((z['frame_numbers']/r['fps']>=r['start'])&(z['frame_numbers']/r['fps']<r['end']))[0]
 for j,n in enumerate(indices[[-min(5,len(indices)), -2, -1]]):
  frame=z['frame_numbers'][n];cap.set(1,int(frame));ok,a=cap.read();assert ok;xy=z['image_xy'][n]
  for chain in [[11,13,15],[12,14,16]]:
   for h,t in zip(chain,chain[1:]):
    if np.isfinite(xy[[h,t]]).all():cv2.line(a,tuple(xy[h].astype(int)),tuple(xy[t].astype(int)),(0,255,0),3)
   for h in chain:
    if np.isfinite(xy[h]).all():cv2.circle(a,tuple(xy[h].astype(int)),4,(0,0,255),-1)
  im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((395,220));x=j*400;y=k%4*250;sheet.paste(im,(x,y+25));ImageDraw.Draw(sheet).text((x+4,y+3),f"{r['review_id']} {frame/r['fps']:.2f}s",fill='black')
 cap.release();sheet.save(O/f'tracking-v2-{k//4+1:02}.jpg',quality=94)
print('QA_SHEETS',4)

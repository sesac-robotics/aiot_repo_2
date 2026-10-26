from pathlib import Path
from fractions import Fraction
import json,cv2,numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/pushup-root-cause-20261024';inventory={r['id']:r for r in json.loads((B/'output/r2d2-review-20261024/inventory.json').read_text())}
for name in ['V28','V30','V33','V35','V36']:
 r=inventory[name];z=np.load(B/'output/integrated-five-20261024/team-observations'/name/'side.npz');c=cv2.VideoCapture(r['copy'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);canvas=Image.new('RGB',(1500,1000),'white')
 for col,time in enumerate([1.5,2.5,3.5]):
  ix=int(np.argmin(abs(z['frame_numbers']/float(Fraction(r['fps']))-time)));c.set(1,int(z['frame_numbers'][ix]));ok,im=c.read();assert ok
  h,w=im.shape[:2];xy=z['image_xy'][ix];finite=xy[np.isfinite(xy).all(1)];xmin,ymin=finite.min(0);xmax,ymax=finite.max(0);xmin=max(0,int(xmin)-50);ymin=max(0,int(ymin)-35);xmax=min(w,int(xmax)+50);ymax=min(h,int(ymax)+35)
  for row,side in enumerate([0,1]):
   overlay=im.copy();ids=[j+side for j in [11,13,15,23,25,27,31]]
   for a,b in [(ids[0],ids[1]),(ids[1],ids[2]),(ids[0],ids[3]),(ids[3],ids[4]),(ids[4],ids[5]),(ids[2],ids[6])]:
    if np.isfinite(xy[[a,b]]).all():cv2.line(overlay,tuple(xy[a].astype(int)),tuple(xy[b].astype(int)),(0,0,255),3)
   for j in ids:
    if np.isfinite(xy[j]).all():cv2.putText(overlay,str(j),tuple(xy[j].astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2)
   tile=Image.fromarray(cv2.cvtColor(overlay[ymin:ymax,xmin:xmax],cv2.COLOR_BGR2RGB));tile.thumbnail((500,440));canvas.paste(tile,(col*500,row*500+40));ImageDraw.Draw(canvas).text((col*500+5,row*500+5),f'{name} {time}s side {side}; confidence '+str(np.round(z['confidence'][ix,ids].mean(0),2)),fill='black')
 c.release();canvas.save(O/(name+'-tracking.jpg'),quality=92)

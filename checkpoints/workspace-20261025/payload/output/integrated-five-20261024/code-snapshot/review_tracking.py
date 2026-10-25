from pathlib import Path
import sys,json,cv2
import numpy as np
from PIL import Image,ImageDraw
from common import B,O,read,save
EDGES=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
def main():
 inv=read(B/'output/r2d2-review-20261024/inventory.json');panels=[];stats=[]
 for r in inv:
  cache=O/'team-observations'/r['id']/'side.npz'
  if not cache.exists():continue
  z=np.load(cache);seq=z['image_xy'];frames=z['frame_numbers'];strip_cache=O/'tracking-strips'/(r['id']+'.png')
  if strip_cache.exists():
   panels.append(Image.open(strip_cache).copy());stats.append(dict(source_id=r['id'],torso_coverage=float(np.isfinite(seq[:,[11,12,23,24]]).all((1,2)).mean())));continue
  cap=cv2.VideoCapture(r['copy']);w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));strip=Image.new('RGB',(240*3,450),'#fff');draw=ImageDraw.Draw(strip);draw.text((6,6),r['id'],fill='black')
  for j,index in enumerate(np.linspace(0,len(frames)-1,5).astype(int)[1:4]):
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(frames[index]));ok,frame=cap.read()
   if not ok:continue
   xy=seq[index];scale=240/w;im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((240,round(h*scale)));d=ImageDraw.Draw(im)
   for a,b in EDGES:
    if np.isfinite(xy[[a,b]]).all():d.line(tuple((float(x*scale),float(y*scale)) for x,y in xy[[a,b]]),fill='#00ff70',width=3)
   strip.paste(im,(j*240,22));draw.text((j*240+5,430),f"{z['time_seconds'][index]:.2f}s",fill='black')
  strip_cache.parent.mkdir(exist_ok=True);strip.save(strip_cache);cap.release();panels.append(strip);stats.append(dict(source_id=r['id'],torso_coverage=float(np.isfinite(seq[:,[11,12,23,24]]).all((1,2)).mean())))
 for page in range((len(panels)+5)//6):
  sheet=Image.new('RGB',(1440,1350),'#eee')
  for j,im in enumerate(panels[page*6:(page+1)*6]):sheet.paste(im,((j%2)*720,(j//2)*450))
  sheet.save(O/f'tracking-review-{page+1:02}.jpg',quality=90)
 save('tracking-observation-statistics.json',stats)
if __name__=='__main__':main()

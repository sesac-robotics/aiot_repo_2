"""Sample original frames with corrected observations, without grade overlays."""
import json,cv2,numpy as np
from PIL import Image,ImageDraw
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/pushup-root-cause-20261024';plans=json.loads((O/'crop-plan-v2.frozen.json').read_text());wanted=['subject_001_push_up_good_side','subject_012_push_up_bad_side','subject_009_push_up_bad_side','V28','V30','V35','V36'];records=[]
if '--test' in __import__('sys').argv:wanted=['V33','V34']
for k,name in enumerate(wanted):
 r=next(r for r in plans if r['id']==name);z=np.load(O/'observations'/name/'primary.npz');xy=z['image_xy'];frames=z['frame_numbers'];canvas=Image.new('RGB',(1500,450),'white');c=cv2.VideoCapture(r['source'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
 for col,j in enumerate(np.linspace(0,len(frames)-1,3).round().astype(int)):
  c.set(1,int(frames[j]));ok,a=c.read();assert ok
  for offset,color in [(0,(0,0,255)),(1,(0,255,0))]:
   for u,v in [(11,13),(13,15),(11,23),(23,25),(25,27),(15,31)]:
    u+=offset;v+=offset
    if np.isfinite(xy[j,[u,v]]).all():cv2.line(a,tuple(xy[j,u].astype(int)),tuple(xy[j,v].astype(int)),color,3)
  x0,y0,x1,y1=r['crop'];tile=Image.fromarray(cv2.cvtColor(a[y0:y1,x0:x1],cv2.COLOR_BGR2RGB));tile.thumbnail((500,415));canvas.paste(tile,(col*500,30));ImageDraw.Draw(canvas).text((col*500+3,5),name+f' f{frames[j]}',fill='black')
 c.release();path=O/(name+'-corrected-tracking.jpg');canvas.save(path,quality=91);records.append(str(path))
(O/('qa-test-sheets.json' if '--test' in __import__('sys').argv else 'qa-train-sheets.json')).write_text(json.dumps(records,indent=2));print('QA_SHEETS',len(records),flush=True)

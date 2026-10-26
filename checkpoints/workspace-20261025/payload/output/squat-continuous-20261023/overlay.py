from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent;S=O.parent/'squat-relabel-expanded-20261023';rs=json.loads((S/'source-manifest.json').read_text());out=[]
for id in ['Vasil_01','Vasil_02','Vasil_08']:
 r=next(r for r in rs if r['id']==id);z=np.load(S/'observations'/id/'front.npz');x=z['image_xy'];frames=z['frame_numbers'];print(id,frames,flush=True);c=cv2.VideoCapture(r['views']['front']['path']);canvas=Image.new('RGB',(6*320,2*410),'#edf3f5');d=ImageDraw.Draw(canvas)
 indexes=np.linspace(0,len(x)-1,6,dtype=int)
 for col,i in enumerate(indexes):
  c.set(cv2.CAP_PROP_POS_FRAMES,int(frames[i]));ok,bgr=c.read()
  if not ok:continue
  for row in [0,1]:
   im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(im)
   if row:
    for a,b in [(23,25),(25,27),(24,26),(26,28),(23,24)]:
     if np.isfinite(x[i,[a,b]]).all():draw.line([tuple(x[i,a]),tuple(x[i,b])],fill='yellow',width=3)
    for k in [23,24,25,26,27,28]:
     if np.isfinite(x[i,k]).all():
      px,py=x[i,k];draw.ellipse((px-5,py-5,px+5,py+5),fill='red');draw.text((px+5,py),str(k),fill='black')
   im.thumbnail((316,375));canvas.paste(im,(col*320,row*410+28));d.text((col*320+3,row*410+3),f'{id} f{frames[i]} '+('pose' if row else 'raw'),fill='black')
 c.release();canvas.save(O/f'{id}-overlay.jpg')

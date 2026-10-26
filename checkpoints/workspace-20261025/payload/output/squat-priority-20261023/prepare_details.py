from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
O=Path(__file__).parent;allr=json.loads((O/'preparation.json').read_text())['selected']+json.loads((O/'ranked-preparation.json').read_text());ids=['00155040','00150869','00113178','00033517','00240934','00074550']
for identifier in ids:
 r=next(r for r in allr if r['id']==identifier);c=cv2.VideoCapture(r['path']);frames=np.linspace(0,min(r['frame_count']-1,100),16,dtype=int).tolist();canvas=Image.new('RGB',(8*220,2*300),'#edf3f5');d=ImageDraw.Draw(canvas)
 for i,f in enumerate(frames):
  c.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=c.read()
  if not ok:continue
  im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((216,266));x=i%8*220;y=i//8*300;canvas.paste(im,(x,y+24));d.text((x+3,y+3),f'f{f}',fill='black')
 c.release();canvas.save(O/f'detail-{identifier}.jpg');r['dense_review_frames']=frames
(O/'dense-preparation.json').write_text(json.dumps([r for r in allr if r['id'] in ids],ensure_ascii=False,indent=2))

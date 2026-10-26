"""피사체 주변을 확대한 시간순 원본 표본. 모델·관절선 없이 검토한다."""
import json,cv2,numpy as np
from pathlib import Path
O=Path(__file__).parent;rs=json.loads((O/'source-manifest.json').read_text())
for person in ['Petra','Terka','Vasil','Mima']:
 tiles=[]
 for r in [r for r in rs if r['person']==person]:
  for view in ['front','side']:
   cap=cv2.VideoCapture(r['views'][view]['path']);ims=[]
   for tick in np.linspace(r['first_frame'],r['last_frame'],9).astype(int):
    cap.set(1,int(tick));ok,f=cap.read()
    if not ok:raise RuntimeError(r['id'])
    h,w=f.shape[:2];f=f[int(h*.30):int(h*.88),int(w*.20):int(w*.95)];scale=min(160/f.shape[1],210/f.shape[0]);f=cv2.resize(f,(int(f.shape[1]*scale),int(f.shape[0]*scale)));tile=np.full((235,160,3),245,np.uint8);y=25+(210-f.shape[0])//2;x=(160-f.shape[1])//2;tile[y:y+f.shape[0],x:x+f.shape[1]]=f;cv2.putText(tile,f'{r["id"]} {view[0]} {tick}',(2,17),0,.35,(0,0,0),1);ims.append(tile)
   cap.release();tiles.append(np.hstack(ims))
 cv2.imwrite(str(O/f'{person}-dense.jpg'),np.vstack(tiles));print(person,flush=True)

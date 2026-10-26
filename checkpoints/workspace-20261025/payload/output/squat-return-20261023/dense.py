from pathlib import Path
import json,cv2,numpy as np
O=Path(__file__).resolve().parent;rs=json.loads((O/'screen-manifest.json').read_text());r=next(r for r in rs if r['id']=='00152596');c=cv2.VideoCapture(r['path']);canvas=np.full((6*220,6*260,3),245,np.uint8)
for j,k in enumerate(np.linspace(0,r['frame_count']-1,36).round().astype(int)):
 c.set(1,int(k));ok,im=c.read()
 if not ok:continue
 scale=min(260/im.shape[1],195/im.shape[0]);im=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)));x=j%6*260;y=j//6*220;canvas[y+24:y+24+im.shape[0],x:x+im.shape[1]]=im;cv2.putText(canvas,f'f{k}',(x+2,y+16),0,.4,(0,0,0),1)
c.release();cv2.imwrite(str(O/'dense-00152596.jpg'),canvas)

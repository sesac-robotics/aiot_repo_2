import json,cv2,numpy as np
from pathlib import Path
O=Path(__file__).resolve().parent;B=O.parents[1];rs=json.loads((B/'output/squat-errors-20261023/pose-search.json').read_text());seen=set();chosen=[]
for r in sorted([r for r in rs if r.get('return_signal') is not None],key=lambda r:r['return_signal'],reverse=True):
 if r['person'] in seen:continue
 seen.add(r['person']);chosen.append(r)
 if len(chosen)==4:break
for r in chosen:
 cap=cv2.VideoCapture(r['path']);canvas=np.full((4*270,4*350,3),242,np.uint8);frames=np.linspace(0,r['frame_count']-1,16).round().astype(int)
 for j,k in enumerate(frames):
  cap.set(1,int(k));ok,im=cap.read()
  if not ok:raise RuntimeError(r['id'])
  scale=min(350/im.shape[1],244/im.shape[0]);im=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)));x=(j%4)*350;y=(j//4)*270;canvas[y+25:y+25+im.shape[0],x:x+im.shape[1]]=im;cv2.putText(canvas,f'{r["id"]} f{k}',(x+3,y+18),0,.4,(0,0,0),1)
 cap.release();path=O/f'return-{r["id"]}.jpg';cv2.imwrite(str(path),canvas);r.update(review_frames=frames.tolist(),new_evidence=str(path))
(O/'return-candidates.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2));print([(r['id'],r['person'],r['return_signal']) for r in chosen])

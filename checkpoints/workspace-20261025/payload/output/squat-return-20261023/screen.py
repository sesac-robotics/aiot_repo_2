from pathlib import Path
import json,cv2,numpy as np
O=Path(__file__).resolve().parent;B=O.parents[1]
rows=json.loads((B/'output/squat-errors-20261023/pose-search.json').read_text());old={r['id'] for r in json.loads((B/'output/squat-error-strengthening-20261023/return-candidates.json').read_text())};rs=[r for r in rows if r['id'] not in old and r.get('return_signal') is not None and r['torso_coverage']>=.8 and .2<r['return_signal']<.9];rs.sort(key=lambda r:('squats - insufficient' not in r['provider_hints'],-r['return_signal']));chosen=rs[:32]
for page in range(8):
 canvas=np.full((4*225,8*180,3),245,np.uint8)
 for row,r in enumerate(chosen[page*4:page*4+4]):
  cap=cv2.VideoCapture(r['path']);frames=np.linspace(0,r['frame_count']-1,8).round().astype(int)
  for col,k in enumerate(frames):
   cap.set(1,int(k));ok,im=cap.read()
   if not ok:continue
   scale=min(180/im.shape[1],200/im.shape[0]);im=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)));x=col*180;y=row*225;canvas[y+24:y+24+im.shape[0],x:x+im.shape[1]]=im;cv2.putText(canvas,f'{r["id"]} f{k}',(x+2,y+17),0,.36,(0,0,0),1)
  cap.release()
 cv2.imwrite(str(O/f'screen-{page:02}.jpg'),canvas)
(O/'screen-manifest.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2))
print([(r['id'],r['person'],round(r['return_signal'],2)) for r in chosen])

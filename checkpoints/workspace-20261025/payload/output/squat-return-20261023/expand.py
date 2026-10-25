from pathlib import Path
import json,cv2,numpy as np,random
O=Path(__file__).resolve().parent;B=O.parents[1]
rows=json.loads((B/'output/squat-priority-20261023/remaining-review.json').read_text())['remaining'];reserved=set(json.loads((B/'output/five-exercise-recovery-20261023/final_test_reservation.json').read_text())['reserved_people'])|{'qevd_'+str(x) for x in json.loads((B/'data/qevd-selected/manifests/official_test_workers.json').read_text())};seen={r['id'] for r in json.loads((O/'screen-manifest.json').read_text())};rows=[r for r in rows if r['person'] not in reserved and r['id'] not in seen and 'squats - insufficient' in r['provider_hints']];random.Random(623).shuffle(rows);chosen=rows[:96]
for page in range(12):
 canvas=np.full((8*165,8*180,3),245,np.uint8)
 for row,r in enumerate(chosen[page*8:page*8+8]):
  cap=cv2.VideoCapture(r['path']);n=int(cap.get(7));r['frame_count']=n;r['fps']=cap.get(5);r['sheet']=str(O/f'extra-{page:02}.jpg');frames=np.linspace(0,n-1,8).round().astype(int)
  for col,k in enumerate(frames):
   cap.set(1,int(k));ok,im=cap.read()
   if not ok:continue
   scale=min(180/im.shape[1],140/im.shape[0]);im=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)));x=col*180;y=row*165;canvas[y+23:y+23+im.shape[0],x:x+im.shape[1]]=im;cv2.putText(canvas,f'{r["id"]} f{k}',(x+2,y+16),0,.36,(0,0,0),1)
  cap.release()
 cv2.imwrite(str(O/f'extra-{page:02}.jpg'),canvas)
(O/'extra-manifest.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2));print('screened',len(chosen),flush=True)

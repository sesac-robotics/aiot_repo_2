"""기존 train/validation의 정면·측면 표본만 준비하며 최종 test는 열지 않는다."""
import json,cv2,numpy as np,hashlib
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/squat-relabel-20261023';m=json.loads((B/'mediapipe-pilot/correction/sample-manifest.json').read_text());manifest=[]
classes=['NORMAL','ERR_KNEES_INSIDE','ERR_TORSO_TO_SIDE','ERR_SHALLOW_SQUAT','ERR_LIFTED_HEELS']
for person in ['Petra','Terka','Vasil','Mima']:
 rows=[]
 for j,label in enumerate(classes,1):
  r=next(r for r in m if r['person']==person and r['label']==label and r['camera']=='2');rid=f'{person}_{j:02d}'
  rec=dict(id=rid,person=person,split=r['split'],first_frame=r['first_frame'],last_frame=r['last_frame'],provider_label_candidate_only=label,views={})
  ims=[]
  for camera,view in [('1','front'),('3','side')]:
   path=B/f'data/rehab26-squat/raw/videos/ex6/ex6_{person}_cam{camera}.mp4';cap=cv2.VideoCapture(str(path));fps=cap.get(5)
   rec['views'][view]=dict(path=str(path),fps=fps,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
   for fnum in np.linspace(r['first_frame'],r['last_frame'],3).astype(int):
    cap.set(1,int(fnum));ok,f=cap.read()
    if not ok:raise RuntimeError(path)
    # 원본 비율을 유지하고 좌우에 여백을 둔다.
    h,w=f.shape[:2];scale=min(192/w,256/h);im=cv2.resize(f,(int(w*scale),int(h*scale)));tile=np.full((286,192,3),245,np.uint8);y=30+(256-im.shape[0])//2;x=(192-im.shape[1])//2;tile[y:y+im.shape[0],x:x+im.shape[1]]=im;cv2.putText(tile,f'{rid} {view} f{fnum}',(3,18),0,.38,(0,0,0),1);ims.append(tile)
   cap.release()
  rows.append(np.hstack(ims));manifest.append(rec)
 cv2.imwrite(str(O/f'{person}-review.jpg'),np.vstack(rows))
(O/'source-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('selected',len(manifest),'no test sources opened')

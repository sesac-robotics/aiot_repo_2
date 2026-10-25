import json,hashlib,cv2,numpy as np
from pathlib import Path
O=Path(__file__).resolve().parent;B=O.parents[1];rs=json.loads((B/'mediapipe-pilot/correction/sample-manifest.json').read_text());rows=[]
for person in ['Petra','Terka','Vasil','Mima']:
 for angle in ['10','20']:
  chosen=[next(r for r in rs if r['person']==person and r['camera']=='1' and r['orientation']==angle and r['label']==label) for label in ['NORMAL','ERR_KNEES_INSIDE','ERR_TORSO_TO_SIDE']]
  canvas=np.full((3*290,9*205,3),242,np.uint8)
  for i,r in enumerate(chosen):
   path=B/'data/rehab26-squat/raw/videos/ex6'/f'ex6_{person}_cam1.mp4';cap=cv2.VideoCapture(str(path));frames=np.linspace(r['first_frame'],r['last_frame'],9).round().astype(int);ident=f'{person}_a{angle}_{i}'
   for j,k in enumerate(frames):
    cap.set(1,int(k));ok,im=cap.read()
    if not ok:raise RuntimeError(ident)
    h,w=im.shape[:2];im=im[int(.30*h):int(.90*h),int(.20*w):int(.95*w)];scale=min(205/im.shape[1],263/im.shape[0]);im=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)));x=j*205+(205-im.shape[1])//2;y=i*290+25;canvas[y:y+im.shape[0],x:x+im.shape[1]]=im;cv2.putText(canvas,f'{ident} f{k}',(j*205+2,i*290+18),0,.38,(0,0,0),1)
   cap.release();rows.append(dict(id=ident,person='rehab26_'+person,source_person=person,split=r['split'],orientation=int(angle),path=str(path),first_frame=r['first_frame'],last_frame=r['last_frame'],provider_hint_not_label=r['label'],review_frames=frames.tolist(),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),evidence=str(O/f'{person}-{angle}.jpg')))
  cv2.imwrite(str(O/f'{person}-{angle}.jpg'),canvas)
(O/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(len(rows))

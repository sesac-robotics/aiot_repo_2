import json,hashlib,sys
from pathlib import Path
import cv2,numpy as np
O=Path(__file__).resolve().parent;B=O.parents[1]
old=json.loads((B/'output/squat-continuous-20261023/review-decisions.json').read_text())['records']
rows=[dict(id=r['id'],person=r['person'],path=r['path'],prior_decision=r['decision'],prior_reason=r['reason'],prior_evidence=r['evidence_pages']) for r in old if r['decision']=='exclude_fine_knee_labels_crop_or_occlusion']
for r in json.loads((B/'output/squat-unified-20261023/qevd-direct-review.json').read_text()):
 rows.append(dict(id=Path(r['path']).stem,person=r['person'],path=r['path'],prior_decision=r['action'],prior_reason=r['reason'],prior_evidence=[r['evidence_image']]))
for r in rows:
 cap=cv2.VideoCapture(r['path']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);frames=np.linspace(0,n-1,16).round().astype(int);canvas=np.full((4*274,4*360,3),240,np.uint8)
 for i,k in enumerate(frames):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(k));ok,im=cap.read()
  if not ok:raise RuntimeError(r['id'])
  h,w=im.shape[:2];scale=min(360/w,245/h);im=cv2.resize(im,(round(w*scale),round(h*scale)));xx=(i%4)*360;yy=(i//4)*274;canvas[yy+25:yy+25+im.shape[0],xx:xx+im.shape[1]]=im;cv2.putText(canvas,f"{r['id']} frame {k}",(xx+5,yy+18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
 cap.release();e=O/(r['id']+'-review.jpg');cv2.imwrite(str(e),canvas);r.update(frame_count=n,fps=fps,review_frames=frames.tolist(),evidence=str(e),source_sha256=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest())
(O/'review-manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print('review sources',len(rows))

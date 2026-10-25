"""Keep full frame when old landmarks cannot locate full-body crop boundaries."""
import json,numpy as np,cv2
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/pushup-root-cause-20261024';rows=json.loads((O/'crop-plan.frozen.json').read_text());changes=[]
for r in rows:
 z=np.load(r['previous_cache']);x=z['image_xy'];coverage=np.isfinite(x).all(2).mean(0);critical=[max(coverage[js]) for js in [[11,12],[23,24],[27,28],[31,32]]]
 if min(critical)>=.8:continue
 c=cv2.VideoCapture(r['source'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);ok,a=c.read();c.release();assert ok;h,w=a.shape[:2];old=r['crop'];r['crop']=[0,0,w,h];r['roi_fallback']='full_frame_missing_torso_or_foot_anchor';changes.append(dict(id=r['id'],old_crop=old,crop=r['crop'],critical_coverage=critical))
p=O/'crop-plan-v2.frozen.json';assert not p.exists() or json.loads(p.read_text())==rows;p.write_text(json.dumps(rows,indent=2));(O/'crop-boundary-review.json').write_text(json.dumps(dict(reason='D034 raw source feet were visible but first-pass ROI excluded them because old pose missed feet. Prevent self-confirming crop using full-frame fallback whenever either foot/torso anchor is insufficient.',changes=changes,test_targets_used=False),indent=2));print('FULL_FRAME_FALLBACKS',len(changes))

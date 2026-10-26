import sys,json,importlib.util,hashlib
from pathlib import Path
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');A=B/'output/live-webcam-20261023';O=B/'project-documents/squat-motion-recovery-20261025';sys.path.insert(0,str(A))
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=load('old_train',O/'movement.before.py');new=load('new_train',O/'movement_candidate.py');profile=json.loads((B/'output/squat-count-Train-profile-20261025/profile.frozen.json').read_text());rows=[]
for src in profile['sources']:
 name=src['id'];p=B/'output/squat-return-20261023'/f'{name}-heavy.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()==src['cache_sha256'];z=np.load(p);ticks=np.rint(z['frame_numbers']*10/float(z['fps'])).astype(int);w=int(z['width']);h=int(z['height']);versions={}
 for version,module in [('before',old),('candidate',new)]:
  t=module.MovementTracker('squats','front');segs=[]
  for tick,xy in zip(ticks,z['image_xy']):
   seg=t.push(int(tick),xy,width=w,height=h)
   if seg:segs.append(dict(start=seg[0][0],end=seg[-1][0],samples=len(seg)))
  versions[version]=dict(count=t.count,segments=segs,uncertain=t.uncertain_attempts,recoveries=t.recovery_receipts if hasattr(t,'recovery_receipts')else [])
 rows.append(dict(source=name,split='train',person=next(r['person']for r in profile['records']if r['source']==name),**versions));print(name,versions['before']['count'],versions['candidate']['count'],flush=True)
(O/'train-replay.json').write_text(json.dumps({'scope':'Train source continuous replay; existing return labels are not count ground truth; no new labels or fit','source_count':len(rows),'rows':rows},ensure_ascii=False,indent=2)+'\n')

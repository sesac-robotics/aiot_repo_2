import json,sys,importlib.util
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[2];sys.path.insert(0,str(B/'tools/error-enrichment'));from common import sha
spec=importlib.util.spec_from_file_location('u',B/'tools/return-root-cause/features.py');u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
rs=json.loads((B/'output/squat-return-20261023/labels-v2.frozen.json').read_text())['labels'];z=np.load(B/'output/squat-return-20261023/squat_diag_w2_5_7-heavy.npz');rx=z['image_xy'][(z['frame_numbers']>=68)&(z['frame_numbers']<=76)];s=rx[:,[11,12]].mean(1);h=rx[:,[23,24]].mean(1);ref=dict(hip_y=float(np.median(h[:,1])),shoulder_y=float(np.median(s[:,1])),center_x=float(np.median(h[:,0])),scale=float(np.median(np.linalg.norm(h-s,axis=1))),torso_x=float(np.median((s-h)[:,0])));out=[]
for r in rs:
 if 'diag_w2' not in r['id']:continue
 vals=[]
 for extra in [0,12]:
  m=(z['frame_numbers']>=r['first_frame'])&(z['frame_numbers']<=r['last_frame']+extra);frames=z['frame_numbers'][m];a=u.extract(z['image_xy'][m],ref)
  vals.append(dict(extra=extra,top_frame=int(frames[a['return_index']]) if a else None,feature=a['x'] if a else None,bottom_frame=int(frames[a['bottom_index']]) if a else None))
 out.append(dict(id=r['id'],label=r['grade'],end_frame=r['last_frame'],contexts=vals))
print(json.dumps(out,indent=2));(B/'output/return-recheck-20261024/context-audit.json').write_text(json.dumps(out,indent=2))

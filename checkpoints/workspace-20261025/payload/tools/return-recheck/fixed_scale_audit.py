import json,sys,importlib.util
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[2];O=B/'output/return-recheck-20261024';sys.path.insert(0,str(B/'tools/return-discrimination'));import features
rs=json.loads((B/'output/return-discrimination-ensemble-20261024/feature-provenance.json').read_text());z=np.load(B/'output/squat-return-20261023/squat_diag_w2_5_7-heavy.npz');r=z['image_xy'][(z['frame_numbers']>=68)&(z['frame_numbers']<=76)];out=[]
for name in ['r05','r08','r11']:
 p=next(a for a in rs if a['id']=='return_squat_diag_w2_5_7_'+name);s=p['return_source_frame'];q=z['image_xy'][(z['frame_numbers']>=s)&(z['frame_numbers']<=s+6)];sides=[]
 for c in [[11,23,25,27],[12,24,26,28]]:
  a=r[:,c];b=q[:,c];leg=lambda x:np.median(np.linalg.norm(x[:,1]-x[:,2],axis=1)+np.linalg.norm(x[:,2]-x[:,3],axis=1));L=leg(a);Q=leg(b);v=[]
  for j in [1,0]:v.append(float((np.median(b[:,j,1]-b[:,3,1])-np.median(a[:,j,1]-a[:,3,1]))/L))
  sides.append(dict(fixed_height=v,return_to_reference_leg_ratio=float(Q/L)))
 out.append(dict(id=p['id'],old_features=p['features'],sides=sides))
(O/'scale-normalization-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

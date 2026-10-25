import sys,json,importlib.util,collections
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[2];O=B/'output/mild-return-mining-v2-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
spec=importlib.util.spec_from_file_location('f',B/'tools/return-root-cause/features.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
read=lambda p:json.loads(p.read_text());labels=read(B/'output/squat-return-20261023/labels-v2.frozen.json')['labels'];old={r['id'] for r in read(B/'output/mild-return-mining-20261024/intake.frozen.json')};rs=[]
for d in labels:
 if d['split']!='train' or 'return_'+d['id'] in old or d['grade']==1:continue
 stem=Path(d['path']).stem;normal=[a for a in labels if a['path']==d['path'] and a['grade']==2];first=369 if 'diag_m4' in stem else min(a['last_frame'] for a in normal)-6
 if d['first_frame']<=first+8:continue
 z=np.load(B/'output/squat-return-20261023'/(stem+'-heavy.npz'));mask=(z['frame_numbers']>=d['first_frame'])&(z['frame_numbers']<=d['last_frame']);refxy=z['image_xy'][(z['frame_numbers']>=first)&(z['frame_numbers']<=first+8)];q=refxy[:,[11,12]].mean(1);h=refxy[:,[23,24]].mean(1);ref=dict(hip_y=float(np.median(h[:,1])),shoulder_y=float(np.median(q[:,1])),center_x=float(np.median(h[:,0])),scale=float(np.median(np.linalg.norm(h-q,axis=1))),torso_x=float(np.median((q-h)[:,0])))
 a=f.extract(z['image_xy'][mask],ref,int(z['width']),int(z['height']))
 if a is None:continue
 if d['grade'] is not None and (a['x'][0]>.16 or a['x'][3]>.35):continue
 frames=z['frame_numbers'][mask];top=int(frames[a['return_index']]);rs.append(dict(id='return_'+d['id'],review_id=f'D{len(rs)+1:02}',split='train',old_grade=d['grade'],person=d['person'],path=d['path'],source_sha256=sha(d['path']),source_frames=[d['first_frame'],d['last_frame']],reference_frame=first+3,reference_window=[first,first+8],return_frame=top,fps=d['fps'],features=a['x'],review_frames=[first+3]+[max(0,top+k) for k in [-6,0,6,12,18,24,30]],selection='unreviewed low-residual Train normals and unassigned Train labels; no development/Test outcomes'))
assert not (O/'intake.frozen.json').exists();(O/'intake.frozen.json').write_text(json.dumps(rs,indent=2));print('NEW CANDIDATES',len(rs),collections.Counter(r['person'] for r in rs))

import json,sys,importlib.util
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[2];O=B/'output/mild-return-mining-v2-20261024';sys.path[:0]=[str(B/'tools/return-discrimination'),str(B/'tools/error-enrichment')]
import features as geom
spec=importlib.util.spec_from_file_location('upper',B/'tools/return-root-cause/features.py');upper=importlib.util.module_from_spec(spec);spec.loader.exec_module(upper)
read=lambda p:json.loads(p.read_text());labels=read(B/'output/squat-return-20261023/labels-v2.frozen.json')['labels'];byname={r['id']:r for r in labels};rows=read(B/'output/mild-return-calibrated-20261024/score-inputs.frozen.json');audit=[]
for r in rows:
 if not r['id'].startswith(('return_squat_','reviewed_return_squat_')):continue
 name=r['id'].removeprefix('reviewed_').removeprefix('return_');d=byname[name];stem=Path(d['path']).stem;normal=[a for a in labels if a['path']==d['path'] and a['grade']==2];first=369 if 'diag_m4' in stem else min(a['last_frame'] for a in normal)-6
 if d['first_frame']<=first+8:continue
 z=np.load(B/'output/squat-return-20261023'/(stem+'-heavy.npz'));mask=(z['frame_numbers']>=d['first_frame'])&(z['frame_numbers']<=d['last_frame']);rx=z['image_xy'][(z['frame_numbers']>=first)&(z['frame_numbers']<=first+8)];q=rx[:,[11,12]].mean(1);h=rx[:,[23,24]].mean(1);ref=dict(hip_y=float(np.median(h[:,1])),shoulder_y=float(np.median(q[:,1])),center_x=float(np.median(h[:,0])),scale=float(np.median(np.linalg.norm(h-q,axis=1))),torso_x=float(np.median((q-h)[:,0])));a=upper.extract(z['image_xy'][mask],ref,int(z['width']),int(z['height']));f=geom.extract(z['image_xy'][mask],rx,a)
 if f:
  g=next(g for i,g in zip(r['items'],r['levels']) if i['bank']=='height_upper_reversal_v1');audit.append(dict(id=r['id'],person=r['person'],split=r['split'],grade=g,feature=f['x'],return_frame=int(z['frame_numbers'][mask][f['return_index']]),reference_window=[first,first+8]))
(O/'support-geometry-audit.json').write_text(json.dumps(audit,indent=2));print('geometry',len(audit))
for r in audit:
 if r['grade'] is not None and r['grade']<2:print(r['split'],r['id'],[round(x,3) for x in r['feature']])

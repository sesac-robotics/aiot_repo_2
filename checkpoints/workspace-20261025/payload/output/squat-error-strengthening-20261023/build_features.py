import json,hashlib,sys
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;B=O.parents[1];sys.path.insert(0,str(B/'output/live-webcam-20261023'));from item_observation import observed_features
from features import stance_context
prior=json.loads((B/'output/squat-item-observation-20261023/feature-rows.json').read_text());rows=[]
for r in prior:
 z=np.load(r['path']);x=z['image_xy'].copy();frames=z['frame_numbers']
 if 'first_frame' in r:x=x[(frames>=r['first_frame'])&(frames<=r['last_frame'])]
 w=int(z['width']);h=int(z['height']);bad=(x[:,:,0]<0)|(x[:,:,0]>=w)|(x[:,:,1]<0)|(x[:,:,1]>=h);x[bad]=np.nan
 r=dict(r,core=observed_features(x,w,h),stance=stance_context(x),old_corpus=True,orientation=0);rows.append(r)
raw=(O/'labels.frozen.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==(O/'labels.sha256').read_text()
for r in json.loads(raw)['records']:
 p=O/'observations'/r['id']/'front.npz';z=np.load(p);x=z['image_xy'].copy();w=int(z['width']);h=int(z['height']);bad=(x[:,:,0]<0)|(x[:,:,0]>=w)|(x[:,:,1]<0)|(x[:,:,1]>=h);x[bad]=np.nan
 rows.append(dict(id=r['id'],person=r['person'],source=[r['source_sha256']],levels=r['levels'],core=observed_features(x,w,h),stance=stance_context(x),old_corpus=False,origin='new_error_review',orientation=r['orientation'],path=str(p)))
for r in rows:
 for key in ['knee_inward','trunk_balance']:
  if key not in r['core']:continue
  core=r['core'][key];r.setdefault('optional',{})[key]=dict(core)
  if key=='knee_inward' and r['stance']:
   r['optional'][key]=dict(x=core['x']+r['stance']['x'],scalar=r['stance']['scalar'],higher=True,coverage=min(core['coverage'],r['stance']['coverage']))
(O/'feature-rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print('rows',len(rows),'new eligible',[(k,sum(not r['old_corpus'] and r['levels'].get(k) is not None and k in r['core'] for r in rows)) for k in ['knee_inward','trunk_balance']])

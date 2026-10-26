import json,sys,hashlib
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;B=O.parents[1];A=B/'output/live-webcam-20261023';sys.path.insert(0,str(A));from item_observation import observed_features as new
sys.path.insert(0,str(B/'output/squat-relabel-expanded-20261023'));from criteria_features import observed_features as old
prior=json.loads((B/'output/squat-priority-20261023/comparison.json').read_text())['feature_rows'];rows=[]
def features(path,a=None,b=None):
 z=np.load(path);x=z['image_xy'];frames=z['frame_numbers'];x=x[(frames>=a)&(frames<=b)] if a is not None else x.copy();w=int(z['width']);h=int(z['height']);outside=(x[:,:,0]<0)|(x[:,:,0]>=w)|(x[:,:,1]<0)|(x[:,:,1]>=h);x[outside]=np.nan
 return x,w,h,old(x),new(x,w,h)
for r in prior:
 base='squat-priority-20261023' if r['origin']=='priority_normal_controls' else 'squat-relabel-expanded-20261023';path=B/'output'/base/'observations'/r['id']/'front.npz'
 x,w,h,b,a=features(path);rows.append(dict(id=r['id'],person=r['person'],source=r['source'],levels=r['levels'],before=b,after=a,origin='legacy',path=str(path)))
raw=(O/'labels.frozen.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==(O/'labels.sha256').read_text();labels=json.loads(raw)['records'];audit=[]
for r in labels:
 path=O/'observations'/r['id']/'front.npz';x,w,h,b,a=features(path,r['first_frame'],r['last_frame'])
 used={k:v for k,v in r['levels'].items() if v is not None and k in a}
 rows.append(dict(id='recovered_'+r['id'],person=r['person'],source=[r['source_sha256']],levels=r['levels'],before=b,after=a,origin='recovered',path=str(path),first_frame=r['first_frame'],last_frame=r['last_frame']))
 audit.append(dict(id=r['id'],person=r['person'],visually_labeled_items=[k for k,v in r['levels'].items() if v is not None],eligible_items=list(a),usable_labeled_items=list(used),before_eligible_items=list(b),not_used={k:'required_joint_or_start_end_reference_unavailable' for k,v in r['levels'].items() if v is not None and k not in a}))
(O/'feature-rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));(O/'eligibility-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print(json.dumps(audit,ensure_ascii=False));print('rows',len(rows))

"""Freeze label-free ROI recipe and exact-source extraction inventory."""
import sys,json,hashlib,collections
from pathlib import Path
import numpy as np,cv2
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/pushup-root-cause-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'));from common import sha
rows=json.loads((B/'output/label-verified-five-expanded-20261024/score-inputs.frozen.json').read_text());by={r['id']:r for r in rows};inventory=json.loads((O/'train-input-inventory.json').read_text());plans=[]
for r in inventory:
 old=by[r['id']]
 if not r['source'] or not any(i['bank'].startswith('side') for i in old['items']):continue
 p=Path(r['cache']);z=np.load(p);meta=json.loads(p.with_suffix('.json').read_text());source=Path(r['source'])
 if sha(source) not in old['sources']:continue
 plans.append(dict(id=r['id'],video_sha256=sha(source),source=str(source),segment=meta['config']['segment'],required_views=['side'],previous_cache=str(p),split=r['split']))
team=json.loads((B/'output/r2d2-review-20261024/inventory.json').read_text());policy=json.loads((B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json').read_text())
for r in team:
 if r['id'] not in ['V28','V29','V30','V31','V32','V33','V34','V35','V36']:continue
 p=B/'output/integrated-five-20261024/team-observations'/r['id']/'side.npz';meta=json.loads(p.with_suffix('.json').read_text());plans.append(dict(id=r['id'],video_sha256=r['sha256'],source=r['copy'],segment=meta['config']['segment'],required_views=['side'],previous_cache=str(p),split=policy['assignments'][r['id']]))
for r in plans:
 z=np.load(r['previous_cache']);xy=z['image_xy'];finite=xy[np.isfinite(xy).all(2)];lo=np.percentile(finite,1,axis=0);hi=np.percentile(finite,99,axis=0);pad=np.maximum((hi-lo)*.2,30);c=cv2.VideoCapture(r['source'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);ok,im=c.read();c.release();assert ok;h,w=im.shape[:2];r['crop']=[max(0,int(lo[0]-pad[0])),max(0,int(lo[1]-pad[1])),min(w,int(hi[0]+pad[0])+1),min(h,int(hi[1]+pad[1])+1)]
p=O/'crop-plan.frozen.json';assert not p.exists() or json.loads(p.read_text())==plans;p.write_text(json.dumps(plans,indent=2));(O/'protocol.frozen.json').write_text(json.dumps(dict(recipe='Source interval landmark bounding box 1st/99th percentile +20percent/min30px padding, clamp to original frame, Heavy960,10Hz; map back original pixels.',targets_and_test_source_assignments_unchanged=True,model_selection='Training person-group CV only; fixed validation for assessment, diagnostic Test used only after freeze.',test_is_diagnostic_previously_exposed=True,main_hypotheses=['Pose tracking misses elbow at bottom despite high confidence.','Body-line model uses knee-angle nuisance correlated with participant.','Low knee clearance at normal bottom phase confused with sustained extra support.'],criteria_reference='https://www.acefitness.org/resources/everyone/exercise-library/41/push-up/'),indent=2));print('ROI_PLANS',len(plans),collections.Counter(r['split'] for r in plans))

"""Knee-score geometry diagnostics; no visual posture truth inferred from poses."""
from pathlib import Path
import json,sys,datetime
import numpy as np
B=Path(__file__).resolve().parents[2];L=B/'output/live-webcam-20261023';SID='1792858741575615610';O=B/'output/squat-retest-review-20261025'/SID
sys.path.insert(0,str(L));from live_observation import mask_frame_border
from item_observation import observed_features
z=np.load(L/'sessions'/SID/'pose.npz');xy=mask_frame_border(z['image_xy'],int(z['width']),int(z['height']));ts=z['frame_numbers'];review=json.loads((O/'independent-review.frozen.json').read_text());saved=json.loads((L/'sessions'/SID/'final-state.json').read_text());out=[]
for seg in review['segments']:
 mask=(ts>=seg['start_tick'])&(ts<=seg['end_tick']);a=xy[mask];ticks=ts[mask];f=observed_features(a,int(z['width']),int(z['height']));hip=abs(a[:,24,0]-a[:,23,0]);signed=a[:,26,0]-a[:,25,0];knee=abs(signed);ratio=knee/np.maximum(hip,1);relative=ratio/np.median(ratio[:3]);stored=saved['repetitions'][seg['number']-1];item=next(i for i in stored['method_comparison']['item_comparisons']if i['item']=='knee_inward');assert np.isclose(np.percentile(relative,10),item['rule_evidence']['value']);bounds=item['rule_evidence']['boundaries'];scale=np.median(np.linalg.norm(a[:,[11,12]].mean(1)-a[:,[23,24]].mean(1),axis=1));steps=np.linalg.norm(np.diff(a[:,[23,24,25,26]],axis=0),axis=2)/scale
 out.append(dict(number=seg['number'],samples=len(a),knee_feature_p10=f['knee_inward']['scalar'],recomputed_scalar_matches_saved=True,knee_relative_min=float(np.min(relative)),knee_relative_end_median3=float(np.median(relative[-3:])),samples_below_partial_boundary=int((relative<bounds[1]).sum()),knee_order_sign_changes=int((np.sign(signed[1:])!=np.sign(signed[:-1])).sum()),max_joint_step_torso_units=float(np.max(steps)),p95_joint_step_torso_units=float(np.percentile(steps,95)),timecourse=[dict(tick=int(t),knee_width_px=float(k),hip_width_px=float(h),relative_feature=float(r))for t,k,h,r in zip(ticks,knee,hip,relative)]))
v=dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),session=SID,rows=out,RGB_available=False,scope='Confirm the exact saved feature and describe time continuity. Smooth finite poses do not prove anatomical tracking correctness.',new_fit=0,threshold_changed=False)
(O/'knee-geometry-review.frozen.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items()if k!='timecourse'}for r in out],ensure_ascii=False,indent=2))

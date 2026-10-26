import sys,json,hashlib
from pathlib import Path
from fractions import Fraction
from collections import Counter
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ytw-work-20261024/boundary-v2';sys.path.insert(0,str(B/'output/live-webcam-20261023'))
from live_observation import mask_frame_border
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.load(open(O/'evaluation.json'));ref=json.load(open(O/'references-v2.frozen.json'));I={x['id']:x for x in json.load(open(B/'output/r2d2-review-20261024/inventory.json'))};policy=json.load(open(B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json'));logo=json.load(open(B/'output/ytw-work-20261024/logo-v1/replay.json'));audits=[];obs={}
for sid in ref['source_video_sha256']:
 s=I[sid];assert policy['assignments'][sid]=='train';assert sha(s['copy'])==s['sha256']==ref['source_video_sha256'][sid]
 p=B/f'output/integrated-five-20261024/team-observations/{sid}/side.npz';z=np.load(p);fps=float(Fraction(s['fps']));ticks=np.rint(z['frame_numbers']/fps*10).astype(int);rot=abs(float(s['rotation'][0]['rotation']))%180;w,h=(s['height'],s['width']) if rot==90 else (s['width'],s['height']);xy=mask_frame_border(z['image_xy'],w,h);valid=np.isfinite(xy[:,[11,12,13,14,15,16,23,24]]).all((1,2));obs[sid]=(ticks,valid)
 aud=dict(source_id=sid,source_role='train_unchanged',source_sha256=s['sha256'],cache_sha256=sha(p),sample_count=len(ticks),border_safe_required_valid=int(valid.sum()),actual_slots=int(ticks[-1]-ticks[0]+1),coverage=float(valid.sum()/(ticks[-1]-ticks[0]+1)),synchronized_views_and_overlap_stay_same_source=True);audits.append(aud)
for x in r['negative_comparisons']:
 a,b=x['interval_seconds'];ticks,valid=obs[x['source_id']];lo,hi=round(a*10),round(b*10);coverage=float(valid[(ticks>=lo)&(ticks<=hi)].sum()/(hi-lo+1));x['required_joint_slot_coverage']=coverage;x['adequately_observed']=coverage>=.8
# No parent source / same person's linked source is fitted in its fold.
rows=[x for x in json.load(open(B/'output/initial-pose-return-v7-20261024/state-inputs.frozen.json')) if x['exercise']=='ytw'];folds=[]
for f in logo['folds']:
 tr=[x for x in rows if x['id'] in f['train_ids']];assert all(x['person']!=f['excluded_person'] for x in tr);assert sha(f['model'])==f['model_sha256']
 folds.append(dict(excluded_person=f['excluded_person'],train_people=sorted({x['person'] for x in tr}),train_source_hashes=sorted({x['source_sha256'] for x in tr}),model_sha256=f['model_sha256'],no_fit_preprocessing=True))
r['summary']['adequately_observed_negative_episodes']=sum(x['adequately_observed'] for x in r['negative_comparisons']);r['summary']['false_completion_in_adequately_observed_negative_episodes']=sum(x['false_completion'] and x['adequately_observed'] for x in r['negative_comparisons'])
r['observation_source_audit']=audits;r['fold_separation_audit']=folds
(O/'evaluation-audited.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
verification=dict(reference_sha256=sha(O/'references-v2.frozen.json'),plan_sha256=sha(O/'plan.frozen.json'),evaluation_sha256=sha(O/'evaluation.json'),audited_evaluation_sha256=sha(O/'evaluation-audited.json'),source_roles_preserved=True,Test_or_finalTest_used=False,models_refit=False,tracker_changed=False,labels_used_for_fit=False,source_audits=audits,summary=r['summary'],activation_gate_passed=False,code_sha256={str(p.relative_to(B)):sha(p) for p in [B/'tools/ytw-work/boundary_review.py',B/'tools/ytw-work/negative_detail.py',B/'tools/ytw-work/freeze_boundary_v2.py',B/'tools/ytw-work/evaluate_boundary_v2.py',B/'tools/ytw-work/audit_boundary_v2.py']})
(O/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(summary=r['summary'],negative_observation=[(x['reference_id'],x['required_joint_slot_coverage']) for x in r['negative_comparisons']]),ensure_ascii=False,indent=2))

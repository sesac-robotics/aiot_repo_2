import json,sys,copy,shutil,datetime
from pathlib import Path
import cv2,numpy as np,joblib
B=Path('/home/jayden/SeSACPAI-Mid');D=B/'output/paired-failure-mining-20261024';OLD=B/'output/paired-enriched-five-v2-20261024';O=B/'output/paired-failure-enriched-five-v2-20261024';O.mkdir(exist_ok=True);SUB=O/'changed-heads-fit';SUB.mkdir(exist_ok=True)
sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/integrated-five'),str(B/'tools/label-recheck'),str(B/'five-exercise-comparison')]
from common import sha
from split_guard import assert_training_allowed,POLICY
from team_features import describe,height_features
from pose_features import arm_pose
from features import shape,bilateral
from recovery_features import describe_ytw
import three_items
rows=json.loads((OLD/'score-inputs.frozen.json').read_text());existing={r['person']:r['group'] for r in rows};labels=json.loads((D/'labels-v2.frozen.json').read_text())['reviewed'];audit=[];newrows=[];changed=set()
for d in labels:
 if not d['train_eligible']:continue
 p=D/'observations'/d['review_id']/'primary.npz';m=json.loads(p.with_suffix('.json').read_text());assert m['config']['source_sha256']==d['source_sha256'];z=np.load(p);assert str(z['fingerprint'])==m['fingerprint'];times=z['frame_numbers']/d['fps'];seq=z['image_xy'][(times>=d['start'])&(times<d['end'])].copy();cap=cv2.VideoCapture(d['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);w,h=cap.get(3),cap.get(4);cap.release();seq=three_items.mask_image_border(seq,w,h);ref=None
 if d['exercise']=='squats':
  reference_seq=three_items.mask_image_border(z['image_xy'][(times>=0)&(times<=.201)],w,h);ref=height_features.reference(reference_seq)
 if d['exercise']=='ytw':
  notes=describe_ytw(seq)
  for n in notes:n['bank']='front'
 else:notes=describe(d['exercise'],seq,'front',d['end']-d['start'],ref,w,h)
 grades=d['grades'].copy()
 def append(i,bank,f):
  g=d['grades'][i];ok=g is not None and f is not None;notes.append(dict(item=notes[i]['item'],bank=bank,feature=f if ok else None,eligible=ok,holds=[] if ok else ['manual_grade_or_required_pose_unobserved'],features_contain_no_rule_grades=True));grades.append(g)
 if d['exercise']=='ytw':
  f=arm_pose(seq)
  for i,kind in enumerate(['Y','T','W']):append(i,'hinged_arm_pose_v2',f);append(i,'hinged_compact_shape_v3',shape(seq,kind))
 if d['exercise']=='shoulder':
  f=arm_pose(seq)
  for i in [0,2]:append(i,'front_press_arm_pose_v2',f)
  append(0,'front_compact_press_v3',shape(seq,'press'));append(2,'front_compact_timing_v3',bilateral(seq))
 for n,g in zip(notes,grades):
  for k in ['rule_grade','measurement','higher_is_better','proxy_only']:n.pop(k,None)
  if g is None:n.update(eligible=False,feature=None,holds=['direct_visual_grade_uncertain_or_unobservable'])
  if n['feature'] is not None and not np.isfinite(n['feature']).all():n.update(eligible=False,feature=None,holds=['nonfinite_observation'])
 if d['exercise']=='squats' and ref is None:notes[1].update(eligible=False,feature=None,holds=['verified_initial_standing_reference_not_observed'])
 r=dict(id=d['id'],exercise=d['exercise'],person=d['person'],group=existing.get(d['person'],d['person']),sources=[d['source_sha256']],recording_groups=[d['person']],split='train',levels=grades,items=notes,family=d['family'],origin='paired_failure_enrichment',view='front',variant=d['variant'],label_origin='direct_visual_draft',reference_total_allowed=False,whole_attempt_observed=True,source_annotation=d['review_id']);assert_training_allowed([r]);rows.append(r);newrows.append(r)
 changed|={(r['exercise'],n['item'],n['bank']) for n,g in zip(notes,grades) if n['eligible'] and g is not None}
 audit.append(dict(id=r['id'],review_id=d['review_id'],cache_sha256=sha(p),metadata_sha256=sha(p.with_suffix('.json')),source_sha256=d['source_sha256'],start=d['start'],end=d['end'],verified_standing_reference=ref,eligible_items=[n['item']+':'+n['bank'] for n in notes if n['eligible']]))
assert len({r['id'] for r in rows})==len(rows)
for name,v in [('score-inputs.frozen.json',rows),('feature-provenance.json',audit)]:
 p=O/name;assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
for name in ['test-inputs.frozen.json','split-audit.json']:shutil.copyfile(OLD/name,O/name)
subset=[]
for r in rows:
 s=copy.deepcopy(r);pairs=[(n,g) for n,g in zip(s['items'],s['levels']) if (r['exercise'],n['item'],n['bank']) in changed];s['items']=[n for n,g in pairs];s['levels']=[g for n,g in pairs]
 if pairs:subset.append(s)
(SUB/'score-inputs.frozen.json').write_text(json.dumps(subset,ensure_ascii=False,indent=2,allow_nan=False))
protocol=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_run=str(OLD),train_input_sha256=sha(O/'score-inputs.frozen.json'),test_input_sha256=sha(O/'test-inputs.frozen.json'),policy_sha256=sha(POLICY),direct_review_sha256=sha(D/'labels-v2.frozen.json'),changed_keys=sorted(changed),selection='Unchanged Train group CV and fixed development gates; retain all unsupported/failed heads as experimental only',test_used_to_select=False,all_five_reliable=False,production_replacement=False)
(O/'MODEL_SELECTION_PROTOCOL.frozen.json').write_text(json.dumps(protocol,indent=2));print('PREPARED',len(rows),'new',len(newrows),'changed_keys',len(changed),flush=True)

from pathlib import Path
import sys,json,copy,collections,cv2,numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024';sys.path.insert(0,str(B/'tools/integrated-five'));import common;common.O=O
from common import read,save,sha
from team_features import describe,height_features
from pose_features import arm_pose,relative_knee_gap,relative_trunk
sys.path.insert(0,str(B/'five-exercise-comparison'));import recovery_train,recovery_features,three_items,upper_support_event_model
rows=copy.deepcopy(read(B/'output/integrated-five-20261024/score-inputs.frozen.json'));labels=read(B/'data/error-enrichment-20261024/annotations/segments.frozen.json')+read(B/'data/error-enrichment-20261024/annotations/rehab-segments.frozen.json');events=[];new=[]
for d in labels:
 if not d['train_eligible']:continue
 p=O/'observations'/d['review_id']/'primary.npz';z=np.load(p);frames=z['frame_numbers'];mask=(frames/d['fps']>=d['start'])&(frames/d['fps']<d['end']);seq=z['image_xy'][mask].copy();cap=cv2.VideoCapture(d['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);width,height=cap.get(3),cap.get(4);cap.release();seq=three_items.mask_image_border(seq,width,height)
 if d['exercise']=='plank':
  f,diag=upper_support_event_model.event_features(seq)
  events.append(dict(id=d['id'],person=d['person'],group=d['person'],split=d['split'],x=f.tolist() if f is not None and np.isfinite(f).all() else None,target=d['visible_upper_interruption'],source_sha256=d['source_sha256'],scope='upper_support_change_not_whole_body_score',diagnostic=diag));continue
 if d['exercise']=='ytw':
  notes=recovery_features.describe_ytw(seq)
  # A missing automatic phase is not filled with a normal score; pooled observations below can learn observed absence.
  for n in notes:n['bank']='front';n.pop('rule_grade',None)
 else:notes=describe(d['exercise'],seq,d['view'],d['end']-d['start'],height_features.reference(z['image_xy'][:8]),width,height)
 for n,g in zip(notes,d['levels']):
  if g is None:n.update(eligible=False,feature=None)
 for n in notes:
  if n['feature'] is not None and not np.isfinite(n['feature']).all():n.update(eligible=False,feature=None)
 r=dict(id=d['id'],exercise=d['exercise'],person=d['person'],sources=[d['source_sha256']],recording_groups=[d['person']],split=d['split'],levels=d['levels'],items=notes,origin='error_enrichment',family=d['family'],view=d['view'],variant=d['variant'],label_origin='direct_visual_draft',reference_total_allowed=all(g is not None for g in d['levels']),whole_attempt_observed=True,source_annotation=d['id'])
 if d['exercise']=='squats' and d['view']=='front':
  for i,name,bank,f in [(2,'knee_gap','front_gap_relative_v2',relative_knee_gap(seq)),(3,'trunk_balance','front_trunk_relative_v2',relative_trunk(seq))]:
   r['levels'].append(d['levels'][i]);r['items'].append(dict(item=name,bank=bank,feature=f,eligible=f is not None and d['levels'][i] is not None,holds=[] if f else ['required_joints_or_initial_reference_not_observed']))
 if d['exercise']=='ytw':
  f=arm_pose(seq)
  for i,name in enumerate(['y_reach','t_reach','w_shape']):r['levels'].append(d['levels'][i]);r['items'].append(dict(item=name,bank='hinged_arm_pose_v2',feature=f,eligible=f is not None and d['levels'][i] is not None,holds=[] if f else ['required_arms_not_observed']))
 rows.append(r);new.append(r)
# New pose pooling for existing upright team YTW, using existing independent labels.
inv={r['id']:r for r in read(B/'output/r2d2-review-20261024/inventory.json')};teamlabs={r['id']:r for r in read(B/'data/r2d2-team-20261023/annotations/labels.frozen.json')['labels']}
for r in rows:
 if r['family']!='R2D2' or r['exercise'] not in ['ytw','squats']:continue
 d=teamlabs[r['id'][5:]];src=inv[d['source_id']];z=np.load(B/'output/integrated-five-20261024/team-observations'/d['source_id']/'side.npz');from fractions import Fraction
 times=z['frame_numbers']/float(Fraction(src['fps']));seq=z['image_xy'][(times>=d['start'])&(times<d['end'])];f=arm_pose(seq)
 if r['exercise']=='ytw':
  for i,name in enumerate(['y_reach','t_reach','w_shape']):r['levels'].append(r['levels'][i]);r['items'].append(dict(item=name,bank='upright_arm_pose_v2',feature=f,eligible=f is not None and r['levels'][i] is not None,holds=[] if f else ['required_arms_not_observed']))
 elif d['view']=='front':
  for i,name,bank,f in [(2,'knee_gap','front_gap_relative_v2',relative_knee_gap(seq)),(3,'trunk_balance','front_trunk_relative_v2',relative_trunk(seq))]:
   r['levels'].append(r['levels'][i]);r['items'].append(dict(item=name,bank=bank,feature=f,eligible=f is not None and r['levels'][i] is not None,holds=[] if f else ['required_joints_or_initial_reference_not_observed']))
# Ensure split roles follow connected people and original content, not the new clip ID.
groups=recovery_train.component_groups(rows)
for r,g in zip(rows,groups):r['group']=g
roles=collections.defaultdict(set)
for r in rows:roles[r['group']].add(r['split'])
assert all(len(v)==1 for v in roles.values())
reserved=read(B/'output/five-exercise-recovery-20261023/final_test_hashed_manifest.json')['records'];assert not {s for r in reserved for s in r['sources']}&{s for r in rows for s in r['sources']}
save('score-inputs.frozen.json',rows);save('upper-events.frozen.json',events);save('intake-audit.json',dict(previous_rows=921,new_score_rows=len(new),event_rows=len(events),new_rows_by_exercise=dict(collections.Counter(r['exercise'] for r in new)),old_source_families_preserved=sorted({r['family'] for r in rows}),final_test_used=False,unobserved_labels_not_filled=True,quarantined_possible_shared_setting=['E069','E074'],provider_native_aihub_tasks_retained_separately=True));print('ROWS',len(rows),'NEW',len(new),'EVENTS',len(events),flush=True)

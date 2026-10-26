"""Append-only audited Train overlay; preserves original annotation files."""
from pathlib import Path
import json,hashlib,datetime
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-expanded-train-v3-20261025';S={r['id']:r for r in json.loads((O/'source-plan.frozen.json').read_text())};I=json.loads((B/'output/shoulder-data-inventory-20261025/INVENTORY.json').read_text());receipts={r['id']:r for r in json.loads((O/'full-review-preparation.json').read_text())};out=[]
returns={'V20':[[3,3.4],[7.7,8.5],[11.8,12.6],[16.5,17.3],[20.9,21.7]],'V21':[[1.6,2.2],[4.5,5.1],[6.8,7.2],[8.8,9.3],[11,11.7]],'V24':[[3,3.6],[5.8,6.3],[8.7,9.2],[11.5,11.9],[14.5,14.9]],'V26':[[3.5,4],[7.8,8.3],[11.5,12],[15,15.8]],'V27':[[3.3,3.8],[6.7,7.5]]}
counts={}
for old in I['team_original_annotation_rows']:
 sid=old['source_id']
 if sid not in S:continue
 ix=counts.get(sid,0);counts[sid]=ix+1;lo=2;body=2;window=[old['start'],old['end']];turn=returns.get(sid,[]);turn=turn[ix] if ix<len(turn) else None;note='어깨 높이 복귀와 중앙 몸통을 실제 운동 구간에서 관찰.'
 if sid=='V21':note='상단 도달 제한은 복귀 오류가 아님. 각 반복의 실제 하단 구간을 별도로 지정.'
 if sid=='V25':
  lo=None;note='앞 두 구간은 한 팔 중심 동작으로 양팔 공통 하단 등급 제외. 이후 구간은 팔을 옆으로 벌렸다 내리는 비표준 동작으로 제외.'
  if ix>=2:body=None
 if sid=='V27' and ix>=2:
  lo=None;note='한 팔 교대/비대칭 동작이므로 양팔 공통 하단 등급 제외.'
  if ix==4:body=None;note+='몸통과 머리·어깨 움직임의 경계가 모호하여 몸통 보류.'
  if ix>=5:body=1;note+='골반에 비해 몸통 축이 옆으로 기우는 구간이 보임. 원본과 관절을 추가 대조한 뒤 학습 허용.'
 out.append(dict(id=old['id']+'-audit-v3',original_annotation_id=old['id'],source_id=sid,person=S[sid]['person'],source_sha256=S[sid]['source_sha256'],sources=S[sid]['sources'],split='train',posture='seated',view='front',original_clip_window_seconds=window,lower_grade=lo,lower_turn_seconds=turn,lower_scope='observed_source_turn_not_entire_original_clip',body_grade=body,body_window_seconds=window,note_ko=note,review_pages=receipts[sid]['pages'],reviewer='AI_visual_draft_not_independent_expert',status='source_visual_review_complete_tracking_admission_separate',labels_inferred_from_person_or_filename=False))
assert len(out)==31
external=[
 dict(source_id='youtube_gpAwfWZBd1o',windows=[[0,28],[144,196.7]],lower_grade=None,body_grade=None,disposition='talking_titles_not_exercise'),
 dict(source_id='youtube_gpAwfWZBd1o',windows=[[28,48],[72,86],[89,101],[105,129]],lower_grade=None,body_grade=None,disposition='setup_side_view_cuts_or_held_pose_no_closed_frontal_return'),
 dict(source_id='youtube_gpAwfWZBd1o',windows=[[53,70],[130,143]],lower_grade=2,body_grade=None,disposition='visual_lower_normal_candidates_hips_cropped_feature_admission_held'),
 dict(source_id='youtube_vlFGTI5JzjI',windows=[[0,77],[86,90],[102,105],[114,133],[141,158],[172,213.2]],lower_grade=None,body_grade=None,disposition='talking_setup_gestures_not_continuous_target_press'),
 dict(source_id='youtube_vlFGTI5JzjI',windows=[[77,86],[105,114]],lower_grade=2,body_grade=None,disposition='side_view_normal_near_arm_only_no_bilateral_frontal_fit'),
 dict(source_id='youtube_vlFGTI5JzjI',windows=[[90,102],[133,141]],lower_grade=None,body_grade=None,disposition='oblique_press_candidate_requires_dense_target_review_not_red_X_grade'),
 dict(source_id='youtube_vlFGTI5JzjI',windows=[[158,172]],lower_grade=None,body_grade=None,disposition='repeated_short_descent_candidate_pending_dense_review_oblique_bank_only')]
for r in external:r.update(source_sha256=S[r['source_id']]['source_sha256'],split='train',review_pages=receipts[r['source_id']]['pages'],reviewer='AI_visual_draft_not_independent_expert')
doc=dict(schema='shoulder-train-item-window-audit-v3',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_plan_sha256=hashlib.sha256((O/'source-plan.frozen.json').read_bytes()).hexdigest(),team_clips=out,external_interval_review=external,initial_ready_grade_not_used=True,original_frozen_labels_modified=False,independent_final_test=False)
p=O/'visual-labels.frozen.json';assert not p.exists();p.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n');print('31 team rows frozen; external screen recorded')

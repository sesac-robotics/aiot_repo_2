from pathlib import Path
import json,hashlib,collections,sys
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-expanded-train-v3-20261025';sys.path.insert(0,str(B/'tools/error-enrichment'));from split_guard import assert_training_allowed
R=json.loads((O/'feature-candidates.json').read_text());Q={r['id']:r for r in json.loads((O/'qa-render-receipts.json').read_text())};rows=[]
# Import only compatible, previously individually reviewed seated inputs, with immutable ancestry.
parent=B/'output/shoulder-rebuild-v2-20261025/training-inputs.frozen.json'
for r in json.loads(parent.read_text()):
 if r['posture']=='seated':rows.append(dict(r,parent_training_input=str(parent),parent_input_sha256=hashlib.sha256(parent.read_bytes()).hexdigest()))
for r in R:
 q=Q[r['id']];assert q['pages'];qa=dict(reviewer='assistant_direct_original_source_overlay_review',expert_adjudicated=False,pages=[str(O/p) for p in q['pages']],all_actual_context_frames_reviewed=True,anatomical_correctness_guaranteed=False,body_joints='shoulders_and_hips_follow_visible_torso_not_head_or_arms',scope='this_exact_context_only')
 for key,item,criterion in [('body','frontal_trunk_hold','frontal_trunk_hold'),('lower','lower_return','observed_anatomical_lower_turn')]:
  f=r[key]
  if f['status']!='features_ready':continue
  rows.append(dict(id=r['id']+'_'+item,source_id=r['source_id'],source_sha256=r['source_sha256'],sources=r['sources'],person=r['person'],source_group='sha256:'+r['source_sha256'],recording_group=r['source_id'],split='train',view='front',posture='seated',item=item,grade=r[key+'_grade'],criterion=criterion,feature_schema=f['version'],features=f['features'],feature_names=f['feature_names'],cache_path=r['cache_path'],cache_sha256=r['cache_sha256'],cache_fingerprint=r['cache_fingerprint'],visual_qa=qa,label_source=str(O/'visual-labels.frozen.json'),feature_window_seconds=r['window_seconds'],event_native_windows=r['event'] if key=='lower' else None,all_context_native_frames=q['frames'],initial_ready_inferred=False,waiting_or_preparation_included=False))
assert_training_allowed(rows)
assert len(rows)==21;assert len({r['id'] for r in rows})==len(rows)
p=O/'training-inputs.frozen.json';assert not p.exists();p.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
# Dense follow-up confirms target-specific errors, not the source video's red X.
followup=dict(schema='shoulder-additional-visual-adjudication-v3',reviewer='AI_visual_draft_not_expert_adjudication',team_body=dict(source='V27',windows=[[17.9,19.7],[20.2,22]],grade=1,confirmed_visible_target='lateral_torso_axis_shift_relative_to_pelvis_not_head_only',qa_pages=Q['V27-review-7']['pages']+Q['V27-review-8']['pages'],independent_people=1),external_lower=dict(source='youtube_vlFGTI5JzjI',view='oblique',grade=1,events=[dict(peak_seconds=[158.9,159.3],turn_seconds=[160.5,160.8],next_ascent_seconds=[161.3,161.6]),dict(peak_seconds=[161.9,162.2],turn_seconds=[163.2,163.5],next_ascent_seconds=[164,164.3]),dict(peak_seconds=[164.2,164.6],turn_seconds=[165.8,166.1],next_ascent_seconds=[166.8,167.1]),dict(peak_seconds=[167.2,167.6],turn_seconds=[168.8,169.1],next_ascent_seconds=[169.8,170.1])],reason='repeated_visible_descent_stops_with_both_elbows_above_shoulder_rack_then_next_ascent',review_pages=[f'colossus-partial-dense-{i}.jpg' for i in range(1,4)],training_admitted=False,hold_reason='oblique_view_no_validated_frontal_feature_bank_or_joint_QA',independent_people=1),severe_grade_zero_added=0)
p=O/'dense-followup-labels.frozen.json';assert not p.exists();p.write_text(json.dumps(followup,ensure_ascii=False,indent=2)+'\n');print(collections.Counter((r['item'],r['grade']) for r in rows))

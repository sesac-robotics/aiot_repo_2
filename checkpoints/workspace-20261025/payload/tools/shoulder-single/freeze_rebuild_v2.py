from pathlib import Path
import json,collections,hashlib,sys
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-rebuild-v2-20261025';rows=json.loads((O/'event-feature-candidates.json').read_text());renders={r['id']:r for r in json.loads((O/'all-context-render-receipts.json').read_text())};sys.path.insert(0,str(B/'tools/error-enrichment'));from split_guard import assert_training_allowed
# Explicit findings after inspecting EVERY rendered context sample for all automatically
# feature-ready events. These decisions apply only to the identified cache and interval.
manual={
 'subject_018_shoulder_bad_front_event_1':{'both':'native66_crossed_arm_shoulder_assignment_and68_collapsed_torso'},
 'subject_013_shoulder_good_front_event_1':{'body':'native28_shoulder_torso_axis_collapse'},
 'subject_006_shoulder_good_front_event_1':{'both':'native448_shoulder_on_face_with_narrow_torso'},
 'subject_007_shoulder_good_front_event_1':{'both':'native26_shoulder_position_uncertain_at_upper_context'},
 'subject_026_shoulder_bad_front_event_2':{'both':'native244_upper_shoulder_alignment_uncertain'},
 'subject_023_shoulder_bad_front_event_3':{'lower':'unilateral_episode_not_bilateral_press_bank'},
}
admitted=[];review=[]
for r in rows:
 for item,status,grade,field,criterion,schema in [
 ('lower_return',r['lower_status'],r['lower_grade'],'lower','observed_anatomical_lower_turn','shoulder-anatomical-lower-turn-v1'),
 ('frontal_trunk_hold',r['body_status'],r['body_grade'],'body','observed_frontal_window_trunk_stability_only','shoulder-frontal-four-joint-window-quality-v2')]:
  holds=[]
  if status!='features_ready':holds.append(r.get(field,{}).get('reason') or r.get('reason') or 'feature_context_unavailable')
  if grade is None:holds.append('no_existing_visual_item_grade')
  finding=manual.get(r['id'],{});reason=finding.get('both') or finding.get(field)
  if reason:holds.append('manual_tracking_or_protocol_hold:'+reason)
  receipt=renders.get(r['id']);reviewed=status=='features_ready'
  entry=dict(event_id=r['id'],item=item,feature_status=status,visual_reviewed_all_context=reviewed,holds=holds,expert_adjudicated=False,original_grade=grade,source_grade_changed=False)
  review.append(entry)
  if holds:continue
  assert receipt and receipt['all_actual_context_samples_rendered'];f=r[field]
  out=dict(id=r['id']+'_'+item,source_id=r['source_id'],source_sha256=r['source_sha256'],sources=r['sources'],person=r['person'],recording_group=r['recording_group'],source_group='sha256:'+r['source_sha256'],split='train',view='front',posture=r['posture'],item=item,grade=grade,criterion=criterion,feature_schema=schema,features=f['features'],feature_names=f['feature_names'],cache_path=r['cache_path'],cache_sha256=r['cache_sha256'],cache_fingerprint=r['cache_fingerprint'],extraction_delegate=r['delegate'],label_source=r['label_source'],original_reviewed_turn_seconds=r['reviewed_turn_seconds'],event_native_windows=r['event'],feature_window_seconds=r['feature_window_seconds'],all_context_native_frames=receipt['native_frames'],visual_qa={'reviewer':'assistant_direct_original_source_overlay_review','expert_adjudicated':False,'pages':[str(O/p) for p in receipt['pages']],'all_actual_context_frames_reviewed':True,'anatomical_correctness_guaranteed':False},initial_ready_inferred=False,waiting_or_preparation_included=False)
  admitted.append(out)
assert_training_allowed(admitted)
for name,data in [('training-inputs.frozen.json',admitted),('item-review-decisions.frozen.json',review)]:
 p=O/name;assert not p.exists();p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
intake=json.loads((O/'additional-error-review-intake.json').read_text());notes={1:'상단폄/양팔타이밍차이는있으나14.8/16.4~16.8어깨옆하단,몸통큰좌우편위미확인',6:'16.6~16.8어깨옆하단,상단좌우차이와하단오류구분',7:'17.2~17.8어깨옆하단,상단동시성차이/머리움직임을몸통좌우실패로바꾸지않음',10:'14.0~14.2/16.0어깨옆하단,몸통중앙',11:'13초하단뒤상승;앉은시상면신전은정면좌우목표오류와다름',13:'21.6~22.4하단후재상승,정면몸통중앙',14:'21.8~22.4하단후재상승,몸통큰좌우편위미확인',16:'12.6~13.0하단,정면몸통중앙',18:'10.6~11.4하단후재상승,몸통큰좌우편위미확인',19:'18.8~19.2하단이조금높게보이나어깨선경계/가림으로1등급확정보류',23:'교대/한팔우세구간.양팔하단/몸통좌우오류확정불가,은행분리필요',26:'24.0~24.4/25.6~26.0하단,상단범위문제와구분'}
for r in intake:
 num=int(r['source_id'].split('_')[1]);r.update(review_status='assistant_direct_5Hz_source_review_not_expert',note_ko=notes[num],new_error_grade=None,training_admitted=False,all_repetitions_reviewed=False)
(O/'additional-error-review-decisions.frozen.json').write_text(json.dumps(intake,ensure_ascii=False,indent=2)+'\n')
summary={'admitted_item_rows':len(admitted),'by_item':dict(collections.Counter(r['item'] for r in admitted)),'source_videos':len({r['source_sha256'] for r in admitted}),'people':len({r['person'] for r in admitted}),'banks':dict(collections.Counter(r['item']+'/'+r['posture'] for r in admitted)),'grade_counts':dict(collections.Counter(str(r['grade']) for r in admitted)),'held_item_rows':sum(bool(r['holds']) for r in review),'new_error_grades':0,'all_roles_train':True,'normal_labels_aligned_not_replaced_by_error':True}
(O/'dataset-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(summary)

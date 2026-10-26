from standing_state import *
rs=read(O/'hard-state-candidates.json');old=read(D/'standing-states-combined.frozen.json');seen={(r['source_sha256'],r['frame']) for r in old};decisions={e:y for e,y in [
('E026',[1,0]),('E027',[1,0]),('E029',[1,0]),('E030',[1,0]),('E032',[0,0]),('E035',[1,0]),('E037',[1,0]),('E040',[1,0]),('E041',[1,0]),('E042',[1,0]),('E043',[1,0]),('V01',[1,0]),('V06',[1,0]),('V10',[1,0]),('squat_diag_m1_6_7',[None,1]),('squat_diag_m4_5_7',[1,0]),('squat_diag_w1_9_3',[None,0]),('squat_front_m3_9_3',[0,0]),('squat_diag_m2_8_6',[1,0])]};added=[];excluded=[]
for r in rs:
 for f,y in zip(r['frames'][1:],decisions[r['review_id']]):
  if y is None:excluded.append(dict(review_id=r['review_id'],frame=f,reason='post_exercise_person_translation_not_stable_calibration_state'));continue
  if (r['source_sha256'],f) in seen:continue
  seen.add((r['source_sha256'],f));added.append(dict(id='hard_'+r['review_id']+'_'+str(f),review_id=r['review_id'],path=r['path'],source_sha256=r['source_sha256'],frame=f,fps=r['fps'],person=r['person'],split=r['split'],state='standing' if y else 'flexed',target=y,sheet=r['sheet'],annotation_status='nonexpert_visual_frame_state_draft',failure_label=False,review_note_ko='같은 영상의 기립 기준 표본과 원본 후보 프레임을 나란히 직접 비교. 모델 예측은 라벨로 복사하지 않음.'))
save('hard-state-review.json',dict(added=added,excluded=excluded,validation_frames_added_to_training=False));(D/'standing-states-hard-reviewed.frozen.json').write_text(json.dumps(old+added,ensure_ascii=False,indent=2));print('hard states added',len(added))

"""Versioned rubric alignment. Existing independent visual drafts are reused, never rule-labeled."""
from pathlib import Path
import sys,json,hashlib,copy,collections
import numpy as np
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent;L=R/'output/live-webcam-20261023';sys.path.insert(0,str(L))
from runtime_snapshot import prepare_snapshot
runtime,d=prepare_snapshot();sys.path.insert(0,str(runtime));from app.scoring import three_items
ids=['descent_range','return_extension','knee_foot','trunk_balance'];weights=[30,20,30,20]
rubric=dict(version='squat-core100-observation-v3',exercise='일반 맨몸 스쿼트',weights=dict(zip(ids,weights)),score_range=[0,100],weights_are_project_policy_not_validated_medical_scale=True,expert_validated=False,target_protocol='기존 병렬 허벅지 참조 라벨은 표준 목표 범위 일치도로만 사용. 개인 목표 범위·통증에 대한 임상 평가가 아님.',items=[dict(id=ids[0],title='목표 하강 범위',weight=30,definition='무릎·엉덩이를 굽혀 설정한 목표 범위에 도달하는지',observation='정확한 깊이 라벨은 측면 또는 동기화 측면을 확인. 정면 골반 이동은 추정 특징.',not_equivalent='골반 이동량이 작다는 이유만으로 실제 깊이 부족 확정 금지'),dict(id=ids[1],title='상승·서기 복귀',weight=20,definition='상승 뒤 무릎·엉덩이를 펴 서기 자세로 복귀',observation='종료 상단의 관절 신전 확인; 영상 잘림·준비·휴식·의도적 홀드를 실패로 라벨링하지 않음',not_equivalent='화면상 골반 높이 일치만으로 신전 정상 확정 금지'),dict(id=ids[2],title='무릎·발 정렬',weight=30,definition='무릎의 뚜렷한 안쪽 붕괴와 실제 발 지지 이탈을 관측',observation='발 방향·발 지지가 안 보이면 합성 항목 전체 정답을 만들지 않음. 나머지 운동 인식·평가는 계속.',not_equivalent='무릎이 발끝 앞으로 나오는 것·좁은 발 간격 자체는 자동 오류가 아님'),dict(id=ids[3],title='몸통 좌우 안정',weight=20,definition='정면에서 뚜렷한 몸통 좌우 기울임·이탈이 있는지',observation='어깨·골반으로 관측. 전후 허리 정렬·호흡·근력·관절 부하는 판정 범위 아님',not_equivalent='자연스러운 전경 또는 시작 대비 작은 변화만으로 감점하지 않음')],annotation_states=['충족','부분 미충족','명확한 미충족','관측 불가','판단 불확실'],uncertain_is_not_partial_grade=True,internal_levels={'2':'충족','1':'부분 미충족','0':'명확한 미충족','null':'관측 불가 또는 판단 불확실; reason 별도'},conversion='항목점수=배점×내부등급/2; AI 연속 예측에도 동일 적용. 전체 네 항목 평가 시에만 합계 /100. 누락은 재정규화·0점·만점 처리하지 않음.',sources=[dict(title='ACE Bodyweight Squat',url='https://www.acefitness.org/resources/everyone/exercise-library/135/bodyweight-squat/'),dict(title='NASM squat biomechanics',url='https://www.nasm.org/resource-center/blog/training/biomechanics-of-the-squat'),dict(title='Fry 2003 knee position and torques',url='https://pubmed.ncbi.nlm.nih.gov/14636100/')],realtime='진행 중 안내는 잠정값. 완료 반복 단위로 확정·저장. 준비·휴식은 점수 구간에서 제외.',rule_scope='기존 단일 각도 경계는 공학 정책이며 보편적 의학 정상 경계가 아님. 새 경계는 학습 사람에서만 보정하고 사람 분리 검증.')
(O/'rubric.json').write_text(json.dumps(rubric,ensure_ascii=False,indent=2))
paths=[R/'output/five-exercise-recovery-20261022/squat_partial_top_trial/candidate_input_manifest.json',R/'output/five-exercise-recovery-20261023/new_feature_rows.json'];by_id={}
for p in paths:
 a=json.loads(p.read_text());a=a if isinstance(a,list) else a['records']
 for r in a:
  if r['exercise']=='squats':by_id[r['id']]=dict(copy.deepcopy(r),origin_file=str(p),alignment='legacy_visual_draft_criterion_audit')
# Record all corresponding source annotations for traceability, conflicts are not silently overwritten.
notes={}
for p in [R/'output/three-exercise-comparison-20261022/score_annotations.json',R/'output/three-exercise-comparison-20261022/additional_score_annotations.json',R/'output/five-exercise-recovery-20261023/labels.json']:
 if p.exists():
  for r in json.loads(p.read_text()):
   if r.get('exercise')=='squats':notes[r['id']]=r
migration=[]
for r in by_id.values():
 a=notes.get(r['id'],{});text=' '.join([str(a.get(k,'')) for k in ['note_ko','labels_descriptive']]);actions=[]
 for i in range(4):
  item=r['items'][i];reason=None
  # Legacy height-based frontal return is not joint extension ground truth.
  if i==1 and item.get('bank')=='front':reason='정면 높이·무릎 투영만으로 실제 복귀 신전 라벨을 검증할 수 없어 별도 보조 관측으로 이동'
  if i==2 and r['levels'][i] is not None and ('발끝 앞' in text or '무릎 전방' in text or 'knees over toes' in text.lower()) and not any(w in text for w in ['뒤꿈치 들','안쪽 붕괴','안쪽 이동','지지 이탈']):reason='전방 무릎 이동을 오류로 취급했을 가능성: 새 기준 정렬·지지 근거 재검토 필요'
  if reason:
   actions.append(dict(item=ids[i],before=r['levels'][i],after=None,reason=reason));r['levels'][i]=None;item['eligible']=False;item['feature']=None
 r['label_origin']='existing_independent_visual_draft_migrated';r['rubric_version']=rubric['version'];r['relabel_actions']=actions
 migration.append(dict(id=r['id'],person=r['person'],origin=r['origin_file'],actions=actions,reused_items=[ids[i] for i in range(4) if r['levels'][i] is not None],expert_validated=False))
# Checklist labels are not a replacement for original depth/return/composite knee targets.
S=R/'output/squat-relabel-expanded-20261023';labelbytes=(S/'labels.frozen.json').read_bytes()
for a in json.loads(labelbytes)['records']:
 z=np.load(S/'observations'/a['id']/'front.npz');x=three_items.mask_image_border(z['image_xy'],float(z['width']),float(z['height']));note=three_items.describe('squats',None,x)[3];note['bank']='front'
 other=[dict(item=k,bank='front',eligible=False,feature=None,holds=['checklist_target_not_equivalent_to_original_item']) for k in ids[:3]]
 key='aligned_rehab26_'+a['id'];r=dict(id=key,exercise='squats',person='rehab26_'+a['person'],split=a['split'],sources=a['sources'],recording_groups=['rehab26_'+a['person']],levels=[None,None,None,a['levels']['trunk_balance']],items=other+[note],reference_total_allowed=False,whole_attempt_observed=True,origin_file=str(S/'labels.frozen.json'),label_origin='independent_visual_draft_same_trunk_semantics',rubric_version=rubric['version']);by_id[key]=r
 migration.append(dict(id=key,person=r['person'],origin=r['origin_file'],actions=[dict(item='descent_range',after=None,reason='기존 별도 깊이 후보를 유지; 이 체크리스트를 원래 모델 라벨로 무조건 변환하지 않음'),dict(item='return_extension',after=None,reason='높이 복귀 체크리스트와 관절 신전은 다름'),dict(item='knee_foot',after=None,reason='무릎 간격·뒤꿈치 라벨을 정렬 합성 정답으로 자동 결합하지 않음')],reused_items=['trunk_balance'],expert_validated=False))
# Newly re-reviewed clips are frozen before feature extraction/model scoring.
from app.scoring.media import load_landmarks
from app.scoring.features import describe
import joblib
old_bundle=joblib.load(runtime/'artifacts/posture_score/squats.joblib')
for v in json.loads((O/'new_visual_labels.json').read_text())['records']:
 seq=load_landmarks(O/'observations'/v['id']/(v['view']+'.npz'))
 notes_v=describe('squats',seq,old_bundle,view=v['view'])
 for i,n in enumerate(notes_v):
  if v['levels'][i] is None:n['eligible']=False;n['feature']=None
 by_id[v['id']]=dict(id=v['id'],exercise='squats',person=v['person'],split='train',sources=[hashlib.sha256(Path(v['source_path']).read_bytes()).hexdigest()],recording_groups=[],levels=v['levels'],items=notes_v,reference_total_allowed=False,whole_attempt_observed=True,origin_file=str(O/'new_visual_labels.json'),label_origin='new_direct_independent_visual_draft',rubric_version=rubric['version'])
 migration.append(dict(id=v['id'],person=v['person'],origin=str(O/'new_visual_labels.json'),actions=[dict(item='return_extension' if v['view']=='side' else 'trunk_balance',before='unreviewed_provider_hint',after=2,reason=v['evidence'])],reused_items=[],expert_validated=False))
rows=list(by_id.values());doc=dict(rubric_version=rubric['version'],expert_validated=False,frozen_before_training=True,rows=rows)
(O/'labels.aligned.frozen.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False));(O/'label-migration.json').write_text(json.dumps(migration,ensure_ascii=False,indent=2));summary=dict(rows=len(rows),people_tokens=len({r['person'] for r in rows}),actions=collections.Counter(a['item'] for r in migration for a in r['actions']),coverage={ids[i]:dict(collections.Counter(str(r['levels'][i]) for r in rows if r['items'][i]['eligible'] and r['levels'][i] is not None)) for i in range(4)},draft_labels_not_expert_reference=True)
(O/'label-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))

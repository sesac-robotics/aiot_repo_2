"""Read-only metadata compatibility audit. Does not train a model or change splits."""
from pathlib import Path
from collections import Counter, defaultdict
import csv,json
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'checks/training_compatibility'
GROUPS={'squats':'squats','pushups':'pushups','elbow plank':'elbow_plank','shoulder press':'shoulder_press'}
def read(p):return json.loads(p.read_text())
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 datasets={};combined=[];worker_sets={};video_sets={}
 for source,root in [('qevd-selected',ROOT.parent/'qevd-selected'),('shoulder-qevd',ROOT)]:
  labels=read(root/'raw/coach/fine_grained_labels.json')
  with (root/'manifests/short_clips.csv').open() as f: manifest={r['video_id']:r for r in csv.DictReader(f)}
  split_workers={s:{r['worker_id'] for r in manifest.values() if r['split']==s} for s in ['train','test']}
  worker_sets[source]=set.union(*split_workers.values());video_sets[source]=set(manifest)
  count_ex=Counter();count_labels=Counter()
  for annotation in labels:
   prefixes={l.split(' - ')[0] for l in annotation['labels']}
   assert len(prefixes)==1
   prefix=next(iter(prefixes));exercise='ytw' if prefix.startswith('standing ') else GROUPS[prefix]
   video_id=Path(annotation['video_path']).stem;row=manifest[video_id]
   assert (root/row['video_path']).is_file()
   assert annotation['split']==row['split']
   count_ex[exercise]+=1;count_labels.update(annotation['labels'])
   combined.append(dict(source=source,video_id=video_id,exercise=exercise,labels=annotation['labels'],split=annotation['split'],worker_group=f"{source}:{row['worker_id']}"))
  datasets[source]=dict(clips=len(labels),exercises=dict(count_ex),unique_labels=len(count_labels),multilabel_clips=sum(len(r['labels'])>1 for r in labels),splits=dict(Counter(r['split'] for r in labels)),train_test_worker_overlap=len(split_workers['train']&split_workers['test']),label_counts=dict(count_labels))
 vocabulary=sorted({l for r in combined for l in r['labels']});label_ids={l:i for i,l in enumerate(vocabulary)};exercise_ids={e:i for i,e in enumerate(['squats','pushups','elbow_plank','ytw','shoulder_press'])}
 # Dry-run encoding all rows using plain Python: no video inference or model training.
 encoded_examples=[]
 for r in combined:
  positive_indices=[label_ids[l] for l in r['labels']]
  target=[0]*len(vocabulary)
  for index in positive_indices:target[index]=1
  assert sum(target)==len(r['labels'])
  assert r['exercise'] in exercise_ids
  if (r['source']=='qevd-selected' and len(r['labels'])>1 and not encoded_examples) or (r['source']=='shoulder-qevd' and 'no obvious issue' not in r['labels'][0] and len(encoded_examples)==1):
   encoded_examples.append({**r,'exercise_id':exercise_ids[r['exercise']],'positive_label_indices':positive_indices,'multi_hot_target':target})
 reviews=read(ROOT/'manifests/annotation_review.json');by_id={r['video_id']:r for r in reviews};support=[]
 for label in sorted(datasets['shoulder-qevd']['label_counts']):
  rows=[r for r in combined if r['source']=='shoulder-qevd' and label in r['labels']]
  support.append(dict(label=label,clips=len(rows),subjects=len({by_id[r['video_id']]['subject_id'] for r in rows}),train=sum(r['split']=='train' for r in rows),test=sum(r['split']=='test' for r in rows)))
 test_good=sum(r['split']=='test' and r['source_quality']=='good' for r in reviews);test_bad=sum(r['split']=='test' and r['source_quality']=='bad' for r in reviews)
 report=dict(
  verdict='schema_merge_and_label_encoding_passed; not_ready_for_validated_joint_error_training',
  training_performed=False,pose_extraction_performed=False,datasets=datasets,
  combined=dict(clips=len(combined),exercises=5,raw_label_vocabulary_size=len(vocabulary),splits=dict(Counter(r['split'] for r in combined)),shoulder_share_percent=round(100*100/len(combined),3),video_id_collisions=len(video_sets['qevd-selected']&video_sets['shoulder-qevd']),worker_id_collisions=len(worker_sets['qevd-selected']&worker_sets['shoulder-qevd']),encoding_dry_run_passed_rows=len(combined)),
  shoulder=dict(annotation_statuses=dict(Counter(r['annotation_status'] for r in reviews)),training_ready=sum(r['training_ready'] for r in reviews),landmarks_extracted=sum(r['landmarks'] is not None for r in reviews),test_good=test_good,test_bad=test_bad,always_predict_good_test_accuracy_percent=100*test_good/(test_good+test_bad),class_support=support,feedback_origin='assistant-authored templates; not expert ground truth'),
  current_training_code=dict(path=str(ROOT.parent/'Server/mediapipe_pose/ai_based/train_gru.py'),input='X_squat_v2.npy, y_squat_v2.npy, groups_squat_v2.npy',output='2 classes, CrossEntropyLoss',compatible_without_changes=False),
  issues=[
   'All 100 shoulder annotations are unverified drafts: 70 source-good weak labels, 10 detailed examples and 20 screening candidates.',
   'Shoulder test contains 15 source-good and only 1 error-candidate clip; 5 of 6 candidate error labels have no test examples.',
   'Uneven arm timing has one participant and one clip; independent train/test evaluation of that label is unavailable.',
   'QEVD contains 6059 multi-label clips; existing binary squat training code cannot directly consume these labels.',
   'QEVD labels include form attributes, normal states, visibility conditions and errors. Do not treat every label as an error.',
   'No common extracted pose feature arrays exist in the packaged datasets; raw video decoding/pose extraction and training-compatible sequence construction are still required.',
   'Only shoulder videos represent shoulder press in this combination, so exercise recognition may exploit background/source artifacts. A cross-camera/live ESP32 evaluation is needed.',
   'Clip labels must not be copied onto every frame/window as exact error timing. Unlabeled error dimensions are not automatically confirmed absent.',
   'The 70 raw strings are an encoding vocabulary, not 70 validated error types. Corrective feedback text is not an image-model training target by itself.'
  ])
 (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 (OUT/'label_vocabulary.json').write_text(json.dumps({'exercise_ids':exercise_ids,'label_ids':label_ids,'note':'Raw annotation vocabulary, not verified error taxonomy'},ensure_ascii=False,indent=2)+'\n')
 (OUT/'encoding_examples.json').write_text(json.dumps(encoded_examples,ensure_ascii=False,indent=2)+'\n')
 md=['# QEVD + 어깨 라벨 공동 학습 가능성 점검','', '판정: **자료 형식 병합과 라벨 인코딩은 가능하지만, 현재 코드·라벨 상태 그대로 신뢰할 수 있는 5종목 오류 학습·평가를 진행할 준비는 안 됨.**','', '모델 학습이나 MediaPipe 추출은 실행하지 않았습니다. 원본 라벨·분할도 변경하지 않았습니다.','', '## 실제 점검 수치','',f'- 전체 18,752개: QEVD 18,652 + 어깨 초안 100개. train 17,769 / test 983개.',f'- 운동 그룹 5개, 원본 라벨 문자열 70개. QEVD에서 복수 라벨을 가진 클립 6,059개.',f'- 18,752개 모두 경로 확인 및 70차원 multi-hot 라벨 인코딩 dry-run 통과. 영상·참여자 ID 충돌 0개.',f'- 각 자료의 train/test 참여자 겹침 0개. 서로 다른 자료의 동일 인물 여부는 확인한 것이 아님.',f'- 어깨는 전체의 {report["combined"]["shoulder_share_percent"]}%이며, 검수 완료 및 training_ready 레코드는 0개.', '', '## 어깨 클래스별 지원 현황','', '| 라벨 | 영상 수 | 참여자 수 | train | test |','|---|---:|---:|---:|---:|']
 for r in support:md.append(f"|{r['label'].split(' - ',1)[1]}|{r['clips']}|{r['subjects']}|{r['train']}|{r['test']}|")
 md+=['','어깨 test는 정상 후보 15개·오류 후보 1개입니다. 전부 정상이라고만 예측해도 정확도 93.75%이므로, 이 숫자를 오류 검출 성능이라고 해석하면 안 됩니다.','', '## 기존 코드와의 연결','',f"`{report['current_training_code']['path']}`는 스쿼트 `.npy` 입력을 읽고 정상/비정상 2개를 출력합니다. QEVD JSON과 세부 오류 라벨을 직접 읽거나 5종목을 예측하는 코드가 아닙니다.",'', '## 필요한 변경','', '1. 어깨 초안 검수: 특히 팔꿈치 간격은 그립·운동 변형인지, 머리 뒤 손 이동은 다른 운동인지 확인. 부족한 오류 유형을 추가 수집.', '2. 공통 입력 생성: 두 자료 모두 같은 포즈 추출기·관절 순서·좌표 정규화·가시성 처리·시간 단위로 변환. 속도/순서 오류에는 실제 타임스탬프를 보존.', '3. 라벨 의미 정의: 정상·동작 특성·오류·판독 불가를 구분. 미검토 항목은 음성 정답으로 강제하지 않고 불확실성 마스크/가중치로 관리.', '4. 모델 출력 변경: 5종목 분류와 종목별 복수 오류 예측을 구분. 운동별 오류 기준을 유지하고 피드백 문구는 예측 코드에서 매핑.', '5. 분할 재설계: QEVD 공식 test 유지. 어깨는 검수·추가 수집 후 사람 기준 분할과 유형별 지원 수를 확인. 검증용 데이터는 학습 참여자에서 별도 분리.', '6. 불균형 처리: 학습 데이터에서 종목·오류별 샘플링/가중치를 결정. test/validation은 학습에 넣지 않음. 증강과 다른 시점 사본을 새로운 참여자로 세지 않음.', '7. 성능 확인: 전체 정확도 하나 대신 운동별·오류별 precision/recall/F1 및 독립 ESP32-CAM 영상으로 측정. 드문 클래스는 평가 불가로 표시.','', '## 현재 권장 범위','', '검수 전 초안을 명시한 학습 파이프라인 실험은 가능합니다. 실시간 교정 정확도를 주장하거나 최종 평가에 사용하기에는 어깨 라벨과 테스트 지원이 부족합니다. 데이터 형식이 같다는 사실이 정답 라벨과 모델의 호환성을 보장하지는 않습니다.','']
 (OUT/'REPORT.md').write_text('\n'.join(md))
 print(json.dumps({'verdict':report['verdict'],'combined':report['combined'],'shoulder_test':{'good':test_good,'bad':test_bad},'report':str(OUT/'REPORT.md')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()

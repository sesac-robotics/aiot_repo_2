import json,sys,collections,datetime
from pathlib import Path
import numpy as np,joblib
from shared_features import pool_item
B=Path(__file__).resolve().parents[2];O=B/'output/return-cycle-aligned-v1-20261024';P=B/'output/return-initial-context-20261024';D=B/'output/return-label-expansion-v3-20261024';sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/paired-evaluation')]
from common import sha
from split_guard import POLICY,assert_training_allowed
from artifact_integrity import equal_state
from train_scores import metrics
read=lambda p:json.loads(p.read_text());rows=read(O/'score-inputs.frozen.json');test=read(O/'test-inputs.frozen.json');oldrows=read(P/'score-inputs.frozen.json');oldtest=read(P/'test-inputs.frozen.json');byid={r['id']:r for r in rows};bytest={r['id']:r for r in test};m=joblib.load(O/'score-candidates.joblib');old=joblib.load(P/'score-candidates.joblib');cycle=read(O/'new-head-training.json');control=read(B/'output/return-cycle-tail-control-v1-20261024/new-head-training.json');cmp=read(O/'comparison.frozen.json');prior=read(P/'comparison.frozen.json');audit=read(O/'feature-provenance.json');labs=read(D/'labels-v2.frozen.json');clips=read(B/'data/return-label-expansion-v3-20261024/clip-manifest-v2.frozen.json')
assert len(rows)==979 and len(oldrows)==976 and len(test)==len(oldtest)==51
for a,b in zip(rows[:len(oldrows)]+test,oldrows+oldtest):
 for k in ['id','person','group','sources','split']:assert a[k]==b[k]
 assert a['items'][:len(b['items'])]==b['items'] and a['levels'][:len(b['levels'])]==b['levels']
assert sha(O/'original-test-inputs.frozen.json')==sha(P/'test-inputs.frozen.json')
assert sha(POLICY)=='7aded6fa543285e7e8926b690317c2cc53e1bc96c7f9ca1c181f3077be23f809'
team_path=B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json';assert sha(team_path)=='fa4e5913e3b48790562702e3ff8751aee159f1435ae4b6e633b3e7a27e678667'
changed=('pushups','return_extension','front');assert len(m['heads'])==59 and len(old['heads'])==58
assert all(equal_state(h,m['heads'][k]) for k,h in old['heads'].items() if k!=changed)
trained={i for h in m['heads'].values() for i in h['trained_ids']};assert_training_allowed([byid[i] for i in trained]);assert not {s for r in rows for s in r['sources']}&{s for r in test for s in r['sources']}
held={r['id'] for r in labs['held']};assert not held&trained;assert len(labs['reviewed'])==3 and sum(r['return_grade']==1 for r in labs['reviewed'])==1
for h in [cycle,control]:
 for fold in h['nested_CV_folds']:
  assert not set(fold['fit_groups'])&set(fold['held_groups'])
  for member in fold['inner_members']:
   assert not set(member['fit_ids'])&set(member['held_ids']);assert not set(member['fit_groups'])&set(member['held_groups']);assert not set(member['fit_ids'])&set(fold['outer_ids'])
 for member in h['component_training']:
  assert_training_allowed([byid[i] for i in member['fit_ids']]);assert not {s for i in member['fit_ids'] for s in byid[i]['sources']}&{s for i in member['held_ids'] for s in byid[i]['sources']}
assert {p['id']:p['target'] for p in cycle['selected_oof_predictions']}=={p['id']:p['target'] for p in control['selected_oof_predictions']}
oldp={(p['id'],p['criterion']):p for p in prior['paired_predictions']};newp={(p['id'],p['criterion']):p for p in cmp['paired_predictions']};assert oldp.keys()<=newp.keys()
for k,a in oldp.items():
 b=newp[k]
 for field in ['target','ai_prediction','rule_prediction']:assert a[field]==b[field]
extra=[newp[k] for k in newp.keys()-oldp.keys()];assert len(extra)==7 and all(p['criterion']=='return_extension' and p['target']==2 for p in extra)
assert cmp['same_bank_paired_items']==163
for p in extra:
 row=bytest[p['id']];assert next(i for i in row['items'] if i['bank']==p['bank'])['eligible']
assert cycle['validation_count']==0 and not cycle['candidate_gate_passed']
# Recount the actual canonical team strata, rather than merely copying a table.
policy=read(POLICY);cells=collections.Counter();source_roles=collections.defaultdict(set)
for r in read(team_path)['labels']:
 grades=[i['grade'] if i['training_mask'] else None for i in r['items']];q='error' if any(g is not None and g<2 for g in grades) else 'normal' if len(grades)==4 and all(g==2 for g in grades) else 'unknown';q=q if r['quality_training_candidate'] else 'unknown';role=policy['assignments'][r['source_id']];cells[r['person'],r['exercise'],q,role]+=1;source_roles[r['source_sha256']].add(role)
baseaudit=read(P/'person-quality-split-audit.json');assert all(cells[r['person'],r['exercise'],r['quality'],r['split']]==r['count'] for r in baseaudit['cells']);assert all(len(v)==1 for v in source_roles.values());assert sum(r['count']==0 for r in baseaudit['cells'])==37
(O/'person-quality-split-audit.json').write_text(json.dumps(dict(baseaudit,recomputed_from_canonical_labels=True,policy_sha256=sha(POLICY),canonical_labels_sha256=sha(team_path)),indent=2))
assert len(clips['clips'])==3 and sum(len(r['pixel_checks']) for r in clips['clips'])==9
for r in clips['clips']:assert sha(r['path'])==r['sha256'] and sha(r['original'])==r['source_sha256'] and r['split']=='train' and all(k['exact_decoded_pixels_equal'] for k in r['pixel_checks'])
for name,n in [('contracts',16),('previous-models',12)]:
 t=(O/f'tests-{name}.log').read_text();assert f'Ran {n} tests' in t and '\nOK' in t
for r in read(B/'output/return-shared-extension-v1-20261024/score-inputs.frozen.json'):
 for i in r['items']:
  if i['bank']=='observed_arm_extension_shared_v1':
   source=next(q for q in r['items'] if q['bank']==i['source_feature_bank'] and q['item']=='return_extension');assert pool_item(source)==i
height=read(B/'output/return-height-only-v1-20261024/new-head-training.json');shared=read(B/'output/return-shared-extension-v1-20261024/comparison.frozen.json');monotone=read(B/'output/return-extension-monotone-v1-20261024/new-head-training.json')
v=dict(status='pushup_return_timing_corrected_diagnostic_coverage_improved_goal_unmet',all_five_reliable=False,production_replacement=False,existing_Train_labels_and_roles_preserved=True,policy_sha256=sha(POLICY),canonical_team_labels_sha256=sha(team_path),new_reviewed_Train_originals=20,new_Train_interval_labels=3,new_partial_failure_labels=1,new_error_people=1,new_normal_return_labels=2,borderline_failure_draft_held_before_fit=1,new_squat_failure_labels=0,new_severe_grade0_labels=0,new_partial_label_used_in_front_candidate=True,new_partial_label_held_by_cycle_extractor=True,cycle_Train_rows=cycle['train_count'],cycle_Train_errors=cycle['grade_counts']['1'],cycle_Train_normals=cycle['grade_counts']['2'],cycle_error_people=cycle['error_groups'],cycle_Train_group_CV=cycle['group_cv'],tail_control_same36_Train_group_CV=control['group_cv'],cycle_development_rows=0,cycle_acceptance_gate_passed=False,Test_rows=51,unchanged_paired_Test_items=156,additional_paired_Test_items=7,paired_Test_items=163,new_return_Test_errors=0,new_return_Test_normals=7,new_return_Test_AI_false_alarms=sum(p['ai_prediction']<1.5 for p in extra),new_return_Test_Rule_false_alarms=sum(p['rule_prediction']<1.5 for p in extra),remaining_pushup_return_Test_held=3,squat_development_error_detected=1,squat_development_error_n=6,squat_development_normal_false_alarms=0,squat_development_normal_n=13,squat_AI_Rule_models_unchanged=True,models=59,prior_heads_preserved=57,refitted_front_head=1,new_cycle_head=1,unique_trained_rows=len(trained),source_overlap=0,exact_person_quality_80_20_achieved=False,empty_team_split_cells_unchanged=37,independent_final_Test_opened=False,diagnostic_Test_historically_exposed=True,Test_used_for_fit_or_numeric_model_selection=False,diagnostic_Test_used_for_root_cause_research=True,regression_tests_passed=28,lossless_clips=3,exact_pixel_checks=9,actual_trained_families=dict(collections.Counter(byid[i]['family'] for i in trained)),report=str(O/'RESULTS.txt'))
(O/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2))
(O/'controlled-tail-vs-cycle.json').write_text(json.dumps(dict(same36_Train_ids_and_targets=True,same_nested_CV_groups=True,same_C_grid_threshold_policy=True,old_tail=control['group_cv'],observed_cycle=cycle['group_cv'],independent_Test=False),indent=2))
lines=['복귀 오류 보강 및 푸시업 복귀 시점 수정 — 2026-10-24','', '현재 목표는 미달이며 실서비스 모델은 교체하지 않았습니다. 스쿼트 복귀 오류 개발 검출은 여전히1/6, 정상 오판0/13입니다. 이 오류6개는 같은 사람의 반복 노출 개발 자료입니다. 이전4/6 보고 정정도 유지합니다.','','라벨 보강','기존에 사용/검토하지 않은 Train 원본20개(20명)를 선별 검토했습니다. 푸시업 부분복귀1점 초안1개와 정상 복귀2개를 추가했습니다. 처음 부분복귀로 검토한 첫 반복1개는 정상에 가까운 경계 사례로 판단해 학습 전에 보류했습니다. 원본/프레임 증거와 초안 및 수정판을 모두 보존했습니다. 전문가 확정 정답은 아닙니다. 새 스쿼트 복귀 실패 및 심한 미복귀0점은 추가 확보하지 못했습니다.','추가 부분복귀1개는 기존 정면 복귀 후보 재학습에 사용했으나, 새 자동 회차 추출기는 그 얕은 구간의 완결 전환을 충분히 확인하지 못해 새 회차 모델에서는 보류합니다. 관측 조건을 무시해 학습 행을 늘리지 않았습니다. 정상2개는 새 회차 모델에도 사용했습니다.','문맥을 포함한 무손실 클립3개와 원본 픽셀 일치9회 확인을 저장했습니다. 클립 문맥이 추가 라벨이나 독립 표본으로 늘어난 것은 아닙니다.','','확인한 원인과 수정','기존 동일 Test156항목에서는 푸시업 복귀10개가 빠졌습니다. 관절은 관측되지만 측면 전용 복귀 head에 정상/오류 그룹이 부족해 모델이 없었습니다. 정면과 측면은 같은 관절각 추출 함수를 사용하므로 같은 측정 단위로 합친 별도 후보를 만들었습니다.','일반 분류 통합 후보는 정상 Test10개 중5개를 오류로 판단했습니다. 정상에 가까운 각도에서도 오류가 나오는 비물리적 반응을 확인했습니다. 이 후보는 학습/개발 기준부터 미달이므로 채택하지 않습니다. 단순히 관측 지원을 늘리는 것만으로 정확도가 해결되지 않습니다.','정상 Train140행 중34행은 기존 끝부분의 팔꿈치90백분위가160도 미만이었습니다. 이는 불일치 의심 수량이며34개 전부를 틀린 라벨로 확정한 뜻은 아닙니다. 팀 V32/V36 원본을 대조해 다음 하강이나 여러 반복이 섞인 고정 구간의 끝을 복귀 순간처럼 측정하는 문제를 확인했습니다.','새 추출기는 첫 하강/바닥/상승/상단 전환을 찾고 최대0.35초 같은 원본 문맥만 사용합니다. 바닥은 원래 구간 안에 있어야 하고, 나중 반복의 정상 복귀로 앞선 부분복귀를 대체하지 않습니다. 상승 중 잘린 구간과 가만히 지지한 구간은 보류합니다. 팔을 더 펴면 오류 위험이 커지지 않도록 학습 계수도 제한하고 관측률은 심각도 학습 입력에서 제외했습니다.','','통제 비교와 검증',f"같은 Train36행(정상30/오류6), 같은 사람/원본 그룹과 학습 후보·경계 선택 절차로 끝부분 특징과 회차 복귀 특징을 통제 비교했습니다. 끝부분 특징 교차검증: 오류{round(control['group_cv']['error_recall']*6)}/6, 정상 오판{round(control['group_cv']['normal_false_alarm']*30)}/30. 수정 후: 오류{round(cycle['group_cv']['error_recall']*6)}/6, 정상 오판{round(cycle['group_cv']['normal_false_alarm']*30)}/30.",'이 수치는 학습 자료 안의 그룹 분리 교차검증입니다. 독립 개발/최종 Test의 실제 오류 검출률로 해석하지 않습니다. 새 회차 모델의 오류/정상 개발 자료가 없어 채택 게이트는 실패입니다.','같은 기존 Test51행의 원본/정답/역할을 유지했고 기존156항목의 AI·Rule 예측도 모두 동일했습니다. 푸시업 복귀7개를 추가로 함께 평가해 총163항목이 됐습니다. 이7개는 모두 정상이고 AI·Rule 정상 오판은 모두0/7입니다. 나머지 복귀3개는 관측/회차 전환 조건 미달로 보류했습니다. 추가 평가된 항목에 오류가 없어 Test 오류 검출 향상을 입증하지 못합니다.','기존 푸시업 정상 오판6개도 해결된 것은 아닙니다. 정상 분모가 늘었다는 이유로 기존 오류가 줄었다고 보고하지 않습니다.','운동 | AI 오류 검출 / Rule 오류 검출 | AI 정상 오판 / Rule 정상 오판 (추가 관측 포함)']
for e,s in cmp['same_bank_by_exercise'].items():
 a=s['ai'];r=s['rule_base'];lines.append(f"{e} | {round(a['error_recall']*a['error_n'])}/{a['error_n']} / {round(r['error_recall']*r['error_n'])}/{r['error_n']} | {round(a['normal_false_alarm']*a['normal_n'])}/{a['normal_n']} / {round(r['normal_false_alarm']*r['normal_n'])}/{r['normal_n']}")
lines+=['','추가 부정 결과','스쿼트 높이4개 특징만 사용한 후보도 오류 개발0/6으로 개선하지 못했습니다. 각도 특징을 제거하는 것만으로 미세 오류 부족이 해결되지 않았습니다. 푸시업 끝부분 특징에 단조 제약만 추가한 후보도 Train 오류0/12로 기준 미달이었습니다. 이런 실패 실험도 별도 보존했습니다.','','분할/남은 작업','팀 사람별 정상/오류 Train/Test 원본 배정과 라벨 해시는 유지했습니다. 실제 팀 라벨을 다시 집계해80칸 중37빈칸이 그대로임을 확인했습니다. 정확한80/20은 아직 미달이며 Test를 Train으로 옮기지 않았습니다. 기존 자료의 실제 학습 계열도 보존했습니다.','별도 미사용 최종 Test는 열지 않았습니다. 기존 진단 Test는 이미 노출되어 원인 분석에 사용됐으며 독립 성능 증명이 아닙니다. 모델 가중치/C/경계 수치의 선택은 Train 그룹 검증에서만 수행했습니다.','새 회차 추출기의 원본 대조 범위를 늘리고, 자동 추출에서 보류한 얕은 오류 및 실제0/1점 완결 자료를 다른 사람에게서 보강해야 합니다. 개발 오류와 정상 자료를 함께 확보해 채택 기준을 통과한 뒤 동일 Test와 라이브 전체 경로를 확인해야 합니다.','기존 스쿼트 미세 복귀 실패 자료 확보표는 data/return-recheck-20261024/CAPTURE_PLAN.txt에 있습니다. 실제 새 촬영은 아직 없습니다.','회귀 검사28개 통과. 실서비스/카메라/UI 파일과 배포 모델은 이 실험에서 변경하지 않았습니다.','근거: verification.json, controlled-tail-vs-cycle.json, comparison.frozen.json, feature-provenance.json, person-quality-split-audit.json.']
p=O/'RESULTS.txt';assert not p.exists();p.write_text('\n'.join(lines)+'\n');(O/'NEXT_SESSION.txt').write_text('현재 스쿼트 개발1/6 FP0/13 그대로. 새 푸시업 회차 모델은 같은36Train 그룹CV6/6 FP0/30, 독립개발0행/게이트실패. 같은Test51행/기존156항목예측유지+정상복귀7추가=163항목, 추가AI/Rule FP0/7 오류0. Test오류성능 향상 주장 금지. 새부분복귀초안1+정상2, 경계1보류. 새오류초안은front재학습에만포함, cycle에서는관측부족보류. 팀37빈칸 유지,Test이동금지,독립최종Test미열람. RESULTS와verification참조.\n')
for name,src in [('cycle_features.frozen.py',B/'tools/return-label-expansion-v3/cycle_features.py'),('models.frozen.py',B/'tools/error-enrichment/models.py'),('train.frozen.py',B/'tools/return-label-expansion-v3/train_cycles.py'),('prepare.frozen.py',B/'tools/return-label-expansion-v3/prepare_cycles.py')]:
 assert not (O/name).exists();(O/name).write_bytes(src.read_bytes())
(O/'artifact-hashes.frozen.json').write_text(json.dumps({p.name:sha(p) for p in sorted(O.iterdir()) if p.is_file() and p.suffix in ['.json','.joblib','.py','.txt']},indent=2))
pointer=dict(run=str(O),parent_run=str(P),policy=str(POLICY),status=v['status'],all_five_reliable=False,production_replacement=False,report=str(O/'RESULTS.txt'),initial_pose_return_experiment=str(B/'output/initial-pose-return-v7-20261024'))
(B/'output/CURRENT_FIVE_EXERCISE_RUN.json').write_text(json.dumps(pointer,indent=2));print(json.dumps(v,ensure_ascii=False,indent=2))

"""Audit root-cause evidence, unchanged assignments, new labels and paired results."""
import collections,json,re,sys
from pathlib import Path
import numpy as np,joblib
B=Path(__file__).resolve().parents[2];O=B/'output/return-root-cause-v3-20261024';D=B/'output/return-root-cause-20261024';OLD=B/'output/squat-side-return-enriched-v2-20261024'
sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/paired-evaluation')]
from common import sha
from split_guard import assert_training_allowed,POLICY
from artifact_integrity import equal_state
read=lambda p:json.loads(p.read_text());rows=read(O/'score-inputs.frozen.json');test=read(O/'test-inputs.frozen.json');oldrows=read(OLD/'score-inputs.frozen.json');oldtest=read(OLD/'test-inputs.frozen.json');byid={r['id']:r for r in rows};model=joblib.load(O/'score-candidates.joblib');old=joblib.load(OLD/'score-candidates.joblib');reports=read(O/'score-training.json');cmp=read(O/'comparison.frozen.json');new=read(O/'new-labels.frozen.json');rules=read(O/'rule-training.json');head=next(h for h in reports['heads'] if h['key']==['squats','standing_height_return','height_reference_pose_v1']);rule=next(h for h in rules['heads'] if h['key']==head['key'])
assert sha(POLICY)=='7aded6fa543285e7e8926b690317c2cc53e1bc96c7f9ca1c181f3077be23f809'
assert sha(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json')=='fa4e5913e3b48790562702e3ff8751aee159f1435ae4b6e633b3e7a27e678667'
assert reports['completed'] and len(model['heads'])==56 and len(new)==2 and len(rows)==976
assert all(equal_state(h,model['heads'][k]) for k,h in old['heads'].items())
trained={i for h in model['heads'].values() for i in h['trained_ids']};assert_training_allowed([byid[i] for i in trained]);assert {r['id'] for r in new}<=trained
assert not {s for r in rows for s in r['sources']}&{s for r in test for s in r['sources']}
assert len(test)==len(oldtest)==51 and sha(O/'original-test-inputs.frozen.json')==sha(OLD/'test-inputs.frozen.json')
for r,p in zip(rows,oldrows):
    for k in ['id','person','group','split','sources']:assert r[k]==p[k]
    assert r['levels'][:len(p['levels'])]==p['levels'] and r['items'][:len(p['items'])]==p['items']
for r,p in zip(test,oldtest):
    for k in ['id','person','split','sources']:assert r[k]==p[k]
    assert r['levels'][:len(p['levels'])]==p['levels'] and r['items'][:len(p['items'])]==p['items']
original=next(h for h in read(OLD/'score-training.json')['heads'] if h['key']==['squats','standing_height_return','height_v4']);old_error={p['id'] for p in original['validation_predictions'] if p['target']<2};new_error={p['id'] for p in head['validation_predictions'] if p['target']<2};assert old_error==new_error and len(new_error)==6
assert len({p['person'] for p in head['validation_predictions']})==1
assert head['candidate_gate_passed'] is False and head['grade_counts']=={'2':110,'1':9}
assert sha(O/'features.frozen.py')==read(O/'MODEL_SELECTION_PROTOCOL.frozen.json')['feature_implementation_sha256']
for key in ['same_bank_by_exercise','existing_rules_by_exercise']:
    for m in cmp[key].values():
        for n in ['n','error_n','normal_n']:assert m['ai'][n]==m['rule_base'][n]
assert cmp['same_bank_paired_items']==156
clips=read(B/'data/return-root-cause-20261024/clip-manifest.json')['clips'];assert len(clips)==3
assert all(sha(c['path'])==c['sha256'] and c['split']=='train' and all(k['exact_decoded_pixels_equal'] for k in c['pixel_checks']) for c in clips)
tests={}
for n,count in [('features',7),('contracts',16),('paired',9)]:
    text=(O/f'tests-{n}.log').read_text();assert re.search(rf'Ran {count} tests\b',text) and '\nOK' in text;tests[n]=count
parent_train=[r for r in oldrows if r['split']=='train'];errors=[]
for r in parent_train:
    for i,g in zip(r['items'],r['levels']):
        if i['bank']=='height_v4' and i['item']=='standing_height_return' and g is not None and g<2 and i['eligible']:errors.append(i['feature'][0])
missed=[p for p in original['validation_predictions'] if p['target']<2 and p['prediction']>=1.5];oldby={r['id']:r for r in oldrows};missedfeatures=[next(i['feature'] for i in oldby[p['id']]['items'] if i['bank']=='height_v4' and i['item']=='standing_height_return') for p in missed]
source_audit=read(O/'feature-provenance.json');corrected=next(r for r in source_audit if r['id']=='return_squat_diag_m4_5_7_r09');oldfeature=next(i['feature'] for i in oldby[corrected['id']]['items'] if i['bank']=='height_v4')
cause=dict(original_fixed_development_rows=23,original_development_error_rows=6,error_people=['cfrep_w2'],same_error_IDs_preserved=sorted(new_error),new_bank_paired_development_rows=19,new_bank_development_normals=13,original_normal_REHAB_rows_not_recomputed=4,no_error_rows_dropped=True,old_Train_partial_hip_residual_range=[min(errors),max(errors)],missed_five_hip_residual_range=[min(f[0] for f in missedfeatures),max(f[0] for f in missedfeatures)],normal_Train_timing_error=dict(id=corrected['id'],actual_upright_original_frame=375,old_selected_frame=393,new_selected_frame=corrected['return_source_frame'],old_shoulder_residual=oldfeature[3],new_shoulder_residual=corrected['features'][3],evidence_sheets=['m4-dense-360.jpg','m4-f376.jpg'],old_last_quarter_restricted_window_confirmed=True),additional_cause='existing grade1 examples concentrated on larger incomplete returns; five mild development errors overlap normal feature ranges. Class-balancing weights already exist, so normal sample count alone is not an adequate explanation.',label_consistency_issue='Prior pelvis-only audit withdrew m4r10/r11. Dense new review explicitly requires initial hip AND shoulder heights, independent from angle-only trunk score; those two Train intervals are restored as partial height return. Existing development/Test grades unchanged.',AI_development_detected=1,rule_development_detected=round(rule['fixed_development']['error_recall']*6),development_error_n=6,AI_development_normal_false_alarm=0,rule_development_normal_false_alarm=round(rule['fixed_development']['normal_false_alarm']*13),development_normal_n=13,not_an_independent_final_Test=True)
(O/'root-cause-audit.json').write_text(json.dumps(cause,indent=2))
decisions=[]
for r in read(D/'mining/intake.frozen.json'):
    decisions.append(dict(id=r['id'],review_id=r['review_id'],split=r['split'],person=r['person'],source_sha256=r['source_sha256'],reviewed=True,review_sheet_sha256=sha(D/'mining'/(r['review_id']+'-01.jpg')),review_method='chronological_approximately_4fps_nonexpert_screen',grade=None,training_eligible=False,reason='No unambiguous completed return failure confirmed; hold/rest/crop/last descending frame not grade0; no blanket normal label.'))
(O/'additional-mining-decisions.frozen.json').write_text(json.dumps(dict(sources=8,decisions=decisions,new_failure_labels_from_this_screen=0),indent=2))
receipt=dict(status='root_cause_timing_fix_and_re_reviewed_Train_partials_development',all_five_reliable=False,production_replacement=False,policy_unchanged=True,team_person_quality_source_assignments_unchanged=True,exact_80_20_achieved=False,missing_split_cells_unchanged=37,newly_used_partial_return_Train_segments=2,new_partial_return_people=1,previously_withdrawn_Train_labels_re_reviewed=True,new_grade0_segments=0,additional_existing_sources_screened=8,actual_new_trained_rows=2,actual_unique_trained_rows=len(trained),actual_trained_families=dict(collections.Counter(byid[i]['family'] for i in trained)),training_input_rows=len(rows),trained_heads=56,new_heads=2,old_heads_preserved=54,source_overlap=0,Test_used_for_fit_or_selection=False,original_Test_rows=51,original_Test_existing_targets_preserved=True,original_Test_bytes_preserved=True,Test_feature_only_augmentation=True,paired_Test_items=156,old_common_Test_items=153,additional_normal_side_return_items=3,existing_rule_compatible_items=134,same_six_development_errors_retained=True,AI_development_error_detected=1,rule_development_error_detected=cause['rule_development_detected'],development_error_n=6,independent_final_Test_opened=False,live_service_replaced=False,total_regression_tests=sum(tests.values()),tests=tests,new_attempt_clips=2,calibration_reference_clips=1,exact_clip_pixel_checks=9,report=str(O/'RESULTS.txt'))
(O/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
def rate(m):return f"오류 {round(m['error_recall']*m['error_n'])}/{m['error_n']}, 정상 오판 {round(m['normal_false_alarm']*m['normal_n'])}/{m['normal_n']}"
lines=['복귀 오류 1/6 원인 분석·수정·추가 라벨·동일 Test 비교','','결과: 실제 특징 추출 오류를 수정하고 보류 Train 부분 복귀2구간을 재검토해 학습했다. 그러나 문제의 같은6개 개발 오류에서 AI 검출은 여전히1/6이다. 개선 완료나 배포 가능 상태가 아니다.','','확인한 원인',
f"1. 정상 라벨의 복귀 시점을 잘못 뽑았다. m4 diagonal r09는 약375프레임에서 섰는데 기존 마지막25% 제한 때문에 다음 숙임393프레임을 골랐다. 새 선택은{corrected['return_source_frame']}프레임이며 어깨 높이 잔차는{oldfeature[3]:.3f}→{corrected['features'][3]:.3f}로 바뀐다. 정상 학습 특징에 다음 동작을 섞는 오류를 확인했다.",
f"2. 기존 Train 부분 복귀7개는 골반 잔차{min(errors):.3f}~{max(errors):.3f}인 큰 오류에 치우쳤다. 놓친5개는{min(f[0] for f in missedfeatures):.3f}~{max(f[0] for f in missedfeatures):.3f}로 정상 범위와 겹친다. 정상 개수만 많은 문제로 설명할 수 없다. 기존 학습은 이미 등급·사람 그룹 가중치를 사용했다.",
'3. 여섯 개발 오류는 모두 cfrep_w2 한 사람에게서 나왔다. 새 특징의 공통 개발 자료19개는 오류6/정상13이며 모두 같은 사람이다. 원래 개발23개 중 REHAB 정상4개는 새 특징을 재계산하지 못했지만 오류6개는 단 하나도 제외하지 않았다.',
'4. 골반만 돌아왔을 때와 골반·어깨의 초기 선 높이를 모두 요구할 때의 라벨 해석이 혼재했다. 앞서 골반 단독 기준으로 보류했던 m4r10/r11을 원본 시간순 약10fps와 초기 완전 기립 장면으로 재검토했다. 이번 명시 기준은 초기 골반 및 어깨 높이의 복귀이며, 몸통 각도만으로 복귀 오류를 만들지 않는다. 개발/Test 정답은 수정하지 않았다.', '',
'수정과 추가 학습',
'고정 마지막25% 대신 하강 후 실제 상단 복귀를 찾고, 다음 하강이 섞여도 이미 관찰한 복귀를 잃지 않도록 수정했다. 원래6개 높이 특징을 보존하면서, 관찰된 한쪽 어깨-골반-무릎-발목의 초기 자세 대비 차이를 추가한 별도 AI 후보도 학습했다. 관절 누락을 채우지 않았고 골반/어깨의 세로 높이를 지우지 않았다. 무릎 각도를 서기 높이 정답으로 바꾼 것이 아니다.',
'm4r10/r11은 어깨 높이가 초기 기립보다 낮은 채 다음 하강을 시작하는 실제 부분 복귀1점2구간이다. 같은 원본·같은 사람의 두 반복이며 독립 사람2명으로 세지 않는다. 과거 초안에 있었다가 보류된 구간을 재검토해 이번 혼합 학습에 새로 사용한 것으로, 처음 발견한 원본2개라는 뜻이 아니다. 전문가 검증 라벨은 아니다.',
'기존 QEVD 후보8원본도 추가 시간순 검토했지만 그 화면들에서 새 복귀 실패를 확정하지 못했다. 실제0점은 이번에도 확보하지 못했다. 마지막 하강·정지·휴식을0점으로 만들지 않았다.',
f'기존974행을 보존해976행, 실제 학습 고유구간{len(trained)}개다. 기존54개 모델을 실제 추정기 상태까지 그대로 보존하고 새 특징 모델2개를 추가했다. 새 후보의 Train 그룹 교차검증은 오류8/9, 정상 오판1/110이지만 개발 AI는1/6, 정상 오판0/13이다. Train 지표를 독립 성능 개선으로 해석하지 않는다.',
f"같은19개 개발 항목에서 보정 Rule은 오류{cause['rule_development_detected']}/6, 정상 오판{cause['rule_development_normal_false_alarm']}/13이다. 이 규칙은 Train으로만 경계를 학습했고 개발/Test 결과로 경계를 다시 맞추지 않았다. AI와 규칙 모두 채택/실패 구별 완료 기준을 통과하지 못했다.",
'새 부분 복귀 클립2개와 그보다 앞선 기립 기준 클립1개를 data/return-root-cause-20261024에 무손실로 저장했다. 원본 해시·프레임·Train 역할·보류 이력을 보존했고 원본 픽셀9건을 확인했다. 기립 기준 클립은 반복 점수 정답으로 세지 않는다.', '',
'동일 Test 비교',
'기존51개 구간 ID·사람·원본 배정·기존 정답을 보존했다. 원래 Test 파일은 original-test-inputs.frozen.json에 바이트 단위로 보존했다. 새 파일은 같은 구간에 수정된 관측 특징만 추가한 것이다. 이전에 별도 진단했던 측면 정상 복귀3항목도 두 방식에 동일하게 제공해 공통153→156항목이 됐다. 분모 증가를 오류 검출 개선으로 주장하지 않는다.']
for ex,m in cmp['same_bank_by_exercise'].items():lines.append(ex+' | AI '+rate(m['ai'])+' | Train 보정 Rule '+rate(m['rule_base']))
lines+=['실제 배포 Rule과 목표가 호환되는 공통134항목 비교도 comparison.frozen.json에 별도로 보존했다. 위 보정 Rule은 배포 Rule과 다르다. 새로 평가된3개는 한 사람의 정상 복귀뿐이며 Test에는 새 복귀 오류 표본이 추가되지 않았다. 이 Test는 이미 노출된 진단용이다. 최종 미사용 Test는 열지 않았다.', '',
'분할·확인',
'P01~P04의 사람·운동·정상/오류별 기존 Train/Test 원본 배정을 그대로 유지했다. 정확한80/20과 기존 빈칸37개는 아직 달성하지 못한 상태다. 새 m4 오류도 원래 Train 사람/원본에 속하며 같은 원본의 정상과 오류는 같은 Train 그룹으로 유지했다. Test 학습 사용0, Train/Test 원본 중복0이다.',
'새 특징/복귀 시점7개, 분할 학습 계약16개, 같은 입력 비교9개 총32개 회귀 검증 통과. 모델54개 보존과 클립 픽셀9건은 별도로 확인했다.', '',
'남은 핵심 문제: 미세한 부분 복귀를 보이는 독립 촬영자들의 실제 Train 사례와 개발·Test 오류가 부족하다. 현재 보강2개는 비교적 큰 어깨 높이 실패라 놓친5개의 작은 높이 차이를 충분히 대표하지 못한다. 실제0점 데이터도 없다. 6개 개발 정답을 완화하거나 정상으로 바꿔 성능을 올리지 않는다. 새 오류의 근거와 기준 일관성을 검증하면서 자료를 보강해야 하며, 현재 후보는 라이브로 교체하지 않았다.']
(O/'RESULTS.txt').write_text('\n'.join(lines)+'\n');(O/'NEXT_SESSION.txt').write_text('최신 '+str(O)+f"\nRESULTS.txt/root-cause-audit.json/verification.json 참조. 마지막25%복귀윈도 버그확인(m4r09정상에다음숙임393프레임사용). 새실제Train부분복귀2(m4과거보류재검토), Grade0신규0. QEVD추가8원본화면실패확정0. 새타이밍+초기지지상대포즈head2개추가, 기존54개보존. 같은6개개발오류AI여전히1/6, 최종v3 Rule{cause['rule_development_detected']}/6(정상FP{cause['rule_development_normal_false_alarm']}/13); 이전v2중간실험 Rule3/6과구분. Test51구간동일기존정답/배정유지, 새특징만추가156항목(기존153+정상측면3). 미사용최종Test미개봉,라이브미교체. 다음에는기준선·작은부분복귀의독립Train촬영자를확보해야함. Test로후속튜닝금지.\n")
pointer=read(B/'output/CURRENT_FIVE_EXERCISE_RUN.json');pointer.update(run=str(O),parent_run=str(OLD),status=receipt['status'],all_five_reliable=False,production_replacement=False,report=str(O/'RESULTS.txt'));(B/'output/CURRENT_FIVE_EXERCISE_RUN.json').write_text(json.dumps(pointer,ensure_ascii=False,indent=2));print(json.dumps(receipt,ensure_ascii=False))

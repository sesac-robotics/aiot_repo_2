"""Audit the corrected split and publish a candid, reproducible run report."""
import collections,json,sys
from pathlib import Path
import joblib
from common import sha
from split_guard import POLICY,assert_training_allowed

B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/per-person-stratified-80-20-20261024'

def write(name,data):
    (O/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))

def main():
    policy=json.loads(POLICY.read_text());training=json.loads((O/'score-inputs.frozen.json').read_text())
    test=json.loads((O/'test-inputs.frozen.json').read_text());allrows={r['id']:r for r in training+test}
    report=json.loads((O/'score-training.json').read_text());assert report['completed']
    model=joblib.load(O/'score-candidates.joblib');assert model['labels_sha256']==sha(O/'score-inputs.frozen.json')
    trained_ids={i for head in model['heads'].values() for i in head['trained_ids']}
    assert_training_allowed([allrows[i] for i in trained_ids])
    assert not trained_ids & {r['id'] for r in test}
    assert not {s for r in training for s in r['sources']} & {s for r in test for s in r['sources']}
    state=joblib.load(O/'standing-monotone-candidate.joblib')
    state_rows={r['id']:r for r in json.loads((O/'standing-state-inputs.frozen.json').read_text())}
    assert_training_allowed([state_rows[i] for i in state['training_ids']])
    shared=joblib.load(O/'shared-frontal-trunk-candidate.joblib')
    assert_training_allowed([allrows[i] for i in shared['training_ids']])
    assignments=json.loads((O/'clip-assignments.frozen.json').read_text())
    assert len(assignments)==281 and len({r['id'] for r in assignments})==281
    assert len({r['source_id'] for r in assignments})==57
    test_classes={(r['exercise'],r['quality_class']) for r in assignments if r['split']=='test'}
    assert {(e,c) for e in ['squats','pushups','shoulder','plank','ytw'] for c in ['normal','error']}<=test_classes
    for person in policy['people']:
        for c in ['normal','error']:
            assert min(person['classes'][c]['train'],person['classes'][c]['test'])>0
    clips=json.loads((B/'data/error-enrichment-20261024/clip-manifest.json').read_text())
    assert len(clips)==54 and all(r['source_unchanged'] and r['frames_decoded']>0 for r in clips)
    evaluation=json.loads((O/'test-evaluation.json').read_text())
    assert evaluation['frozen_fingerprints']['model_sha256']==sha(O/'score-candidates.joblib')
    assert sha(B/'data/r2d2-team-20261023/annotations/labels.frozen.json')=='6b340faf5a330ef0dd9deb5fbf1a16faa27f5124d47c94d1570fa9b559051ef9'
    people=collections.Counter(allrows[i]['person'] for i in trained_ids if allrows[i]['family']=='R2D2')
    assert set(people)=={'r2d2_P01','r2d2_P02','r2d2_P03','r2d2_P04'}
    history=[]
    for old in ['integrated-five-20261024','error-enrichment-20261024','per-person-80-20-20261024']:
        oldmodel=joblib.load(B/'output'/old/'score-candidates.joblib')
        oldids={i for h in oldmodel['heads'].values() for i in h['trained_ids']}
        history.append(dict(run=old,new_test_rows_previously_trained=len(oldids & {r['id'] for r in test})))
    files=['score-candidates.joblib','standing-monotone-candidate.joblib','shared-frontal-trunk-candidate.joblib','score-inputs.frozen.json','test-inputs.frozen.json','test-evaluation.json']
    audit=dict(status='passed',team_sources=57,team_labeled_segments=281,
        assigned_counts=dict(collections.Counter(r['split'] for r in assignments)),
        class_counts={c:dict(collections.Counter(r['split'] for r in assignments if r['quality_class']==c)) for c in ['normal','error','unknown']},
        all_people_have_normal_and_error_in_both_sides=True,all_five_exercises_have_both_classes_in_test=True,
        actual_trained_unique_rows=len(trained_ids),actual_trained_team_rows=dict(people),
        actual_trained_source_families=dict(collections.Counter(allrows[i]['family'] for i in trained_ids)),
        test_rows_previously_trained_by_current_models=0,source_overlap_with_training=0,
        standing_and_shared_models_test_protection_checked=True,original_team_labels_unchanged=True,
        new_labeled_segments=54,new_project_score_candidate_segments=47,new_clips_decoded=54,
        annotation_status='non_expert_visual_draft_with_eight_documented_item_holds',
        independent_unseen_final_test=False,historical_exposure=history,
        models_promoted_to_live_service=False,all_five_reliable_evaluation_completed=False,
        verification_tests='16 contract tests; existing service 41 tests and 13 subtests passed',
        files={n:dict(path=str(O/n),sha256=sha(O/n)) for n in files},policy_sha256=sha(POLICY))
    write('verification.json',audit)
    table=['사람 | 정상 Train/Test | 오류 Train/Test | 판정 보류 Train/Test']
    for p in policy['people']:
        pairs=['%s/%s'%(p['classes'][c]['train'],p['classes'][c]['test']) for c in ['normal','error','unknown']]
        table.append(p['person']+' | '+' | '.join(pairs))
    text='''현재 유효한 작업: 사람별 정상/오류 층화 분할 후 처음부터 재학습

팀 원본 57개, 281개 구간 모두 Train 또는 Test에 배정했습니다.
이전의 전체 개수만 맞춘 분할과 해당 평가는 최종 결과에서 제외합니다.
라벨을 비율에 맞춰 바꾸지 않았습니다. 정상은 주요 4항목 모두 관찰상 정상,
오류는 한 항목 이상 관찰된 오류, 나머지는 판정 보류입니다.
판정 보류라도 개별 항목이 관찰되면 그 항목만 학습할 수 있습니다.
라벨은 4fps 장면표를 통한 비전문가 시각 검토 초안이며, 전문가 확정본이 아닙니다.

'''+ '\n'.join(table)+'''

전체 정상: Train 86 / Test 24. 오류: Train 103 / Test 27.
판정 보류: Train 34 / Test 7. 총 Train 223 / Test 58.
한 원본 영상의 모든 구간을 같은 쪽에 묶었습니다. P01 정상 Test는 10/35=28.6%입니다.
정상 플랭크 한 원본에서 나온 연속 10구간을 Test 쪽에 함께 두어 발생한 차이입니다.
정확히 20%를 맞추려고 이 10구간을 양쪽에 섞으면 매우 비슷한 장면으로 평가하게 됩니다.
각 사람의 정상/오류는 양쪽에 있고, 전체 Test에는 5종 각각 정상/오류가 있습니다.
사람×운동×항목×오류 정도까지 모든 세부 조합에 충분한 Test가 있는 것은 아닙니다.

기존 QEVD, REHAB 계열, 미국 해군 및 기존 직접 검토 자료도 계속 포함했습니다.
AI-Hub는 제공 Training 2D 관절·자세 조건으로 학습한 3개 표현 모델을 특징으로 활용했습니다.
AI-Hub 제공 Validation을 Training으로 옮기거나 제공 조건을 프로젝트 점수로 바꾸지 않았습니다.
원시 영상 인식 신경망 전체나 MediaPipe 자체를 재학습했다는 뜻은 아닙니다.

모델은 기존 가중치를 이어받지 않고 새 Train으로 처음부터 적합했습니다.
Test 원본/구간은 점수·서기 상태·공통 몸통 모델 학습에서 제외한 것을 검사했습니다.
모델 선택은 Train 내부 사람 그룹 교차검증으로 하고, 모델을 고정한 뒤 Test를 평가했습니다.
이전 모델이 보았던 자료라는 이력은 남아 있습니다. 새 사람에 대한 독립 최종 시험은 아닙니다.

학습·평가 실행 완료가 5종 모든 항목의 신뢰성 확보를 의미하지는 않습니다.
test-evaluation.json에는 관찰 가능하고 모델이 있는 항목의 결과와 보류 항목이 함께 기록돼 있습니다.
지원하지 않는 항목을 정상 점수로 채우지 않고, 기존 서비스 모델도 자동 교체하지 않았습니다.
서기 높이 복귀의 0등급, 어깨 하강 복귀 오류, YTW 일부 오류와 관찰 불가 항목은 보강이 필요합니다.
Test를 보고 모델을 다시 고른 뒤 동일 Test를 독립 성능처럼 재사용하지 않습니다.

다음 작업: 새 Train에서 오류 유형별 라벨 근거를 추가 검토하고, 부족한 유형을 확보합니다.
최종 서비스 반영은 항목별 오류 탐지·정상 오경보·보류율 및 별도 미사용 검증으로 판단합니다.
'''
    (O/'RESULTS.txt').write_text(text)
    (O/'NEXT_SESSION.txt').write_text('현재 유효한 실행: '+str(O)+'\n분할 정책: '+str(POLICY)+'\nRESULTS.txt와 verification.json을 먼저 읽으십시오.\n이전 count-only 분할은 사용하지 마십시오.\nTest 결과로 후보를 재선택하지 마십시오. 현재도 5종 모두 신뢰성 확보가 완료된 상태는 아닙니다.\n')
    write('model-and-data-manifest.json',dict(models_and_inputs=audit['files'],policy=str(POLICY),policy_sha256=audit['policy_sha256'],candidate_only=True))
    (B/'output/CURRENT_FIVE_EXERCISE_RUN.json').write_text(json.dumps(dict(run=str(O),policy=str(POLICY),status='retrained_and_tested_experimental',all_five_reliable=False,report=str(O/'RESULTS.txt')),ensure_ascii=False,indent=2))
    for old in ['per-person-80-20-20261024','rejected-stratified-no-ytw-20261024']:
        p=B/'output'/old
        if p.exists():(p/'SUPERSEDED.txt').write_text('최종 결과에서 제외: 분할 조건 미충족. 현재 실행: '+str(O)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k in ['status','assigned_counts','class_counts','actual_trained_unique_rows','actual_trained_team_rows','actual_trained_source_families']},ensure_ascii=False))

if __name__=='__main__':main()

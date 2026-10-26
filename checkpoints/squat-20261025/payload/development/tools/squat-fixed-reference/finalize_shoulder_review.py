"""Audit and freeze the 72-source, selected-window visual review checkpoint."""
import collections
import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

B = Path(__file__).resolve().parents[2]
R = B/'output/shoulder-common-support-audit-20261024/full-review-v1'
sys.path.insert(0, str(B/'tools/error-enrichment'))
from split_guard import POLICY, assert_training_allowed

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(name, value):
    p = R/name
    assert not p.exists(), p
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def main():
    labels = json.loads((R/'dense-visual-labels.journal.json').read_text())
    inventory = json.loads((B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json').read_text())['sources']
    own = {x['id']:x for x in inventory if x['owner']=='model_evaluation'}
    assert len(labels)==len(own)==72
    assert len({x['id'] for x in labels})==72
    dispositions=[]
    for row in labels:
        src=own[row['id']]
        for k in ['source_sha256','person','recording_group','split','view']:
            assert row[k]==src[k], (row['id'],k)
        assert sha(src['path'])==src['source_sha256']
        for p in row['reviewed_pages']+[row['screen']]:
            assert sha(p)==row['evidence_sha256'][Path(p).name]
        assert not row['label_inferred_from_provider'] and not row['fitted']
        if row['view']!='front': assert row['frontal_trunk_grade'] is None
        if row['split']=='train': assert_training_allowed([dict(row,sources=[row['source_sha256']])])
        for target,grade in [('lower_return',row['lower_return_grade']),('frontal_trunk',row['frontal_trunk_grade'])]:
            reasons=[]
            if grade is None:
                reasons.append('projection_not_frontal' if target=='frontal_trunk' and row['view']!='front' else 'movement_protocol_or_return_event_ambiguous_or_not_press')
            if row['split']!='train': reasons.append('frozen_'+row['split']+'_evaluation_only')
            if target=='lower_return' and row['view']=='side': reasons.append('near_arm_only_not_bilateral_grade')
            reasons.append('pose_window_alignment_and_target_tracking_QA_required_before_fit')
            dispositions.append(dict(id=row['id'],person=row['person'],source_sha256=row['source_sha256'],
                split=row['split'],view=row['view'],criterion=target,visual_grade=grade,
                visual_label_status='AI_assisted_selected_window_draft' if grade is not None else 'held',
                actual_training_used=False,training_hold_reasons=reasons,note_ko=row['note_ko']))
    assert sha(POLICY)=='7aded6fa543285e7e8926b690317c2cc53e1bc96c7f9ca1c181f3077be23f809'
    team=B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json'
    assert sha(team)=='fa4e5913e3b48790562702e3ff8751aee159f1435ae4b6e633b3e7a27e678667'
    save('dense-visual-labels.frozen.json',labels)
    save('source-dispositions.frozen.json',dispositions)
    save('protocol-label-correction.frozen.json',dict(id='subject_005_shoulder_bad_side',
        previous_file='protocol-model-v1/labels.frozen.json',previous_label='shoulder_press',
        corrected_label='non_press_arm_extension',evidence='subject_005_shoulder_bad_side-supplement-upper.jpg',
        new_review_window=[1.4,5.4],old_snapshot_preserved=True,
        model_fitted_from_old_labels=False,old_snapshot_not_approved_for_fit=True))
    lower=collections.Counter('held' if x['lower_return_grade'] is None else str(x['lower_return_grade']) for x in labels)
    body=collections.Counter('held' if x['frontal_trunk_grade'] is None else str(x['frontal_trunk_grade']) for x in labels)
    verification=dict(source_count=72,all_source_hashes_verified=True,evidence_hashes_verified=True,
        all_72_whole_video_1Hz_screens_and_selected_5Hz_panels_visually_reviewed=True,
        every_repetition_or_every_frame_adjudicated=False,expert_verified=False,
        roles=dict(collections.Counter(x['split'] for x in labels)),roles_changed=False,
        lower_return_labels=dict(lower),frontal_trunk_labels=dict(body),
        new_partial_return_error_drafts=1,new_error_draft_source_role='validation',
        new_Train_error_labels=0,new_training_used_rows=0,new_grade0_labels=0,
        team_labels_and_policy_hashes_preserved=True,independent_final_Test_opened=False,
        historically_exposed_Test_review_only=True,production_changes=False,
        protocol_old_press_label_corrected_prior_to_any_fit=True)
    save('review-verification.frozen.json',verification)
    report='''숄더 원본 72개 선택 구간 검토 체크포인트

72개 모두 전체 길이 1Hz 선별과 선택 회차의 5Hz 원본 표본을 직접 검토하고 근거/시간/사유를 기록했다. 모든 반복·모든 프레임의 전문가 판정 완료를 뜻하지 않는다. 원본 전체에 한 등급을 복사하지 않는다.
하단 복귀 초안: 정상65, 부분복귀1, 보류6. 정면 몸통: 정상24, 다른 투영48 보류. 측면 하단은 보이는 근측팔만의 초안이며 양팔 정답으로 확대하지 않는다. 몸통 측면을 정면 좌우 항목으로 학습하지 않는다.
새 부분복귀는012 bad_front 1.8~2.4초: 상단→하강→다음 상승에도 상완/덤벨이 높은 상태로 남는다. 같은 사람 good_front 1.6~2초의 어깨옆 하단과 native 원본을 대조했다. validation/앉은뷰/초기top시작이라 Train 이동과 초기준비 특징의 강제 계산을 금지한다. 새 Train 오류 및0점은0개다.
005 bad_side 상체 확대에서 머리뒤 팔꿈치폄 경로를 확인했다. 앞선 프로토콜 실험 입력의press 초안은 새정정판에서 철회한다. 옛 스냅샷을 보존하고 해당 라벨로 fit한 모델은 없다. 004 세뷰는 초반 상승과 이후 팔꿈치폄이 섞여 새하단 등급 보류. 011 큰몸통접힘 추정 철회도 유지한다.
실제 학습 투입은0행이다. 새 시간창과 관절 추정의 원본 겹침검사, 특징 계약 및 학습 등급 지원을 통과해야 한다. 후보를 추출했다는 이유만으로 학습 완료나 점수 지원을 선언하지 않는다.
Train48/validation18/기존노출Test6 역할과 원본해시를 확인했다. 팀 정책/라벨 해시는 불변이며 사람별 정상·오류 정확80/20 미달(37빈칸)도 그대로다. 독립 최종Test 미개봉, 라이브 미변경.
근거: dense-visual-labels.frozen.json, source-dispositions.frozen.json, review-verification.frozen.json, 012-front-lower-native-comparison.jpg, protocol-label-correction.frozen.json.
'''
    p=R/'REVIEW_RESULTS.txt';assert not p.exists();p.write_text(report)
    progress=dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),decoded=72,
        whole_video_1Hz_screen_reviewed=72,selected_5Hz_panel_sources_reviewed=72,
        selected_window_annotation_rows=72,all_repetitions_adjudicated=False,
        labels_frozen=True,actual_fit_rows=0,new_partial_return_drafts=1,new_Train_error_labels=0,
        next='새오류validation보호/72와파트너78의자료별사유병합/좌표원본QA후학습입력별도동결')
    (R/'review-progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
    s=B/'project-coordination/status/model-evaluation.json';status=json.loads(s.read_text())
    status['updated_at_utc']=progress['updated_at_utc']
    status['shoulder_full_review'].update(state='72_selected_window_labels_frozen_not_all_repetitions_adjudicated',
        whole_video_screens_reviewed=72,selected_dense_sources_reviewed=72,annotation_rows=72,
        new_partial_return_drafts=1,new_Train_errors=0,models_refit=False,
        report=str((R/'REVIEW_RESULTS.txt').relative_to(B)))
    status['evidence'].extend(str((R/n).relative_to(B)) for n in ['REVIEW_RESULTS.txt','review-verification.frozen.json','dense-visual-labels.frozen.json'])
    s.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(verification,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

"""Apply the offline target contract to the two actually reviewed 012 events."""
import datetime
import hashlib
import json
import sys
from pathlib import Path
from return_target_contract import ANATOMICAL, READY_RELATIVE, assess

B=Path(__file__).resolve().parents[2]
R=B/'output/return-target-contract-v1-20261025'
S=B/'output/shoulder-common-support-audit-20261024/full-review-v1'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,value):
    p=R/name;assert not p.exists(),p
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    R.mkdir(exist_ok=True)
    registry={x['id']:x for x in json.loads((S/'decoded-manifest.frozen.json').read_text())['sources']}
    reviews={x['id']:x for x in json.loads((S/'dense-visual-labels.frozen.json').read_text())}
    output=[]
    for quality,grade,peak,turn,following in [('bad',1,[0,12],[27,36],[39,45]),('good',2,[0,9],[24,30],[33,42])]:
        sid=f'subject_012_shoulder_{quality}_front';src=registry[sid];review=reviews[sid]
        assert src['split']=='validation' and review['lower_return_grade']==grade
        assert sha(src['path'])==src['source_sha256']
        for path in review['reviewed_pages']:
            assert sha(path)==review['evidence_sha256'][Path(path).name]
        annotation=dict(id=sid,source_sha256=src['source_sha256'],person=src['person'],
            split=src['split'],view=src['view'],posture='seated',exercise='shoulder',
            criterion=ANATOMICAL,protocol='shoulder_press',laterality='bilateral',
            grade=grade,label_origin='direct_source_visual_review',
            review_status='reviewed_AI_assisted',expert_adjudicated=False,
            scope='selected_event_only',rubric='anatomical_rack_range_v1',
            event_source_sha256=src['source_sha256'],peak_frames=peak,turn_frames=turn,
            next_ascent_frames=following,closure='observed_next_ascent',
            actual_pixels_visible=True,interrupted_or_clipped=False,
            evidence_sha256=review['evidence_sha256'],
            parent_visual_label_sha256=sha(S/'dense-visual-labels.frozen.json'),
            note_ko='기존선택구간시각판정을새해부학적하단가동범위목표로명시해기록. 초기준비와의차이라고해석하지않음. 원본라벨/역할보존.')
        admission=assess(annotation,src)
        relative=dict(annotation,criterion=READY_RELATIVE)
        contrast=assess(relative,src)
        assert admission['visual_label_eligible']
        assert not admission['training_eligible'] and not admission['evaluation_input_eligible']
        assert not contrast['visual_label_eligible']
        output.append(dict(annotation=annotation,anatomical_admission=admission,
            ready_relative_admission=contrast,actual_fit=False))
    save('reviewed-events.frozen.json',output)
    team=B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json'
    policy=B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json'
    assert sha(team)=='fa4e5913e3b48790562702e3ff8751aee159f1435ae4b6e633b3e7a27e678667'
    assert sha(policy)=='7aded6fa543285e7e8926b690317c2cc53e1bc96c7f9ca1c181f3077be23f809'
    save('verification.frozen.json',dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        direct_visual_events=2,both_validation=True,source_and_evidence_hashes_verified=True,
        anatomical_visual_labels_admitted=2,relative_visual_labels_admitted=0,
        actual_feature_or_grade_model_fits=0,new_extractor_registered=False,
        contract_tests_passed=24,tests_are_functional_not_accuracy=True,
        team_labels_and_policy_hashes_unchanged=True,source_roles_changed=False,
        final_Test_opened=False,production_changed=False,
        contract_sha256=sha(B/'tools/squat-fixed-reference/return_target_contract.py')))
    text='''초기 준비 없는 부분 복귀 자료 사용 계약 / 2026-10-25

결정
초기 준비가 없다는 이유로 관측된 부분 복귀 영상 전체를 버리지 않는다. 다만 서로 다른 두 목표를 분리한다.
1) observed_ready_relative_lower_return: 같은 원본의 운동 이전 준비 자세와 실제 복귀의 차이. 준비 기준이 상단보다 먼저 있어야 한다. 나중의 기립/다른good영상/정상참조평균을 과거 기준으로 넣지 않는다.
2) observed_anatomical_lower_turn: 양팔을 어깨 옆 하단까지 내렸는지의 관측된 가동범위. 초기 준비가 없어도 선행상단→실제하강→하단반전→다음상승이 원본에 보이면 그 사건의 시각등급 초안을 허용한다. 별도모델목표·특징버전·정면/앉음뷰은행이 필요하다. 첫하단을 보지 않고 뒤정상하단으로 바꾸지 않는다.

판정 범위
현재 rubric은프로젝트AI보조시각초안이다. 2는관측양팔이어깨옆하단으로돌아온선택사건,1은내려오긴했지만상완/덤벨이높게남은채다음상승으로반전한선택사건이다. 머리뒤팔꿈치폄같은다른운동,한팔만가시,모자이크/누락,중간잘림,사용자종료는그자체로1점/0점아니다. 0점의실제학습근거는아직없어0점지원으로확대하지않는다. 수치각도문턱/임상등급을이문서에서만들지않는다.
정상하단이원본에서안정적으로보여완결되면다음상승없는마지막정상회차도2점시각정답후보가될수있다. 부분오류는후속상승으로하단반전이닫혀야하며단순종료로오류를만들지않는다.

학습 적합성
시각라벨가능/특징가능/Train투입/점수출력을각각구분했다. 실제10Hz시간축·경계2%·관측80%·추적원본QA·정확한사건시점·특징버전·뷰/앉음은행이모두맞아야한다. 신규해부학적특징추출기는아직등록하지않았으므로현재fit보류다. confidence통과만으로어깨오연결을정상/오류자세로학습하지않는다. 공통split_guard와고정source역할검사도별도로반드시통과한다.
012badfront1.8~2.4초부분복귀1점과012goodfront1.6~2초정상2점의실제검토사건에적용했다. 둘다validation그대로이며해부학적시각초안2건허용,초기준비비교목표는2건모두보류,실제fit0건이다. 두영상의동작을비교한것은시각판정근거이며다른원본좌표를초기특징으로복사하지않는다.
기존head와criterion/schema/bank가다르면재사용금지. 정상/실제오류등급의Train지원,원본그룹중복감사,사람제외개발및같은Test비교,등급범위/보정/입력QA를통과한새모델을받은뒤에만통합심사한다. 이번계약/기능검사/정상reference학습이45점지원완료를뜻하지않는다.

전달 인터페이스
tools/squat-fixed-reference/return_target_contract.py의assess(annotation,canonical_source,observation=receipt,registered_schemas=reviewed_extractors).
반환:visual_label_eligible,training_eligible,evaluation_input_eligible와각보류사유/criterion/schema/bank. 점수/등급을새로생성하지않는다. compatible_with_head는목표·버전·은행의일치만검사하며배포허가를뜻하지않는다.
새observationreceipt는원본해시/사건peak-turn-next_ascent프레임/실제슬롯·관측/추적QA/뷰·앉음/특징버전이필요하다. 라벨봉합이나보간을허용하지않는다.

검증
24기능검사통과:미래/타원본준비차단,초기준비없는해부학적라벨허용,validation/test학습차단,회차순서/종료/누락/편측/nonpress/버전·은행/추적QA차단,마지막정상랙허용,입력불변. 실제정확도검증/새모델학습은아니다. 기존라벨/분할/모델/서비스/카메라불변,독립최종Test미개봉.
'''
    p=R/'CONTRACT.txt';assert not p.exists();p.write_text(text)
    save('artifact-hashes.frozen.json',{str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file()})
    status_path=B/'project-coordination/status/model-evaluation.json'
    status=json.loads(status_path.read_text());status['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    status['return_target_contract']=dict(run=str(R.relative_to(B)),tests=24,
        visual_events=2,roles='validation_only',actual_fit=0,production_changed=False,
        next='숄더담당관측·경계수정결과와새해부학적특징버전조율후실제Train등급지원감사')
    status['evidence'].append(str((R/'CONTRACT.txt').relative_to(B)))
    status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print('Frozen actual reviewed events2; validation preserved; fit0.')

if __name__=='__main__':main()

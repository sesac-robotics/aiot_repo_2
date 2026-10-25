"""Read-only audit of frozen labels; propose additions without repartitioning data."""
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

B = Path(__file__).resolve().parents[2]
O = B / 'output/data-readiness-20261024'
O.mkdir(exist_ok=True)
sys.path.insert(0, str(B / 'tools/error-enrichment'))
from split_guard import assert_training_allowed

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

policy_path = B / 'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json'
labels_path = B / 'output/label-recheck-20261024/team-labels-reviewed.frozen.json'
run = Path(read(B / 'output/CURRENT_FIVE_EXERCISE_RUN.json')['run'])
protected = [policy_path, labels_path, run / 'test-inputs.frozen.json', run / 'score-candidates.joblib']
before = {str(p): sha(p) for p in protected}
assert sha(policy_path) == '7aded6fa543285e7e8926b690317c2cc53e1bc96c7f9ca1c181f3077be23f809'
assert sha(labels_path) == 'fa4e5913e3b48790562702e3ff8751aee159f1435ae4b6e633b3e7a27e678667'
policy = read(policy_path)
labels = read(labels_path)['labels']
exercises = ['squats', 'pushups', 'shoulder', 'ytw', 'plank']
names = dict(squats='스쿼트', pushups='푸시업', shoulder='숄더 프레스', ytw='YTW', plank='플랭크')
cells = collections.defaultdict(list)
source_roles = collections.defaultdict(set)
for r in labels:
    role = policy['assignments'][r['source_id']]
    grades = [i['grade'] if i['training_mask'] else None for i in r['items']]
    observed = 'error' if any(g is not None and g < 2 for g in grades) else 'normal' if len(grades) == 4 and all(g == 2 for g in grades) else 'unknown'
    quality = observed if r['quality_training_candidate'] else 'unknown'
    cells[r['person'], r['exercise'], quality, role].append(r)
    source_roles[r['source_sha256']].add(role)
assert all(len(v) == 1 for v in source_roles.values())
matrix = []
plans = []
for person in ['P01', 'P02', 'P03', 'P04']:
    for exercise in exercises:
        for quality in ['normal', 'error']:
            a, z = [cells[person, exercise, quality, role] for role in ['train', 'test']]
            # Minimum add-only integer solution to exact 4:1 interval counts.
            # A planning lower bound, NOT a capture guarantee or automatic role assignment.
            target_test = max(len(z), (len(a) + 3) // 4, 1)
            row = dict(person=person, exercise=exercise, quality=quality,
                       train=len(a), test=len(z),
                       train_sources=sorted({r['source_id'] for r in a}),
                       test_sources=sorted({r['source_id'] for r in z}),
                       test_fraction=len(z)/(len(a)+len(z)) if a or z else None,
                       add_train_lower_bound=4*target_test-len(a),
                       add_test_lower_bound=target_test-len(z))
            matrix.append(row)
            if row['add_train_lower_bound'] or row['add_test_lower_bound']:
                plans.append(row)
missing = [dict(person=r['person'], exercise=r['exercise'], quality=r['quality'], split=s)
           for r in matrix for s in ['train', 'test'] if r[s] == 0]
assert len(missing) == 37
rows = read(run / 'score-inputs.frozen.json')
assert_training_allowed([r for r in rows if r['split'] == 'train'])
assert not {s for r in rows if r['split'] == 'train' for s in r['sources']} & {s for r in read(run/'test-inputs.frozen.json') for s in r['sources']}
heads = read(run / 'score-training.json')['heads']
support = []
for h in heads:
    if h['key'][1] not in ['standing_height_return', 'return_extension', 'lower_return']:
        continue
    support.append({k: h.get(k) for k in ['key', 'status', 'train_count', 'validation_count', 'grade_counts', 'normal_groups', 'error_groups', 'candidate_gate_passed', 'acceptance']})
for name, data in [('split-matrix.csv', matrix), ('add-only-count-lower-bounds.csv', plans)]:
    with (O/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(data)
receipt = dict(matrix=matrix, missing_cells=missing, missing_cell_count=len(missing),
               source_overlap=0, roles_modified=False, labels_modified=False, models_modified=False,
               independent_final_Test_opened=False, support=support, protected_hashes=before,
               addition_lower_bound_warning='Arithmetic interval counts only. New independent recordings must be assigned roles before grading. Do not split adjacent repetitions across Train/Test, discard records to force ratios, or guarantee counts before observing actual labels.',
               quality_definition='Observed criterion grades plus quality_training_candidate. Unknown is not normal. Item-level faults retained separately.')
(O/'audit.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
lines = ['5종 운동: 현재 자료 점검과 즉시 보강 목록', '',
         '기존 라벨/모델/Test는 변경하지 않았다. 아래는 검토된 팀 구간의 전체 품질 기준이다. 보류 구간은 정상으로 채우지 않았으며 개별 항목 오류는 별도로 유지한다.',
         '사람×운동×정상/오류×분할 80칸 중 빈칸 37개. 같은 원본의 Train/Test 중복 0개.', '',
         '사람 / 운동 / 정상 Train:Test / 오류 Train:Test']
for person in ['P01', 'P02', 'P03', 'P04']:
    for e in exercises:
        pair = [next(r for r in matrix if r['person']==person and r['exercise']==e and r['quality']==q) for q in ['normal','error']]
        lines.append(f"{person} / {names[e]} / " + ' / '.join(f"{r['train']}:{r['test']}" for r in pair))
lines += ['', '우선 확보할 자료',
          '1. P04 푸시업과 P02 숄더 프레스: 기존에 Train 원본이 없어, 정상과 오류를 모두 포함한 별도 Train 촬영이 우선이다.',
          '2. 스쿼트 서기 높이, 푸시업 팔 펴기, 숄더 랙 복귀: 실제 완료 시도의 0점/1점/2점을 구분한다. 완전한 복귀 전 다음 반복을 시작한 사례를 시간순으로 확인한다. 쉬기·중단·영상 잘림은 실패 정답으로 만들지 않는다.',
          '3. 정면 푸시업 복귀: 개발용 오류 0개. 최소 오류 3구간/2개 사람·원본 그룹이 필요하며 이는 채택 기준의 최소량일 뿐 충분한 성능 보장이 아니다.',
          '4. 숄더 복귀: 정면 Train 104구간이 모두 정상 복귀여서 오류 학습 그룹이 없다. bad 영상의 양팔 시차·상단 범위를 복귀 실패로 바꾸지 않는다.',
          '5. 스쿼트 높이 복귀: 개발 오류가 한 그룹에 몰려 있고 검출 1/6으로 부족하다. 별도 개발 오류 촬영자가 필요하다.',
          '6. YTW는 Y→T→W 모양/순서, 플랭크는 지지 유지의 실제 오류를 확보한다. 서기 복귀 라벨을 붙이지 않는다.', '',
          '80/20 보강 계획 사용법',
          'split-matrix.csv는 현재 원본 ID와 구간 수, add-only-count-lower-bounds.csv는 기존 배정을 보존하며 4:1이 되기 위한 산술적 추가 구간 하한이다. 촬영 횟수나 확정 정답 수가 아니다. 한 원본의 반복들은 같은 역할을 유지해야 하므로 실제 추가량은 더 커질 수 있다.',
          '새 원본은 먼저 Train/Test 역할을 배정하고, 각 사람·운동에서 정상과 오류를 각각 확보한다. 오류 0/1점도 가능하면 양쪽에 확보한다. 학습 중 선택은 Train 안의 그룹 교차검증/개발 자료로 한다. 새 Test를 모델 선택에 사용하지 않는다.',
          '기존 Test는 이미 노출된 진단용이다. 새 최종 검증은 별도로 잠그고 마지막에 1회 평가한다. 사람 ID는 현재 시각적 연속성에 기반한 임시 식별이며 실명 확정이 아니다.', '',
          '바로 가능한 작업: 미검토 Train/개발 원본의 전체 반복을 촘촘히 재검토하고, 관찰 근거가 있는 오류만 추가한다. 새 라벨이 없는 상태의 반복 재학습은 진행하지 않는다.']
(O/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
assert before == {str(p): sha(p) for p in protected}
print(json.dumps(dict(team_intervals=len(labels), matrix_rows=len(matrix), missing_cells=len(missing), return_heads=len(support), protected_files_unchanged=True, report=str(O/'RESULTS.txt')), ensure_ascii=False))

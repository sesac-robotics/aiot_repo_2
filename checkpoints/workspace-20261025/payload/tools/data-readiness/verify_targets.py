"""Verify criterion target lineage without confusing missing features with bad labels."""
import collections
import hashlib
import json
import sys
import os
from pathlib import Path
import joblib

B = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(B/'tools/error-enrichment'))
O = Path(os.environ.get('DATA_READINESS_OUTPUT',str(B/'output/data-readiness-20261024')))
O.mkdir(parents=True,exist_ok=True)
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
run = Path(read(B/'output/CURRENT_FIVE_EXERCISE_RUN.json')['run'])
test_path = run/'test-inputs.frozen.json'
before = sha(test_path)
labels = {r['id']:r for r in read(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json')['labels']}
policy = read(B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json')
rows = read(run/'score-inputs.frozen.json') + read(test_path)
aliases = {
    'squats': {'trunk_balance':'frontal_trunk_stability'},
    'pushups': {'trunk_line':'trunk_alignment', 'return_extension':'extension_return'},
    'shoulder': {'frontal_trunk_hold':'frontal_trunk_stability'},
    'ytw': {'y_reach':'Y_shape', 't_reach':'T_shape', 'w_shape':'W_shape', 'upright_trunk_hold':'upright_trunk_stability'},
    'plank': {'trunk_line':'trunk_alignment', 'leg_and_body_support':'support', 'elbow_sagittal_position':'elbow_position', 'support_continuity':'continuity'},
}
compared = matched = 0
unobservable = []
unmapped = []
represented = set()
conflicts = []
team_rows = [r for r in rows if r['family']=='R2D2']
for r in team_rows:
    assert len(r['items'])==len(r['levels'])
    d = labels[r['id'][5:]]
    represented.add(d['id'])
    assert r['person']=='r2d2_'+d['person']
    assert r['split']==policy['assignments'][d['source_id']]
    assert r['sources']==[d['source_sha256']]
    grades = {i['id']:i['grade'] if i['training_mask'] else None for i in d['items']}
    for item, grade in zip(r['items'],r['levels']):
        name = aliases[r['exercise']].get(item['item'],item['item'])
        record = dict(id=r['id'], source_id=d['source_id'], view=d['view'], split=r['split'], item=item['item'], bank=item['bank'], input_grade=grade, eligible=item['eligible'], holds=item['holds'])
        if name not in grades:
            # Legacy geometry with no corresponding manual criterion cannot train.
            assert grade is None and item['eligible'] is False
            unmapped.append(record)
            continue
        compared += 1
        record['reviewed_grade'] = grades[name]
        if grade==grades[name]:
            matched += 1
        elif grade is None and not item['eligible'] and item['feature'] is None and item['holds']:
            unobservable.append(record)
        else:
            conflicts.append(record)
assert not conflicts, conflicts
models = joblib.load(run/'score-candidates.joblib')
byid = {r['id']:r for r in rows}
fit_checks = 0
for key, head in models['heads'].items():
    for rid in head['trained_ids']:
        r = byid[rid]
        assert r['split']=='train'
        if r['family']!='R2D2':
            continue
        criterion, bank = key[1:]
        candidates = [(i,g) for i,g in zip(r['items'],r['levels']) if i['item']==criterion and i['bank']==bank]
        assert candidates and any(i['eligible'] and g is not None for i,g in candidates)
        fit_checks += 1
omitted = [dict(id=d['id'],source_id=d['source_id'],person=d['person'],exercise=d['exercise'],split=policy['assignments'][d['source_id']],complete_cycle=d['complete_cycle'],quality_training_candidate=d['quality_training_candidate'],observed_items=[i['id'] for i in d['items'] if i['training_mask']],reason='partial_test_item_diagnostic_queue_without_changing_fixed_Test' if d['complete_cycle'] else 'incomplete_cycle_not_a_complete_attempt_target') for d in labels.values() if d['id'] not in represented]
assert len(team_rows)==264 and len(omitted)==16
assert all(r['input_grade'] is None and not r['eligible'] for r in unobservable+unmapped)
assert all(r['split']=='test' for r in omitted if r['complete_cycle'])
receipt = dict(team_labels=len(labels),team_input_rows=len(team_rows),compared_feature_bank_targets=compared,exactly_matching_targets=matched,wrong_grade_conflicts=conflicts,observable_training_label_conflicts=0,feature_unobservable_targets=unobservable,unsupported_legacy_targets=unmapped,omitted_segments=omitted,trained_team_head_memberships_checked=fit_checks,test_input_sha256=before,test_modified=False,labels_modified=False,models_modified=False,independent_final_Test_opened=False)
(O/'target-lineage.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
(O/'remaining-review-queue.json').write_text(json.dumps(dict(feature_unobservable_targets=unobservable,omitted_segments=omitted,queue_only_not_training_labels=True,existing_roles_must_be_preserved=True),ensure_ascii=False,indent=2))
assert sha(test_path)==before
print(json.dumps(dict(team_inputs=len(team_rows),compared_targets=compared,matching=matched,wrong_grades=len(conflicts),unobservable=len(unobservable),unsupported_legacy=len(unmapped),omitted=len(omitted),trained_memberships_checked=fit_checks),ensure_ascii=False))

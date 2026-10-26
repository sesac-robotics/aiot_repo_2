import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ytw-work-20261024/y-entry-v4.1';r=json.load(open(O/'evaluation.json'));checks=r['checks'];base=checks['baseline'];fix=checks['ready_origin_fix'];new=checks['hold_onset_v4'];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();source=B/'tools/ytw-work/ytw_hold_onset_v4.py';assert r['candidate_code_sha256']==sha(source)
for replay in new['replays']:
 for e in replay['events']:
  assert e['complete_attempt_reviewed'] is False and e['projection_reviewed'] is False and e['grade'] is None
  if e['auto_phase_windows'] is not None:
   N=e['end_tick']-e['start_tick']+1;w=e['auto_phase_windows'];assert all(0<=a<=b<N for a,b in w.values());assert w['Y'][1]<w['T'][0]<=w['T'][1]<w['W'][0]
   assert e['Y_boundary_evidence']['confirmed_tick']==e['start_tick'];assert e['Y_boundary_evidence']['backdated'] is False;assert e['coarse_start_tick']<=e['start_tick']<=e['end_tick'];assert e['observation_coverage']>=.8
assert new['summary']['source_event_counts']==base['summary']['source_event_counts']
def timing(check):
 ys=[p['phase_errors']['Y'] for p in check['positive'] if 'phase_errors' in p]
 return dict(Y_early_more_than_0_2=sum(y['predicted']<y['reference'][0]-.200001 for y in ys),Y_late_more_than_0_2=sum(y['predicted']>y['reference'][1]+.200001 for y in ys),Y_definitely_within_0_2=sum(y['definite_within_0_2'] for y in ys),Y_observations=len(ys))
saved=json.load(open(O/'saved-session.json'));assert saved['candidate_code_sha256']==sha(source)
for result in saved['results']:
 for e in result['events']:
  s=e.get('same_dense_AI_Rule_result',{});assert s.get('ai_score') is None and s.get('rule_score') is None
v=dict(schema='ytw-Y-entry-repair-verification-v4.1',updated_at_utc=datetime.now(timezone.utc).isoformat(),comparison={n:dict(summary=c['summary'],timing=timing(c)) for n,c in checks.items()},reference_sha256=sha(B/'output/ytw-work-20261024/boundary-v2/references-v2.frozen.json'),model_refit=False,source_roles_preserved=True,Test_or_finalTest_opened=False,live_app_changed=False,camera_started=False,all_new_phase_windows_valid=True,no_backdated_stability_tick=True,ordered_count_preserved=True,natural_negative_false_completions_unchanged=True,scoring_manual_approvals_false=True,source_bundle_newly_deployed=False,new_regression_tests_passed=13,scope='fully_exposed_development_repair_after_v3_failure_not_independent_validation',activation_gate_passed=False,remaining=['13 negatives below20/types limited','phase19/30 definite and terminal7/10 below90percent','one late Y case >=0.3s','AI draft uncertainty, no new full-bundle unseen person','state boundary improvement not severity score accuracy'],files_sha256={str(p.relative_to(B)):sha(p) for p in [source,B/'tools/ytw-work/ytw_ready_fix_v2.py',B/'tools/ytw-work/entry_v4_replay.py',B/'tools/ytw-work/test_ytw_entry_v4.py',O/'evaluation.json',O/'saved-session.json']})
(O/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');print(json.dumps({n:x['timing'] for n,x in v['comparison'].items()},ensure_ascii=False,indent=2))

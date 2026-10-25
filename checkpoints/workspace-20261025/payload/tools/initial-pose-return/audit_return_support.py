"""Freeze return-grade support and person/quality partition checks, without fit."""
import json,sys,collections
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/initial-pose-return-v7-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import POLICY,assert_training_allowed
read=lambda p:json.loads(Path(p).read_text())
assert sha(POLICY)=='7aded6fa543285e7e8926b690317c2cc53e1bc96c7f9ca1c181f3077be23f809'
team_path=B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json'
assert sha(team_path)=='fa4e5913e3b48790562702e3ff8751aee159f1435ae4b6e633b3e7a27e678667'
policy=read(POLICY);team=read(team_path)['labels'];cells=collections.Counter();observed_cells=collections.Counter();sources=collections.defaultdict(set);team_return=collections.defaultdict(collections.Counter)
for r in team:
 role=policy['assignments'][r['source_id']];grades=[i['grade'] if i['training_mask'] else None for i in r['items']];observed_quality='error' if any(g is not None and g<2 for g in grades) else 'normal' if len(grades)==4 and all(g==2 for g in grades) else 'unknown';quality=observed_quality if r['quality_training_candidate'] else 'unknown';cells[(r['person'],r['exercise'],quality,role)]+=1;observed_cells[(r['person'],r['exercise'],observed_quality,role)]+=1;sources[r['source_sha256']].add(role)
 for item in r['items']:
  if item['id'] in ['standing_height_return','extension_return','lower_return'] and item['training_mask'] and item['grade'] is not None:team_return[(r['exercise'],role)][item['grade']]+=1
assert all(len(v)==1 for v in sources.values())
current=read(B/'output/CURRENT_FIVE_EXERCISE_RUN.json');run=Path(current['run']);training=read(run/'score-training.json');rows=read(run/'score-inputs.frozen.json');tr=[r for r in rows if r['split']=='train'];assert_training_allowed(tr)
heads=[h for h in training['heads'] if h['key'][1] in ['standing_height_return','return_extension','lower_return']]
support=[{k:h.get(k) for k in ['key','status','train_count','grade_counts','normal_groups','error_groups','source_families','candidate_gate_passed','group_cv','fixed_development_validation']} for h in heads]
people={}
for p in ['P01','P02','P03','P04']:
 people[p]={}
 for q in ['normal','error','unknown']:
  train=sum(n for (person,ex,quality,role),n in cells.items() if person==p and quality==q and role=='train');test=sum(n for (person,ex,quality,role),n in cells.items() if person==p and quality==q and role=='test');people[p][q]=dict(train=train,test=test,test_fraction=test/(train+test) if train+test else None)
missing=[dict(person=p,exercise=e,quality=q,split=s) for p in people for e in ['squats','pushups','shoulder','ytw','plank'] for q in ['normal','error'] for s in ['train','test'] if not cells[(p,e,q,s)]]
report=dict(policy_sha256=sha(POLICY),reviewed_team_labels_sha256=sha(team_path),source_roles_unchanged=True,same_original_source_in_train_and_test=False,requested_test_fraction=.2,exact_80_20_achieved=False,whole_source_grouping_preserved=True,person_quality_counts=people,person_exercise_quality_split_cells=[dict(person=p,exercise=e,quality=q,split=s,count=n) for (p,e,q,s),n in sorted(cells.items())],missing_normal_error_cells=missing,missing_normal_error_cell_count=len(missing),team_return_grades={e+'/'+s:dict(c) for (e,s),c in team_return.items()},mixed_data_return_severity_heads=support,no_severe_grade_zero_return_labels_in_these_heads=not any(int(g)==0 and n for h in heads for g,n in h.get('grade_counts',{}).items()),Test_used_for_fit_or_model_selection=False,existing_Test_is_historically_exposed_diagnostic=True,independent_final_Test_opened=False,score_run=str(run),score_model_sha256=sha(run/'score-candidates.joblib'),test_inputs_sha256=sha(run/'test-inputs.frozen.json'))
report.update(operational_quality_definition='Observed primary grades plus existing quality_training_candidate flag. Excluded/incomplete whole-quality intervals remain unknown for whole-quality strata; item-level observed errors remain in labels.',item_observation_counts_before_whole_quality_eligibility=[dict(person=p,exercise=e,quality=q,split=s,count=n) for (p,e,q,s),n in sorted(observed_cells.items())],whole_quality_exclusion_does_not_erase_observed_item_faults=True)
(O/'return-grade-and-split-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
intake=read(B/'output/return-outcome-audit-20261024/shoulder-intake.frozen.json');decisions=[]
for r in intake:
 assert r['split']=='train'
 decisions.append(dict(id=r['id'],person=r['person'],source_sha256=r['source_sha256'],split='train',review_sheet=r['review_sheet'],review_sheet_sha256=sha(r['review_sheet']),review_frames=r['review_frames'],review_scope='12_frame_chronological_screen_nonexpert',return_grade=None,training_eligible=False,decision='needs_dense_complete_cycle_review_before_new_return_failure_label',reason='Video-wide bad label is not a return-error label. Sampled frames show lowering/repeated lifts, insufficient overhead reach or trunk compensation; no unambiguous completed failed-return grade is established by this screen.'))
(O/'new-shoulder-return-screen.frozen.json').write_text(json.dumps(dict(reviewed=decisions,new_confirmed_failure_labels=0,guessed_failure_labels_not_added=True),indent=2))
print('AUDIT',len(team),'team intervals;',len(missing),'missing normal/error cells;',len(support),'return feature heads;',len(decisions),'additional shoulder screens')

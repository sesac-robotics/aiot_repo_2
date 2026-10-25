import json,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ytw-work-20261024/logo-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[r for r in json.load(open(B/'output/initial-pose-return-v7-20261024/state-inputs.frozen.json')) if r['exercise']=='ytw']
r=json.load(open(O/'replay.json'));plan=json.load(open(O/'plan.frozen.json'))
# Directly inspected all eleven 0.25s contact sheets. These are conservative
# observed hold interiors, not exact onset/offset or severity truth.
windows={
'V37':[[[3.75,4.25],[4.75,5.5],[6,7.5]],[[8.25,8.75],[9.25,10.25],[11,12]],[[13,13.75],[14.5,15],[15.75,16.5]],[[17.5,18],[18.75,19.5],[20.25,21.25]]],
'V39':[[[0,.5],[1.25,1.75],[2.25,3.75]],[[4.5,5.75],[6.5,7.25],[7.75,8.75]],[[9.5,10.25],[11,11.5],[12.25,12.75]],[[13.5,14.75],[15.5,16],[16.75,17.5]],[[18.25,18.75],[19.75,20.25],[20.75,21.25]],[[21.75,22.25],[23.25,24],[24.5,25]]],
'V41':[[[0,.5],[1,1.5],[2.25,3]],[[3.75,4.5],[5.25,5.75],[6.5,7.25]],[[8,8.5],[9,9.5],[10,10.75]],[[11.5,12],[12.5,13],[13.75,14.25]],[[15,15.5],[16,16.5],[17,17.75]]],
'V43':[[[0,1.5],[2.25,3.25],[4,5]],[[6,6.75],[7.25,7.75],[8.5,9.5]],[[10.25,10.75],[11.25,11.75],[12.25,13]],[[13.75,15],[15.75,16.25],[17,17.75]],[[18.5,19.25],[19.75,20.25],[21,21.75]],[[22.25,23],[23.75,24.25],[25,25.5]]]
}
review=dict(schema='ytw-video-observed-hold-interiors-v1-draft',reviewed_at_utc=datetime.now(timezone.utc).isoformat(),review_method='all_11_time_ordered_0.25s_video_contact_sheets_visually_inspected',reviewer='AI_assisted_draft_not_expert',labels_used_for_fit=False,prediction_exposure='prior_full_model_candidate_counts_seen_and_LOPO_counts_logged_before_annotation_not_blind',boundary_truth_available=False,severity_truth_available=False,cycles={sid:[dict(id=sid+'_order_'+str(i+1),Y=y,T=t,W=w,quality_grade=None,exact_boundaries=None) for i,(y,t,w) in enumerate(ws)] for sid,ws in windows.items()},ambiguous=[dict(source_id='V37',interval=[0,3.25],reason='initial_Y_already_in_progress_and_short_observed_hold; not_failure_truth')],negative_intervals=[dict(source_id='V43',interval=[26.5,28.75],kind='arms_down_wait',quality_grade=None)],missing_negative_types=['wrong_order','occlusion','time_gap','aborted_attempt'],sheets_sha256={p.name:sha(p) for p in sorted((O/'video-review').glob('*.jpg'))})
p=O/'video-hold-review-v1.frozen.json'
if not p.exists():p.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
else:review=json.load(open(p))
audits=[]
for fold in r['folds']:
 tr=[x for x in rows if x['id'] in fold['train_ids']];va=[x for x in rows if x['id'] in fold['excluded_ids']]
 assert len(tr)+len(va)==76
 assert set(x['person'] for x in tr).isdisjoint(x['person'] for x in va)
 assert set(x['source_id'] for x in tr).isdisjoint(x['source_id'] for x in va)
 assert set(x['source_sha256'] for x in tr).isdisjoint(x['source_sha256'] for x in va)
 assert set(x['cache_sha256'] for x in tr).isdisjoint(x['cache_sha256'] for x in va)
 assert all(x['split']=='train' for x in tr+va)
 audits.append(dict(excluded_person=fold['excluded_person'],fit_people=fold['train_people'],fit_rows=len(tr),excluded_rows=len(va),fit_source_hashes=sorted({x['source_sha256'] for x in tr}),excluded_source_hashes=sorted({x['source_sha256'] for x in va}),fit_source_ids=sorted({x['source_id'] for x in tr}),excluded_source_ids=sorted({x['source_id'] for x in va}),sync_or_overlap_exclusion='all_YTW_labelled_rows_of_excluded_person_removed; no_other_view_or_interval_reintroduced',fit_preprocessing='none; deterministic_per_frame_pose_only; no_scaler_fit',required_class_support={'target':sorted({x['target'] for x in tr}),'phase':sorted({x['phase'] for x in tr})},no_inherited_gate_flags=True,model_sha256=fold['model_sha256']))
result=dict(schema='ytw-logo-development-review-v1',plan_sha256=sha(O/'plan.frozen.json'),input_sha256=plan['inputs_sha256'],scope='existing_settings_fixed_person_excluded_development_not_nested_independent',all_four_people_used_in_prior_settings_selection=True,fold_audits=audits,candidate_events={f['source_id']:len(f['events']) for f in r['folds']},video_draft_order_attempts={sid:len(v) for sid,v in windows.items()},sequence_accuracy_metrics=None,boundary_accuracy_metrics=None,negative_intervals_reviewed=1,gates_passed=False,reasons=['only_1_natural_negative_interval_not_20','no_exact_onset_offset_boundary_truth','ordered_attempt_drafts_not_normal_quality_truth','settings_previously_selected_using_all_people','AI_assisted_exposed_draft_not_blind_expert_validation'],score_model_evaluated=False,Test_used=False,production_replacement=False,code_sha256={str(p.relative_to(B)):sha(p) for p in [B/'tools/ytw-work/logo_development.py',B/'tools/ytw-work/auto_contract_v1.py',B/'tools/ytw-work/review_video.py']})
(O/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['fold_audits','code_sha256']},ensure_ascii=False,indent=2))

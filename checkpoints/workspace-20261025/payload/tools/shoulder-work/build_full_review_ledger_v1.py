from pathlib import Path
import json,csv,collections,datetime
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-work-20261024/full-review-v1'
fit_inputs=json.loads((B/'output/shoulder-work-20261024/normal-reference-v3/inputs.frozen.json').read_text())
inv=json.loads((O/'inventory.frozen.json').read_text())['sources'];own_labels=json.loads((O/'dense-visual-labels.json').read_text());model_label_path=B/'output/shoulder-common-support-audit-20261024/full-review-v1/dense-visual-labels.frozen.json';model_labels=json.loads(model_label_path.read_text());assert len(model_labels)==72
labels={x['id']:x for x in own_labels+model_labels};journal={x['id']:x for x in json.loads((O/'shoulder-review-journal.json').read_text())}
run=Path(json.loads((B/'output/CURRENT_FIVE_EXERCISE_RUN.json').read_text())['run']);previous=json.loads((run/'score-inputs.frozen.json').read_text())+json.loads((run/'test-inputs.frozen.json').read_text());out=[]
for s in inv:
 l=labels.get(s['id']);hist=[r for r in previous if s['source_sha256'] in r['sources']];itemhist={i:sum(1 for r in hist for a,g in zip(r['items'],r['levels']) if a['item']==i and a['eligible'] and g is not None) for i in ['lower_return','frontal_trunk_hold']}
 row={k:s[k] for k in ['id','path','source_sha256','person','recording_group','view','provider_tag','split','owner']}; row.update(review_status='full_timeline_and_selected_dense_reviewed' if l else 'other_owner_review_in_progress',historical_offline_rows=len(hist),historical_item_eligible_rows=itemhist,new_fit_status='not_yet_fitted',identity_review='no_identity_assertion_from_filename',synchronized_view_status='not_verified_do_not_assume_synchronized')
 if l:
  row.update(label_evidence=str(O/'dense-visual-labels.json') if s['owner']=='shoulder' else str(model_label_path),movement_protocol=l['movement_protocol'],selected_observed_events=l['observed_events'],evidence_note_ko=l['note_ko'])
  for name,key in [('lower_return','lower_return_grade'),('frontal_trunk_hold','frontal_trunk_grade')]:
   g=l[key]
   if g is not None:reason='관측한 선택구간의 정상등급 새라벨. '+('Train 정상학습입력 후보; 0/1 오류판별지원 근거 없음.' if s['split']=='train' else '기존 '+s['split']+' 평가전용; 학습금지.')
   elif l['movement_protocol']=='non_press_arm_extension':reason='상완고정 머리뒤팔꿈치폄: 숄더프레스 점수정답으로미사용; 별도프로토콜분류 '+('학습후보' if s['split']=='train' else '평가전용')
   elif name=='frontal_trunk_hold' and s['view']!='front':reason='정면몸통 목표와 구도불일치; 측면전후굽힘과 좌우기울임을 합치지 않음.'
   elif s['view']=='side':reason='먼팔 중첩/골반crop/목표프로토콜불확실로 양팔복귀등급 미사용; 근측팔 경로는 프로토콜자료로보존.'
   else:reason='선택구간 관측/확정근거 부족; 누락이나 비대칭을 오류등급으로 만들지 않음.'
   usage=[x for x in fit_inputs if x['bank']!='front_seated' and x['source_id']==s['id'] and (x['item']==name or name=='frontal_trunk_hold' and x['item']=='frontal_trunk_hold_window_reference')]
   if usage:reason+=' 정상참조 학습실행 '+str(len(usage))+'행; 사람제외검증 일반화미달/비배포.'
   elif g is not None and s['split']=='train':
    if s['view']=='side':reason+=' 근측팔만 관측한 등급이므로 공통 양팔복귀 등급으로 학습하지 않음.'
    elif s['owner']=='model_evaluation':reason+=' 이번정상참조v3 학습입력동결 뒤 검토결과가 전달되어 해당후보에는 미편입.'
    elif 'subject_023_' in s['id']:reason+=' 앉은참조는Train1명뿐이라 별도bank학습보류.'
    else:reason+=' 시각/관측조건/원본추적QA를 통과한 입력을 확보하지 못해 미편입(원본폐기아님).'
   row[name]={'laterality':l.get('lower_laterality','bilateral_visual_draft' if s['view']!='side' else 'ungraded_far_arm_occlusion') if name=='lower_return' else None,'common_bilateral_grade':None if name=='lower_return' and s['view']=='side' else g,'normal_reference_fit_rows':len(usage),'normal_reference_variant_ids':[x['id'] for x in usage],'grade':g,'scope':'selected_visually_observed_windows_only','disposition_reason_ko':reason}
 else:
  for name in ['lower_return','frontal_trunk_hold']:row[name]={'grade':None,'scope':'pending_other_owner','disposition_reason_ko':'모델담당72개 원본검토 진행중; 미검토를 미사용확정으로표시하지않음.'}
 if any(row[n].get('normal_reference_fit_rows',0)>0 for n in ['lower_return','frontal_trunk_hold']):row['new_fit_status']='normal_reference_fit_completed_research_only'
 out.append(row)
report={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':out,'total':len(out),'reviewed_here':len(own_labels),'reviewed_model_owner':len(model_labels),'reviewed_total':len(labels),'other_owner_pending_merge':len(out)-len(labels),'split_counts':dict(collections.Counter(s['split'] for s in inv)),'historical_measurement_authority':str(run),'expert_adjudicated':False,'new_fit_completed':True,'normal_reference_total_inputs':len(fit_inputs),'normal_reference_actual_fitted_Train_rows':sum(x['split']=='train' and x['bank']!='front_seated' for x in fit_inputs),'normal_reference_development_rows':sum(x['split']=='validation' for x in fit_inputs),'normal_reference_single_person_held_rows':sum(x['bank']=='front_seated' for x in fit_inputs),'normal_reference_production_eligible':False,'production_changed':False}
(O/'source-dispositions.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
with (O/'source-dispositions.csv').open('w') as f:
 fields=['id','split','view','provider_tag','review_status','movement_protocol','lower_grade','lower_reason','trunk_grade','trunk_reason','new_fit_status','evidence_note_ko']; w=csv.DictWriter(f,fields);w.writeheader()
 for r in out:w.writerow({**{k:r.get(k,'') for k in fields if k not in ['lower_grade','lower_reason','trunk_grade','trunk_reason']},'lower_grade':r['lower_return']['grade'],'lower_reason':r['lower_return']['disposition_reason_ko'],'trunk_grade':r['frontal_trunk_hold']['grade'],'trunk_reason':r['frontal_trunk_hold']['disposition_reason_ko']})
assert len(labels)==150 and len(out)==150
for s in inv:
 if s['id'] in labels:assert s['source_sha256']==labels[s['id']]['source_sha256'] and s['split']==labels[s['id']]['split']
assert not ({s['source_sha256'] for s in inv if s['split']=='train'} & {s['source_sha256'] for s in inv if s['split']!='train'})
print({k:v for k,v in report.items() if k!='sources'})

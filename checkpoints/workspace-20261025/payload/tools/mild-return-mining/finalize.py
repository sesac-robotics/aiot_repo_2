"""Record an unsuccessful sensitivity experiment without promoting it."""
import json,collections,re,sys
from pathlib import Path
import joblib
B=Path(__file__).resolve().parents[2];O=B/'output/mild-return-calibrated-20261024';OLD=B/'output/return-root-cause-v3-20261024';D=B/'output/mild-return-mining-20261024';sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/paired-evaluation')]
from common import sha
from split_guard import POLICY,assert_training_allowed
from artifact_integrity import equal_state
read=lambda p:json.loads(p.read_text());rows=read(O/'score-inputs.frozen.json');before=read(OLD/'score-inputs.frozen.json');test=read(O/'test-inputs.frozen.json');model=joblib.load(O/'score-candidates.joblib');parent=joblib.load(OLD/'score-candidates.joblib');reports=read(O/'score-training.json')['heads'];cmp=read(O/'comparison.frozen.json');target='return_squat_front_w1_9_3_r08';changes=[]
assert len(rows)==len(before)==976 and len(model['heads'])==56
for r,p in zip(rows,before):
 for k in ['id','person','group','split','sources','items']:assert r[k]==p[k]
 for j,(g,old) in enumerate(zip(r['levels'],p['levels'])):
  if g!=old:
   assert r['id']==target and r['split']=='train' and r['items'][j]['item']=='standing_height_return' and old==2 and g==1;changes.append(r['items'][j]['bank'])
assert len(changes)==3
assert sha(O/'test-inputs.frozen.json')==sha(OLD/'test-inputs.frozen.json')
assert sha(POLICY)=='7aded6fa543285e7e8926b690317c2cc53e1bc96c7f9ca1c181f3077be23f809'
team_path=B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json';assert sha(team_path)=='fa4e5913e3b48790562702e3ff8751aee159f1435ae4b6e633b3e7a27e678667'
changed={tuple(h['key']) for h in read(O/'changed-heads-fit/score-training.json')['heads']};assert sum(equal_state(h,parent['heads'][k]) for k,h in model['heads'].items() if k not in changed)==53
byid={r['id']:r for r in rows};trained={i for h in model['heads'].values() for i in h['trained_ids']};assert_training_allowed([byid[i] for i in trained]);assert len(trained)==815 and target in trained
assert not {s for r in rows for s in r['sources']}&{s for r in test for s in r['sources']}
selection=read(B/'output/mild-return-combined-20261024/TRAIN_SELECTION.frozen.json');assert selection['selected']=='upper'
head=next(h for h in reports if h['key']==['squats','standing_height_return','height_upper_reversal_v1']);rp=next(h for h in read(O/'rule-training.json')['heads'] if h['key']==head['key']);ai=head['validation_predictions'];rule=rp['validation_predictions'];a={p['id']:p for p in ai};b={p['id']:p for p in rule};assert a.keys()==b.keys() and len(ai)==19
oldhead=next(h for h in read(OLD/'score-training.json')['heads'] if h['key']==['squats','standing_height_return','height_reference_pose_v1']);old_errors={p['id'] for p in oldhead['validation_predictions'] if p['target']<2};assert old_errors=={p['id'] for p in ai if p['target']<2} and len(old_errors)==6
paired=[dict(id=i,target=p['target'],ai=p['prediction'],rule_base=b[i]['prediction']) for i,p in a.items()];assert all(p['target']==b[p['id']]['target'] for p in ai)
clips=read(B/'data/mild-return-expansion-20261024/clip-manifest.json')['clips'];assert len(clips)==2 and all(sha(c['path'])==c['sha256'] and c['split']=='train' and all(p['exact_decoded_pixels_equal'] for p in c['pixel_checks']) for c in clips)
review=read(D/'review-decisions.frozen.json')['reviewed'];assert len(review)==32 and sum(r['new_grade']==1 for r in review)==1
for e,m in cmp['same_bank_by_exercise'].items():
 for k in ['n','error_n','normal_n']:assert m['ai'][k]==m['rule_base'][k]
assert cmp['same_bank_paired_items']==156
# Recompute all 80 team participant/exercise/quality/split cells without treating unknown as normal.
policy=read(POLICY);labels=read(team_path)['labels'];counts=collections.Counter();unknown=collections.Counter()
for r in labels:
 split=policy['assignments'][r['source_id']];gs=[i['grade'] for i in r['items'] if i['training_mask'] and i['grade'] is not None];quality='error' if r['quality_training_candidate'] and any(g<2 for g in gs) else 'normal' if r['quality_training_candidate'] and len(gs)==4 and all(g==2 for g in gs) else 'unknown'
 if quality=='unknown':unknown[(r['person'],r['exercise'],split)]+=1
 else:counts[(r['person'],r['exercise'],quality,split)]+=1
cells=[dict(person=p,exercise=e,quality=q,split=s,count=counts[p,e,q,s]) for p in ['P01','P02','P03','P04'] for e in ['squats','pushups','shoulder','ytw','plank'] for q in ['normal','error'] for s in ['train','test']]
empty=sum(c['count']==0 for c in cells)
# Preserve and expose actual empty-cell counts; don't assert a rounded 80/20 split.
(O/'person-quality-split-audit.json').write_text(json.dumps(dict(source_assignments_unchanged=True,exact_80_20_achieved=False,cells=cells,empty_cells=empty,unknown=[dict(person=k[0],exercise=k[1],split=k[2],count=v) for k,v in unknown.items()]),indent=2))
checks={}
for name,n in [('calibration',4),('contracts',16),('paired',9)]:
 t=(O/f'tests-{name}.log').read_text();assert f'Ran {n} tests' in t and '\nOK' in t;checks[name]=n
actual=sum(p['target']<2 and p['prediction']<1.5 for p in ai);ractual=sum(p['target']<2 and p['prediction']<1.5 for p in rule);fp=sum(p['target']==2 and p['prediction']<1.5 for p in ai);rfp=sum(p['target']==2 and p['prediction']<1.5 for p in rule)
return_test=[p for p in cmp['paired_predictions'] if p['criterion']=='standing_height_return'];assert len(return_test)==6 and all(p['target']==2 for p in return_test)
verification=dict(status='Train_label_correction_and_nested_calibration_no_selected_detection_gain',all_five_reliable=False,production_replacement=False,policy_unchanged=True,team_labels_unchanged=True,exact_80_20_achieved=False,missing_split_cells=empty,new_Train_label_revisions=1,new_rows=0,new_grade0=0,screened_Train_intervals=32,screened_sources=len({r['source_sha256'] for r in read(D/'intake.frozen.json')}),unique_trained_rows=len(trained),trained_heads=56,heads_refitted=3,unchanged_heads_preserved=53,source_overlap=0,Test_used_for_fit_or_selection=False,Test_bytes_unchanged=True,Test_rows=len(test),same_Test_paired_items=156,standing_return_Test_normals=6,standing_return_Test_errors=0,selected_bank=head['key'][2],same_six_development_errors_retained=True,development_errors=6,AI_error_detected=actual,Rule_error_detected=ractual,development_normals=13,AI_normal_false_alarms=fp,Rule_normal_false_alarms=rfp,selected_model_recall_improved=actual>1,independent_final_Test_opened=False,tests=checks,total_tests=sum(checks.values()),actual_trained_families=dict(collections.Counter(byid[i]['family'] for i in trained)),clip_pixel_checks=6)
(O/'verification.json').write_text(json.dumps(verification,indent=2));(O/'same-development-ai-rule.json').write_text(json.dumps(dict(same_ids_targets=True,predictions=paired,selection_on_development_forbidden=True),indent=2))
lines=['미세 복귀 오류 재검토 및 재학습 결과','',f'32개 Train 후보/10개 원본 재검토: 정상으로 잘못 라벨링된 1개를 부분 복귀 1점으로 수정했습니다. 기존 학습 인물 CFRep w1의 실제 반복 동작이며, 새 사람/새 독립 표본으로 세지 않습니다.','수정: return_squat_front_w1_9_3_r08, 원본313–351프레임. 초기67프레임과 비교해 최대 복귀348/351프레임에서도 완전히 일어서지 않고354프레임부터 다음 하강을 시작합니다. 다른31개는 변경할 증거가 부족해 기존 라벨을 유지했습니다. 전문가 검수 전 AI 영상 검토 라벨입니다.','실제 복귀 0점은 새로 확인하지 못했습니다. 정지·잘림·중간 상승을 실패로 만들지 않았습니다.','',f'Train 내부 사람 제외 중첩 검증으로 확률 경계를 학습했습니다. 선택된 height_upper_reversal_v1은 여전히 동일 개발 오류6개 중 {actual}개 검출, 정상13개 중 {fp}개 오판입니다. 같은19개 개발 항목에서 Rule Base는 오류{ractual}/6, 정상 오판{rfp}/13입니다.','선택된 AI의 실제 개발 검출률은 개선되지 않았습니다. 사람별 정상 편차와 미세 오류의 겹침이 아직 해결되지 않았습니다.','기준 자세 특징 후보는 오류3/6, 기존 높이 특징 후보는4/6을 검출했지만, 후자는 정상17개 중2개 오판했습니다. 이 개발 결과를 보고 후보를 고르지 않았습니다. 기준 자세/높이 결합도 Train 검증에서 채택 근거가 부족했습니다.','이6개 오류는 한 사람 w2의 반복 사용한 개발 자료입니다. 독립 최종 성능으로 일반화할 수 없습니다.','',f'기존 자료 함께 사용: 815개 실제 학습 행, 56개 항목 모델 중3개 재학습/53개 동일 유지. QEVD, MultiView, R2D2, CFRep, REHAB24/26, US Navy 등의 기존 학습 행을 유지했습니다.','AI-Hub 이미지는 이 영상 복귀 학습에 새로 추가하지 않았습니다. 정지 사진만으로 완료된 복귀 실패를 확정하지 않았습니다.','',f'원본 영상의 사람별 Train/Test 배정과 Test 파일 전체 바이트 동일성을 확인했습니다. Test 학습 원본 중복0. 정상·오류를 따로 감사했으나 기존 빈 조합{empty}/80은 남아 있어 정확한80/20은 미달입니다. 사람을 통째로 정상/오류로 판단하지 않았습니다.','동일 Test: 51개 행,156개 공통 평가 항목. 이는 운동 반복 수가 아닌 평가 항목 수입니다. 이미 과거 검토된 진단 자료이며 독립 최종 Test는 열지 않았습니다.','서기 높이 복귀 Test는 정상6개/오류0개이므로 아래 스쿼트 전체 오류 검출률을 복귀 실패 검출률이라고 해석하면 안 됩니다.','','운동 | 오류검출 AI / Rule | 정상오판 AI / Rule']
for e,m in cmp['same_bank_by_exercise'].items():
 def count(m,k,n):return round(m[k]*m[n])
 a=m['ai'];r=m['rule_base'];lines.append(f"{e} | {count(a,'error_recall','error_n')}/{a['error_n']} / {count(r,'error_recall','error_n')}/{r['error_n']} | {count(a,'normal_false_alarm','normal_n')}/{a['normal_n']} / {count(r,'normal_false_alarm','normal_n')}/{r['normal_n']}")
lines+=['','회귀 검증29개 통과, 잘라낸 동작/기준 자세 영상2개에 원본 픽셀 일치6회 확인.','실서비스 모델은 교체하지 않았습니다. 이 실험은 미세 오류 개선 미달로 보관합니다.','다음 과제: 개발w2/Test와 독립인 기존 Train 영상에서 실제 약한 복귀 오류를 추가 확보하고 초기 자세 대비 관절별 복귀 특징을 검증. 독립 복귀 오류 Test/0점 원본 부족도 해소해야 합니다. Test를 Train으로 옮기거나 정답을 바꿔 수치를 높이지 않습니다.']
(O/'RESULTS.txt').write_text('\n'.join(lines)+'\n');(O/'NEXT_SESSION.txt').write_text('실험은 완료했으나 사용자 목표 미달. RESULTS/verification 참조. 선택 모델 오류1/6 그대로. 더 나은 개발 후보를 Test로 선택하지 말 것. 정상 라벨1개 수정은 유효하고 클립 data/mild-return-expansion-20261024에 있음. 0점 미확보, 빈 분할37개 유지. 독립 기존 Train 미세 실패 추가 검토가 다음 작업.\n')
print(json.dumps(verification,ensure_ascii=False,indent=2))

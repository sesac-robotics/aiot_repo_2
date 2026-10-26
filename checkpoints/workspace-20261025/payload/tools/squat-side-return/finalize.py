import collections,json,re,sys
from pathlib import Path
import numpy as np,joblib
B=Path(__file__).resolve().parents[2];O=B/'output/squat-side-return-enriched-v2-20261024';OLD=B/'output/return-failure-scores-20261024';SIDE=B/'output/squat-side-return-v5-20261024';MINING=B/'output/squat-side-return-v3-20261024/mining'
sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/paired-evaluation')]
from common import sha
from split_guard import assert_training_allowed,POLICY
from artifact_integrity import equal_state
read=lambda p:json.loads(p.read_text());rows=read(O/'score-inputs.frozen.json');test=read(O/'test-inputs.frozen.json');model=joblib.load(O/'score-candidates.joblib');old=joblib.load(OLD/'score-candidates.joblib');report=read(O/'score-training.json');byid={r['id']:r for r in rows};cmp=read(O/'comparison.frozen.json');side=read(SIDE/'feature-provenance.json');supp=read(O/'supplemental-side-Test-results.frozen.json');labels=read(MINING/'labels-v2.frozen.json');clips=read(B/'data/squat-side-return-expansion-20261024/clip-manifest.json')['clips']
assert report['completed'] and len(rows)==974 and len(test)==51 and len(side)==35 and all(r['recovered'] for r in side) and supp['recovered']==3
assert sha(POLICY)=='7aded6fa543285e7e8926b690317c2cc53e1bc96c7f9ca1c181f3077be23f809'
assert sha(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json')=='fa4e5913e3b48790562702e3ff8751aee159f1435ae4b6e633b3e7a27e678667'
assert sha(O/'test-inputs.frozen.json')==sha(OLD/'test-inputs.frozen.json')
assert not {s for r in rows for s in r['sources']}&{s for r in test for s in r['sources']}
trained={i for h in model['heads'].values() for i in h['trained_ids']};assert_training_allowed([byid[i] for i in trained]);newids={r['id'] for r in labels['reviewed']};assert newids<=trained
changed={('squats','standing_height_return','height_v4'),('squats','descent_range','front')}
assert all(equal_state(h,old['heads'][k]) for k,h in model['heads'].items() if k not in changed)
parent=read(OLD/'score-inputs.frozen.json');assert len(parent)==970
for r,p in zip(rows,parent):
    assert r['id']==p['id'] and r['split']==p['split'] and r['sources']==p['sources'] and r['person']==p['person']
    for n,g,oldn,oldg in zip(r['items'],r['levels'],p['items'],p['levels']):
        if r['id'] in {a['id'] for a in side} and n['item']=='standing_height_return':assert g==2
        else:assert n==oldn and g==oldg
for a in side:
    cp=B/'output/integrated-five-20261024/team-observations'/a['source']/'side.npz';z=np.load(cp);meta=read(cp.with_suffix('.json'));assert str(z['fingerprint'])==meta['fingerprint'] and meta['config']['source_sha256'] in byid[a['id']]['sources']
assert len(clips)==4 and all(c['id'] in trained and c['split']=='train' and sha(c['path'])==c['sha256'] for c in clips)
assert sum(len(c['pixel_checks']) for c in clips)==12 and all(k['exact_decoded_pixels_equal'] for c in clips for k in c['pixel_checks'])
assert all(r['target']==2 and r['ai_prediction']==2 for r in supp['records'])
assert cmp['same_bank_paired_items']==153 and cmp['existing_compatible_rule_paired_items']==134
for m in cmp['same_bank_by_exercise'].values():assert m['ai']['n']==m['rule_base']['n']
tests={}
for name,n in [('side',13),('contracts',16),('paired',9)]:
    t=(O/f'tests-{name}.log').read_text();assert re.search(rf'Ran {n} tests\b',t) and '\nOK' in t;tests[name]=n
screen=[]
for r in read(MINING/'intake.frozen.json'):
    reason={'S01':'shallow descent, return observed','S02':'return failure not established; do not label whole video normal','S03':'lower body cropped; descent outcome not established','S04':'returns upright; no unambiguous return failure','S05':'shallow descent, return observed','S06':'front projection/crop insufficient to assign new depth/return error'}[r['review_id']]
    screen.append(dict(id=r['id'],review_id=r['review_id'],person=r['person'],split=r['split'],source_sha256=r['source_sha256'],reviewed=True,selection_not_a_label=True,whole_video_grade=None,new_complete_segments=sum(d['review_id']==r['review_id'] for d in labels['reviewed']),new_return_failure_grade=None,note=reason))
(O/'mining-review-decisions.frozen.json').write_text(json.dumps(dict(records=screen,new_descent_error_segments=4,new_return_failure_segments=0,independent_people=2,expert_verified=False),indent=2))
head=next(h for h in report['heads'] if h['key']==['squats','standing_height_return','height_v4']);descent=next(h for h in report['heads'] if h['key']==['squats','descent_range','front'])
verification=dict(status='side_return_observation_recovered_and_draft_error_labels_trained',all_five_reliable=False,production_replacement=False,old_side_held_items=38,recovered_Train_side_items=35,recovered_supplemental_diagnostic_Test_side_items=3,original_fixed_Test_modified=False,original_Test_is_historically_exposed=True,final_independent_Test_opened=False,source_overlap=0,team_source_assignments_unchanged=True,original_team_labels_unchanged=True,exact_person_exercise_normal_error_80_20_achieved=False,missing_split_cells_unchanged=37,new_shallow_descent_labels=4,new_return_failure_labels=0,new_independent_people=2,training_candidate_rows=len(rows),actual_unique_trained_rows=len(trained),actual_trained_families=dict(collections.Counter(byid[i]['family'] for i in trained)),refitted_heads=2,preserved_model_heads=52,total_heads=len(model['heads']),side_return_grade0_supported=False,standing_return_gate_passed=head['candidate_gate_passed'],descent_gate_passed=descent['candidate_gate_passed'],same_Test_paired_items=153,supplemental_same_input_AI_Rule_items=3,supplemental_AI_normal_correct=3,supplemental_rule_normal_correct=0,supplemental_only_normal_single_person=True,tests=tests,total_tests=sum(tests.values()),new_clips=4,exact_pixel_checks=12,live_service_replaced=False,report=str(O/'RESULTS.txt'))
(O/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2))
def rate(m):return f"오류 {round(m['error_recall']*m['error_n'])}/{m['error_n']}, 정상 오판 {round(m['normal_false_alarm']*m['normal_n'])}/{m['normal_n']}"
lines=['스쿼트 측면 복귀 미평가 개선 및 기존 자료 오류 보강', '',
'결과: 기존 측면 미평가38항목 전부 복귀 특징을 관측할 수 있게 했다. Train35개와 별도 진단 Test3개이며, 점수 정확성/0·1·2점 전 등급 검증 완료나 라이브 교체를 뜻하지 않는다.', '',
'원인과 수정',
'기존 코드가 측면 구도를 일괄 제외했다. 또한 영상 시작부터 앉아 있던 경우를 첫 기립 기준으로 삼는 문제가 있어, 이전의 실제 선 자세를 원본 이미지에서 확인한 뒤 해당 반복 전에 관찰된 좌표만 기준으로 사용했다. 최초 기준을 미래 복귀에서 역산하지 않았다.',
'측면 스쿼트에서 자연스러운 골반의 수평 이동을 카메라 이동과 같게 처리하던 조건을 고쳤다. 발이 관찰되면 위치 안정성을 검사하고, 발이 안 보이면 복귀 시점의 골반 위치 제한을 유지한다. 골반·어깨의 세로 높이는 중심화하거나 지우지 않는다. 필요한 몸통 관절이 누락되면 채우지 않고 보류한다.',
'V14의 일부 구간은 골반이 거의 낮아지지 않는 몸통 숙이기였다. 올바른 스쿼트 하강이 있어야만 복귀도 관찰할 수 있게 묶여 있던 조건을 풀어, 이 경우 관찰된 어깨 움직임으로 복귀 시점을 찾는다. 하강 오류와 복귀 높이를 별개로 평가하며 상태를 확인했다는 이유로 2점을 고정하지 않는다.', '',
'오류 라벨과 실제 학습',
'기존 QEVD 자료6원본을 약4fps로 재검토하고, 2원본은 약7.5fps 시간순 장면과 원본 크기로 다시 확인했다. 2명에게서 얕은 하강4구간을 추가했다. 정면 대리 관찰 하강1점/서기 높이 복귀2점이며, 나머지 항목은 null이다. 전문가 확정 라벨은 아니다. 이4개는 복귀 실패가 아니라 하강 오류다. 새 복귀 실패0·1점 라벨은 이번에 추가하지 못했다.',
'새4구간을 data/squat-side-return-expansion-20261024/clips/squats에 무손실 분리했다. 처음/중간/마지막 원본 픽셀 일치12건을 확인했다. 원본 해시·프레임 대응·사람·Train 배정을 함께 보존했다.',
f'기존970행을 유지해 총974행, 실제 학습 고유구간{len(trained)}개다. 기존 팀·QEVD·REHAB·미국 해군·MultiView·CFRep 자료와 AI-Hub 전이 특징을 유지했다. 전체54개 모델 중 스쿼트 하강/높이 복귀2개만 재학습했고52개는 실제 추정기 상태가 같음을 검증했다. 포즈 신경망 자체의 전체 재학습은 아니다.',
f"복귀 학습은 {head['train_count']}구간(정상{head['grade_counts']['2']}, 부분 복귀{head['grade_counts']['1']})이다. Train 그룹 교차검증 오류5/7, 정상 오판6/171이지만 별도 개발 오류는1/6 검출에 머물러 채택하지 않았다. 추가 측면35개는 모두 기존 복귀2점 정답이므로 측면 실패 구별을 검증했다고 주장하지 않는다.",
'하강 모델도 정상 오판6/29와 개발 정상 표본1개로 채택 기준을 통과하지 못했다. Test 결과를 보고 모델이나 규칙 경계를 다시 맞추지 않았다.', '',
'같은 기존 진단 Test51구간에서 AI/Train 보정 실험 Rule 비교(153개 공통 항목)']
for ex,m in cmp['same_bank_by_exercise'].items():lines.append(ex+' | AI '+rate(m['ai'])+' | 보정 Rule '+rate(m['rule_base']))
lines+=['','기존 Test 입력은 이전 실행과 바이트 단위로 동일하게 유지했다. 위 분모는 독립 사람 수가 아닌 구간별 평가 항목 수다. 보정 Rule은 실제 배포 Rule과 다르며 AI의 예측을 규칙 입력으로 사용하지 않는다.',
'측면 Test V15의 기존 미평가3개는 원래153개 분모에 섞지 않고 모델·규칙 동결 뒤 별도 진단했다. 첫 구간의 초기 기립0~0.22초는 기준 수립에만 쓰고 그 뒤부터 복귀 특징을 계산했다. 3개 모두 추출 가능해졌고 복귀 정답은2점, AI는2점3/3, 보정 Rule은1점3/3으로 정상 오판했다. 이3개는 한 사람의 정상 복귀만 있으므로 실패 구별/0·1점 성능 증거가 아니다. 규칙은 이 Test 결과로 수정하지 않았다.', '',
'분할과 검증',
'P01~P04의 기존 사람/운동/정상·오류별 원본 배정을 변경하지 않았다. Train/Test 원본 중복0, Test 학습 사용0. 정확한80/20과 빈칸37개 문제는 아직 남아 있으며, 기존 Test를 옮겨 맞추지 않았다. 별도의 미사용 최종 Test는 열지 않았다.',
'측면 복귀13개, 분할·학습 계약16개, 같은 입력 비교9개 총38개 검증 통과. 새 클립 픽셀12개와 모델52개 보존도 별도로 확인했다.', '',
'남은 작업: 실제 측면/정면 복귀 실패0·1점과 독립 개발 오류 그룹을 추가해야 한다. 개발 검증 기준을 통과한 후 실시간 기준 수립과 UI까지 연결해야 한다. 이번 변경은 실험 점수 파이프라인 개선이며 배포 서비스는 교체하지 않았다.']
(O/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
(O/'NEXT_SESSION.txt').write_text('최신 실험 '+str(O)+'\nRESULTS.txt/verification.json 참조. 측면35Train+3진단Test 모든 특징 복구, 얕은하강4실제학습(복귀오류0). 54heads중2refit52보존. 개발복귀오류1/6으로미채택, live미교체. 기존Test입력동일,추가측면Test3은별도정상전용진단. 새Test결과로모델/규칙조정금지. 실제복귀0/1실패와독립개발사람필요. 새클립 data/squat-side-return-expansion-20261024.\n')
pointer=read(B/'output/CURRENT_FIVE_EXERCISE_RUN.json');pointer.update(run=str(O),parent_run=str(OLD),status=verification['status'],all_five_reliable=False,production_replacement=False,report=str(O/'RESULTS.txt'));(B/'output/CURRENT_FIVE_EXERCISE_RUN.json').write_text(json.dumps(pointer,ensure_ascii=False,indent=2))
print(json.dumps(verification,ensure_ascii=False))

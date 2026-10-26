"""Save reviewed screening decisions and check the immutable parent checkpoint."""
import collections, datetime, json, sys
from pathlib import Path
B=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import POLICY, assert_training_allowed
P=B/'output/squat-fixed-reference-combined-v3-20261024'
O=B/'output/squat-remaining-diagnostic-v4-20261024'
read=lambda p:json.loads(p.read_text())
def save(n,x):
 p=O/n;assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def main():
 protocol=read(O/'protocol.frozen.json');assert protocol['parent_model_sha256']==sha(P/'new-head.joblib')
 for name,digest in read(P/'artifact-hashes.frozen.json').items():assert sha(P/name)==digest,name
 assert sha(POLICY)=='7aded6fa543285e7e8926b690317c2cc53e1bc96c7f9ca1c181f3077be23f809'
 labels=B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json'
 assert sha(labels)=='fa4e5913e3b48790562702e3ff8751aee159f1435ae4b6e633b3e7a27e678667'
 rows=read(O/'new-Train-review/intake.frozen.json')+read(O/'additional-Train-review/intake.frozen.json')
 assert_training_allowed(rows)
 test=read(P/'test-inputs.frozen.json');test_sources={s for r in test for s in r['sources']}
 assert not {r['source_sha256'] for r in rows}&test_sources
 reasons={
 'N01':'초기 기립 후 두 차례 상하 운동과 상단 복귀가 보이나 하체 일부가 화면 아래에서 가려져 정밀 높이 정답 추가 보류. 실패 확정 없음.',
 'N02':'대기 후 두 차례 하강·상승과 상단 복귀가 보임. starting late 제공자 태그는 복귀 실패가 아니며 새 실패 없음.',
 'N03':'초기/복귀에서 머리와 발이 잘리고 마지막은 서 있는 상태. 신장 높이 부족과 화면 잘림을 구분할 근거 부족. 실패 보류.',
 'N04':'두 차례 하강 후 초기 상단 높이로 돌아오는 표본이 보임. 이후 세 번째 하강에서 영상 끝. 잘린 마지막 하강은 복귀 실패 아님.',
 'F01':'insufficient 태그지만 표본은 작은 범위의 무릎 굽힘 후 초기 상단으로 복귀. 운동 깊이 오류를 서기 높이 실패로 복사하지 않음.',
 'F02':'상승 상단 표본은 초기 상단과 비슷하며 하강 중 몸통 전경이 뚜렷함. 몸통 태그와 복귀 실패 구분, 새 실패 확정 없음.',
 'F03':'첫 상승 상단 후 두 번째 하강에서 종료. 발/발목이 영상 아래에서 잘려 지지점 기준의 미세 높이 부족을 확정하지 않음.',
 'F04':'두 차례 상단 복귀 후 세 번째 하강 중 영상 종료. 허리 굽힘/얕은 깊이 태그만으로 복귀 실패를 부여하지 않음.'}
 decisions=[]
 for r in rows:
  assert sha(r['path'])==r['source_sha256']
  sub='new-Train-review' if r['review_id'].startswith('N') else 'additional-Train-review'
  sheet=O/sub/(r['review_id']+'.jpg')
  decisions.append(dict(review_id=r['review_id'],person=r['person'],source_path=r['path'],
      source_sha256=r['source_sha256'],split='train',screening_hz=5,
      sheet=str(sheet),sheet_sha256=sha(sheet),decision='no_confirmed_return_failure',
      severity_grade=None,fitted=False,reason_ko=reasons[r['review_id']],
      reviewer='AI_assisted_visual_screening',expert_verified=False,
      limits='5Hz 선별 표본 검토. 모든 프레임을 전문가가 검증한 정답이 아님. 정상 점수 라벨도 본 선별만으로 확정하지 않음.'))
 save('review-decisions.frozen.json',dict(reviewed_sources=8,confirmed_new_failure_labels=0,
      confirmed_new_normal_labels=0,all_new_sources_not_fitted=True,decisions=decisions))
 # Recount the canonical quality cells; never redistribute the samples.
 policy=read(POLICY);cells=collections.Counter()
 for r in read(labels)['labels']:
  grades=[i['grade'] if i['training_mask'] else None for i in r['items']]
  q='error' if any(g is not None and g<2 for g in grades) else 'normal' if len(grades)==4 and all(g==2 for g in grades) else 'unknown'
  if not r['quality_training_candidate']:q='unknown'
  cells[r['person'],r['exercise'],q,policy['assignments'][r['source_id']]]+=1
 audit=read(P/'person-quality-split-audit.json')
 assert all(cells[a['person'],a['exercise'],a['quality'],a['split']]==a['count'] for a in audit['cells'])
 assert audit['empty_cells']==37
 d=read(O/'fixed-model-diagnostic.json');miss=d['missed']
 conflict=next(r for r in d['samples'] if r['id'].endswith('diag_w2_5_7_r12'))
 assert miss['prediction']==2 and miss['target']==1 and conflict['target']==2
 assert conflict['normal_probability']<miss['normal_probability']
 cmp=read(P/'comparison.frozen.json');paired=cmp['paired_predictions'];assert len(paired)==163
 assert len({(r['id'],r['criterion']) for r in paired})==163
 save('verification.json',dict(parent_artifact_manifest_verified=True,parent_model_unchanged=True,
     parent_labels_and_split_unchanged=True,canonical_person_quality_counts_recomputed=True,empty_cells=37,
     exact80_20_achieved=False,new_Train_sources_screened=8,new_failure_labels=0,new_training=0,
     Test_new_sources_overlap=0,AI_and_Rule_same_Test_items=163,AI_and_Rule_model_or_rule_changes=False,
     diagnostic_error_probability_ordering_requires_at_least_one_normal_false_alarm_to_detect_all6=True,
     development_error_detected=5,development_error_n=6,development_normal_false_alarms=0,
     development_normal_n=13,independent_final_Test_opened=False,production_replacement=False))
 text='''스쿼트 잔여 미검출 진단 / 2026-10-24

기존 개선 후보 유지
고정된 초기 다리 길이와 복귀 자세 문맥을 사용한 후보는 같은 노출 개발19개에서 AI 오류5/6, 정상 오판0/13이다. Rule Base는1/6,0/13이다. 기존 높이 정규화 상쇄 문제를 고친 후보이며 이번 추가 진단에서는 모델·문턱·정답을 변경하지 않았다.

남은 r09 원인
고정 모델의 r09 정상 확률은0.762297, Train에서 선택한 정상 문턱은0.658821이다. 가장 가까운 정상 특징은 r05다. 골반 높이 부족은 r09 평균0.0212/최대0.0390으로 정상 r05의0.0265/0.0438보다 작고, 어깨 높이도0.0596/0.0667 대0.0604/0.0653으로 겹친다. 수치는 초기 다리 길이로 나눈 값이며 임상 허용 기준이 아니다.
원본 상단 매 프레임 표본과 픽셀 확대 비교에서는 r09의 몸통 전경이 남지만 단일 높이/추정 관절 통계가 정상 r05와 가까운 상태다. 측정 특징과 자세 정답의 구분 능력에 한계가 있다. 영상 검토는 AI 보조이며 전문가 정답 검증은 아니다. 기존 라벨을 판정에 맞춰 바꾸지 않았다.
정상 r12의 정상 확률0.690942가 r09보다 낮다. 따라서 이 모델의 문턱만 올려6개 오류를 모두 잡으면 정상 r12도 오류가 된다. 개발 정답으로 문턱을 다시 선택하지 않았다. feature별 성분 logit 차이도 저장했으며 앙상블 확률에 대한 인과적 기여도로 과장하지 않는다.

실제 오류 자료 보강 시도
새 QEVD Train 원본8개/8명을 역할 동결 후5Hz 시간순 표본으로 추가 검토했다. 초기 기립/상단 복귀, 깊이 부족, 몸통 전경, 화면 아래 발 잘림, 마지막 다음 하강 종료를 구분했다. 이번 선별에서 확정한 실제 새 복귀 실패는0개다. 정상·오류 점수 정답을 새로 학습한 것도0개다. 이8개를 학습에 넣지 않았으며 제공자 insufficient/back-not-straight/starting-late 태그를 복귀 실패로 복사하지 않았다.

분할·동일 Test
부모 실험의 전체 산출물 해시, 팀 라벨 해시, 분할 정책 해시를 재검증했다. 사람·운동·정상/오류·Train/Test 수를 다시 집계했고 기존80칸 중37빈칸과 배정은 그대로다. 정확80/20 달성을 선언하지 않는다. 새 검토8원본과 Test 중복0. 독립 최종 Test는 열지 않았다.
부모의 동일 Test51행/163개 AI·Rule 대응 항목과 고정 결과를 유지했다. 모델 변화가 없어 Test 재선택/재학습을 하지 않았다. 스쿼트 복귀 Test는 정상6/오류0이므로 독립 오류 검출률은 평가할 수 없다. AI 정상오판0/6, Rule2/6이다. 최신 후보는 미배포이며5종 신뢰성 목표는 미달이다.

다음 작업
다른 사람의 초기 기립과 발 지지점이 관측되는 완결된 미세 복귀 실패, 몸통만 숙인 정상 복귀의 구별 정답을 확보해야 한다. 상단 연속 구간의 관절 추정 오차·골반/어깨의 실제 변화와 전문가 라벨 합의를 확인한 뒤 Train만으로 특징·모델을 보강하고 동일 진단 Test와 별도 독립 평가를 수행한다.
근거: fixed-model-diagnostic.json, visual-review-intake.frozen.json, review-decisions.frozen.json, verification.json. 성능 근거는 부모 squat-fixed-reference-combined-v3-20261024/RESULTS.txt.
'''
 p=O/'RESULTS.txt';assert not p.exists();p.write_text(text)
 save('artifact-hashes.frozen.json',{str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()})
 status=B/'project-coordination/status/model-evaluation.json';s=read(status)
 s['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 s['summary']='스쿼트 후보개발5/6 FP0/13유지. r09 특징이정상r05와겹치며 문턱만올리면정상r12오판. 추가Train8원본선별실패확정0. 동일Test163/기존분할보존/미배포.'
 s['evidence'] += [str((O/n).relative_to(B)) for n in ['RESULTS.txt','verification.json','fixed-model-diagnostic.json','review-decisions.frozen.json']]
 s['current_work']['additional_diagnostic']=str(O.relative_to(B))
 s['current_work']['additional_Train_sources_screened']=8
 s['current_work']['additional_confirmed_failure_labels']=0
 s['ytw_review'].update(state='person_excluded_development_reviewed_activation_held',
     followup_run='output/ytw-work-20261024/logo-v1',functional_tests=25,
     limitation='기존v7설정전체4명선택/정밀경계없음/자연비완료1개/횟수일치정확도아님/미배포',
     followup='YTW담당이사용자추가지시로boundary-v2기존Train영상정밀경계검토진행')
 status.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(read(O/'verification.json'),indent=2))
if __name__=='__main__':main()

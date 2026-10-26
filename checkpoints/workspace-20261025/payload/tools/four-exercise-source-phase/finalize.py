from pathlib import Path
import json,hashlib,collections,datetime,subprocess,csv,re
B=Path('/home/jayden/SeSACPAI-Mid');P=B/'output/four-exercise-source-phase-v2-20261025';O=P/'clock-posture-corrected-v4';D=B/'project-documents/project-handbook';A=D/'data-audit-20261025'
rows=json.loads((O/'score-inputs.frozen.json').read_text());report=json.loads((O/'score-training.json').read_text());folds=json.loads((O/'folds.frozen.json').read_text());baseline=json.loads((O/'paired-baseline-used-v2.frozen.json').read_text());prev=json.loads((B/'output/four-exercise-review-train-v1-20261025/reconciled-v2/score-training.json').read_text());links={r['id']:r for r in json.loads((P/'all-resolved-inputs.frozen.json').read_text())};labels=json.loads((P/'labels-audit.frozen.json').read_text());reused=json.loads((P/'reused-shoulder-item-windows.frozen.json').read_text())
def support(rs):return {str(g):dict(rows=sum(y==g for id,y in rs),groups=len({folds['group_by_id'][id] for id,y in rs if y==g})) for g in [0,1,2]}
# Baseline bookkeeping follows the same visually verified posture bank transfer.
for h in report['heads']:
 if not h['bank'].startswith('four_joint_body_v2/'):continue
 eligible=[(r['id'],y) for r in baseline for it,y in zip(r['items'],r['levels']) if (it['item'],it['bank'])==(h['item'],h['bank']) and y is not None and it['eligible'] and it.get('feature') is not None]
 for a in h['algorithms']:
  for f in a['folds']:
   tr=[p for p in eligible if folds['fold_by_id'][p[0]]!=f['fold']];s=support(tr);tr=[p for p in tr if s[str(p[1])]['groups']>=2];f['baseline']['grade_support']=support(tr)
(O/'score-training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
native=json.loads((B/'output/four-exercise-review-train-v1-20261025/data-utilisation-receipt.json').read_text());usage={};all_used={i for s in report['summary'].values() for i in s['actual_fit_ids']}
with (O/'input-use-register.jsonl').open('w') as f:
 for r in rows:
  models=['|'.join((h['exercise'],h['item'],h['bank'],a['name'])) for h in report['heads'] if h['exercise']==r['exercise'] for a in h['algorithms'] if r['id'] in a['actual_fit_ids']]
  value=dict(id=r['id'],exercise=r['exercise'],person=r['person'],split=r['split'],sources=r['sources'],source_observation=links.get(r['id']),actual_fit=bool(models),actual_models=models,items=[dict(item=it['item'],bank=it['bank'],grade=y,eligible=it['eligible'],holds=it.get('holds',[])) for it,y in zip(r['items'],r['levels'])]);f.write(json.dumps(value,ensure_ascii=False)+'\n')
for ex,s in report['summary'].items():
 rr=[r for r in rows if r['exercise']==ex];fit=[r for r in rr if r['id'] in all_used];old=set(prev['summary'][ex]['actual_fit_ids']);new=set(s['actual_fit_ids']);n=native[ex]
 for item in n.get('aihub_native_models_referenced',[])+n.get('qevd_native_models_referenced',[]):assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
 usage[ex]=dict(existing_interval_candidates=sum(r['id'] in links for r in rr),reused_precise_item_window_records=sum(r['id'] not in links for r in rr),all_source_links_restored=True,additional_native_interval_reviews=sum(r['exercise']==ex for r in labels),actual_fit_unique_existing_intervals=len(fit),actual_fit_people=len({r['person'] for r in fit}),actual_fit_source_hashes=len({h for r in fit for h in r['sources']}),trained_head_algorithm_count=s['trained_head_algorithms'],added_fit_ids=sorted(new-old),removed_fit_ids=sorted(old-new),all_actual_fit_ids=sorted(new),all_unfitted_ids=s['unfitted_ids'],native_models_reused=n.get('aihub_native_models_referenced',[])+n.get('qevd_native_models_referenced',[]),native_models_refitted=False,native_provider_targets_not_project_grades=True)
(O/'data-utilisation-receipt.json').write_text(json.dumps(usage,ensure_ascii=False,indent=2)+'\n')
with (O/'all-head-comparison.tsv').open('w') as f:
 w=csv.writer(f,delimiter='\t');fields=['labelled_targets','evaluated','abstained','coverage','errors_detected','observed_errors','all_labelled_errors','normal_false_alarms','observed_normals','exact_grade_correct','exact_grade_total'];w.writerow(['exercise','item','bank','algorithm','version']+fields+['development_gate_passed','gate_failures'])
 for h in report['heads']:
  for a in h['algorithms']:
   for role,k in [('revised','candidate_cv'),('baseline','baseline_cv_same_targets')]:w.writerow([h['exercise'],h['item'],h['bank'],a['name'],role]+[a[k][z] for z in fields]+[a['development_gate_passed'],';'.join(a['gate_failures'])])
summary=f'''4종 기존 Train 원본·구간·관절·학습 후속 실행 — 2026-10-25
최종 연구 후보: {O}
이 결과는 노출된 Train 내부 4분할 진단이다. 독립 최종 Test나 배포 성능이 아니다.

자료 연결과 검토
기존 Train609구간 전체의 원본·관절 캐시·실제 프레임·구간을 연결했다.
이전 미연결74개 = 별칭/주석 연결55개 + 새 관절 추출19개. 파생 영상9개는 원본 해시와 시작/중간/끝 픽셀27건을 확인했다.
추가 원본 검토74구간(숄더1/푸시업36/YTW26/플랭크11), 의심8구간은 모든 캐시 관측을 확인했다. 모든 영상 프레임을 검토한 것은 아니다.
숄더42개 추출 원본은 앉음/섬 변형을 원본 표본으로 별도 확인했다. 운동 심각도를 이 표본에서 일괄 부여하지 않았다.
5구간의12개 항목·특징 조합 등급 변경,12구간의 관련 관절/구도 입력 보류. Y의 굽힌 대각선3구간은 정상2→부분1, 잘린 푸시업 복귀1구간은2→null, 숄더 운동 몸통 관찰1구간은null→2다.
기존 정확한 숄더 항목 창21개도 재사용 후보에 연결했다. 겹친 구간을 독립 사람/원본 수로 추가하지 않았다.

실제 사용
기존609구간 + 재사용 항목 창21행 = 후보630행. 실제 분류 모델 학습은 기존532개 고유 구간이다.
숄더103/푸시업154/YTW156/플랭크119. 이전531개에서 적격8개를 복원하고 오연결·구도·잘림7개를 제외했다.
52개 항목·특징 은행×고정된2학습법=104개 연구 분류 모델을 학습했다.83개 은행 전체를 검증했고 지원 불가 은행도 남겼다.
같은 사람·기존 그룹·원본 SHA가 연결된 모든 구간과 동기화 구도는 같은 fold다. 모델은 등급별 사람/원본 연결그룹2개 이상일 때만 해당 등급을 학습했다.
새로운 특징: 실제 하강 후 상단 복귀47구간, Y/T/W 연속5관측 창(단계별 적격120/135/125), 숄더4관절 몸통61구간 관측+기존 정확 창12개. 항목 간 중복 수를 더해 독립 구간으로 세지 않는다.
플랭크 새 추출11구간은 기존 측면 지지·정렬 계약을 같은 단위로 재계산했다. 짧은0.58초 구간에3초 지속 정상 라벨을 만들지 않았다.
AI-Hub·QEVD 기존 보조 모델22개는 해시를 대조해 재사용 근거로 연결했다. 이번에 다시 학습한 것이 아니며 제공자 조건을 세부0/1/2 정답으로 복사하지 않았다.

구현 중 발견한 오류와 수정
첫 연구 버전의 몸통 연결은15fps 원본 시각을10Hz로 반올림하면서 관측 순번이 중복되는 문제가 있었다. 실제 추출 일정의 원본 프레임↔관측 순번으로 고쳐21구간 관측 보류를 복원했다.
최종 v4는42원본의 자세 변형도 직접 대조했다. 이 수정은 아직 분류 학습이 승인되지 않은 몸통 입력에만 영향을 준다. 이미 학습한104개 모델의 학습 특징·등급이 정확히 같음을 검사하고 모델을 재사용했다. 수정 때104개를 새로 학습했다고 중복 집계하지 않는다.
최초 비교 실행은 단위가 다른 기존 푸시업 특징을 같은 baseline에 넣으려다 중단됐다. 최종 baseline은8차원 각도/관측률만 사용하고, YTW는17차원 기존 compact 계약만 사용한다. 실패 실행 기록도 보존했다.

검증 결과와 채택 판단
새 특징의 모든 은행은 개발 채택 기준을 통과하지 못했다. 전체4종 성능 향상이나 운영 완료를 주장하지 않는다.
푸시업 복귀: ExtraTrees 오류5/5검출·정상 오탐6/42, 전체 라벨181중47평가(26.0%). 공통 관측44구간에서는 새 오류5/5·오탐6/39, 기존4/5·오탐4/39. 복귀 측정 개선과 오탐 증가·관측 보류를 함께 기록한다.
YTW 전경 W logistic: 공통85구간에서 새 오류34/55·오탐15/30, 기존30/55·오탐17/30. 일부 개선은 있지만 운영 기준 미달. 다른 항목에는 악화도 있으며 전체 비교표에 모두 기록했다.
YTW 직립 T logistic: 오류4/5·정상 오탐0/18이지만34라벨 중23평가라서 관측/학습 지원 기준 미달. 심각도 정확 일치는18/23이며 오류검출률과 구분한다.
숄더 정확한 정면·앉은 하단 창9개는 정상만, 앉은4관절 몸통 오류4행은 겹친 창을 포함해 동일1사람 연결그룹이다. 이 두 항목은 다등급 분류 학습/운영 지원 보류다. 전체 보유 자료 부족이라는 뜻이 아니다.
기존 AI-Hub 전이 푸시업 몸통 logistic 은행1개만 개발 수치 기준을 충족했지만 새 특징 개선의 통과로 세지 않았다. 독립 최종 Test/라이브 채택은 하지 않았다.
미평가를0점 또는 정상2로 바꾸지 않았다. 심각도0의 지원을 정상·부분 학습에서 추정하지 않았다.

다음 적용 범위
숄더: 기존Train의 아직 항목별로 확정하지 않은 실제 복귀/몸통 이탈 구간을 우선하고 앉음/섬·양팔 프로토콜별로 구분한다. 한 사람의 겹친 오류 창을 늘려 그룹 수를 채우지 않는다.
푸시업:181라벨의 관측 보류134개를 완결 구간/말단 가림/관절 오연결/작은 움직임으로 나눈 뒤 원본에서 닫힌 반복 경계를 확인한다. 종료를 확인하지 못한 복귀를 실패로 가르치지 않는다.
YTW: 전경형 정상의 투영 차이와 단계별 정상 오탐을 원본에 대조한다. 기존 제공자 오류 설명·Y-only 보조 과제를 후보 탐색에 사용하되 전체YTW 등급으로 복사하지 않는다.
플랭크: 교정된 바닥 추가 접촉·전환·정상 대조 구간을 유지하고 항목별 오판과 전이 특징을 비교한다. 같은 원본의 잘못된 초반 라벨을 후반 정상 구간에 복사하지 않는다.
동일 자료/동일 특징을 이유 없이 반복 재학습하지 않는다. 현재 미검토·미연결 보유 목록은 기존 프로젝트 기준서에 연결되어 있다.

보존과 문서
기존 스쿼트/라이브 보호1087파일, 부모 입력·모델·공통CURRENT·동결분할 해시와Git 상태를 별도 검증한다.
31개 최종 검사(원본/관측9+최종학습/시간/변형9+숄더회귀13)를 통과했다. 이는 자세 정확도31개 사례 검증이 아니다.
기존 프로젝트 기준서 관련 절을 갱신하고Git에는 추가하지 않는다. Confluence 게시·카메라·라이브 교체·전원 종료는 하지 않았다.
'''
(O/'RESULTS.txt').write_text(summary)
names={'shoulder':'숄더 프레스','pushups':'푸시업','ytw':'YTW','plank':'플랭크'}
detail={
 'shoulder':'하단의 정확한 정면·앉은 창9개는 정상만이며, 앉은 몸통 오류4행은 겹친 구간을 포함해 동일1사람 그룹이다. 42개 추출 원본의 앉음/섬을 원본 표본으로 확인했다. 몸통 시간 변환을 실제 추출 순번으로 고쳐21구간 보류를 복원했지만 다등급 모델 지원은 입증하지 못했다.',
 'pushups':'새 실제 복귀 특징은47구간이다. ExtraTrees는 평가된 오류5/5를 검출했으나 정상 오탐6/42, 전체181라벨 중47평가에 그쳐 운영 채택 보류다. 공통44구간 비교에서 오탐은 기존4/39→새6/39이므로 전체 성능 개선이라고 쓰지 않는다. 손목 잘림·얼굴/어깨 오연결·상승 중 종료를 분리했다.',
 'ytw':'Y/T/W 각각 연속5관측 창의 적격 입력은120/135/125개다. 전경/직립을 구분했고 Y의 굽힌 대각선3구간을2→1로 수정했다. 전경W logistic 공통85구간의 오류검출30/55→34/55·오탐17/30→15/30이지만 여전히 기준 미달이다. 직립T의 일부 개선도 관측 보류를 포함하면 통과하지 못한다.',
 'plank':'새로 추출한11구간의 정렬·팔꿈치·무릎/골반 지지를 기존 단위로 재계산했다.0.58초 구간의 정렬 관측과3초 지속을 구분했다. 추가 접촉0과 나쁜 자세 지속2를 별도 유지하며 전체 채택 개선으로 주장하지 않는다.'}
common=f'추가 검토74구간·5구간12항목/특징 라벨 변경·12구간 관련 입력 보류를 마쳤다. 원본/관절/구간 연결은 기존 Train609개 전체다. 최종 연구 후보는 [{O.name}]({O}/RESULTS.txt)이며 공통 CURRENT·운영 모델과 구분한다.'
for ex,label in names.items():
 p=A/ex/'README.md';s=p.read_text();s=s.replace('이번 작업은 신규 라벨링이나 모델 변경을 포함하지 않는다.','최초 목록 감사와 후속 라벨링·학습 실행은 아래에서 구분한다.')
 u=usage[ex];section=f'''## 2026-10-25 원본 연결 복구·실제 구간 특징 후속 검증

{common}

{label}: 기존 Train {u['existing_interval_candidates']}구간, 이번 추가 원본 검토 {u['additional_native_interval_reviews']}구간, 실제 분류 학습 {u['actual_fit_unique_existing_intervals']}개 고유 구간·{u['actual_fit_people']}명·원본 해시{u['actual_fit_source_hashes']}개다. 숄더 정확 창21행은 별도 재사용 후보이며 독립 자료 수로 합산하지 않는다. 전체 실제 학습은532구간으로, 이전 대비 새로8구간을 복원하고 부적합7구간을 제외한 결과다.

{detail[ex]}

[최종 실제 학습 ID·항목별 분리 검증]({O}/score-training.json) · [구간별 사용·보류 전체 목록]({O}/input-use-register.jsonl) · [기존 보조 모델과 이번 사용량]({O}/data-utilisation-receipt.json) · [추가 라벨 근거]({P}/labels-audit.frozen.json). 보유 자료 전체를 부족하다고 결론 내리지 않으며, 위 보유 목록·미검토 목록의 다음 활용 범위를 유지한다. 독립 최종 Test와 라이브 적용은 완료하지 않았다.

'''
 anchor='## 과거 실험과 모델 근거';assert anchor in s;s=s.replace(anchor,section+anchor);p.write_text(s)
for p in [D/'README.md',A/'README.md']:
 s=p.read_text();s+='\n## 원본 연결과 실제 운동 구간을 반영한 최신 후속 실행\n\n'+common+f' 실제 학습은 숄더103·푸시업154·YTW156·플랭크119=532개 고유 구간이다.104개 연구 모델을 학습·검증했으나 새 특징은 운영 채택 기준을 통과하지 못했다. 몸통 시간/변형 수정 후 동일 학습 특징의 모델을 재사용했으며 추가104개 학습으로 세지 않았다. [입력별 사용 목록]({O}/input-use-register.jsonl)과[전체 비교표]({O}/all-head-comparison.tsv)를 확인한다. 스쿼트·운영 모델·기존 분할은 보존했다.\n';p.write_text(s)
p=A/'WORK_RULES.md';s=p.read_text();s+='\n## 이번 후속 검증에서 확정한 입력·보고 규칙\n\n15fps 원본 시각을10Hz로 단순 반올림해 관측 순번을 만들지 않는다. 실제 추출 일정의 원본 프레임과 관측 순번을 직접 대응시킨다. 앉음/섬 변형은 원본 근거로 확정하며 기본값으로 섞지 않는다. 서로 다른 차원·단위의 기존 특징을 비교 baseline에 혼합하지 않는다. 평가 비율의 분모에는 관측 보류와 학습 불가 fold를 포함한 전체 항목 라벨 수를 쓴다. 공통 관측 비교와 전체 관측률을 함께 기록하고, 오류검출과0/1/2 정확 일치를 분리한다.\n';p.write_text(s)
p=D/'four-exercise-criteria.md';s=p.read_text();s+='\n## 실제 구간·관측 적용 기록\n\n최신 후속 연구는 '+f'[최종 실행 기록]({O}/RESULTS.txt)'+'에 연결한다. 푸시업 복귀는 실제 하강 뒤 닫힌 상단을 확인하고, YTW는 단계별 연속 관측을 사용한다. 숄더 몸통은 양어깨·양골반4관절과 확인된 운동 창을 사용한다. 화면 잘림·가림·오연결은 심각도 등급과 분리하며, 일부 특징 개선을 전체4종 채택으로 해석하지 않는다.\n';p.write_text(s)
# Existing mutable document checksum index, never frozen historical inventories.
hp=A/'document-artifact-hashes.json';hashes=json.loads(hp.read_text())
for name in hashes:
 path=A/name
 if path.is_file():hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
hp.write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
protected=json.loads((P/'preserved-before.json').read_text());base=json.loads(Path(protected['protected_baseline']).read_text());fail=[]
for path,h in base['files'].items():
 f=B/path
 if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:fail.append(str(f))
for path,h in protected['source_hashes'].items():
 f=Path(path)
 if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:fail.append(str(f))
git=subprocess.check_output(['git','-C',str(B/'team-posture-scoring'),'status','--porcelain'],text=True);tag=subprocess.check_output(['git','-C',str(B/'team-posture-scoring'),'rev-parse','squat-verified-20261025^{commit}'],text=True).strip();assert not fail;assert git==protected['git_status'];assert tag=='dae3bc73dc008edfa5a11044040a8fc7ccad2047'
checks=dict(protected_live_squat_files=len(base['files']),changed_protected_files=fail,parent_model_inputs_CURRENT_split_hashes_unchanged=True,git_status_unchanged=True,squat_tag=tag,source_contract_tests=9,final_training_clock_posture_tests=9,shoulder_regression_tests=13,tests_total=31,independent_final_Test_run=False,live_changed=False,documents_outside_Git=True,confluence_published=False)
(O/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
stpath=B/'project-coordination/status/shoulder-single.json';st=json.loads(stpath.read_text());st.update(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='four_exercise_source_phase_research_cycle_complete_adoption_held',active_four_exercise_run=str(O),current_work='기존Train609원본/관절구간연결복구, 추가74구간및숄더42원본변형검토,5구간12항목정정/12구간관련입력보류,532고유구간104연구모델학습검증. 새특징운영채택미달,스쿼트/라이브보존.',next_action='기존Train미검토원본에서숄더실제복귀/몸통이탈의사람분산확인,푸시업완결경계/관측보류유형분리,YTW전경정상투영및단계오탐검토. 동일특징반복fit금지. 독립최종Test/라이브는미완료.');st['latest_source_phase_cycle']=dict(run=str(O),source_links=609,additional_visual_intervals=74,shoulder_posture_source_reviews=42,actual_fit_unique_intervals=532,models_trained=104,grade_changed_intervals=5,grade_changed_item_banks=12,input_hold_intervals=12,verification=checks);st['limitations']=[x.replace('34구간라벨정정후','이전38구간라벨정정후') for x in st.get('limitations',[])];stpath.write_text(json.dumps(st,ensure_ascii=False,indent=2)+'\n')
(P/'CURRENT_RESEARCH_CANDIDATE.json').write_text(json.dumps(dict(run=str(O),scope='research_only_exposed_Train_CV',common_CURRENT_unchanged=True,live_deployed=False),indent=2)+'\n')
artifact_hashes={str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in O.rglob('*') if p.is_file() and p.name!='artifact-hashes.json'};(O/'artifact-hashes.json').write_text(json.dumps(artifact_hashes,indent=2)+'\n');print(checks);print({k:{z:u[z] for z in ['actual_fit_unique_existing_intervals','actual_fit_people','actual_fit_source_hashes']} for k,u in usage.items()})

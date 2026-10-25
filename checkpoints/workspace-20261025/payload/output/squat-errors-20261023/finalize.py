from pathlib import Path
import json,hashlib,sys
O=Path(__file__).resolve().parent;R=O.parents[1];P=O.parent/'squat-priority-20261023'
def read(p):return json.loads(p.read_text())
def save(name,x):(O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2))
records=read(O/'selected.json')+read(O/'ranked.json')+read(O/'final-ranked.json');assert len({r['id'] for r in records})==176
specific={'00063280':'촘촘한 프레임에서 양 무릎은 대체로 발과 같은 바깥 방향. 좁은 발 간격 자체를 무릎 모임 오류로 라벨링하지 않음.','00061758':'복귀 후 기립이 확인되며 하단의 무릎 위치는 저해상도·원근 영향을 받음. 명확한 안쪽 붕괴를 확정하지 않음.','00021440':'좁은 스탠스에서 반복하지만 양 무릎과 발의 관계만으로 붕괴를 확정하지 못함. 복귀 기립 표본이 확인됨.','00187626':'전경이 크고 골반 아래가 잘려 실제 신전과 좌우 안정 오류를 확정하기 어려움.'}
decisions=[]
for r in records:
 reason=specific.get(r['id'])
 if reason is None:
  hint=r.get('ranked_by',r.get('selection_hint'))
  reason=('시작·중간 기립이 보이고 영상 끝 하강/중간 동작은 잘림과 구별되지 않음. 끝 프레임이 낮다는 이유만으로 복귀 실패를 라벨링하지 않음.' if hint=='return_signal' else '좁거나 넓은 스탠스·하강 중 투영 변화와 실제 안쪽 붕괴를 구분. 표본에서 명확한 무릎 오류를 확정하지 못함.' if hint in ['knee_signal','squats - narrow'] else '전후 숙임·얕은 하강·카메라 기울기·잘림을 현재 기준의 몸통 좌우 오류나 복귀 실패로 자동 치환하지 않음. 표본에서 새 목표 오류를 확정하지 못함.')
  if r['id']=='00159645':reason='제공자 표시에 없던 넓은 스탠스·플리에 유사 변형이 보임. 표준 양발 스쿼트 오류로 변환하지 않음.'
 decisions.append(dict(id=r['id'],person=r['person'],path=r['path'],source_sha256=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest(),review_frames=r['review_frames'],evidence=r['sheet'],sampling_review=True,continuous_review=False,expert_validated=False,decision='needs_adjudication_not_training',reason=reason,criterion_version='squat-core100-observation-v3',return_extension=None,knee_inward=None,trunk_balance=None,used_for_training=False))
save('review-decisions.json',dict(records=decisions,new_confirmed_errors=0,new_normal_labels=0,new_training_labels=0,expert_validated=False))
queue=read(P/'remaining-review.json')['remaining'];seen={r['id'] for r in decisions};assert seen<={r['id'] for r in queue};left=[r for r in queue if r['id'] not in seen];assert len(left)==3974
save('remaining-review.json',dict(previous_directly_unreviewed=4150,new_source_sample_reviews=176,still_directly_unreviewed=3974,reviewed_but_unlabelled=len(decisions),complete_relabelling=False,remaining=left,pending_adjudication_ids=sorted(seen)))
old=read(P/'verification.json');actual={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in old['original_model_sha256']};assert actual==old['original_model_sha256'];sys.path.insert(0,str(R/'output/live-webcam-20261023'));from runtime_snapshot import read_source,digest
assert digest(read_source())==old['source_digest']
protected=set(read(R/'output/five-exercise-recovery-20261023/final_test_reservation.json')['reserved_people'])|{'qevd_'+str(x) for x in read(R/'data/qevd-selected/manifests/official_test_workers.json')};assert not {r['person'] for r in records+read(O/'pose-search.json')}&protected
ai=read(O/'error-focus-comparison.json');ai['limitations']=['기존 개발 자료를 재사용한 추가 실험이며 독립 최종 검증이 아님','이번 원본 검토에서 새 학습 라벨을 확보하지 못해 새 자료로 재학습한 결과가 아님'];save('error-focus-comparison.json',ai)
save('verification.json',dict(original_models_preserved=True,original_model_sha256=actual,team_source_digest=old['source_digest'],previous_candidate_sha256=hashlib.sha256((P/'candidate.joblib').read_bytes()).hexdigest(),new_candidate_sha256=hashlib.sha256((O/'error-focus-candidate.joblib').read_bytes()).hexdigest(),new_normal_training_rows=0,new_error_training_rows=0,runtime_models_changed=False,runtime_rules_changed=False,criterion_unchanged=True,protected_people_excluded=True,automatic_pose_screened=512,sampled_source_reviews=176,unreviewed=3974,expert_validated=False,independent_final_test=False,additional_recording=False))
rows=''
for method,file in [('AI Model','error-focus-comparison.json'),('Rule Base','rule-focus-comparison.json')]:
 for h in read(O/file)['reports']:
  b=h['before'];a=h['after'];rows+=f'<tr><td>{"무릎 모임" if h["item"]=="knee_inward" else "몸통 좌우"}</td><td>{method}</td><td>{b["missed_errors"]}/{b["error_n"]} → {a["missed_errors"]}/{a["error_n"]}</td><td>{b["normal_flagged"]}/{b["normal_n"]} → {a["normal_flagged"]}/{a["normal_n"]}</td><td>기존 유지</td></tr>'
style=(P/'index.html').read_text().split('<style>')[1].split('</style>')[0]
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>스쿼트 오류 탐색 · 개선 미확인</title><style>STYLE</style><main>
<header><span class="tag">추가 촬영 없음 · 오류 중심 탐색 · 기존 모델 보존</span><h1>오류 누락 감소는 아직 확인하지 못했습니다</h1><p>정상 사례는 추가 학습하지 않았습니다.<br>오류 가중치를 높여 재학습했지만 누락이 줄지 않아 실행 버전은 유지합니다.</p></header>
<div class="cards"><div class="card"><span class="number">512</span>추가 관절 자동 선별</div><div class="card"><span class="number">176</span>원본 표본 프레임 검토</div><div class="card"><span class="number">0</span>새로 확정한 오류 라벨</div><div class="card"><span class="number">3,974</span>직접 미검토 원본</div></div>
<section><h2>오류를 놓친 수가 줄었는가</h2><p>같은 수행자 ID는 같은 그룹에 묶어 5개 그룹으로 검증했습니다. 오류 가중치 1·2·4 중 선택은 각 학습 그룹 안에서 다시 사람별로 나눈 내부 검증으로 수행했습니다. 바깥 검증 라벨로 가중치를 고르지 않았습니다.</p><table><thead><tr><th>항목</th><th>방식</th><th>오류 누락 / 오류 수 ↓</th><th>정상에서 오류 표시 / 정상 수 ↓</th><th>적용</th></tr></thead><tbody>ROWS</tbody></table><small>내부 진단 경계 1.5를 사용한 개발 지표입니다. 운동 화면의 자세 총점·실제 교정 경고 횟수·임상 진단이 아닙니다. 검증 오류 수가 매우 적습니다.</small><details><summary>비교 모델과 한계</summary><p>AI 비교 전은 직전 기하 특징 후보의 학습 방법을 같은 그룹에서 다시 학습한 결과입니다. 실행 중 모델을 직접 측정한 수치가 아닙니다. Rule Base는 기존 상대 무릎 간격·몸통 변화 특징의 임계값을 매 학습 그룹에서 다시 맞췄습니다. 기존 후보와 원본은 그대로 보관했습니다.</p><p>이전에 사용한 개발 자료를 재사용했습니다. 독립 최종 시험 성능으로 주장하지 않습니다. 수행자 ID 분리는 제공된 데이터셋 ID 기준입니다.</p></details></section>
<section><h2>새 오류 라벨을 억지로 만들지 않았습니다</h2><p>4,150개 목록에서 제공자 오류 태그를 우선한 96개와 자동 선별 후보 중 80개를 확인했습니다. 합계 176개는 시간순 표본 검토이며 연속 영상 전체 검토·전문가 확정은 아닙니다. 4개 의심 원본은 각각 24개 프레임으로 추가 확인했습니다.</p><ul><li><b>복귀:</b> 영상이 하강 중 끝나는 것을 복귀 실패로 처리하지 않았습니다.</li><li><b>무릎:</b> 좁은 스탠스·무릎 전방 이동·투영 변화만으로 무릎 안쪽 붕괴를 확정하지 않았습니다.</li><li><b>몸통:</b> 제공자 ‘등이 곧지 않음’ 표시는 전후 숙임을 포함합니다. 이를 현재의 정면 좌우 안정 오류 라벨로 바꾸지 않았습니다.</li></ul><p class="notice">새 오류 라벨 확보와 그 라벨을 추가한 재학습은 아직 이루지 못했습니다. 이번 재학습은 기존 오류 사례의 가중치를 조정한 실험입니다. 검토한 176개도 정상 학습 정답으로 채택하지 않았으며 추가 판정 대상으로 남겼습니다.</p></section>
<section><h2>누락 사례에서 확인한 병목</h2><p>무릎 AI가 계속 놓치는 2건은 Vasil_02·Vasil_08로 같은 수행자에게 집중됩니다. 기존 시각 라벨은 오류이지만 후보는 2.0으로 예측합니다. Vasil_02는 관절이 모두 관측되어도 상대 무릎 간격 특징이 감소 신호를 잘 담지 못했습니다. 단순 관절 누락이나 학습 가중치 문제만으로 설명할 수 없습니다.</p><p>몸통 AI는 Terka_03의 부분 오류 라벨을 놓칩니다. 오류 가중치를 바꿔도 사람별 검증의 오류 사례 오차가 악화해 채택하지 않았습니다. 개선을 위해서는 이런 누락 동작을 실제로 담는 관절·영상 특징과 더 다양한 수행자의 오류 근거가 필요합니다.</p><p>100점 기준(하강30·복귀20·무릎·발30·몸통20)은 변경하지 않았습니다. 기존 모델·규칙·실행 화면도 보존했습니다.</p></section>
<section><h2>검증 기록</h2><p><a href="error-focus-comparison.json">AI Model 전체 비교</a> · <a href="rule-focus-comparison.json">Rule Base 전체 비교</a> · <a href="review-decisions.json">176개 검토 사유</a> · <a href="remaining-review.json">미검토·추가 판정 목록</a> · <a href="verification.json">원본 보존 기록</a> · <a href="http://127.0.0.1:8793/">직전 결과</a></p></section></main></html>'''.replace('STYLE',style).replace('ROWS',rows)
(O/'index.html').write_text(page)
print('sources',len(records),'remaining',len(left),'new error labels',0)

from pathlib import Path
import json,hashlib,sys,html
O=Path(__file__).resolve().parent;R=O.parents[1];U=R/'output/squat-unified-20261023'
def read(p):return json.loads(p.read_text())
def save(name,x):(O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2))
new=read(O/'new-people/ranked-preparation.json');dense={r['id']:r for r in read(O/'new-people/dense-preparation.json')};decisions=[]
for r in new:
 detail=dense.get(r['id']);reason='표본에서 새 명확한 복귀·무릎·좌우 몸통 오류를 확정하지 못함. 촬영 투영·잘림과 동작 오류를 구분하고 학습 대상에서는 제외.'
 if r['id']=='00166686':reason='상승 정점에도 무릎 굽힘이 남아 보이는 부분범위 후보. 완전 기립 기준과 수행 의도가 불명확해 복귀 오류로 단정하지 않음.'
 if r['id']=='00261858':reason='깊은 하강에서 무릎 간격이 달라지나 양 무릎은 대체로 바깥 방향을 유지. 정면 투영 변화만으로 안쪽 붕괴를 확정하지 않음.'
 if r['id']=='00197393':reason='얕은 동작 후 기립과 자세 전환이 섞임. 지정 목표 없는 영상에서 미완료·복귀 오류로 변환하지 않음.'
 decisions.append(dict(id=r['id'],person=r['person'],source_sha256=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest(),action='reviewed_uncertain_not_training',reason=reason,review_frames=(detail or r).get('dense_review_frames',r['review_frames']),evidence=str(O/f'detail-{r["id"]}.jpg') if detail else r['sheet'],continuous_video_review=False,expert_validated=False,levels=None,used_in_training=False,model_output_used_as_target=False))
save('new-people/decisions.json',decisions)
first=read(O/'labels.frozen.json');ids={r['id'] for r in first['decisions']}|{r['id'] for r in decisions};queue=[r for r in read(U/'review-queue.json') if r['source']=='QEVD'];assert len(ids)==96 and ids<={r['id'] for r in queue}
left=[r for r in queue if r['id'] not in ids];save('remaining-review.json',dict(original_unreviewed=4246,sampled_frame_reviewed=len(ids),not_reviewed=len(left),remaining=left,complete_corpus_review=False));assert len(left)==4150
reserved=set(read(R/'output/five-exercise-recovery-20261023/final_test_reservation.json')['reserved_people']);official={'qevd_'+str(x) for x in read(R/'data/qevd-selected/manifests/official_test_workers.json')};protected=reserved|official
all_screen=read(O/'pose-screen.json')+read(O/'new-people/pose-screen.json');assert not {r['person'] for r in all_screen}&protected
old=read(U/'verification.json');actual={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in old['original_model_sha256']};assert actual==old['original_model_sha256']
sys.path.insert(0,str(R/'output/live-webcam-20261023'));from runtime_snapshot import read_source,digest
assert digest(read_source())==old['team_digest']
r=read(O/'comparison.json');assert r['labels_sha256']==hashlib.sha256((O/'labels.frozen.json').read_bytes()).hexdigest();assert r['candidate_sha256']==hashlib.sha256((O/'candidate.joblib').read_bytes()).hexdigest()
for h in r['heads']:
 for f in h['folds']:assert not set(f['train_people'])&set(f['held_people'])
accepted=[h['item'] for h in r['heads'] if h['approved_ai']]
audit=dict(reviewed_sources=96,new_observational_normal_intervals=len(first['labels']),direct_review='sampled raw frames, not expert or continuous video review',automatic_pose_screened=len(all_screen),remaining_unreviewed=len(left),new_confirmed_error_sources=0,protected_people_excluded=True,models_preserved=True,source_digest=old['team_digest'],original_model_sha256=actual,accepted_development_candidate_ai=accepted,active_runtime_models_changed=False,independent_final_test=False,additional_recording=False,rule_base_preserved=True)
save('verification.json',audit)
rows=''
for h in r['heads']:
 for method in ['ai','rule']:
  m=h['outer_metrics'];b=m['before_'+method];a=m['after_'+method];approved=h['approved_'+method]
  rows+=f'<tr><td>{"무릎 모임 보조" if h["item"]=="knee_inward" else "몸통 좌우 안정"}</td><td>{"AI Model" if method=="ai" else "Rule Base"}</td><td>{b["n"]}건 / 오류 {b["error_n"]}건</td><td>{b["MAE"]*50:.2f} → {a["MAE"]*50:.2f}</td><td>{b["error_MAE"]*50:.2f} → {a["error_MAE"]*50:.2f}</td><td class="{"ok" if approved else "hold"}">{"개발 검증 통과" if approved else "기존 유지"}</td></tr>'
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>스쿼트 데이터 보강 · 중간 검증</title><style>
body{margin:0;background:#edf3f6;color:#17364a;font:16px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:40px 24px}header{background:#153d52;color:white;padding:34px;border-radius:22px}h1{font-size:32px;margin:8px 0}h2{font-size:23px}p{margin:10px 0}.tag{color:#aee3d0;font-size:14px}section{background:white;border-radius:18px;padding:24px;margin:20px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:20px}.card{background:white;padding:20px;border-radius:16px}.number{display:block;font-size:32px;font-weight:750}small{color:#617988}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:14px 8px;border-bottom:1px solid #dce6eb}.ok{color:#137353;font-weight:bold}.hold{color:#946016}.notice{border-left:5px solid #d6a447;padding-left:15px}a{color:#126e98}details{margin-top:18px}li{margin:8px 0}@media(max-width:700px){.cards{grid-template-columns:repeat(2,1fr)}section{overflow:auto}}</style><main>
<header><span class="tag">저장 자료만 사용 · 중간 결과 · 기존 모델 보존</span><h1>정면 스쿼트, 무엇이 개선됐나</h1><p>무릎 모임 AI 후보는 사람별 분리 개발 검증을 통과했습니다.<br>몸통 AI와 새 규칙은 악화되는 항목이 있어 기존 버전을 유지합니다.</p></header>
<div class="cards"><div class="card"><span class="number">96</span>원본 표본 프레임 검토</div><div class="card"><span class="number">35</span>정상 관측 구간 라벨 초안</div><div class="card"><span class="number">0</span>새로 확정한 오류 사례</div><div class="card"><span class="number">4,150</span>아직 직접 미검토한 원본</div></div>
<section><h2>같은 사람을 학습·검증 양쪽에 넣지 않았습니다</h2><p>데이터셋의 수행자 ID를 기준으로 5개 그룹 교차 검증에서 같은 수행자의 녹화는 한쪽에만 배치했습니다. 아래 값은 같은 검증 사례에 대한 개선 전후 평균 절대 오차입니다. 낮을수록 라벨에 가깝습니다.</p><table><thead><tr><th>항목</th><th>방식</th><th>검증 구성</th><th>전체 오차 ↓</th><th>오류 사례 오차 ↓</th><th>판정</th></tr></thead><tbody>ROWS</tbody></table><small>비교를 위해 내부 0~2 목표 오차를 ×50해 0~100 범위로 표시했습니다. 사용자 자세 총점이나 정확도(%)가 아닙니다. 무릎 오류 5건·몸통 오류 4건뿐인 작은 개발 표본입니다.</small>
<details><summary>추가 검증과 비교 범위</summary><p>기존 검증 수행자 Mima의 10건에서도 무릎 AI 전체 오차가 42.36 → 23.62로 감소했습니다. 이 10건의 오류 사례는 1건뿐이며, 이미 사용해 온 개발 검증입니다. 독립 최종 시험으로 주장하지 않습니다. 수행자 분리는 제공된 ID 기준이며, 데이터셋 사이의 실제 신원을 별도로 확인한 것은 아닙니다.</p><p>그룹 검증의 개선 전 모델은 기존 라벨만으로 매 폴드 다시 학습했습니다. 고정 10건 비교는 보존한 기존 모델을 사용했습니다. 정상 대조군 추가와 기하 특징 변경을 함께 평가했으므로 각각의 효과를 분리해 확정할 수 없습니다.</p></details></section>
<section><h2>AI Model과 Rule Base가 달라진 이유</h2><ul><li><b>무릎:</b> 무릎 간격만 보면 카메라 투영 때문에 정상 동작도 오류처럼 보일 수 있습니다. AI 후보에 골반–무릎–발목의 상대 위치를 추가해 이 혼동을 줄였습니다. 다만 기존 오류 5건 중 내부 정상 경계로 놓치는 비율은 40%로 그대로여서 오류 검출이 해결됐다는 뜻은 아닙니다.</li><li><b>Rule Base:</b> 새 위치 기준의 임계값을 학습했지만 정상에서의 오판이 증가했습니다. 무릎 규칙은 전체 오차가 커져 채택하지 않았습니다.</li><li><b>몸통:</b> AI 전체 평균은 좋아졌지만 오류 사례 오차가 49.12 → 51.87로 악화했습니다. 정상만 잘 맞히는 개선을 피하려고 기존 모델을 유지합니다.</li><li><b>복귀:</b> 부분 범위 동작·영상 잘림을 복귀 실패로 만들지 않았습니다. 이번 검토에서 명확한 새 복귀 오류를 확보하지 못해 오류 판별 모델의 재학습은 진행하지 않았습니다.</li></ul></section>
<section><h2>라벨링과 적용 범위</h2><p>제공자 태그와 관절 추정값은 후보 선별에만 사용했습니다. 실제 원본의 시간 순서 프레임을 보고 관측 가능한 항목만 라벨링했습니다. 기존 개발 수행자의 영상 64개와 새 수행자의 영상 32개를 검토했고 후자의 32개는 학습에 사용하지 않았습니다.</p><p>35개 정상 관측 구간 중 몸통 34개·무릎 18개가 관절 조건을 통과해 재학습에 사용됐습니다. 연속 영상 전체 검토나 전문가 확정 라벨은 아닙니다.</p><p class="notice"><b>100점 기준은 그대로입니다.</b> 하강30 · 복귀20 · 무릎·발30 · 몸통20. 이번 무릎 AI는 안쪽 이동 관측을 돕는 보조 후보입니다. 발목이 필요한 후보이며, 발 관측이 없는 입력의 성능은 검증하지 않았습니다. 이를 무릎·발30점 전체로 대체하거나 미관측 점수를 채우지 않습니다.</p><p>후보는 별도 파일로 보존했습니다. 실행 중인 운동 화면의 모델·규칙은 바꾸지 않았으며, 원본 모델 해시와 팀 소스가 이전 상태와 같은 것을 확인했습니다.</p></section>
<section><h2>확인 자료</h2><p><a href="comparison.json">사람별 개선 전후 전체 수치</a> · <a href="labels.frozen.json">학습 전 고정한 라벨</a> · <a href="new-people/decisions.json">새 수행자 검토 사유</a> · <a href="verification.json">모델 보존·검증 기록</a> · <a href="remaining-review.json">남은 검토 목록</a> · <a href="http://127.0.0.1:8791/?replay=unified">기존 100점 평가 화면</a></p></section></main></html>'''.replace('ROWS',rows)
(O/'index.html').write_text(page)
print(json.dumps(audit,ensure_ascii=False,indent=2))

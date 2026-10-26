from pathlib import Path
import sys,json,hashlib,html,collections,re
import numpy as np
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent;L=R/'output/live-webcam-20261023';sys.path.insert(0,str(L))
from runtime_snapshot import read_source,digest
from server import score_samples
from rubric100 import assessment
inv=json.loads((O/'source-inventory.json').read_text());comp=json.loads((O/'comparison.json').read_text());labels=json.loads((O/'labels.aligned.frozen.json').read_text());summary=json.loads((O/'label-summary.json').read_text());rubric=json.loads((O/'rubric.json').read_text())
assert comp['source_digest']==digest(read_source())
model_paths=[p for folder in [L/'front-candidate',R/'output/squat-relabel-20261023',R/'output/squat-relabel-expanded-20261023',R/'team-posture-scoring/Server/AIService/artifacts/posture_score'] for p in folder.glob('*.joblib')]
preserved={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in model_paths}
# The earlier complete saved session permits all six existing kinematic segments to replay.
audit=json.loads((R/'output/squat-replay-improvement-20261023/comparison.json').read_text());session=Path(audit['source_session']);z=np.load(session/'pose.npz');times=(z['frame_numbers']-z['frame_numbers'][0])/float(z['fps']);replays=[]
for event in audit['new_events']:
 mask=(times>=event['start_seconds']-.001)&(times<=event['end_seconds']+.001);segment=z['image_xy'][mask]
 result=score_samples(list(enumerate(segment)),float(z['width']),float(z['height']),'squats','front',True)
 row=dict(number=event['number'],duration=event['end_seconds']-event['start_seconds'],items=result['items'],front_candidate=result['front_candidate'],method_comparison=result['method_comparison'],accepted_items=result['accepted_items'],coaching=result['coaching'],score_100=result['score_100'])
 replays.append(row)
direct_reviews=json.loads((O/'qevd-direct-review.json').read_text())
excluded_review_ids={r['id'].split('_')[1] for r in direct_reviews if r['action']=='exclude_from_repetition_score_training'}
# Quantify what is actually reviewed, rather than equating an archive listing to labeling.
qevd_reviewed={m.group(0) for r in labels['rows'] if (m:=re.search(r'(?<=qevd_)\d{8}',r['id']))}
pending=dict(qevd_stored_squat_originals=4308,qevd_originals_in_migrated_rows=len(qevd_reviewed),qevd_not_in_migrated_rows=4308-len(qevd_reviewed),qevd_directly_reviewed_excluded=len(excluded_review_ids),qevd_remaining_unreviewed=4308-len(qevd_reviewed)-len(excluded_review_ids),all_4308_visually_relabelled=False,all_554_REHAB_repetitions_visually_relabelled=False,whole_corpus_fully_relabelled=False,reason='원본 전체 메타데이터 목록화와 선별 시각 라벨을 구분. 미검토 원본의 제공자 오류명을 100점 정답으로 변환하지 않음.')
state=dict(exercise='squats',view='front',collecting=False,status='저장 자료 재실행 · 새 100점 계약',generation='unified-v3',movement=dict(count=len(replays),phase='저장 결과'),repetitions=replays,last_rep=replays[-1],source_commit='55fae6974bee80a71c50c8acbf2ce5fa782c58a6',loaded_code_model_sha256=digest(read_source()),source_session=str(session),additional_recording=False,independent_video_ground_truth=False)
(O/'replay-state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False))
receipt=dict(team_digest=digest(read_source()),team_unchanged=True,original_model_sha256=preserved,original_models_overwritten=False,rows_aligned=len(labels['rows']),new_direct_label_rows=len(json.loads((O/'new_visual_labels.json').read_text())['records']),expert_validated=False,additional_recording=False,independent_final_test=False,source_pose_sha256=hashlib.sha256((session/'pose.npz').read_bytes()).hexdigest(),saved_replay_repetitions=len(replays),adopted_ai_heads=[h['item']+':'+h['bank'] for h in comp['heads'] if h.get('promoted_ai')],adopted_rule_heads=[h['item']+':'+h['bank'] for h in comp['heads'] if h.get('promoted_rule')],pending=pending)
(O/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));(O/'remaining-review.json').write_text(json.dumps(pending,ensure_ascii=False,indent=2))
# Persist a candidate queue for every metadata source, including the reason it cannot yet be fitted.
queue=[]
for r in inv['records']:
 if r['source']=='QEVD' and r['id'] not in qevd_reviewed and r['id'] not in excluded_review_ids:
  label=r.get('provider_labels',[]);priority=0 if any('insufficient' in s for s in label) else 1 if any('back not straight' in s or 'knees over toes' in s for s in label) else 2
  queue.append(dict(id=r['id'],source=r['source'],path=r['path'],person=r.get('person'),priority=priority,status='unreviewed_not_training_target',provider_hints=label))
 elif r['source'] in ['YouTube','AIHub231']:queue.append(dict(id=r['id'],source=r['source'],path=r['path'],person=r.get('person'),priority=2,status='metadata_only_needs_motion_variant_and_item_review'))
(O/'review-queue.json').write_text(json.dumps(sorted(queue,key=lambda r:r['priority']),ensure_ascii=False,indent=2))
e=lambda v:html.escape(str(v));names=dict((i['id'],i['title']) for i in rubric['items'])
body='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>스쿼트 자료 통합 · 100점 체계 검증</title><style>body{font:17px/1.65 system-ui;background:#edf3f5;color:#20394b;max-width:1160px;margin:36px auto;padding:0 22px}h1{font-size:30px}h2{font-size:23px}section{background:white;border-radius:18px;padding:25px;margin:20px 0}.tag{background:#fff0cf;border-radius:10px;padding:14px}table{width:100%;border-collapse:collapse;font-size:15px}td,th{text-align:left;padding:12px 8px;border-bottom:1px solid #dfe8ec}small{display:block;color:#58707d}a,summary{color:#126674}summary{cursor:pointer}.scroll{overflow:auto}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.metric{background:#e8f5f3;border-radius:12px;padding:16px}.metric strong{display:block;font-size:26px}@media(max-width:750px){.metrics{display:block}}</style><h1>스쿼트 · 보유 자료 통합과 100점 체계</h1><p class="tag">중간 개발 결과 · 추가 촬영 없음 · 기존 모델 보존<br><b>전체 원본을 모두 재라벨링한 상태는 아닙니다.</b> 목록화·라벨 정합성 검토·직접 시각 재검토·재학습을 구분해 표시합니다.</p><p><a href="/?replay=unified">저장 동작 6회 · 새 점수 화면</a> · <a href="/?saved=latest">최신 7회 기존 기록</a> · <a href="/">실시간 운동 화면</a></p>'''
body+=f'<section><div class="metrics"><div class="metric"><small>기준에 맞춰 대조한 기존·신규 라벨</small><strong>{len(labels["rows"])}개 구간</strong></div><div class="metric"><small>이번 새 시각 라벨</small><strong>3개 반복·항목 초안</strong></div><div class="metric"><small>별도 보존한 재학습 모델 묶음</small><strong>{sum(h["trained"] for h in comp["heads"])}개 항목·구도</strong></div></div><p>통합 구간 전부를 이번에 새로 영상 검토했다는 의미가 아닙니다. 기존 시각 초안의 기준 대응을 감사하고, 복귀 오류 후보 1원본의 2반복과 QEVD 후보 8클립을 새로 시각 검토했습니다. QEVD 중 1반복의 몸통 항목만 추가하고 7클립은 학습에서 제외했습니다. 라벨은 전문가 확정 정답이 아닙니다.</p></section>'
body+='<section><h2>고정한 평가 기준 · 합계 100점</h2><table><tr><th>항목</th><th>배점</th><th>동작 기준과 측정 범위</th></tr>'
for i in rubric['items']:body+=f'<tr><td>{e(i["title"])}</td><td>{i["weight"]}점</td><td>{e(i["definition"])}<small>{e(i["observation"])}</small></td></tr>'
body+='</table><p>배점은 기존 프로젝트 체계이며 의학적으로 검증된 표준 점수가 아닙니다. 무릎이 발끝 앞으로 나오는 것만으로 감점하지 않습니다. 관측 불가·판단 불확실을 중간 수행 등급으로 바꾸지 않습니다.</p><small>0·1·2는 내부 라벨이며 운동 화면에서는 항목 배점에 따른 점수를 표시합니다.</small></section>'
body+='<section><h2>보유 자료 전체 목록화</h2><table><tr><th>출처</th><th>확인 범위</th><th>현재 의미</th></tr>'
source_text={'QEVD':('스쿼트 원본 클립','제공자 라벨은 후보 선별에만 사용. 전체 자동 정답화 금지'),'REHAB26':('반복·카메라 행','554개 고유 반복 × 3카메라. 사람 5명이며 독립 반복 수를 1,662회로 부풀리지 않음'),'MultiView':('스쿼트 파일 경로','Exercise-wise / Quality-wise 복사본 포함 가능. 경로 수는 고유 사람·반복 수가 아님'),'REHAB24':('영상 파일','다른 REHAB 계열과 신원 대응이 미확정이라 검증에서 같은 가족 그룹으로 보수적으로 묶음'),'PoseComparison':('영상 파일','정상·오류 점수 정답으로 자동 사용하지 않음'),'UserWebcam':('촬영 폴더','실제 영상이 있는 자료와 중단·앉은 장면을 별도 취급'),'UserPoseSession':('관절 저장 세션','실제 영상이 없는 세션을 독립 시각 정답으로 부르지 않음'),'YouTube':('메타데이터 후보','실제 파일 존재 여부와 스쿼트 관련성을 개별 목록에 기록. 제목·자막은 점수 정답이 아님')}
for source,n in inv['summary']['source_counts'].items():
 title,note=source_text.get(source,('후보','개별 검토 필요'));body+=f'<tr><td>{e(source)}</td><td>{n:,} {title}</td><td>{e(note)}</td></tr>'
body+='</table><p><b>AI Hub:</b> 확보한 validation_body_01.zip의 2,240개 JSON(3D 기록 포함)을 확인했습니다. 운동 메타데이터에는 푸시업·플랭크·복근 운동·Y Exercise 등이 있으며 스쿼트는 없었습니다. AI Hub 전체 데이터셋에 스쿼트가 없다는 뜻은 아닙니다.</p><p><b>미국 해군:</b> 기존 라벨에서 확인한 자료는 플랭크입니다. 스쿼트 학습량으로 합산하지 않았습니다.</p></section>'
body+='<section><h2>재라벨링·학습 입력 정리</h2><p>정면 높이 복귀를 원래 관절 신전 정답으로 사용하던 항목은 제외했습니다. 무릎 간격 변화·뒤꿈치 단독 관측을 원래 무릎·발 정렬 합성 정답으로 자동 결합하지 않았습니다. 몸통 좌우 안정처럼 의미가 대응되는 라벨은 기존 시각 근거와 함께 재사용했습니다.</p><p><b>새 직접 검토:</b> subject_014의 성긴 개요 표본은 부분 복귀처럼 보였지만, 촘촘한 f24–54·f54–84 표본에서는 서기 복귀가 확인됐습니다. 두 반복은 하강 범위 부분 미충족·복귀 충족으로 라벨링하고, 측면에서 확인할 수 없는 정면 항목은 비웠습니다. 모델 점수로 정답을 만들지 않았습니다.</p><table><tr><th>항목</th><th>학습 가능한 관측 초안 분포</th></tr>'
for k,v in summary['coverage'].items():body+=f'<tr><td>{e(names[k])}</td><td>충족 {v.get("2",0)} · 부분 미충족 {v.get("1",0)} · 명확한 미충족 {v.get("0",0)}</td></tr>'
body+='</table><small>구간·촬영 중복을 사람 그룹으로 묶습니다. 위 수치는 독립 사람 수가 아닙니다.</small></section>'
body+='<section><h2>AI Model / Rule Base 개선 전후</h2><p>동일 수행자·원본·촬영 그룹을 연결한 뒤 그룹 단위 외부 교차검증을 수행했습니다. 후보 설정 선택은 각 학습 폴드 내부의 그룹 교차검증에서만 했습니다. REHAB24·26의 신원 중복 불확실성은 두 계열을 함께 보류하는 방식으로 처리했습니다.</p><small>아래는 재학습한 대조 설정 → 후보 설정의 사람 그룹 분리 개발 검증입니다. 과거 동결 모델은 학습에 이미 본 자료가 섞일 수 있어 별도의 재실행 수치로 보관했으며, 독립 최종 시험으로 제시하지 않습니다.</small><div class="scroll"><table><tr><th>항목·구도</th><th>AI 평균 점수 오차<br>대조 → 후보</th><th>규칙 평균 점수 오차<br>기존 → 후보</th><th>반영 판단</th></tr>'
for h in comp['heads']:
 if not h['trained']:body+=f'<tr><td>{e(names[h["item"]])} · {e(h["bank"])}</td><td colspan="2">{e(h["reason"])}</td><td>기존 보존</td></tr>';continue
 m=h['metrics'];scale=h['weight']/2;fmt=lambda n:'비교 불가' if n['MAE_grade'] is None else f'{n["MAE_grade"]*scale:.2f}점';decision='후보 반영' if h['promoted_ai'] or h['promoted_rule'] else '교체하지 않음'
 body+=f'<tr><td>{e(names[h["item"]])} · {e(h["bank"])}</td><td>{fmt(m["prior_recipe_ai"])} → {fmt(m["after_ai"])}</td><td>{fmt(m["prior_rule"])} → {fmt(m["after_rule"])}</td><td>{decision}<small>오류 사람 그룹 {h["error_groups"]}개</small></td></tr>'
body+='</table></div><p>평균 오차만 줄고 오류 누락 또는 정상 동작의 오경고가 늘어난 후보도 반영하지 않습니다. 이번 후보는 채택 조건을 통과하지 못해 기존 모델·규칙을 교체하지 않았습니다. 복귀는 새로 정리한 관측 라벨도 정상뿐이라 오류 판별 AI가 개선됐다고 주장하지 않습니다.</p></section>'
body+='<section><h2>저장 동작 6회 · 100점 계약 재실행</h2><p>전체 관절이 보존된 이전 세션을 사용했습니다. 최신 7회와는 다른 세션이며, 관절 기반 구간 재분할 결과이지 독립 영상 정답 횟수는 아닙니다.</p><table><tr><th>회차</th><th>AI Model /100</th><th>Rule Base /100</th></tr>'
for r in replays:
 s=r['score_100'];fmt=lambda m:(f'{m["total"]:.1f}/100' if m['total'] is not None else f'총점 미산출 · 평가된 {m["assessed_weight"]}점 중 {m["earned_subtotal"]:.1f}점');body+=f'<tr><td>{r["number"]}회</td><td>{fmt(s["ai"])}</td><td>{fmt(s["rule"])}</td></tr>'
body+='</table><p>하강은 기존 동기화 측면 라벨 기반 정면 추정임을 표시합니다. 높이 복귀나 무릎 간격만으로 원래 신전·정렬 항목 점수를 채우지 않으므로 현재 정면 구간은 100점 총점이 완성되지 않습니다. 발 미관측 때문에 횟수나 다른 관측 항목까지 중단하지 않습니다.</p><a href="/?replay=unified">회차별 점수·차이 설명 화면 열기</a></section>'
body+=f'<section><h2>아직 남은 검토</h2><p>QEVD 스쿼트 4,308개 중 이번 통합 라벨에 대응한 원본은 {len(qevd_reviewed)}개이며, {4308-len(qevd_reviewed):,}개는 이번 통합 라벨에 포함되지 않았습니다. 전체 554개 REHAB 반복도 새 기준으로 모두 직접 재검토한 상태는 아닙니다.</p><p>미검토 자료는 사람·출처·제공자 후보 설명과 함께 검토 대기 목록에 저장했습니다. 이를 정상·오류 정답으로 자동 채우거나, 재학습 완료 자료로 계산하지 않았습니다. 이번에 제외한 QEVD 7클립은 별도 직접 검토 기록에 남겼습니다. 추가 촬영보다 이 기존 자료의 항목별 검토가 우선입니다.</p><small>전문가 확정 라벨·독립 최종 시험은 아직 없습니다.</small></section></html>'
(O/'index.html').write_text(body)
print(json.dumps(receipt,ensure_ascii=False,indent=2))

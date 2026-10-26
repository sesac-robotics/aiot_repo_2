"""고정된 동일 녹화의 정책 적용 전후를 비교하고 보존 지문을 확인한다."""
import json,hashlib
from pathlib import Path
import numpy as np
from front_policy import apply_front_policy,frontal_motion_observations,VERSION
from runtime_snapshot import read_source,digest
R=Path(__file__).parent;P=R/'recordings/20261023-110200-964840'
before=digest(read_source())
d=json.loads((P/'legacy-comparison.json').read_text());z=np.load(P/'pose.npz')
assert before==d['recording_model_digest'],'최신 소스가 달라짐: 다시 기준 모델과 비교 필요'
rows=[]
for row in d['repetitions']:
 a,b=row['start_frame'],row['end_frame'];raw=row['legacy_result'];after=apply_front_policy(raw,'squats','front')
 obs=frontal_motion_observations(z['image_xy'][a:b+1])
 rows.append(dict(number=row['number'],before=raw,after=after,observations=obs))
result=dict(policy_version=VERSION,source_preserved=before==digest(read_source()),source_digest=before,model_retrained=False,rule_boundaries_changed=False,labels_used_to_fit=False,comparison_type='동일 입력의 판정 보류·표시 정책 비교; 정확도 향상 실험 아님',repetitions=rows)
assert result['source_preserved']
(P/'policy-before-after.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
trs=''
for r in rows:
 o=r['observations'];trs+=f"<tr><td>{r['number']}</td><td>0 / 2* / 보류 / 2*</td><td>보류 / 보류 / 보류 / 보류</td><td>보류 / 보류 / 보류 / 2</td><td>{o['hip_vertical_range_torso_ratio']}</td><td>{o['return_height_delta_torso_ratio']}</td><td>{100*o['all_six_fraction']:.1f}%</td></tr>"
(P/'policy-before-after.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>스쿼트 개선 전후</title><style>body{font:16px/1.7 sans-serif;color:#173749;max-width:1150px;margin:40px auto;padding:20px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}h1{font-size:28px}strong{color:#08796e}</style><h1>정면 스쿼트 5회 · 보완 정책 적용 전후</h1><p><strong>기존 모델·규칙 경계·팀 소스 보존 확인. 재학습 없음.</strong> 현재 최신 커밋 55fae69 및 촬영 당시 스냅샷과 호환.</p><p>정면에서 검증되지 않은 깊이·신전 AI 출력과 단일 등급 학습 출력을 보류한다. 기존 모델의 원본 등급은 raw_grades에 보관한다. 기존 몸통 규칙은 유지한다. 100점 재환산이나 누락 만점 처리는 하지 않는다.</p><p>등급 순서: 깊이 / 복귀·펴기 / 무릎·발 / 몸통. *는 정상 등급만 학습한 AI 출력. 규칙 결과는 전후 동일하다.</p><table><tr><th>회차</th><th>이전 AI 원본</th><th>적용 후 AI</th><th>규칙 전후 동일</th><th>골반 이동/몸통 길이</th><th>복귀 높이 차/몸통 길이</th><th>6관절 관측</th></tr>'''+trs+'''</table><p>골반 이동량은 영상 세로축의 95–5백분위 범위/몸통 길이이며 실제 깊이가 아니다. 복귀 차이는 종료·시작 5프레임 골반 높이 중앙값의 차이/몸통 길이이다. 양수는 화면에서 종료 위치가 더 아래라는 뜻이며, 자세 등급으로 변환하지 않았다.</p><h2>개선의 범위</h2><ul><li>줄어든 것은 근거가 부족한 확정 점수이다. 모델 예측 정확도가 향상됐다는 결과가 아니다.</li><li>기존 웹 화면도 단일 등급 AI 득점을 이미 숨기고 있었다. 이번에는 결과 사전에도 보류를 적용하고, 정면 깊이·신전의 측정 한계와 원본 등급 기록을 추가했다.</li><li>3·4회의 오른쪽 무릎 누락은 남아 있다. 몸통 관측은 유지하며 무릎 관련 점수를 임의로 채우지 않았다.</li><li>이 녹화는 한 사람의 한 그룹이며 개발 진단용이다. 정상 유사 5회만으로 새로운 임계값을 맞추거나 오류 구분 모델을 훈련하지 않았다.</li><li>공개 ScoringEngine·PoseSequence 계약은 그대로 사용하고 별도 어댑터에서 사본을 변환한다. 팀 소스 갱신 시 기존 버전 감지·재로딩 절차를 따른다.</li></ul><p>검증: 원본 입력 불변, 다른 운동·구도 무변경, 중복 정책 적용 안정성, 발목 없이 관측값 계산, 누락 보존 등 총 19개 테스트 통과.</p><p><a href="policy-before-after.json">전체 전후 결과</a> · <a href="analysis.html">독립 라벨·원인 분석</a></p></html>''')
print(json.dumps(dict(source_preserved=result['source_preserved'],observations=[dict(number=r['number'],**r['observations']) for r in rows]),ensure_ascii=False))

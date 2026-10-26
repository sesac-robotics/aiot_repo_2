from pathlib import Path
import sys,json,hashlib,html
import numpy as np,joblib
ROOT=Path(__file__).resolve().parents[2];LIVE=ROOT/'output/live-webcam-20261023';OUT=Path(__file__).parent
sys.path.insert(0,str(LIVE))
from method_comparison import compare_candidate
from front_candidate import predict
from checklist_comparison import models
from runtime_snapshot import read_source,digest
heads,extract=models();old=joblib.load(ROOT/'output/squat-relabel-20261023/candidate.joblib');expanded=joblib.load(ROOT/'output/squat-relabel-expanded-20261023/candidate.joblib')
paths=[LIVE/'front-candidate/model.joblib',LIVE/'front-candidate/torso-model.joblib',ROOT/'output/squat-relabel-20261023/candidate.joblib',ROOT/'output/squat-relabel-expanded-20261023/candidate.joblib']
preserved={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};team=digest(read_source())
rows=json.loads((ROOT/'output/squat-relabel-expanded-20261023/feature-rows.json').read_text());validation=[]
for key in ('knee_inward','trunk_balance'):
    cases=[]
    for r in rows:
        if r['split']!='validation' or key not in r['features'] or r['levels'][key] is None:continue
        f=r['features'][key];entry={'id':r['id'],'target':r['levels'][key]}
        for name,b in [('before',old),('expanded',expanded)]:
            h=b['heads'][key];ai=float(h['model'].predict([f['x']])[0]);good,partial=h['boundaries'];v=f['scalar'];rule=2 if (v>=good if h['higher'] else v<=good) else 1 if (v>=partial if h['higher'] else v<=partial) else 0
            entry[name]={'ai':ai,'rule':rule}
        cases.append(entry)
    stats={n:{m:float(np.mean([abs(c[n][m]-c['target']) for c in cases])) for m in ['ai','rule']} for n in ['before','expanded']}
    validation.append(dict(item=key,n=len(cases),people=['Mima'],grade_MAE=stats,selected='before' if key=='knee_inward' else 'expanded',cases=cases))
# Re-use a complete captured session and its existing reviewed kinematic boundaries.
audit=json.loads((ROOT/'output/squat-replay-improvement-20261023/comparison.json').read_text());session=Path(audit['source_session']);z=np.load(session/'pose.npz');xy=z['image_xy'];fps=float(z['fps']);times=(z['frame_numbers']-z['frame_numbers'][0])/fps;replays=[]
for e in audit['new_events']:
    segment=xy[(times>=e['start_seconds']-.001)&(times<=e['end_seconds']+.001)]
    c=predict(segment);comparison=compare_candidate(c,segment)
    without=segment.copy();without[:,27:]=np.nan;nofoot=compare_candidate(predict(without),without)
    assert [(r.get('ai_grade'),r.get('rule_grade')) for r in comparison['item_comparisons']]==[(r.get('ai_grade'),r.get('rule_grade')) for r in nofoot['item_comparisons']]
    replays.append(dict(**e,frames=len(segment),comparison=comparison))
assert team==digest(read_source());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in preserved.items())
report=dict(team_digest=team,models_preserved=preserved,team_preserved=True,validation=validation,replay_session=str(session),replay_pose_sha256=hashlib.sha256((session/'pose.npz').read_bytes()).hexdigest(),replay_repetitions=replays,additional_recording=False,expert_validated=False,final_test_opened=False,return_AI_supported=False,return_reason='31 train + 10 validation are normal-only; no fabricated error labels',foot_removal_invariance_verified=True,latest_seven_limitation='최신 7회 중 처음 3회의 원시 관절은 기존 버퍼 잘림으로 없음. 이번 6회 재실행은 전체 관절이 보존된 이전 저장 세션임.')
(OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
esc=lambda v:html.escape(str(v))
body='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>스쿼트 항목별 AI Model / Rule Base 검증</title><style>body{background:#edf3f5;color:#20394b;font:17px/1.65 system-ui;max-width:1140px;margin:40px auto;padding:0 24px}h1{font-size:30px}h2{font-size:23px}section{background:white;border-radius:18px;padding:26px;margin:20px 0}table{width:100%;border-collapse:collapse}td,th{padding:12px 8px;text-align:left;border-bottom:1px solid #dce7ec}small{color:#58707d}summary{cursor:pointer;color:#126674}a{color:#126674}.badge{background:#e6f5ee;border-radius:9px;padding:6px 12px}</style><h1>스쿼트 · 같은 항목으로 비교하기</h1><p class="badge">중간 개발 검증 · 추가 촬영 없음 · 기존 모델 보존 · GPT 없이 자동 설명</p><section><h2>무엇을 비교하나요?</h2><p>하강 범위 · 무릎 모임 · 몸통 좌우 안정은 같은 동작 구간에서 AI Model과 Rule Base를 비교합니다. 0·1·2 관측 등급을 사용하며 합산 총점을 만들지 않습니다.</p><p><b>복귀:</b> Rule Base의 화면상 높이 복귀 평가는 유지합니다. 정상 학습 31회에 복귀 오류가 없어 AI 비교는 미지원으로 표시합니다.</p><small>정면 무릎 간격 변화와 몸통 좌우 기울임의 추정입니다. 발끝 방향·무릎 완전 신전·허리 전후 정렬을 확정하는 평가가 아닙니다.</small></section><section><h2>기존 모델과 비교 · 같은 검증자 10회</h2><small>수치는 0–2 등급의 평균 절대 오차(MAE), 낮을수록 좋습니다. 이미 사용한 개발 검증 자료이며 최종 시험이 아닙니다.</small><table><tr><th>항목</th><th>AI 이전 → 확장</th><th>규칙 이전 → 확장</th><th>이번 연결</th></tr>'''
for v in validation:
    a=v['grade_MAE'];body+=f"<tr><td>{esc({'knee_inward':'무릎 모임','trunk_balance':'몸통 좌우 안정'}[v['item']])}</td><td>{a['before']['ai']:.3f} → {a['expanded']['ai']:.3f}</td><td>{a['before']['rule']:.3f} → {a['expanded']['rule']:.3f}</td><td>{'이전 무릎 모델·규칙 유지' if v['selected']=='before' else '확장 몸통 모델·규칙 비교'}</td></tr>"
body+='</table><p>무릎 AI는 확장 후 오차가 커져 이전 후보를 유지했습니다. 몸통 AI는 오차가 줄어든 확장 후보를 연결했습니다. 규칙도 같은 라벨 기준의 경계를 사용하며, 기존 10도 안내 규칙과 구분합니다.</p></section><section><h2>전체 관절이 보존된 저장 동작 6회 재실행</h2><small>최신 7회 세션과는 다른 이전 저장 세션입니다. 횟수는 기존 관절 재분할 결과이며 독립 영상 정답 횟수는 아닙니다. 발 관절을 모두 제거해도 아래 세 항목 결과는 같았습니다.</small><table><tr><th>회차</th><th>하강 AI / 규칙<br>/30점</th><th>무릎 AI / 규칙<br>/2등급</th><th>몸통 AI / 규칙<br>/2등급</th><th>자동 설명</th></tr>'
for r in replays:
    c=r['comparison'];items={i['item']:i for i in c['item_comparisons']};k=items['knee_inward'];t=items['trunk_balance'];details=''
    for it in c['item_comparisons']:
        details+=f"<p><b>{esc(it['title'])}</b>: {esc(it['summary'])}</p>"
        for evidence in ('rule_evidence','model_evidence'):
            if evidence in it:details+=f"<small>{esc(it[evidence]['explanation'])}</small><br>"
    fmt=lambda i: f"{i['ai_grade']:.2f} / {i['rule_grade']}" if i.get('ai_grade') is not None else '관측 부족'
    body+=f"<tr><td>{r['number']}회</td><td>{c['ai_points']} / {c['rule_points']}</td><td>{fmt(k)}</td><td>{fmt(t)}</td><td><details><summary>차이의 이유 보기</summary>{details}<small>특징의 기여는 계산 설명이며 실제 자세 오류 원인을 확정하지 않습니다.</small></details></td></tr>"
body+='</table></section><section><h2>확인된 한계</h2><p>복귀 오류 AI는 아직 검증할 수 없습니다. 무릎·몸통도 한 명 10회의 개발 검증이고 전문가 확정 라벨이 아니므로 정확한 일반화 성능을 보장하지 않습니다.</p><p>최신 7회의 처음 3회는 원시 관절이 남지 않아 새 항목을 소급 계산하지 않았습니다. 향후 운동은 비교 입력·결과·이유를 회차별로 함께 저장합니다.</p><a href="/?saved=latest">최신 7회 원래 결과</a> · <a href="/">운동 화면</a> · <a href="/?developer=1">개발자 상세</a></section></html>'
(OUT/'index.html').write_text(body)
print(json.dumps(dict(validation=[{k:v for k,v in a.items() if k!='cases'} for a in validation],replay_count=len(replays),team_preserved=True),ensure_ascii=False,indent=2))

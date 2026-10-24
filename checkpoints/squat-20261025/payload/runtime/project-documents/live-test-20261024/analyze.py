from pathlib import Path
import sys,json,hashlib,html
from datetime import datetime,timezone
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid'); L=B/'output/live-webcam-20261023'; O=B/'project-documents/live-test-20261024'
sys.path.insert(0,str(L))
from movement import MovementTracker
sys.path.insert(0,str(L/'runtime/c76334937f21ef24a27aa1c1687de6c9577bf6249e5cac606151a328893746f7'))
from app.scoring.three_items import mask_image_border
rows=[]
for f in sorted((L/'sessions').iterdir()):
 if not f.name.isdigit() or int(f.name)<1792845000000000000:continue
 d=json.loads((f/'final-state.json').read_text()); p=np.load(f/'pose.npz'); xy=p['image_xy']; ticks=p['frame_numbers']; tr=MovementTracker(d['exercise'],d['view']); segs=[]
 for tick,points in zip(ticks,xy):
  seg=tr.push(int(tick),points)
  if seg:segs.append(seg)
 req=[11,12,23,24,13,14,15,16] if d['exercise'] in ('shoulder','pushups') else [11,12,23,24]
 reps=[]
 for i,r in enumerate(d['repetitions']):
  segxy=np.array([v for _,v in segs[i]]); masked=mask_image_border(segxy,int(p['width']),int(p['height']))
  reps.append({'number':r['number'],'duration':r['duration'],'ai_total':r.get('ai',{}).get('total'),'rule_total':r.get('rule',{}).get('total'),'items':[{k:a.get(k) for k in ['id','title','weight','ai_points','rule_points']} for a in r.get('score_100',{}).get('items',[])],'messages':(r.get('coaching') or {}).get('messages',[]),'required_raw_coverage':float(np.isfinite(segxy[:,req]).all((1,2)).mean()),'required_border_safe_coverage':float(np.isfinite(masked[:,req]).all((1,2)).mean()),'border_safe_joint_coverage':{str(j):round(float(np.isfinite(masked[:,j]).all(1).mean()),4) for j in req}})
 rows.append({'session':str(f),'exercise':d['exercise'],'view':d['view'],'samples':len(ticks),'seconds':round(float((ticks[-1]-ticks[0])/10),1),'saved_count':d['movement']['count'],'replayed_count':tr.status()['count'],'same_count':d['movement']['count']==tr.status()['count'],'phase':d['movement']['phase'],'uncertain_attempts':d['movement']['uncertain_attempts'],'required_simultaneous_coverage':float(np.isfinite(xy[:,req]).all((1,2)).mean()),'joint_coverage':{str(j):round(float(np.isfinite(xy[:,j]).all(1).mean()),4) for j in [11,12,13,14,15,16,23,24,25,26,27,28]},'repetitions':reps,'video_saved':any(f.glob('*.mp4')),'sha256':{n:hashlib.sha256((f/n).read_bytes()).hexdigest() for n in ['pose.npz','state.json','final-state.json']},'runtime_digest':d['loaded_code_model_sha256']})
sq=next(r for r in rows if r['exercise']=='squats')
means={m:round(float(np.mean([r[m+'_total'] for r in sq['repetitions']])),1) for m in ['ai','rule']}
report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'현재 배포 구현의 저장 결과 및 관절 재생 기능 분석; 실제 동작 정답 검증 아님','sessions':rows,'squat_mean':means,'limitations':['영상 저장 없음: 독립 육안 실제 횟수·정답 확인 불가','기립 상태 신뢰도×20은 복귀 실패 심각도 점수로 검증된 값이 아님','플랭크 저장 세션 없음','원본 모델·라벨·분할·실서비스 코드 변경 없음']}
(O/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
name={'squats':'스쿼트','shoulder':'숄더 프레스','pushups':'푸시업','ytw':'YTW'}
summary=''.join(f'<tr><td>{name[r["exercise"]]}</td><td>{r["seconds"]}초 / {r["samples"]}표본</td><td>{r["saved_count"] if r["saved_count"] is not None else "미지원"}</td><td>{len(r["repetitions"])}개</td><td>{html.escape(r["phase"])}</td></tr>' for r in rows)
scores=''.join(f'<tr><td>{r["number"]}</td><td>{r["ai_total"]}</td><td>{r["rule_total"]}</td>'+''.join(f'<td>{a["ai_points"]} / {a["rule_points"]}</td>' for a in r['items'])+'</tr>' for r in sq['repetitions'])
sh=next(r for r in rows if r['exercise']=='shoulder')['repetitions'][0]
text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>실시간 테스트 분석 · 2026-10-24</title><style>body{{font:17px/1.65 sans-serif;color:#18323d;background:#f2f6f7;max-width:1200px;margin:45px auto;padding:0 24px}}section{{background:white;padding:26px;margin:22px 0;border-radius:16px}}table{{border-collapse:collapse;width:100%;font-size:16px}}td,th{{padding:12px;border-bottom:1px solid #dbe4e7;text-align:left}}h1,h2{{line-height:1.3}}.notice{{background:#fff1d8;padding:18px;border-radius:10px}}small{{color:#526870}}</style><h1>현재 구현 실시간 테스트 분석</h1><p>2026년 10월 24일 · 저장 세션 4개 · 원본 관절 자료로 반복 검출 재생 확인</p><div class="notice">스쿼트 5회 점수·저장은 작동했습니다. 다른 운동은 관측 조건 또는 구현 범위에 막혔습니다. 영상이 없어 실제 수행 횟수와 자세 정답은 확정하지 않습니다.</div><section><h2>운동별 결과</h2><table><tr><th>운동</th><th>수집 구간</th><th>시스템 횟수</th><th>저장 평가 행</th><th>종료 상태</th></tr>{summary}</table><p>네 세션 모두 pose.npz·state.json·final-state.json 저장 확인. 원본 관절 재생에서 동일 횟수 재현. 플랭크 저장 세션은 없습니다.</p></section><section><h2>스쿼트: AI 평균 {means['ai']} / Rule Base 평균 {means['rule']}</h2><p>네 항목을 모두 산출한 정면 관측 추정 점수입니다. 전신 자세 정답률이 아닙니다.</p><table><tr><th>회차</th><th>AI /100</th><th>Rule /100</th><th>하강 /30<br>AI / Rule</th><th>복귀 /20<br>AI / Rule</th><th>무릎 /30<br>AI / Rule</th><th>몸통 /20<br>AI / Rule</th></tr>{scores}</table><p>Rule Base는 다섯 회 모두 하강 15/30, 복귀 20/20, 무릎 30/30입니다. 몸통은 1·4·5회 10/20, 2·3회 20/20입니다. 5회 몸통 측정 2.491°는 경계 2.458°와 가까워 경계 민감도가 있습니다.</p><p>AI 복귀는 기립 상태 신뢰도×20의 실험 점수입니다. 정상 복귀 확인과 별개로 15.1~16.8점이 나옵니다. 이 차이를 실제 복귀 부족 3.2~4.9점으로 해석할 근거는 없습니다. 무릎 간격은 발끝 방향·무릎 전체 정렬과 다르고, 정면 몸통은 전후 허리 정렬을 판정하지 않습니다.</p><p>위치 이동으로 보류된 추가 구간 1개(1.6초)는 횟수에서 제외됐습니다. 미완료·가림·실제 추가 반복 중 어느 경우인지는 영상 없이 확정할 수 없습니다.</p></section><section><h2>나머지 운동의 막힌 지점</h2><p><b>숄더 프레스:</b> 반복 1회는 인식했지만 모든 점수가 미산출입니다. 검출 구간의 필요 관절 원시 좌표 동시 관측은 {sh['required_raw_coverage']:.0%}, 영상 경계 2% 보호 적용 후에는 {sh['required_border_safe_coverage']:.0%}였습니다. 화면 밖/경계 관절 추정값을 검출기는 허용하고 점수기는 제외하는 조건 차이를 확인했습니다. 이는 0점이 아닙니다. 정상 한 등급만 학습한 복귀·몸통 AI 항목의 오류 판별 제한도 별도로 남습니다.</p><p><b>푸시업:</b> 시스템 0회. 필요 어깨·팔꿈치·손목·골반의 동시 관측 부족으로 준비 자세가 끊겼습니다. 측면에서도 현재 검출기는 양팔·양쪽 골반을 요구해 가려진 쪽 때문에 막힐 수 있습니다. 실제 푸시업 0회를 뜻하지 않습니다.</p><p><b>YTW:</b> 관절 497표본은 저장했지만 자동 반복 구분이 코드에서 미지원이므로 완료 평가 행이 생성되지 않았습니다. 구도만으로 해결되지 않는 구현 제약입니다.</p></section><section><h2>우선 수정할 사항</h2><ol><li>운동 선택 시 실제 지원 범위를 표시하고 YTW 자동 평가 미지원을 시작 전에 안내.</li><li>숄더 프레스의 검출/점수 관측 조건을 맞추고, 화면 경계와 누락 관절을 항목별로 안내.</li><li>푸시업 측면 한쪽 관측 허용 여부를 실제 입력·모델 계약에 맞춰 검증.</li><li>스쿼트의 기립 신뢰도와 복귀 심각도 점수 의미를 분리하고 몸통 경계 민감도를 검증.</li></ol><p>이번 분석에서는 모델·원본 결과·서비스 코드를 수정하지 않았습니다. 학습 성능 개선과 독립 최종 평가의 증거로 사용하지 않습니다.</p><small>상세 근거: analysis.json 및 각 세션의 pose.npz / final-state.json</small></section></html>'''
(O/'index.html').write_text(text)
print(json.dumps({'squat_mean':means,'shoulder_rep_diagnostic':sh,'session_coverage':[(r['exercise'],r['required_simultaneous_coverage']) for r in rows],'report':str(O/'index.html')},ensure_ascii=False))

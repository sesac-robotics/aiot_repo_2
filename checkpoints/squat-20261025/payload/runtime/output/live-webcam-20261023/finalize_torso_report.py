"""원본 팀 평가·1차 후보·2차 후보와 민감도 진단을 보존한다."""
import sys,json,importlib.util
from pathlib import Path
import cv2,numpy as np
from front_candidate import predict
from runtime_snapshot import digest,read_source
R=Path(__file__).parent;O=R/'front-candidate';P=R/'recordings/20261023-110200-964840';meta=json.loads((P/'metadata.json').read_text());before=digest(read_source())
spec=importlib.util.spec_from_file_location('preserved_v1',O/'front_candidate_v1.py');v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1);v1.MODEL=O/'model-v1.joblib'
z=np.load(P/'pose.npz');labels=json.loads((P/'labels.frozen.json').read_text());local=[]
for rep in labels['repetitions']:
 a,b=rep['start_frame'],rep['end_frame'];xy=z['image_xy'][a:b+1]
 shifts=[]
 for shift in [-5,0,5]:
  pts=z['image_xy'][a+shift:b+shift+1];old=v1.predict(pts);new=predict(pts)
  shifts.append(dict(shift_frames=shift,v1_ai_assisted=old['ai_assisted_points'],v2_ai_assisted=new['ai_assisted_points'],v1_return=old['return_height_grade'],v2_return=new['return_height_grade']))
 local.append(dict(number=rep['number'],v1=v1.predict(xy),v2=predict(xy),boundary_test=shifts))
sys.path.insert(0,str(R/'runtime'/meta['code_model_sha256']))
from app.scoring.engine import ScoringEngine
from app.scoring.sequence import PoseSequence
engine=ScoringEngine(bundle=R/'runtime'/meta['code_model_sha256']/'artifacts/posture_score')
rows=json.loads((O/'training-rows.json').read_text());torso=json.loads((O/'torso-training.json').read_text());actual=[]
for r in rows:
 zz=np.load(r['cache_path']);cap=cv2.VideoCapture(r['front_source']);w=int(cap.get(3));h=int(cap.get(4));cap.release();ticks=np.rint((zz['time_seconds']-zz['time_seconds'][0])*10).astype(int);xy=np.full((ticks[-1]+1,33,2),np.nan);xy[ticks]=zz['image_xy'];seq=PoseSequence(xy,np.arange(len(xy)),10.,w,h)
 result=engine.score('squats',seq,view='front',complete_attempt=True)
 t=next(x for x in torso['training' if r['split']=='train' else 'validation']['rows'] if x['id']==r['id'])
 actual.append(dict(id=r['id'],split=r['split'],target=t['target'],original_body_ai=result['items'][3]['ai_grade'],original_body_rule=result['items'][3]['rule_grade'],candidate_body_ai=t['ai'],candidate_body_rule=t['new_rule']))
known=[r for r in actual if r['split']=='validation' and r['target'] is not None and r['original_body_rule'] is not None]
metrics={key:float(np.mean([abs(r[key]-r['target']) for r in known])) for key in ['original_body_ai','original_body_rule','candidate_body_ai','candidate_body_rule']}
out=dict(source_preserved=before==digest(read_source())==meta['code_model_sha256'],local=local,original_comparison=actual,validation_body_MAE=metrics,validation_body_cases=len(known),training_error_case_only=next(r for r in actual if r['target']==0),extra_recording=False)
(O/'torso-comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('body validation',metrics);print('training error only',out['training_error_case_only']);print('source preserved',out['source_preserved'])
rows_html=''.join(f"<tr><td>{r['number']}회</td><td>{r['v1']['ai_assisted_points']} → {r['v2']['ai_assisted_points']}</td><td>{r['v2']['rule_points']}</td><td>{r['v2']['return_height_grade']}</td><td>{r['v2']['trunk_grade']}</td><td>{r['v2']['body_ai_diagnostic_grade']}</td></tr>" for r in local)
page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>복귀·몸통 보강 및 3·4회 원인 분석</title><style>body{{font:16px/1.7 sans-serif;background:#f0f5f7;color:#183749;max-width:1120px;margin:30px auto;padding:20px}}section{{padding:24px;background:white;border-radius:14px;margin:20px 0}}h1{{font-size:28px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}strong{{color:#087a75}}a{{color:#087a75}}</style><h1>복귀·몸통 보강 및 3·4회 점수 진단</h1><p>추가 촬영 없음 · 기존 팀 모델·규칙·1차 후보 보존 · 정면 실험 후보 v2</p><section><h2>결론</h2><p><strong>3·4회의 낮은 하강 점수는 무릎 누락만으로 설명되지 않는다.</strong> 무릎을 사용하지 않고 계산한 골반 영상 세로 이동 범위도 3·4회 약 32픽셀, 1·2·5회 약 46~54픽셀이다. 따라서 영상상 움직임 범위가 작다는 근거는 있다. 실제 해부학적 깊이·피로 여부는 이 정면 영상으로 확정하지 않는다.</p><ul><li>무릎 누락: 정상 회차에 4회의 오른쪽 무릎 누락 패턴을 적용해도 점수 변화 0점. 한쪽 무릎만 쓰는 비교도 유효한 구간에서 최대 0.2점 차이. 누락이 낮은 점수의 주원인이라는 증거는 없다.</li><li>구간 경계: 3회의 +0.5초 이동에서 1차 후보는 54.6→40.5점으로 변했다. 복귀 규칙을 강건화한 v2에서는 54.6→50.5점. 복귀 오감점은 제거됐지만 하강 AI의 4.1점 민감도는 남아 있다.</li><li>촬영 영향: 일정 크기 확대와 영상 원점 이동에서 점수가 유지됐다. 고정 배경 패치는 초기 약 4픽셀 이동 뒤 2~5회 비교에서 거의 같은 위치였다. 3·4회에만 큰 카메라 흔들림이 있었다는 근거는 없지만, 위에서 내려다보는 투영과 사람의 앞뒤 이동 영향은 교정·배제하지 못했다.</li></ul></section><section><h2>저장 영상 5회 결과</h2><p>점수는 /70. AI 보조 = 하강 AI + 복귀·몸통 규칙. 몸통 AI는 별도 진단이며 합계에 넣지 않는다.</p><table><tr><th>회차</th><th>AI 보조 v1 → v2</th><th>규칙 v2 /70</th><th>복귀 /2</th><th>몸통 규칙 /2</th><th>몸통 AI 진단 /2</th></tr>{rows_html}</table><p>원래 경계에서는 점수가 그대로 유지됐다. 점수를 높이려는 변경이 아니라 경계를 조금 달리 잡아도 복귀 결과가 유지되게 보강했다. 최초 기존 모델은 하강 AI 모두 0, 복귀·몸통 AI 모두 2, 복귀 규칙 보류였으며, 이 파일·파라미터는 수정하지 않았다.</p></section><section><h2>복귀·몸통에 적용한 보강</h2><ul><li>복귀: 처음·끝 3프레임 대신 각 끝의 최대 1초에서 상단에 가까운 세 관측의 중앙값을 비교한다. 기존 정상 학습 14구간으로 정상 허용 범위를 보정했다. 관측 허용 비율 80%는 유지하고 무릎·발목을 복귀 계산에 요구하지 않는다.</li><li>몸통: 기존 학습 정상 13구간·오류 1구간으로 규칙 경계와 ExtraTrees AI 진단 후보를 학습했다. 새 시각 정답·합성 실패 라벨을 만들지 않았다.</li><li>정상 허용 복귀 경계 {torso['return_boundaries'][0]:.3f} / 부분 경계 {torso['return_boundaries'][1]:.3f} (몸통 길이 비율). 정상 경계는 학습 자료로 보정했고 부분·실패 경계는 공학 가정이다. 실패 정답이 없어 복귀 실패 AI는 훈련하지 않았다.</li><li>몸통 규칙 경계 {torso['body_rule_boundaries'][0]:.2f}/{torso['body_rule_boundaries'][1]:.2f}도. 오류는 한 사람의 한 구간뿐이므로 새 모델·규칙의 오류 일반화 성능은 미검증이다.</li></ul></section><section><h2>기존 모델과 동일 자료 비교</h2><p>기존 개발 검증의 라벨·관측 교집합 {len(known)}구간(한 사람, 모두 몸통 정상)의 0~2등급 MAE: 기존 몸통 AI {metrics['original_body_ai']:.3f}, 기존 몸통 규칙 {metrics['original_body_rule']:.3f}; 새 AI 진단 {metrics['candidate_body_ai']:.3f}, 새 규칙 {metrics['candidate_body_rule']:.3f}.</p><p>기존 AI는 정상만 학습해서 정상 검증에서 오차가 낮아도 오류 구분 증거가 아니다. 학습 오류 1구간은 기존 AI 2등급→새 진단 0등급이지만 학습 사례 재현이며 검증 성능이 아니다. 새로운 AI 진단을 배포 AI나 합계로 승격하지 않았다.</p><p>독립 최종 시험 없음. 저장 5회는 학습에 사용하지 않음. 원본 팀 코드·모델 지문 보존: {out['source_preserved']}. 총 25개 회귀 검사 통과.</p></section><section><h2>근거 파일</h2><p><a href="/candidate">1차 하강 점수 결과</a></p><p>로컬 front-candidate/torso-comparison.json, torso-training.json, sensitivity.json, camera-background-check.json에 전후·민감도·학습 근거를 보존했다.</p></section></html>'''
(O/'torso.html').write_text(page)

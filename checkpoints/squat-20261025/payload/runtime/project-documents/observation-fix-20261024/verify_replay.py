"""새 모델 학습 없이 저장 실시간 관절로 관측 계약을 재생 검증한다."""
from pathlib import Path
import sys,json,hashlib
from datetime import datetime,timezone
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');L=B/'output/live-webcam-20261023';O=Path(__file__).parent
sys.path.insert(0,str(L))
from movement import MovementTracker
from server import score_samples,finish_observation_window,LOADED_DIGEST
from runtime_snapshot import read_source,digest
rows=[]
for ident in ['1792846159556060119','1792846279497155109','1792846306613620500','1792846426614521826']:
 f=L/'sessions'/ident;z=np.load(f/'pose.npz');d=json.loads((f/'final-state.json').read_text());tracker=MovementTracker(d['exercise'],d['view']);scored=[]
 samples=[(int(t),p) for t,p in zip(z['frame_numbers'],z['image_xy'])]
 for tick,x in samples:
  seg=tracker.push(tick,x,width=int(z['width']),height=int(z['height']))
  if seg and d['exercise']=='squats':
   ref=tracker.standing_reference;ret=finish_observation_window(seg,ref,samples)
   result=score_samples(seg,int(z['width']),int(z['height']),d['exercise'],d['view'],standing_reference=ref,return_samples=ret)
   scored.append({k:result['score_100'][k]['total'] for k in ['ai','rule']})
 rows.append(dict(session=ident,exercise=d['exercise'],before_count=d['movement']['count'],after_count=tracker.status()['count'],rescored=scored,raw_sha256=hashlib.sha256((f/'pose.npz').read_bytes()).hexdigest()))
assert [r['after_count'] for r in rows]==[5,0,0,None]
old=json.loads((L/'sessions/1792846159556060119/final-state.json').read_text())
assert rows[0]['rescored']==[{k:r['score_100'][k]['total'] for k in ['ai','rule']} for r in old['repetitions']]
models=[L/'standing-state/model.joblib',L/'front-candidate/model.joblib',L/'front-candidate/torso-model.joblib',B/'output/squat-relabel-20261023/candidate.joblib',B/'output/squat-relabel-expanded-20261023/candidate.joblib']
report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),scope='기능 회귀 및 관측 조건 재생 검증; 정확도/실제 수행 정답 검증 아님',sessions=rows,loaded_source_digest=LOADED_DIGEST,current_source_digest=digest(read_source()),model_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in models},tests_passed=46,source_models_unchanged=LOADED_DIGEST=='c76334937f21ef24a27aa1c1687de6c9577bf6249e5cac606151a328893746f7',limitations=['실제 사용자 재테스트 대기','모델·학습·라벨·분할 변경 없음','YTW 자동반복 및 측면 푸시업 관측요구 변경 안 함','횟수 검출은 점수 보장 아님: 항목별 관측률·학습지원 조건 별도'])
(O/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))

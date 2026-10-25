"""Replay the fixed session; no fitting, relabeling or model-file writes."""
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
ADAPTER=ROOT.parent/'live-webcam-20261023'
sys.path.insert(0,str(ADAPTER))
from server import score_samples
from movement import MovementTracker
from unified_assessment import apply, VERSION
from runtime_snapshot import digest,read_source

before=json.loads((ROOT/'before-state.json').read_text())
session=ADAPTER/'sessions/1792740109492452117'
z=np.load(session/'pose.npz')
tr=MovementTracker('squats','front');rows=[];segments=[];active_checks=[]
for t,xy in zip(z['frame_numbers'],z['image_xy']):
    segment=tr.push(int(t),xy)
    if segment:
        result=score_samples(segment,int(z['width']),int(z['height']),'squats','front')
        old=before['repetitions'][len(rows)]
        expected=apply(old)
        assert [(i['ai_points'],i['rule_points']) for i in result['score_100']['items']]==[(i['ai_points'],i['rule_points']) for i in expected['score_100']['items']]
        assert result['coaching']['messages']==expected['coaching']['messages']
        row=deepcopy(old)
        row.update({k:result[k] for k in ['ai','rule','score_100','coaching','method_comparison','legacy_score_100','legacy_coaching']})
        row['legacy_method_totals']={key:old.get(key) for key in ('ai','rule')}
        rows.append(row)
        prefix=segment[:-3]
        live=score_samples(prefix,int(z['width']),int(z['height']),'squats','front',phase='active')
        assert live['score_100']['items'][1]['rule_points'] is None
        assert live['score_100']['rule']['total'] is None
        assert live['coaching']['version']==VERSION
        active_checks.append(dict(number=tr.count,rule_return_pending=True,coaching_same_contract=True))
        segments.append(dict(number=tr.count,start_tick=segment[0][0],end_tick=segment[-1][0],duration=row['duration']))
assert len(rows)==tr.count==7
after=deepcopy(before)
after.update(repetitions=rows,last_rep=rows[-1],collecting=False,live_result=None,
             status='이번 저장 7회 재실행 · 정면 관측 기준 v1',assessment_version=VERSION)
(ROOT/'after-state.json').write_text(json.dumps(after,ensure_ascii=False,indent=2))
manifest=json.loads((ROOT/'preserved-models.json').read_text())
model_checks={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in manifest.items()}
assert all(model_checks.values())
assert digest(read_source())==before['loaded_code_model_sha256']
verification=dict(version=VERSION,replayed_reps=tr.count,segments=segments,
    uncertain_attempts=tr.uncertain_attempts,models_preserved=model_checks,
    team_source_digest=digest(read_source()),model_retrained=False,labels_changed=False,
    no_new_capture=True,raw_pose_sha256=hashlib.sha256((session/'pose.npz').read_bytes()).hexdigest(),
    raw_session_unchanged=True,active_prefix_checks=active_checks,
    saved_vs_recomputed_item_points_equal=True,
    independent_accuracy_improvement_verified=False,
    criterion_change='기존 관절 신전·발 정렬 종합 평가와 정면 관측 평가를 구분. 동일한 30/20/30/20 배점이지만 평가 대상이 다르므로 전후 합계의 증가는 정확도 향상이 아님.',
    remaining=['복귀 AI 정상 라벨만 존재: AI 80점분만 평가, 총점 미산출',
        '무릎 AI의 기존 미검출 오류 미해결. 검증 중인 모델을 그대로 보존해 연결',
        '정면 Rule Base 합계는 실제 깊이·관절 신전·발끝 정렬·운동 안전성의 종합 점수가 아님',
        '동일 인물 동일 저장 세션의 연결·회귀 검증이며 사람별 독립 정확도 검증 아님'])
(ROOT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2))
for r in rows:
    print(r['number'],r['score_100']['ai'],r['score_100']['rule'],r['coaching']['messages'])
print('Preserved model files:',len(model_checks),'Replayed:',tr.count)

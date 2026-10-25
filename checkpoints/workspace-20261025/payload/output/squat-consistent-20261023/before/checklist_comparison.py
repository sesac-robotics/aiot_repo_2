"""Same-target, same-window experimental comparison. No remote/LLM calls or total."""
from pathlib import Path
from functools import lru_cache
import hashlib
import joblib
import numpy as np
from accepted_items import load
ROOT=Path(__file__).resolve().parent
TITLES={'knee_inward':'무릎 모임','trunk_balance':'몸통 좌우 안정','standing_return':'서기 복귀'}
NAMES={'knee_inward':['시작 대비 무릎 간격 하위값','시작 대비 무릎 간격 중앙값','시작 대비 무릎 간격 상위값','골반 대비 무릎 간격 하위값','골반 대비 무릎 간격 중앙값','골반 대비 무릎 간격 상위값'], 'trunk_balance':['시작 대비 몸통 기울임 하위값','시작 대비 몸통 기울임 중앙값','시작 대비 몸통 기울임 상위값','시작 대비 어깨 기울임 하위값','시작 대비 어깨 기울임 중앙값','시작 대비 어깨 기울임 상위값','몸통 절대 기울임 하위값','몸통 절대 기울임 중앙값','몸통 절대 기울임 상위값']}
@lru_cache(maxsize=1)
def models():
    expanded,extract=load()
    old_path=ROOT.parent/'squat-relabel-20261023/candidate.joblib'
    old=joblib.load(old_path)
    expanded_path=ROOT.parent/'squat-relabel-expanded-20261023/candidate.joblib'
    # Keep the earlier knee candidate: the expanded candidate worsened held-person MAE.
    return {'knee_inward':(old['heads']['knee_inward'],old_path,old['label_sha256']), 'trunk_balance':(expanded['trunk_balance'],expanded_path,joblib.load(expanded_path)['label_sha256'])},extract

def compare_items(candidate,xy=None,evidence=None):
    from method_comparison import tree_path_contributions
    heads,extract=models()
    observed=extract(np.asarray(xy,float)) if xy is not None and len(xy)>=10 else (evidence or {})
    rows=[]
    for key in ('knee_inward','standing_return','trunk_balance'):
        row=dict(item=key,title=TITLES[key],maximum=2,experimental=True,ai_grade=None,rule_grade=None,whole_posture_total=False)
        obs=observed.get(key)
        if key=='standing_return':
            # Normal-only envelope is retained as a policy, never presented as an error-trained AI.
            tf=(candidate or {}).get('torso_features',[])
            if len(tf)==4 and np.isfinite(tf).all():
                b=joblib.load(ROOT/'front-candidate/torso-model.joblib');good,partial=map(float,b['return_boundaries']);v=float(tf[0]);grade=2 if v<=good else 1 if v<=partial else 0
                row.update(rule_grade=grade,rule_evidence=dict(value=v,boundaries=[good,partial],explanation=f'시작·종료 상단 높이 차 {v:.3f}, 정상 허용 경계 {good:.3f}·{partial:.3f}. 무릎 완전 신전의 판정은 아닙니다.'))
            row.update(status='insufficient_error_labels',summary='복귀 AI 비교 미지원: 학습 라벨 31회가 모두 정상이며 복귀 오류가 없습니다. Rule Base는 화면상 높이 복귀만 평가합니다.',training_support={'0':0,'1':0,'2':31})
            rows.append(row);continue
        if not obs:
            row.update(status='unobserved',summary='동일 구간의 관절 특징이 없어 비교하지 않습니다. 무릎은 양쪽 무릎, 몸통은 양 어깨·골반 관측이 필요합니다.')
            rows.append(row);continue
        head,path,labelsha=heads[key];x=np.asarray(obs.get('x',[]),float)
        if x.shape!=(len(NAMES[key]),) or not np.isfinite(x).all() or obs.get('coverage',0)<.8:
            row.update(status='unobserved',summary='관측률 또는 특징 검증 실패: 정상으로 간주하지 않습니다.');rows.append(row);continue
        raw=float(head['model'].predict([x])[0]);ai=float(np.clip(raw,0,2));good,partial=map(float,head['boundaries']);v=float(obs['scalar']);higher=head['higher'];rule=2 if (v>=good if higher else v<=good) else 1 if (v>=partial if higher else v<=partial) else 0
        base,deltas=tree_path_contributions(head['model'],x);assert abs(base+deltas.sum()-raw)<1e-6
        diff=round(round(ai,2)-rule,2);dominant=int(np.argmin(deltas) if diff<0 else np.argmax(deltas));near=min(abs(v-good),abs(v-partial))<=.1*abs(good-partial)
        rn=f'{NAMES[key][0 if key=="knee_inward" else 2]} {v:.3f}, 정상·부분 경계 {good:.3f}·{partial:.3f}에 따라 {rule}등급입니다.'
        mn=f'AI 예측 경로에서 {NAMES[key][dominant]}의 {"하향" if diff<0 else "상향"} 기여가 가장 큽니다. 모델은 {len(x)}개 특징을 함께 사용합니다.'
        row.update(status='different' if diff else 'close',ai_grade=round(ai,2),rule_grade=rule,difference_grade=diff,coverage=obs['coverage'],summary=f'{TITLES[key]}: AI Model {ai:.2f}/2 · Rule Base {rule}/2. '+('단일 경계와 여러 특징의 조합 때문에 출력이 다릅니다.' if diff else '표시 등급이 같습니다.')+(' 규칙 경계 근처입니다.' if near else ''),rule_evidence=dict(value=v,boundaries=[good,partial],higher=higher,near_boundary=near,explanation=rn),model_evidence=dict(baseline=base,prediction=raw,reconstructed=float(base+deltas.sum()),contributions=[dict(name=n,value=float(x[i]),contribution=float(deltas[i])) for i,n in enumerate(NAMES[key])],explanation=mn),model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),label_sha256=labelsha,criterion_version='squat-observation-checklist-v2',training_support=head['train_support'])
        rows.append(row)
    return dict(items=rows,feature_evidence=observed,version='same-target-checklist-v1',automatic=True,gpt_used=False,total=None,limitations=['0·1·2 영상 관측 등급의 개발 검증이며 임상적 정답 또는 100점 총점이 아님','몸통은 좌우 기울임만 평가; 전후 허리 정렬을 판정하지 않음','무릎 간격의 시작 대비 변화를 추정; 발끝 방향 판정과 다름','트리 기여는 계산 설명이며 실제 오류 원인의 확정이 아님'])

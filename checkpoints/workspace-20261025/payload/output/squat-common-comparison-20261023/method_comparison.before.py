"""Deterministic AI/Rule comparison and tree-path attribution. No GPT or external service."""
from functools import lru_cache
from pathlib import Path
import hashlib
import joblib
import numpy as np
MODEL=Path(__file__).parent/'front-candidate/model.joblib'
FEATURE_NAMES=['골반 높이 변화 폭','시작 기준 골반 하강량','무릎–골반 높이 차 하위값','무릎–골반 높이 차 중앙값','허벅지 수직 투영 하위값','허벅지 수직 투영 중앙값','몸통 절대 기울기','시작–종료 골반 높이 차']
@lru_cache(maxsize=1)
def bundle():
    return joblib.load(MODEL),hashlib.sha256(MODEL.read_bytes()).hexdigest()

def tree_path_contributions(model,x):
    """Average root value plus feature split deltas exactly reconstruct forest prediction."""
    x=np.asarray(x,dtype=np.float32)
    contributions=np.zeros(len(x));base=0.
    for estimator in model.estimators_:
        tree=estimator.tree_;node=0;base+=float(tree.value[0].ravel()[0])
        while tree.children_left[node]!=tree.children_right[node]:
            feature=int(tree.feature[node]);child=int(tree.children_left[node] if x[feature]<=tree.threshold[node] else tree.children_right[node])
            contributions[feature]+=float(tree.value[child].ravel()[0]-tree.value[node].ravel()[0]);node=child
    n=len(model.estimators_)
    return base/n,contributions/n

def compare_candidate(candidate):
    result=dict(version='ai-rule-comparison-v1',labels={'ai':'AI Model','rule':'Rule Base'},scope='하강 범위 · 정면 투영 추정',item='descent_range',maximum=30,whole_posture_total=False,gpt_used=False,automatic=True)
    if not candidate or candidate.get('error'):
        return dict(result,status='unavailable',summary='관측이 부족해 두 방식 비교를 기다립니다.')
    x=np.asarray(candidate.get('features',[]),float)
    if x.shape!=(8,) or not np.isfinite(x).all():
        return dict(result,status='unavailable',summary='같은 입력 특징을 확인할 수 없어 비교하지 않습니다.')
    b,sha=bundle();raw=float(b['model'].predict([x])[0]);ai=float(np.clip(raw,0,2));idx=int(b['rule_feature']);low,high=map(float,b['rule_boundaries']);value=float(x[idx]);rule=2 if value>=high else 1 if value>=low else 0
    if abs(ai-float(candidate.get('ai_depth_grade',-99)))>.002 or rule!=candidate.get('rule_depth_grade'):
        return dict(result,status='source_mismatch',summary='저장 점수와 현재 후보의 출력이 일치하지 않아 비교를 보류합니다.',model_sha256=sha)
    base,deltas=tree_path_contributions(b['model'],x);reconstruction=base+float(deltas.sum())
    if abs(raw-reconstruction)>1e-6:
        return dict(result,status='explanation_mismatch',summary='모델 설명 검증에 실패해 비교를 보류합니다.')
    difference=ai-rule;near=min(abs(value-low),abs(value-high))<=.1*(high-low)
    relation='상단 경계 이상' if rule==2 else '두 경계 사이' if rule==1 else '하단 경계 미만'
    rule_reason=f'골반 높이 변화 폭 {value:.3f}이 경계 {low:.3f}·{high:.3f}의 {relation}이어서 {rule}등급입니다.'
    # Only compare displayed points; no unvalidated clinical agreement threshold.
    delta_points=round(round(15*ai,1)-15*rule,1)
    status='different' if delta_points!=0 else 'close'
    direction='작게' if difference<0 else '크게';dominant=int(np.argmin(deltas) if difference<0 else np.argmax(deltas))
    summary=(f'AI Model이 Rule Base보다 {abs(delta_points):.1f}점 {"낮습니다" if delta_points<0 else "높습니다"}. Rule Base는 골반 이동량의 경계로 분류하고, AI Model은 여러 관절 특징을 함께 평가합니다.' if status=='different' else '두 방식의 표시 점수가 같습니다. AI Model은 연속값, Rule Base는 단계 등급을 출력합니다.')
    if near:summary+=' Rule Base의 경계 근처라 등급 차이에 주의하세요.'
    return dict(result,status=status,summary=summary,ai_grade=round(ai,4),rule_grade=rule,ai_points=round(15*ai,1),rule_points=15*rule,difference_points=delta_points,model_sha256=sha,
        rule_evidence=dict(feature_index=idx,name=FEATURE_NAMES[idx],value=value,boundaries=[low,high],near_boundary=near,explanation=rule_reason),
        model_evidence=dict(method='mean_tree_path_contributions',baseline=base,prediction=raw,reconstructed=reconstruction,contributions=[dict(feature_index=i,name=name,value=float(x[i]),contribution=float(deltas[i])) for i,name in enumerate(FEATURE_NAMES)],largest_directional_feature=FEATURE_NAMES[dominant],explanation=f'이번 AI 예측 경로에서 {FEATURE_NAMES[dominant]} 특징이 가장 큰 {"하향" if difference<0 else "상향"} 기여를 했습니다.'),
        limitations=['예측 계산의 차이 설명이며 실제 자세 오류의 인과관계나 어느 방식이 정답인지 확정하지 않음','정면 하강 범위 추정 후보이며 전체 자세 총점 비교가 아님','같은 시간 구간·같은 특징·같은 배점만 수치 비교'],
        excluded_items=[dict(item='standing_return',reason='복귀 오류 AI 검증 부족'),dict(item='trunk_balance',reason='새 몸통 AI와 관측 경고 규칙은 라벨·판정 기준이 달라 총점 비교에 합산하지 않음'),dict(item='knee_inward',reason='동일 기준 AI 미적용'),dict(item='heel_support',reason='Rule Base 추가 항목, 배포 AI 비교 대상 없음'),dict(item='spine_control',reason='정면 관측으로 확정하지 않음')])

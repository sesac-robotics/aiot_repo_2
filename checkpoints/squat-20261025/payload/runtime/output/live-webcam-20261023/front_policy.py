"""기존 평가기를 변경하지 않고 정면에서 근거가 부족한 점수를 보류한다."""
from copy import deepcopy
import numpy as np
VERSION='front-observation-policy-v1'

def apply_front_policy(result, exercise, view):
    """공개 출력 계약을 유지하며 점수 보류 사유와 원본 등급을 추가한다.

    Args:
        result: 현재 팀 ScoringEngine.score의 출력 사전.
        exercise: 기존 운동 코드.
        view: 사용자가 확인한 촬영 방향.

    Returns:
        dict: 원본을 수정하지 않는 정책 적용 사본. 비대상 운동·구도는 동일 값.
    """
    out=deepcopy(result)
    if exercise!='squats' or view!='front': return out
    out['assessment_policy_version']=VERSION
    for item in out['items']:
        item.setdefault('raw_grades',{k:item.get(k) for k in ('ai_grade','rule_grade')})
        reasons=list(item.get('reasons',[]))
        if item['id'] in ('descent_range','return_extension'):
            item['ai_grade']=None
            item['rule_grade']=None
            reasons.append('frontal_sagittal_measurement_unvalidated')
        if item.get('single_grade_training'):
            item['ai_grade']=None
            reasons.append('single_grade_ai_withheld')
        item['reasons']=list(dict.fromkeys(reasons))
        a,b=item.get('ai_grade'),item.get('rule_grade')
        item['ai_minus_rule']=None if a is None or b is None else a-b
    # 항목 하나라도 보류면 만점 환산하지 않고 기존 완결 확인 조건을 유지한다.
    if any(i.get('ai_grade') is None for i in out['items']):out['ai_score']=None
    if any(i.get('rule_grade') is None for i in out['items']):out['rule_score']=None
    return out


def frontal_motion_observations(xy):
    """발목을 요구하지 않고 화면 내 골반 이동을 점수 없이 보고한다."""
    xy=np.asarray(xy,dtype=float)
    req=[11,12,23,24,25,26]
    coverage={str(i):round(float(np.isfinite(xy[:,i]).all(1).mean()),3) for i in req}
    valid=np.isfinite(xy[:,[11,12,23,24]]).all((1,2))
    if valid.sum()<10:return dict(status='insufficient_observation',joint_coverage=coverage)
    shoulder=xy[:,[11,12]].mean(1);hip=xy[:,[23,24]].mean(1)
    scale=float(np.median(np.linalg.norm(hip[valid]-shoulder[valid],axis=1)))
    if scale<1:return dict(status='insufficient_observation',joint_coverage=coverage)
    # 시작·끝의 강건 중앙값 및 관측 범위. 실제 깊이나 해부학적 각도로 변환하지 않는다.
    y=hip[valid,1];start=float(np.median(y[:5]));end=float(np.median(y[-5:]))
    return dict(status='observed_proxy_only',joint_coverage=coverage,
        all_six_fraction=round(float(np.isfinite(xy[:,req]).all((1,2)).mean()),3),
        hip_vertical_range_torso_ratio=round(float((np.percentile(y,95)-np.percentile(y,5))/scale),3),
        return_height_delta_torso_ratio=round((end-start)/scale,3),
        grade=None,limitation='화면 골반 이동·복귀 높이 대리값. 실제 깊이·신전 각도·정답 점수가 아님.')
